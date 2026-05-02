/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ADR-0238: VmafPicture pool for the public Vulkan preallocation
 *  surface (`VmafVulkanPicturePreallocationMethod`). Mirrors the SYCL
 *  pool in libvmaf/src/sycl/picture_sycl.cpp. Host-method backs each
 *  picture with regular `vmaf_picture_alloc()`; Device-method backs
 *  the luma plane with a host-visible VmafVulkanBuffer (VMA
 *  `AUTO_PREFER_HOST`) so the caller's host writes land in the same
 *  memory the kernel descriptor sets bind.
 */

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/picture.h"
#include "picture.h"
#include "ref.h"

#include "picture_vulkan.h"
#include "vulkan_internal.h"

/* Per-picture release closure: a single VmafVulkanBuffer was attached
 * to pic->data[0]. Stored as the picture's `cookie` so the unref path
 * can free the buffer without going through the pool. */
struct VmafVulkanPicReleaseCookie {
    VmafVulkanContext *ctx;
    VmafVulkanBuffer *buf;
};

static int release_device_picture(VmafPicture *pic, void *cookie)
{
    if (!pic || !cookie)
        return -EINVAL;
    struct VmafVulkanPicReleaseCookie *c = cookie;
    if (c->buf)
        vmaf_vulkan_buffer_free(c->ctx, c->buf);
    free(c);
    pic->data[0] = NULL;
    pic->stride[0] = 0;
    return 0;
}

struct VmafVulkanPicturePool {
    VmafVulkanContext *ctx;
    enum VmafVulkanPoolMethod method;
    enum VmafPixelFormat pix_fmt;
    unsigned w, h, bpc;

    unsigned pic_cnt;
    unsigned curr_idx;
    pthread_mutex_t lock;
    VmafPicture *pic;
};

static int pool_alloc_one_host(VmafVulkanPicturePool *pool, unsigned i)
{
    return vmaf_picture_alloc(&pool->pic[i], pool->pix_fmt, pool->bpc, pool->w, pool->h);
}

static int pool_alloc_one_device(VmafVulkanPicturePool *pool, unsigned i)
{
    assert(pool->ctx != NULL);
    assert(pool->bpc >= 8u && pool->bpc <= 16u);

    const size_t bpp = (pool->bpc + 7u) / 8u;
    const size_t plane_size = (size_t)pool->w * pool->h * bpp;

    VmafVulkanBuffer *buf = NULL;
    int err = vmaf_vulkan_buffer_alloc(pool->ctx, &buf, plane_size);
    if (err)
        return err;

    VmafPicture *pic = &pool->pic[i];
    memset(pic, 0, sizeof(*pic));
    pic->data[0] = vmaf_vulkan_buffer_host(buf);
    pic->stride[0] = (int)((size_t)pool->w * bpp);
    pic->w[0] = pool->w;
    pic->h[0] = pool->h;
    pic->bpc = pool->bpc;
    pic->pix_fmt = pool->pix_fmt;

    VmafPicturePrivate *priv = calloc(1, sizeof(*priv));
    if (!priv) {
        vmaf_vulkan_buffer_free(pool->ctx, buf);
        pic->data[0] = NULL;
        return -ENOMEM;
    }
    priv->buf_type = VMAF_PICTURE_BUFFER_TYPE_VULKAN_DEVICE;
    priv->release_picture = release_device_picture;

    struct VmafVulkanPicReleaseCookie *cookie = calloc(1, sizeof(*cookie));
    if (!cookie) {
        free(priv);
        vmaf_vulkan_buffer_free(pool->ctx, buf);
        pic->data[0] = NULL;
        return -ENOMEM;
    }
    cookie->ctx = pool->ctx;
    cookie->buf = buf;
    priv->cookie = cookie;

    pic->priv = priv;

    err = vmaf_ref_init(&pic->ref);
    if (err) {
        free(cookie);
        free(priv);
        vmaf_vulkan_buffer_free(pool->ctx, buf);
        pic->data[0] = NULL;
        pic->priv = NULL;
        return err;
    }
    return 0;
}

static void pool_unwind(VmafVulkanPicturePool *pool, unsigned up_to)
{
    for (unsigned j = 0u; j < up_to; j++) {
        VmafPicture *p = &pool->pic[j];
        if (pool->method == VMAF_VULKAN_POOL_DEVICE && p->priv) {
            VmafPicturePrivate *priv = (VmafPicturePrivate *)p->priv;
            if (priv->release_picture)
                (void)priv->release_picture(p, priv->cookie);
            free(priv);
            p->priv = NULL;
        } else {
            (void)vmaf_picture_unref(p);
        }
        if (p->ref) {
            vmaf_ref_close(p->ref);
            p->ref = NULL;
        }
    }
}

int vmaf_vulkan_picture_pool_init(VmafVulkanPicturePool **pool_out, VmafVulkanContext *ctx,
                                  unsigned pic_cnt, unsigned w, unsigned h, unsigned bpc,
                                  enum VmafPixelFormat pix_fmt, enum VmafVulkanPoolMethod method)
{
    if (!pool_out || !ctx)
        return -EINVAL;
    if (pic_cnt == 0u || w == 0u || h == 0u)
        return -EINVAL;
    if (bpc < 8u || bpc > 16u)
        return -EINVAL;
    if (method != VMAF_VULKAN_POOL_HOST && method != VMAF_VULKAN_POOL_DEVICE)
        return -EINVAL;

    VmafVulkanPicturePool *pool = calloc(1, sizeof(*pool));
    if (!pool)
        return -ENOMEM;

    pool->ctx = ctx;
    pool->method = method;
    pool->pix_fmt = pix_fmt;
    pool->w = w;
    pool->h = h;
    pool->bpc = bpc;
    pool->pic_cnt = pic_cnt;
    pool->curr_idx = 0u;
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        free(pool);
        return -ENOMEM;
    }

    pool->pic = calloc(pic_cnt, sizeof(*pool->pic));
    if (!pool->pic) {
        pthread_mutex_destroy(&pool->lock);
        free(pool);
        return -ENOMEM;
    }

    for (unsigned i = 0u; i < pic_cnt; i++) {
        const int err = (method == VMAF_VULKAN_POOL_HOST) ? pool_alloc_one_host(pool, i) :
                                                            pool_alloc_one_device(pool, i);
        if (err) {
            pool_unwind(pool, i);
            free(pool->pic);
            pthread_mutex_destroy(&pool->lock);
            free(pool);
            return err;
        }
    }

    *pool_out = pool;
    return 0;
}

int vmaf_vulkan_picture_pool_fetch(VmafVulkanPicturePool *pool, VmafPicture *pic_out)
{
    if (!pool || !pic_out)
        return -EINVAL;

    pthread_mutex_lock(&pool->lock);
    const unsigned idx = pool->curr_idx;
    pool->curr_idx = (pool->curr_idx + 1u) % pool->pic_cnt;
    pthread_mutex_unlock(&pool->lock);

    return vmaf_picture_ref(pic_out, &pool->pic[idx]);
}

int vmaf_vulkan_picture_pool_close(VmafVulkanPicturePool *pool)
{
    if (!pool)
        return 0;

    pool_unwind(pool, pool->pic_cnt);
    free(pool->pic);
    pthread_mutex_destroy(&pool->lock);
    free(pool);
    return 0;
}
