/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Minimal ONNX-protobuf scanner. See onnx_scan.h for rationale.
 *
 *  All parsing is bounds-checked on every read; on malformed input we
 *  return -EBADMSG rather than deref into undefined memory. The scanner
 *  only descends three fixed levels (Model → Graph → Node → op_type
 *  string); it does not recurse into arbitrary nested messages.
 */

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "onnx_scan.h"
#include "op_allowlist.h"

/* Protobuf wire types (see developers.google.com/protocol-buffers/docs/encoding). */
enum {
    PB_WIRE_VARINT = 0,
    PB_WIRE_64BIT = 1,
    PB_WIRE_LEN = 2,
    PB_WIRE_32BIT = 5,
};

/* ONNX field numbers we care about. */
enum {
    MODEL_GRAPH_FIELD = 7,  /* ModelProto.graph (GraphProto) */
    GRAPH_NODE_FIELD = 1,   /* GraphProto.node (repeated NodeProto) */
    NODE_OP_TYPE_FIELD = 4, /* NodeProto.op_type (string) */
};

/* Varint max length for uint64 is 10 bytes (64 bits / 7 bits-per-byte, rounded up). */
#define PB_VARINT_MAX_BYTES 10

/* Read a varint from buf[*off..len). Advance *off past the varint on
 * success. Returns 0 on success, -EBADMSG on truncated / overlong varint. */
static int pb_read_varint(const unsigned char *buf, size_t len, size_t *off, uint64_t *out)
{
    assert(buf != NULL);
    assert(off != NULL);
    assert(out != NULL);

    uint64_t value = 0;
    unsigned shift = 0;
    size_t i = *off;
    size_t consumed = 0;

    while (consumed < PB_VARINT_MAX_BYTES) {
        if (i >= len) {
            return -EBADMSG;
        }
        const unsigned char byte = buf[i++];
        consumed++;
        value |= (uint64_t)(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0u) {
            *off = i;
            *out = value;
            return 0;
        }
        shift += 7;
        if (shift >= 64u) {
            return -EBADMSG;
        }
    }
    /* Ran past PB_VARINT_MAX_BYTES without hitting a terminator. */
    return -EBADMSG;
}

/* Skip the value portion of a field whose tag we already consumed, given
 * the wire type. Advances *off. Returns 0 or -EBADMSG. */
static int pb_skip_field(const unsigned char *buf, size_t len, size_t *off, unsigned wire_type)
{
    assert(buf != NULL);
    assert(off != NULL);

    switch (wire_type) {
    case PB_WIRE_VARINT: {
        uint64_t scratch = 0;
        return pb_read_varint(buf, len, off, &scratch);
    }
    case PB_WIRE_64BIT: {
        if (*off + 8u > len) {
            return -EBADMSG;
        }
        *off += 8u;
        return 0;
    }
    case PB_WIRE_LEN: {
        uint64_t sub_len = 0;
        const int err = pb_read_varint(buf, len, off, &sub_len);
        if (err != 0) {
            return err;
        }
        if (sub_len > (uint64_t)(len - *off)) {
            return -EBADMSG;
        }
        *off += (size_t)sub_len;
        return 0;
    }
    case PB_WIRE_32BIT: {
        if (*off + 4u > len) {
            return -EBADMSG;
        }
        *off += 4u;
        return 0;
    }
    default:
        /* Groups (wire types 3, 4) are deprecated in proto3 and not emitted
         * by ONNX exporters. Reject rather than attempt to parse them. */
        return -EBADMSG;
    }
}

/* Check one op_type string extracted from a NodeProto. Returns 0 if the
 * op is allowed, -EPERM otherwise. On first rejection populates *first_bad. */
static int check_op_name(const char *op_name, size_t slen, char **first_bad)
{
    if (vmaf_dnn_op_allowed(op_name)) {
        return 0;
    }
    if (first_bad && *first_bad == NULL) {
        char *copy = (char *)malloc(slen + 1u);
        if (!copy) {
            return -ENOMEM;
        }
        memcpy(copy, op_name, slen + 1u);
        *first_bad = copy;
    }
    return -EPERM;
}

/* Read the op_type string from a NodeProto field whose tag was already
 * consumed. Advances *off past the string on success. */
static int read_op_type(const unsigned char *buf, size_t len, size_t *off, char **first_bad)
{
    uint64_t slen = 0;
    int err = pb_read_varint(buf, len, off, &slen);
    if (err != 0) {
        return err;
    }
    if (slen > (uint64_t)(len - *off)) {
        return -EBADMSG;
    }
    /* ONNX op names are short (longest in ONNX opset is ~40 chars). Reject
     * pathologically long names; they aren't a real op we'd allow anyway. */
    if (slen == 0u || slen > 127u) {
        return -EPERM;
    }
    char op_name[128];
    memcpy(op_name, buf + *off, (size_t)slen);
    op_name[slen] = '\0';
    *off += (size_t)slen;
    return check_op_name(op_name, (size_t)slen, first_bad);
}

/* Scan one NodeProto for its op_type field. Returns 0 if all op_types
 * seen are allowed, otherwise propagates the error. */
static int scan_node(const unsigned char *buf, size_t len, char **first_bad)
{
    size_t off = 0;
    while (off < len) {
        uint64_t tag = 0;
        int err = pb_read_varint(buf, len, &off, &tag);
        if (err != 0) {
            return err;
        }
        const unsigned field = (unsigned)(tag >> 3);
        const unsigned wire = (unsigned)(tag & 0x7u);

        if (field == NODE_OP_TYPE_FIELD && wire == PB_WIRE_LEN) {
            err = read_op_type(buf, len, &off, first_bad);
        } else {
            err = pb_skip_field(buf, len, &off, wire);
        }
        if (err != 0) {
            return err;
        }
    }
    return 0;
}

/* Scan one GraphProto: iterate its NodeProto children. */
static int scan_graph(const unsigned char *buf, size_t len, char **first_bad)
{
    size_t off = 0;
    while (off < len) {
        uint64_t tag = 0;
        int err = pb_read_varint(buf, len, &off, &tag);
        if (err != 0) {
            return err;
        }
        const unsigned field = (unsigned)(tag >> 3);
        const unsigned wire = (unsigned)(tag & 0x7u);

        if (field == GRAPH_NODE_FIELD && wire == PB_WIRE_LEN) {
            uint64_t nlen = 0;
            err = pb_read_varint(buf, len, &off, &nlen);
            if (err != 0) {
                return err;
            }
            if (nlen > (uint64_t)(len - off)) {
                return -EBADMSG;
            }
            err = scan_node(buf + off, (size_t)nlen, first_bad);
            if (err != 0) {
                return err;
            }
            off += (size_t)nlen;
        } else {
            err = pb_skip_field(buf, len, &off, wire);
            if (err != 0) {
                return err;
            }
        }
    }
    return 0;
}

int vmaf_dnn_scan_onnx(const unsigned char *buf, size_t len, char **first_bad)
{
    if (!buf || len == 0u) {
        return -EINVAL;
    }
    assert(buf != NULL);
    assert(len > 0u);
    if (first_bad) {
        *first_bad = NULL;
    }

    size_t off = 0;
    int graph_found = 0;
    while (off < len) {
        uint64_t tag = 0;
        int err = pb_read_varint(buf, len, &off, &tag);
        if (err != 0) {
            return err;
        }
        assert(off <= len);
        const unsigned field = (unsigned)(tag >> 3);
        const unsigned wire = (unsigned)(tag & 0x7u);

        if (field == MODEL_GRAPH_FIELD && wire == PB_WIRE_LEN) {
            uint64_t glen = 0;
            err = pb_read_varint(buf, len, &off, &glen);
            if (err != 0) {
                return err;
            }
            if (glen > (uint64_t)(len - off)) {
                return -EBADMSG;
            }
            assert(off + (size_t)glen <= len);
            err = scan_graph(buf + off, (size_t)glen, first_bad);
            if (err != 0) {
                return err;
            }
            graph_found = 1;
            off += (size_t)glen;
        } else {
            err = pb_skip_field(buf, len, &off, wire);
            if (err != 0) {
                return err;
            }
        }
    }
    assert(off <= len);
    return graph_found ? 0 : -ENOENT;
}
