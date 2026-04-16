# Base: NVIDIA CUDA 13.0.2 devel on Ubuntu 24.04. Digest-pinned for
# reproducible builds. Gives us nvcc + cudart-dev without Ubuntu's
# stale 'nvidia-cuda-toolkit' apt package (22.04 shipped CUDA 11.5).
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a

ARG NV_CODEC_TAG="876af32a202d0de83bd1d36fe74ee0f7fcf86b0d"
ARG FFMPEG_TAG=n8.1
# Broadened gencode: Turing baseline (sm_75) + Ampere (sm_80) + Hopper
# (sm_90) + Blackwell consumer (sm_120). CUDA 13 dropped sm_50/60/70.
ARG NVCC_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 -O2"
ARG ENABLE_SYCL=false

ENV DEBIAN_FRONTEND=noninteractive

# ---------- system dependencies ----------
# CUDA toolkit + cudart are already present in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    nasm \
    doxygen \
    python3 \
    python3-pip \
    python3-venv \
    xxd \
    clang \
    wget \
    unzip \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------- Intel oneAPI (SYCL, optional) ----------
RUN if [ "$ENABLE_SYCL" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gnupg2 ca-certificates && \
        wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
        echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
        apt-get update && apt-get install -y --no-install-recommends \
            intel-oneapi-compiler-dpcpp-cpp-2025.3 \
            intel-oneapi-runtime-libs-2025.3 \
            libva-dev \
            libva-drm2 \
            level-zero-dev && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ---------- nv-codec-headers ----------
RUN wget -q https://github.com/FFmpeg/nv-codec-headers/archive/${NV_CODEC_TAG}.zip && \
    unzip -q ${NV_CODEC_TAG}.zip && \
    cd nv-codec-headers-${NV_CODEC_TAG} && make && make install && \
    cd / && rm -rf ${NV_CODEC_TAG}.zip nv-codec-headers-${NV_CODEC_TAG}

# ---------- build libvmaf ----------
COPY . /vmaf
WORKDIR /vmaf
ENV PATH=/vmaf:/vmaf/libvmaf/build/tools:$PATH

# when disabling NVCC, CUDA kernels are JIT-compiled at runtime
RUN make clean && make ENABLE_NVCC=true && make install

# ---------- build FFmpeg ----------
RUN wget -q https://github.com/FFmpeg/FFmpeg/archive/${FFMPEG_TAG}.zip && \
    unzip -q ${FFMPEG_TAG}.zip && rm ${FFMPEG_TAG}.zip

COPY patches/ffmpeg-libvmaf-sycl.patch /tmp/
RUN cd FFmpeg-${FFMPEG_TAG} && \
    (git apply /tmp/ffmpeg-libvmaf-sycl.patch 2>/dev/null || patch -p1 < /tmp/ffmpeg-libvmaf-sycl.patch)

RUN SYCL_FLAG="" && \
    if [ "$ENABLE_SYCL" = "true" ]; then SYCL_FLAG="--enable-libvmaf-sycl"; fi && \
    cd FFmpeg-${FFMPEG_TAG} && ./configure \
        --enable-libnpp \
        --enable-nonfree \
        --enable-nvdec \
        --enable-nvenc \
        --enable-cuvid \
        --enable-cuda \
        --enable-cuda-nvcc \
        --enable-libvmaf \
        --enable-ffnvcodec \
        --disable-stripping \
        --nvccflags="${NVCC_FLAGS}" \
        ${SYCL_FLAG} && \
    make -j$(nproc) && \
    make install

# ---------- python tools ----------
RUN pip3 install --no-cache-dir -r /vmaf/python/requirements.txt
ENV PYTHONPATH=python

RUN mkdir -p /data
ENTRYPOINT ["ffmpeg"]
