# Documentation

This is an overview of the available documentation in the VMAF repository.

## Getting started

- [Installation](getting-started/install/) – per-OS install scripts (Ubuntu, Fedora, Arch, Alpine, macOS, Windows)
- [Building on Windows](getting-started/building-on-windows.md) – build VMAF from source on Windows (upstream guide)

## Usage

- [Python library](usage/python.md) – explains the Python wrapper for VMAF
- [FFmpeg](usage/ffmpeg.md) – how to use VMAF in conjunction with FFmpeg
- [Docker](usage/docker.md) – how to run VMAF with Docker
- [MATLAB](usage/matlab.md) – running other quality algorithms (ST-RRED, ST-MAD, SpEED-QA, and BRISQUE) with MATLAB
- [External resources](usage/external-resources.md) – e.g. software using VMAF

## Metrics

- [Features](metrics/features.md) – VMAF's core features (metrics)
- [CAMBI](metrics/cambi.md) – contrast-aware multiscale banding index
- [Confidence Interval](metrics/confidence-interval.md) – bootstrapping for CI estimates of VMAF scores
- [Bad Cases](metrics/bad-cases.md) – how to report cases of VMAF not working well
- [AOM CTC](metrics/ctc/aom.md) – running VMAF under [AOM](http://aomedia.org/) common test conditions
- [NFLX CTC](metrics/ctc/nflx.md) – running VMAF under NFLX common test conditions

## Models

- [Overview](models/overview.md) – summary of the available pre-trained models
- [Datasets](models/datasets.md) – the two publicly available datasets for training custom models

## Backends

GPU / SIMD backend notes under [backends/](backends/):

- [x86 SIMD (AVX2 / AVX-512)](backends/x86/avx512.md)
- [CUDA](backends/cuda/overview.md) + [NVTX profiling](backends/nvtx/profiling.md)
- [SYCL / oneAPI](backends/sycl/overview.md) + [self-contained bundling](backends/sycl/bundling.md)

## Tiny-AI

- [Tiny-AI docs](ai/) – overview, training, inference, benchmarks, security

## Development

- [Engineering principles](principles.md) – NASA Power-of-10 + JPL + CERT + MISRA, golden gate, quality policy
- [Release](development/release.md) – how to perform a new release

## Reference

- [FAQ](reference/faq.md)
- [References](reference/references.md) – a list of links and papers
- [Papers](reference/papers/) and [Presentations](reference/presentations/)
