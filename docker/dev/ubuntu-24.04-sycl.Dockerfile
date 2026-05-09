# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu24.04@sha256:295792b8d5d577f70c5469b12d52fd0bf33ee85e31d256613a60ef015269505f

ENV DEBIAN_FRONTEND=noninteractive \
    INSTALL_LINTERS=1 \
    ENABLE_SYCL=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

COPY scripts/setup/ubuntu.sh /tmp/ubuntu.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo \
    && bash /tmp/ubuntu.sh \
    && rm -rf /var/lib/apt/lists/* /tmp/ubuntu.sh

SHELL ["/bin/bash", "-c"]
WORKDIR /src
CMD ["bash", "-lc", "source /opt/intel/oneapi/setvars.sh && exec bash"]
