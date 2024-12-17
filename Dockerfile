FROM nvcr.io/nvidia/tensorrt:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    SHELL=/bin/bash
ARG UID=1000
ARG GID=${UID}
ARG UHOME=/home/ubuntu
ARG WORKDIR=${UHOME}/app

# Update apt repositories & installl dependencies
RUN apt update -y && \
    apt install -y libopencv-dev cppcheck clang-format bash-completion && \
    rm -rf /var/lib/apt/lists/*

# Change uid & gid of pre-built user "ubuntu" to the same ids of the host user
RUN groupmod -g ${GID} ubuntu && \
    usermod -u ${UID} -g ${GID} ubuntu && \
    chown -R ${UID}:${GID} ${UHOME}/

# Switch to non-root user
USER ubuntu

# Set workdir
WORKDIR ${WORKDIR}
