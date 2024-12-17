FROM nvcr.io/nvidia/tensorrt:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    SHELL=/bin/bash
ARG UID=1000
ARG GID=${UID}
ARG UHOME=/home/ubuntu
ARG WORKDIR=${UHOME}/app
ARG PKGDIR=${UHOME}/external_packages
ARG CXX_STANDARD=17

# Update apt repositories & installl dependencies
RUN apt update -y && \
    apt install -y libopencv-dev libgflags-dev cppcheck clang-format bash-completion && \
    rm -rf /var/lib/apt/lists/*

# Install abseil
RUN git clone https://github.com/abseil/abseil-cpp.git ${PKGDIR}/abseil-cpp && \
    cd ${PKGDIR}/abseil-cpp/ && mkdir build && cd build && \
    cmake -DABSL_BUILD_TESTING=OFF -DABSL_USE_GOOGLETEST_HEAD=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=${CXX_STANDARD} .. && \
    make -j && make install

# Change uid & gid of pre-built user "ubuntu" to the same ids of the host user
RUN groupmod -g ${GID} ubuntu && \
    usermod -u ${UID} -g ${GID} ubuntu && \
    chown -R ${UID}:${GID} ${UHOME}/

# Switch to non-root user
USER ubuntu

# Set workdir
WORKDIR ${WORKDIR}
