ARG TRT_CONTAINER_VERSION=24.10-py3
FROM nvcr.io/nvidia/tensorrt:${TRT_CONTAINER_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    SHELL=/bin/bash
ARG UID=1000
ARG GID=${UID}
ARG UNAME=ffswp
ARG UHOME=/home/${UNAME}
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

# Create or modify a non-root user based on the provided UID/GID
RUN set -eux; \
    # Check if a user with the specified UID/GID already exists
    EXISTING_USER=$(getent passwd ${UID} | cut -d: -f1 || true); \
    EXISTING_GROUP=$(getent group ${GID} | cut -d: -f1 || true); \
    if [ -n "${EXISTING_USER}" ]; then \
        usermod -l ${UNAME} ${EXISTING_USER}; \
        groupmod -n ${UNAME} ${EXISTING_GROUP}; \
    else \
        groupadd -g ${GID} ${UNAME}; \
        useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME}; \
    fi; \
    cp /etc/skel/.bashrc ${UHOME}/.bashrc; \
    chown -R ${UID}:${GID} ${UHOME};

# Switch to non-root user
USER ${UNAME}

# Set workdir
WORKDIR ${WORKDIR}
