# fastfaceswapper-tensorrt

The FastFaceSwapper framework aims to provide blazingly fast face-swapping runtimes highly optimized by TensorRT.
The framework consists of two cascaded DNN-based networks: (1) detection network and (2) swapping network.
Currently, this repository provides the swapping part only (input images and pre-computed bounding boxes of human faces are required); we are working hard on adding support for the detection part ASAP (sorry for the inconvenience).

We support [deep_privacy2](https://github.com/hukkelas/deep_privacy2/) models as the swapping network backbone.

## DEMO Instructions

### 1.a Build binaries (docker: recommended)
Create a `.env` file at the repository root and set your preferred version of the official TensorRT docker image. If not set, `24.10-py3` (CUDA 12.6 + TensorRT 10.5) will be used by default.

```bash
# .env
TRT_CONTAINER_VERSION=24.10-py3
```

**Validated versions:**
* 24.10-py3 (CUDA 12.6 + TensorRT 10.5)
* 24.02-py3 (CUDA 12.3 + TensorRT 8.6)
* 22.12-py3 (CUDA 11.8 + TensorRT 8.5)

Then build the selected image and build the binaries:

```bash
# At workspace root
docker build . -t ffswp:latest
docker compose run app bash

# Build binaries (At container workdir root)
mkdir build && cd build && cmake .. && make -j
```

### 1.b Build binaries (local)

```bash
# Install OpenCV & gflags
sudo apt install -y libopencv-dev libgflags-dev

# Install abseil
# NOTE: headers and libraries will be installed at /usr/local/include/ & /usr/local/lib respectively.
# Optionally add paths to them if not set up yet.
git clone https://github.com/abseil/abseil-cpp.git && \
cd abseil-cpp && mkdir build && cd build && \
cmake -DABSL_BUILD_TESTING=OFF -DABSL_USE_GOOGLETEST_HEAD=OFF -DCMAKE_BUILD_TYPE=Release .. && \
make -j && sudo make install

# Install TensorRT
# Supported version: >= 8.5
# See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian for installation

# Build binaries (At workspace root)
mkdir build && cd build && cmake .. && make -j
```

### 2. Download ONNX

The ONNX model is created from the face swapping model (resolution: 128x128) provided by [deep_privacy2](https://github.com/hukkelas/deep_privacy2/tree/master/media).

```bash
mkdir onnx && sh download_onnx.sh
```

### 3. Build TensorRT inference engine from ONNX

```bash
./build/tools/build_engine -onnx_path onnx/dp2.onnx -out_path onnx/dp2.engine
```

### 4. Run face swapping demo

Output images will be created in the directory specified by `-out_dir`.

```bash
./build/runtime/anonymize -engine_path onnx/dp2.engine -data_dir data/test/dataset -out_dir data/test/output
```

*Input* (this picture is taken from [deep_privacy2 official repository](https://github.com/hukkelas/deep_privacy2/tree/master/media))

<p align="center">
  <img src="data/test/dataset/images/regjeringen.jpg" alt="input" width="1280px">
</p>

*Output*

<p align="center">
  <img src="data/test/output/regjeringen.jpg" alt="output" width="1280px">
</p>

## TODOs

* Add detection networks.
* Add a real-time face-swapping demo with a webcam.
* Add docs for integrating the FastFaceSwapper module into your own project.
* Add more swapping networks.
