# fastfaceswapper-tensorrt

## DEMO Instructions

### 1. Build docker image (it would take < 10mins)

```bash
# At workspace root
docker build . -t ffswp:latest
```

### 2. Launch container and enter it

```bash
# At workspace root
docker compose run app bash
```

### 3. Build binaries

```bash
# At container workdir root
mkdir build && cd build && cmake .. && make -j
```

### 4. Download onnx

The onnx is created from the face swapping model (resolution: 128x128) provided by [deep_privacy2](https://github.com/hukkelas/deep_privacy2/tree/master/media).

```bash
# At container workdir root
mkdir onnx && sh download_onnx.sh
```

### 5. Build tensorRT inference engine from onnx

```bash
# At container workdir root
./build/tools/build_engine -onnx_path onnx/dp2.onnx -out_path onnx/dp2.engine
```

### 6. Run face swapping demo

Output images will be created in the directory specified by `-out_dir`.

```bash
# At container workdir root
./build/runtime/anonymize -engine_path onnx/dp2.engine -data_dir data/test/dataset -out_dir data/test/output
```

Input (this picture is taken from [deep_privacy2 official repository](https://github.com/hukkelas/deep_privacy2/tree/master/media))

<p align="center">
  <img src="data/test/dataset/images/regjeringen.jpg" alt="input" widt="1280px">
</p>

Output

<p align="center">
  <img src="data/test/output/regjeringen.jpg" alt="input" widt="1280px">
</p>


