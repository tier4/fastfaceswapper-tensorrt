add_executable(fswp fswp.cpp)

target_link_libraries(fswp PRIVATE absl::base absl::strings absl::status absl::statusor absl::log gflags nvinfer nvinfer_plugin nvonnxparser cuda cudart)

