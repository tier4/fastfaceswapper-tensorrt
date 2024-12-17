
add_executable(build_engine build_engine.cpp)

target_link_libraries(build_engine PRIVATE absl::base absl::strings absl::status absl::statusor absl::log gflags nvinfer nvinfer_plugin nvonnxparser cuda cudart)
