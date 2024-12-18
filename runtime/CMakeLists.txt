add_executable(anonymize anonymize.cpp)

target_link_libraries(anonymize PRIVATE absl::base absl::strings absl::status absl::statusor absl::log gflags nvinfer nvinfer_plugin nvonnxparser cuda cudart)

