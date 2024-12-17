
add_executable(build_engine build_engine.cpp ${CMAKE_SOURCE_DIR}/include/trt_utils.hpp)

target_include_directories(build_engine PRIVATE /usr/local/cuda/include)
target_link_libraries(build_engine PRIVATE absl::base absl::strings absl::status absl::statusor absl::log gflags nvinfer nvinfer_plugin nvonnxparser cuda cudart)
