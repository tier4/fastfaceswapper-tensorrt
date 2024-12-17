#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "../include/trt_utils.hpp"

DEFINE_string(onnx_path, "", "Path to the input ONNX file (required).");
DEFINE_string(out_path, "", "Path to the output TensorRT engine file (required).");

typedef struct ParsedArgs {
  std::filesystem::path onnx_path;
  std::filesystem::path out_path;
} ParsedArgs;

absl::StatusOr<ParsedArgs> validateFlags() {
  ParsedArgs args;
  if (FLAGS_onnx_path.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --onnx_path");
  }
  if (FLAGS_out_path.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --out_path");
  }
  if (!std::filesystem::exists(FLAGS_onnx_path)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Could not find ONNX file at %s", FLAGS_onnx_path));
  }
  args.onnx_path = FLAGS_onnx_path;
  args.out_path = FLAGS_out_path;
  return args;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate flags
  const auto args = validateFlags();
  if (!args.ok()) {
    LOG(FATAL) << args.status();
  }
  const auto parsed = args.value();

  // Initialize logger
  auto logger = trt_utils::Logger(nvinfer1::ILogger::Severity::kINFO);

  // Initialize builder
  auto builder = nvinfer1::createInferBuilder(logger);
  const auto explicit_batch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = builder->createNetworkV2(explicit_batch);
  auto config = builder->createBuilderConfig();
  auto parser = nvonnxparser::createParser(*network, logger);
  if (!parser->parseFromFile(parsed.onnx_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
    LOG(FATAL) << "Failed to parse ONNX file";
  }
  auto serialized = builder->buildSerializedNetwork(*network, *config);
  if (!serialized) {
    LOG(FATAL) << "Failed to build TensorRT engine";
  }

  std::ofstream engineFile(parsed.out_path, std::ios::binary);
  engineFile.write(static_cast<char *>(serialized->data()), serialized->size());
  LOG(INFO) << "TensorRT engine saved to: " << parsed.out_path;
}
