// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_split.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "tensorrt_utils/tensorrt_utils.hpp"

// Define command line flags
DEFINE_string(onnx_path, "", "[Required] Path to the input ONNX file.");
DEFINE_string(out_path, "", "[Required] Path to the output TensorRT engine file.");
DEFINE_int32(
    opt, 3,
    "[Optional] Set optimization level from 0 to 5. Higher levels enable more "
    "optimizations but take more time. Takes effect only when using TensorRT 8.6 or later.");
DEFINE_bool(sparsity, false, "[Optional] Enable 2:4 sparsity optimization?");
DEFINE_string(dtype, "fp16",
              "[Optional] Precision to use for the engine. Choose one from {fp32, fp16}.");
DEFINE_string(bs, "1,16,32", "[Optional] Minimum, optimal, and maximum batch sizes.");

// Parses a comma-separated string of batch sizes and returns a tuple of three integers.
absl::StatusOr<std::tuple<std::int32_t, std::int32_t, std::int32_t>> parseBatchSizes(
    const std::string &bs) {
  std::vector<std::int32_t> batch_sizes;
  std::vector<std::string> items = absl::StrSplit(bs, ',');
  for (const auto &item : items) {
    try {
      batch_sizes.push_back(std::stoi(item));
    } catch (const std::invalid_argument &e) {
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          absl::StrFormat("Invalid batch size value: %s", item));
    }
  }
  if (batch_sizes.size() != 3) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Batch sizes must contain exactly three values: min, opt, max");
  }
  return std::make_tuple(batch_sizes[0], batch_sizes[1], batch_sizes[2]);
}

// Structure to hold parsed arguments
typedef struct ParsedArgs {
  std::filesystem::path engine_path;
  std::filesystem::path out_path;
  std::int32_t opt;
  bool sparsity;
  std::string dtype;
  std::tuple<std::int32_t, std::int32_t, std::int32_t> bs;
} ParsedArgs;

// Function to validate command line flags
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
  if (FLAGS_opt < 0 || FLAGS_opt > 5) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Optimization level must be between 0 and 5");
  }
  if (FLAGS_dtype != "fp16" && FLAGS_dtype != "fp32") {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Precision must be either 'fp16' or 'fp32'");
  }
  const auto bs = parseBatchSizes(FLAGS_bs);
  if (!bs.ok()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, bs.status().message());
  }
  args.engine_path = FLAGS_onnx_path;
  args.out_path = FLAGS_out_path;
  args.opt = FLAGS_opt;
  args.sparsity = FLAGS_sparsity;
  args.dtype = FLAGS_dtype;
  args.bs = bs.value();
  return args;
}

int main(int argc, char **argv) {
  // Parse command line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate flags
  const auto args = validateFlags();
  if (!args.ok()) {
    LOG(FATAL) << args.status();
  }
  const auto parsed = args.value();

  // Initialize logger
  auto logger = tensorrt_utils::Logger(nvinfer1::ILogger::Severity::kINFO);

  // Initialize TensorRT plugins
  // This function is automatically called in TensorRT 8.6 and later
#if NV_TENSORRT_MAJOR <= 8 && NV_TENSORRT_MINOR <= 5
  initLibNvInferPlugins(&logger, "");
#endif

  // Initialize TensorRT builder
  auto builder = nvinfer1::createInferBuilder(logger);
#if NV_TENSORRT_MAJOR >= 10
  // Since TensorRT 10 implicit batch mode is no longer supported and explicit batch mode is default
  auto network = builder->createNetworkV2(0U);
#else
  // Need to explicitly set kEXPLICIT_BATCH flag for TensorRT 8 and below
  auto network = builder->createNetworkV2(
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif
  auto config = builder->createBuilderConfig();

  // Set optimization level
#if NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 6
  config->setBuilderOptimizationLevel(parsed.opt);
  LOG(INFO) << "Optimization level set to: " << parsed.opt;
#endif

  // Enable sparsity if specified
  if (parsed.sparsity) {
    config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    LOG(INFO) << "2:4 sparsity optimization enabled";
  }

  // Set precision constraints
  config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
  if (parsed.dtype == "fp32") {
    // Do nothing, default is fp32
  } else if (parsed.dtype == "fp16") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  LOG(INFO) << "Precision set to " << parsed.dtype;

  // Parse ONNX model
  auto parser = nvonnxparser::createParser(*network, logger);
  if (!parser->parseFromFile(parsed.engine_path.c_str(),
                             static_cast<std::int32_t>(nvinfer1::ILogger::Severity::kINFO))) {
    LOG(ERROR) << "Failed to parse ONNX file";
    for (std::int32_t i = 0; i < parser->getNbErrors(); ++i) {
      LOG(ERROR) << parser->getError(i)->desc();
    }
    exit(1);
  }

  // Set placeholder dimensions for dynamic batch shapes
  for (std::int32_t i = 0; i < network->getNbInputs(); ++i) {
    auto tensor = network->getInput(i);
    auto shape = tensor->getDimensions();
    LOG(INFO) << absl::StrFormat("Input (%s): %s", tensor->getName(),
                                 tensorrt_utils::dimsToStr(shape));
    if (shape.d[0] > 0) {
      shape.d[0] = -1;
      tensor->setDimensions(shape);
    }
  }

  // Create optimization profile
  auto profile = builder->createOptimizationProfile();

  // Configure dynamic batch shapes
  const auto [minBs, optBs, maxBs] = parsed.bs;
  LOG(INFO) << "Batch sizes set to: " << minBs << ", " << optBs << ", " << maxBs;
  for (std::int32_t i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    auto shape = input->getDimensions();
    shape.d[0] = minBs;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, shape);
    shape.d[0] = optBs;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, shape);
    shape.d[0] = maxBs;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, shape);
  }

  // Set optimization profile
  config->addOptimizationProfile(profile);

  // Build TensorRT engine
  auto serialized = builder->buildSerializedNetwork(*network, *config);
  if (!serialized) {
    LOG(FATAL) << "Failed to build TensorRT engine";
  }

  // Save the engine to file
  std::ofstream engineFile(parsed.out_path, std::ios::binary);
  engineFile.write(static_cast<char *>(serialized->data()), serialized->size());
  LOG(INFO) << "TensorRT engine saved to: " << parsed.out_path;
}
