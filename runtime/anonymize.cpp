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

#include <gflags/gflags.h>

#include <ffswp/ffswp.hpp>
#include <filesystem>
#include <iostream>
#include <tensorrt_utils/inference_engine.hpp>

DEFINE_string(engine_path, "", "Path to the input tensorRT engine (required).");

// Structure to hold parsed arguments
typedef struct ParsedArgs {
  std::filesystem::path engine_path;
} ParsedArgs;

// Function to validate command line flags
absl::StatusOr<ParsedArgs> validateFlags() {
  ParsedArgs args;
  if (FLAGS_engine_path.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Missing required flag --engine_path");
  }
  if (!std::filesystem::exists(FLAGS_engine_path)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrFormat("Could not find tensorRT engine at %s", FLAGS_engine_path));
  }
  args.engine_path = FLAGS_engine_path;
  return args;
}

int main(int argc, char** argv) {
  // Parse command line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate flags
  const auto args = validateFlags();
  if (!args.ok()) {
    LOG(FATAL) << args.status();
  }
  const auto parsed = args.value();

  // Initialize logger
  tensorrt_utils::Logger logger(nvinfer1::ILogger::Severity::kINFO);

  // Create InferenceEngine object
  ffswp::FastFaceSwapper faceSwapper(parsed.engine_path, logger);
  auto imgSize = faceSwapper.getImgSize();
  auto imgC = faceSwapper.getImgChannels();
  LOG(INFO) << "Image size: " << imgSize;
  LOG(INFO) << "Image channels: " << imgC;

  return 0;
}
