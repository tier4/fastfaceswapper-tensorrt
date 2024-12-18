#include <gflags/gflags.h>

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
  tensorrt_utils::InferenceEngine engine(parsed.engine_path, logger);

  return 0;
}
