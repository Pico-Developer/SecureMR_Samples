// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SECUREMR_UTILS_UTILS_H_
#define SECUREMR_UTILS_UTILS_H_

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "serialization.h"
#include "tensor.h"

namespace SecureMR {

struct TensorBinding {
  std::string name;
  std::vector<int> qnnDims;
  std::string qnnType;
  TensorAttribute attr{};
  std::shared_ptr<GlobalTensor> global;
};

struct ModelPackagePipeline {
  Json manifest;
  Json modelJson;
  Json pipelineJson;
  std::shared_ptr<Pipeline> pipeline;
  std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> tensorMap;
  std::unordered_map<std::string, std::shared_ptr<GlobalTensor>> globalTensorMap;
  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> submitBindings;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::string detectionTensor;
};

struct ModelPackagePipelineBundle {
  Json manifest;
  Json modelJson;
  std::unordered_map<std::string, ModelPackagePipeline> pipelines;
  std::unordered_map<std::string, std::shared_ptr<GlobalTensor>> globalTensorMap;
  std::string detectionTensor;
};

class SecureMrUtils {
 public:
  static size_t BytesPerElement(XrSecureMrTensorDataTypePICO dataType);
  static size_t ElementCount(const TensorAttribute& attr);
  static std::optional<Json> LoadModelJson(const std::filesystem::path& jsonPath);
  static bool LoadAssetToBuffer(const std::string& assetPath, std::vector<char>& out, std::string* outError = nullptr);
  static std::optional<Json> LoadJsonAsset(const std::string& assetPath, std::string* outError = nullptr);
  static bool PrepareBindings(const Json& jsonSpec,
                              std::vector<TensorBinding>& inputBindings,
                              std::vector<TensorBinding>& outputBindings,
                              std::string& modelName);
  static bool LoadModelPackagePipelinesFromAssets(
      const std::string& packageAssetRoot,
      const std::shared_ptr<FrameworkSession>& session,
      const std::unordered_map<std::string, std::shared_ptr<GlobalTensor>>& externalGlobals,
      ModelPackagePipelineBundle& outBundle,
      std::string& outError);
};

}  // namespace SecureMR

#endif  // SECUREMR_UTILS_UTILS_H_
