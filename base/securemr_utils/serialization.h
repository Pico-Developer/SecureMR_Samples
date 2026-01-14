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

#ifndef SECUREMR_UTILS_SERIALIZATION_H_
#define SECUREMR_UTILS_SERIALIZATION_H_

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace SecureMR {
struct TensorAttribute;
class FrameworkSession;
class Pipeline;
class PipelineTensor;

using Json = nlohmann::json;

Json TensorAttributeToJson(const TensorAttribute& attr);
Json TensorAttributeVariantToJson(const std::variant<std::monostate, TensorAttribute>& attr);
Json TensorListToJson(const std::vector<std::string>& tensors);
Json MappedTensorListToJson(const std::vector<std::pair<std::string, std::string>>& mapping);
bool WriteJsonToFile(const std::filesystem::path& filePath, const Json& spec);

// Set top-level pipeline inputs/outputs in a JSON spec
void SetInputs(Json& spec, const std::vector<std::string>& inputs);
void SetOutputs(Json& spec, const std::vector<std::string>& outputs);

bool JsonToTensorAttribute(const Json& j, TensorAttribute& out);
std::vector<std::string> ParseTensorList(const Json& arr);
std::vector<std::pair<std::string, std::string>> ParseMappedTensorList(const Json& arr);
bool JsonToFloatArray(const Json& arr, std::array<float, 6>& dest);
Json LoadJsonFromFile(const std::filesystem::path& filePath);
std::string FormatOperatorType(const std::string& typeName);

struct PipelineDeserializationResult {
  std::shared_ptr<Pipeline> pipeline;
  std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> tensorMap;
};

struct PipelineDeserializationOptions {
  std::function<bool(const Json& opSpec,
                     const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor,
                     const std::shared_ptr<Pipeline>& pipeline,
                     std::string& error)>
      customOperatorHandler;
};

bool DeserializePipelineFromJson(const Json& spec,
                                 const std::shared_ptr<FrameworkSession>& session,
                                 PipelineDeserializationResult& outResult,
                                 std::string& outError,
                                 const PipelineDeserializationOptions& options = {});

}  // namespace SecureMR

#endif  // SECUREMR_UTILS_SERIALIZATION_H_
