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

#include "securemr_utils/utils.h"

#include <exception>
#include <limits>
#include <utility>

#include "common.h"
#include "logger.h"

namespace SecureMR {
namespace {

std::string JoinInts(const std::vector<int>& values) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    out.append(std::to_string(values[i]));
    if (i + 1 < values.size()) {
      out.push_back('x');
    }
  }
  return out;
}

XrSecureMrTensorDataTypePICO MapQnnType(const std::string& type, bool& warnedFloat16) {
  if (type == "QNN_DATATYPE_FLOAT_32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "QNN_DATATYPE_FLOAT_16") {
    warnedFloat16 = true;
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "QNN_DATATYPE_INT_32") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;
  }
  if (type == "QNN_DATATYPE_INT_16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO;
  }
  if (type == "QNN_DATATYPE_INT_8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO;
  }
  if (type == "QNN_DATATYPE_UINT_16") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO;
  }
  if (type == "QNN_DATATYPE_UINT_8") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO;
  }
  return XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO;
}

std::optional<TensorBinding> BuildBinding(const Json& info) {
  if (!info.is_object()) {
    return std::nullopt;
  }

  TensorBinding binding;
  if (auto nameIt = info.find("name"); nameIt != info.end() && nameIt->is_string()) {
    binding.name = nameIt->get<std::string>();
  }
  if (auto typeIt = info.find("dataType"); typeIt != info.end() && typeIt->is_string()) {
    binding.qnnType = typeIt->get<std::string>();
  }
  if (auto dimsIt = info.find("dimensions"); dimsIt != info.end() && dimsIt->is_array()) {
    for (const auto& dim : *dimsIt) {
      if (dim.is_number_integer()) {
        binding.qnnDims.push_back(dim.get<int>());
      }
    }
  }
  if (binding.name.empty() || binding.qnnType.empty() || binding.qnnDims.empty()) {
    return std::nullopt;
  }

  bool warnedFloat16 = false;
  binding.attr.dataType = MapQnnType(binding.qnnType, warnedFloat16);
  if (binding.attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO) {
    Log::Write(Log::Level::Error,
               Fmt("Tensor %s has unsupported data type %s", binding.name.c_str(), binding.qnnType.c_str()));
    return std::nullopt;
  }

  std::vector<int> dims = binding.qnnDims;
  if (dims.size() > 1 && dims.front() == 1) {
    dims.erase(dims.begin());
  }

  int channels = 1;
  if (dims.size() >= 2) {
    channels = dims.back();
    dims.pop_back();
  } else if (!dims.empty()) {
    channels = 1;
  }
  if (dims.empty()) {
    dims.push_back(1);
  }

  if (channels <= 0 || channels > std::numeric_limits<int8_t>::max()) {
    Log::Write(Log::Level::Error,
               Fmt("Tensor %s has unsupported channel count %d", binding.name.c_str(), channels));
    return std::nullopt;
  }

  if (channels > 4) {
    dims.push_back(channels);
    channels = 1;
  }

  binding.attr.dimensions = dims;
  binding.attr.channels = static_cast<int8_t>(channels);

  if (binding.attr.dimensions.size() <= 1 && binding.attr.channels == 1) {
    binding.attr.usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO;
  } else {
    binding.attr.usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  }

  if (binding.attr.usage == XR_SECURE_MR_TENSOR_TYPE_MAT_PICO && binding.attr.dimensions.size() < 2) {
    binding.attr.dimensions.insert(binding.attr.dimensions.begin(), 1);
    Log::Write(Log::Level::Warning,
               Fmt("Tensor %s mapped to MAT but had 1 dimension; promoting shape to 1x%d to satisfy MAT requirements",
                   binding.name.c_str(), binding.attr.dimensions.back()));
  }

  if (warnedFloat16) {
    Log::Write(Log::Level::Warning,
               Fmt("Tensor %s uses QNN float16; mapping to FLOAT32 for SecureMR tensor", binding.name.c_str()));
  }

  Log::Write(Log::Level::Info,
             Fmt("Tensor %s | qnn dims=%s type=%s -> attr dims=%s channels=%d type=%d",
                 binding.name.c_str(), JoinInts(binding.qnnDims).c_str(), binding.qnnType.c_str(),
                 JoinInts(binding.attr.dimensions).c_str(), binding.attr.channels, binding.attr.dataType));

  return binding;
}

bool ParseBindings(const Json& graphInfo, const char* key, std::vector<TensorBinding>& outBindings) {
  auto tensorsIt = graphInfo.find(key);
  if (tensorsIt == graphInfo.end() || !tensorsIt->is_array()) {
    Log::Write(Log::Level::Error, Fmt("Model JSON missing %s array", key));
    return false;
  }

  for (const auto& entry : *tensorsIt) {
    if (!entry.is_object()) {
      continue;
    }
    auto infoIt = entry.find("info");
    if (infoIt == entry.end()) {
      continue;
    }
    auto binding = BuildBinding(*infoIt);
    if (binding.has_value()) {
      outBindings.emplace_back(std::move(*binding));
    }
  }
  if (outBindings.empty()) {
    Log::Write(Log::Level::Error, Fmt("No valid %s entries found", key));
    return false;
  }
  return true;
}

}  // namespace

size_t SecureMrUtils::BytesPerElement(XrSecureMrTensorDataTypePICO dataType) {
  switch (dataType) {
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO:
      return 1;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
      return 2;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO:
      return 4;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO:
      return 8;
    default:
      return 0;
  }
}

size_t SecureMrUtils::ElementCount(const TensorAttribute& attr) {
  size_t count = 1;
  for (int dim : attr.dimensions) {
    count *= static_cast<size_t>(dim);
  }
  count *= static_cast<size_t>(attr.channels);
  return count;
}

std::optional<Json> SecureMrUtils::LoadModelJson(const std::filesystem::path& jsonPath) {
  try {
    return LoadJsonFromFile(jsonPath);
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("Failed to read %s: %s", jsonPath.string().c_str(), e.what()));
    return std::nullopt;
  }
}

bool SecureMrUtils::PrepareBindings(const Json& jsonSpec,
                                    std::vector<TensorBinding>& inputBindings,
                                    std::vector<TensorBinding>& outputBindings,
                                    std::string& modelName) {
  inputBindings.clear();
  outputBindings.clear();

  auto infoIt = jsonSpec.find("info");
  if (infoIt == jsonSpec.end() || !infoIt->is_object()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON missing top-level info");
    return false;
  }
  auto graphsIt = infoIt->find("graphs");
  if (graphsIt == infoIt->end() || !graphsIt->is_array() || graphsIt->empty()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON missing graphs array");
    return false;
  }
  const auto& graph = (*graphsIt)[0];
  auto graphInfoIt = graph.find("info");
  if (graphInfoIt == graph.end() || !graphInfoIt->is_object()) {
    Log::Write(Log::Level::Error, "ModelInspect: model JSON graph missing info");
    return false;
  }
  if (auto nameIt = graphInfoIt->find("graphName"); nameIt != graphInfoIt->end() && nameIt->is_string()) {
    modelName = nameIt->get<std::string>();
  }

  if (!ParseBindings(*graphInfoIt, "graphInputs", inputBindings)) {
    return false;
  }
  if (!ParseBindings(*graphInfoIt, "graphOutputs", outputBindings)) {
    return false;
  }
  return true;
}

}  // namespace SecureMR
