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

#include "securemr_utils/serialization.h"

#include <fstream>
#include <algorithm>
#include <cctype>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <system_error>
#include <stdexcept>
#include <unordered_map>

#include "oxr_utils/common.h"
#ifndef SECUREMR_SERIALIZATION_PARSE_ONLY
#include "oxr_utils/logger.h"
#include "pipeline.h"
#include "rendercommand.h"
#endif
#include "tensor.h"

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/asset_manager.h>
extern AAssetManager* g_assetManager;
#endif

namespace SecureMR {

namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\n\r");
  return value.substr(first, last - first + 1);
}

XrSecureMrTensorDataTypePICO ParseDataType(
    const Json& value,
    XrSecureMrTensorDataTypePICO defaultType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO) {
  if (value.is_number_integer()) {
    return static_cast<XrSecureMrTensorDataTypePICO>(value.get<int>());
  }
  if (!value.is_string()) {
    return defaultType;
  }
  const std::string type = ToLower(value.get<std::string>());
  if (type == "float32" || type == "fp32" || type == "float" || type == "securemrtensordatatype.float") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  }
  if (type == "float64" || type == "double" || type == "securemrtensordatatype.double") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO;
  }
  if (type == "int32" || type == "int" || type == "securemrtensordatatype.int") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;
  }
  if (type == "int16" || type == "short" || type == "securemrtensordatatype.short") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO;
  }
  if (type == "int8" || type == "sbyte" || type == "securemrtensordatatype.sbyte") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO;
  }
  if (type == "uint16" || type == "ushort" || type == "securemrtensordatatype.ushort") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO;
  }
  if (type == "uint8" || type == "byte" || type == "securemrtensordatatype.byte") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO;
  }
  if (type == "dynamic_texture_byte" || type == "dynamic_texture_uint8" ||
      type == "securemrtensordatatype.dynamictexturebyte") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO;
  }
  if (type == "dynamic_texture_float" || type == "dynamic_texture_float32" ||
      type == "securemrtensordatatype.dynamictexturefloat") {
    return XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO;
  }
  return defaultType;
}

XrSecureMrTensorTypePICO ParseTensorUsage(const Json& value,
                                          XrSecureMrTensorTypePICO defaultUsage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO) {
  if (value.is_number_integer()) {
    return static_cast<XrSecureMrTensorTypePICO>(value.get<int>());
  }
  if (!value.is_string()) {
    return defaultUsage;
  }
  const std::string usage = ToLower(value.get<std::string>());
  if (usage == "point" || usage == "xr_secure_mr_tensor_type_point_pico") return XR_SECURE_MR_TENSOR_TYPE_POINT_PICO;
  if (usage == "scalar" || usage == "xr_secure_mr_tensor_type_scalar_pico") {
    return XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO;
  }
  if (usage == "slice" || usage == "xr_secure_mr_tensor_type_slice_pico") return XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO;
  if (usage == "color" || usage == "xr_secure_mr_tensor_type_color_pico") return XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO;
  if (usage == "timestamp" || usage == "xr_secure_mr_tensor_type_timestamp_pico") {
    return XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO;
  }
  if (usage == "mat" || usage == "matrix" || usage == "xr_secure_mr_tensor_type_mat_pico") {
    return XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  }
  if (usage == "gltf" || usage == "xr_secure_mr_tensor_type_gltf_pico") return XR_SECURE_MR_TENSOR_TYPE_GLTF_PICO;
  if (usage == "dynamic_texture" || usage == "mat_dynamic_texture" ||
      usage == "xr_secure_mr_tensor_type_mat_dynamic_texture_pico") {
    return XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO;
  }
  return defaultUsage;
}

bool JsonToSpecialTensorAttribute(const Json& j, TensorAttribute& out) {
  auto typeIt = j.find("tensor_type");
  if (typeIt == j.end() || !typeIt->is_string()) {
    return false;
  }
  const std::string tensorType = ToLower(typeIt->get<std::string>());
  const size_t size = j.value("size", 1);
  const auto dataTypeIt = j.find("data_type");
  const XrSecureMrTensorDataTypePICO dataType =
      dataTypeIt == j.end() ? XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO : ParseDataType(*dataTypeIt);

  if (tensorType == "scalar_array") {
    out = TensorAttribute_ScalarArray{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "point2_array") {
    out = TensorAttribute_Point2Array{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "point3_array") {
    out = TensorAttribute_Point3Array{.size = size, .dataType = dataType};
    return true;
  }
  if (tensorType == "rgba_array") {
    out = TensorAttribute_RGBA_Array{.size = size};
    return true;
  }
  if (tensorType == "rgb_array") {
    out = TensorAttribute_RGB_Array{.size = size};
    return true;
  }
  if (tensorType == "slice_array") {
    out = TensorAttribute_SliceArray{.size = size,
                                     .hasSkip = j.value("has_skip", false),
                                     .dataType = dataType};
    return true;
  }
  if (tensorType == "timestamp") {
    out = TensorAttribute_TimeStamp{};
    return true;
  }
  if (tensorType == "dynamic_texture" || tensorType == "dynamic_texture_byte" ||
      tensorType == "dynamic_texture_uint8" || tensorType == "dynamic_texture_float" ||
      tensorType == "dynamic_texture_float32") {
    std::vector<int> dimensions;
    if (auto dimsIt = j.find("dimensions"); dimsIt != j.end() && dimsIt->is_array()) {
      for (const auto& dim : *dimsIt) {
        if (!dim.is_number_integer()) {
          return false;
        }
        dimensions.push_back(dim.get<int>());
      }
    } else {
      dimensions.push_back(static_cast<int>(size));
    }
    out = TensorAttribute{.dimensions = dimensions,
                          .channels = static_cast<int8_t>(j.value("channels", 1)),
                          .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO,
                          .dataType = tensorType == "dynamic_texture_float" || tensorType == "dynamic_texture_float32"
                                          ? XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO
                                          : dataType};
    if (out.dataType != XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO &&
        out.dataType != XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO) {
      out.dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO;
    }
    return true;
  }
  return false;
}

#ifndef SECUREMR_SERIALIZATION_PARSE_ONLY

struct TensorReference {
  std::shared_ptr<PipelineTensor> tensor;
  std::optional<PipelineTensor::Slice> slice;
};

TensorReference ResolveTensorReference(
    const std::string& expression,
    const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  const std::string ref = Trim(expression);
  const auto open = ref.find('[');
  if (open == std::string::npos) {
    return {.tensor = requireTensor(ref), .slice = std::nullopt};
  }
  if (ref.empty() || ref.back() != ']') {
    throw std::runtime_error(Fmt("malformed tensor reference '%s'", expression.c_str()));
  }

  const std::string tensorName = Trim(ref.substr(0, open));
  auto tensor = requireTensor(tensorName);
  const std::string sliceSpec = ref.substr(open + 1, ref.size() - open - 2);
  if (sliceSpec.find(':') == std::string::npos && sliceSpec.find(',') == std::string::npos) {
    return {.tensor = tensor, .slice = (*tensor)[std::stoi(Trim(sliceSpec))]};
  }

  std::vector<std::vector<int>> dims;
  size_t start = 0;
  while (start <= sliceSpec.size()) {
    const size_t comma = sliceSpec.find(',', start);
    const std::string part =
        Trim(sliceSpec.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
    std::vector<int> values;
    size_t partStart = 0;
    while (partStart <= part.size()) {
      const size_t colon = part.find(':', partStart);
      const std::string number =
          Trim(part.substr(partStart, colon == std::string::npos ? std::string::npos : colon - partStart));
      values.push_back(std::stoi(number));
      if (colon == std::string::npos) break;
      partStart = colon + 1;
    }
    dims.push_back(values);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return {.tensor = tensor, .slice = (*tensor)[dims]};
}

Pipeline::ElementwiseOp ParseElementwiseOp(const std::string& op) {
  const std::string value = ToLower(op);
  if (value == "min") return Pipeline::ElementwiseOp::MIN;
  if (value == "max") return Pipeline::ElementwiseOp::MAX;
  if (value == "multiply" || value == "mul") return Pipeline::ElementwiseOp::MULTIPLY;
  if (value == "or") return Pipeline::ElementwiseOp::OR;
  if (value == "and") return Pipeline::ElementwiseOp::AND;
  throw std::runtime_error(Fmt("unsupported elementwise op '%s'", op.c_str()));
}

XrSecureMrComparisonPICO ParseComparison(const std::string& compare) {
  const std::string value = ToLower(compare);
  if (value.empty() || value == ">" || value == "gt" || value == "larger_than" || value == "greater_than") {
    return XR_SECURE_MR_COMPARISON_LARGER_THAN_PICO;
  }
  if (value == ">=" || value == "ge" || value == "larger_or_equal" || value == "greater_or_equal" ||
      value == "larger_equal" || value == "greater_equal") {
    return XR_SECURE_MR_COMPARISON_LARGER_OR_EQUAL_PICO;
  }
  if (value == "<" || value == "lt" || value == "smaller_than" || value == "less_than") {
    return XR_SECURE_MR_COMPARISON_SMALLER_THAN_PICO;
  }
  if (value == "<=" || value == "le" || value == "smaller_or_equal" || value == "less_or_equal" ||
      value == "smaller_equal" || value == "less_equal") {
    return XR_SECURE_MR_COMPARISON_SMALLER_OR_EQUAL_PICO;
  }
  if (value == "==" || value == "eq" || value == "equal" || value == "equal_to") {
    return XR_SECURE_MR_COMPARISON_EQUAL_TO_PICO;
  }
  if (value == "!=" || value == "ne" || value == "not_equal" || value == "not_equal_to") {
    return XR_SECURE_MR_COMPARISON_NOT_EQUAL_PICO;
  }
  throw std::runtime_error(Fmt("unsupported comparison '%s'", compare.c_str()));
}

XrSecureMrModelTypePICO ParseModelType(const std::string& modelType) {
  const std::string value = ToLower(modelType);
  if (value.empty() || value == "qnn" || value == "qnn_context_binary" || value == "qnn_context_binary_pico" ||
      value == "qnn_context" || value == "xr_secure_mr_model_type_qnn_context_binary_pico") {
    return XR_SECURE_MR_MODEL_TYPE_QNN_CONTEXT_BINARY_PICO;
  }
  if (value == "litert" || value == "lite_rt" || value == "lite_rt_model" || value == "tflite" ||
      value == "tensorflow_lite" || value == "xr_secure_mr_model_type_lite_rt_model_pico") {
    return XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO;
  }
  throw std::runtime_error(Fmt("unsupported model_type '%s'", modelType.c_str()));
}

XrSecureMrModelTargetPICO ParseModelTarget(const std::string& modelTarget) {
  const std::string value = ToLower(modelTarget);
  if (value.empty() || value == "npu" || value == "xr_secure_mr_model_target_npu_pico") {
    return XR_SECURE_MR_MODEL_TARGET_NPU_PICO;
  }
  if (value == "gpu" || value == "xr_secure_mr_model_target_gpu_pico") {
    return XR_SECURE_MR_MODEL_TARGET_GPU_PICO;
  }
  if (value == "cpu" || value == "xr_secure_mr_model_target_cpu_pico") {
    return XR_SECURE_MR_MODEL_TARGET_CPU_PICO;
  }
  throw std::runtime_error(Fmt("unsupported model_target '%s'", modelTarget.c_str()));
}

Pipeline::NormalizeType ParseNormalizeType(const std::string& normalizeType) {
  const std::string value = ToLower(normalizeType);
  if (value.empty() || value == "l2" || value == "xr_secure_mr_normalize_type_l2_pico") return Pipeline::NormalizeType::L2;
  if (value == "l1" || value == "xr_secure_mr_normalize_type_l1_pico") return Pipeline::NormalizeType::L1;
  if (value == "inf" || value == "infinity" || value == "xr_secure_mr_normalize_type_inf_pico") {
    return Pipeline::NormalizeType::INF;
  }
  if (value == "minmax" || value == "min_max" || value == "xr_secure_mr_normalize_type_minmax_pico") {
    return Pipeline::NormalizeType::MINMAX;
  }
  throw std::runtime_error(Fmt("unsupported normalize_type '%s'", normalizeType.c_str()));
}

XrSecureMrMatrixSortTypePICO ParseMatrixSortType(const std::string& sortType) {
  const std::string value = ToLower(sortType);
  if (value.empty() || value == "row" || value == "xr_secure_mr_matrix_sort_type_row_pico") {
    return XR_SECURE_MR_MATRIX_SORT_TYPE_ROW_PICO;
  }
  if (value == "column" || value == "col" || value == "xr_secure_mr_matrix_sort_type_column_pico") {
    return XR_SECURE_MR_MATRIX_SORT_TYPE_COLUMN_PICO;
  }
  throw std::runtime_error(Fmt("unsupported sort_type '%s'", sortType.c_str()));
}

XrSecureMrAudioFormatPcmPICO ParseAudioPcmFormat(const std::string& pcmType) {
  const std::string value = ToLower(pcmType);
  if (value.empty() || value == "int16" || value == "pcm_16bit" || value == "pcm16" ||
      value == "xr_secure_mr_audio_format_pcm_16bit_pico") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_16BIT_PICO;
  }
  if (value == "int32" || value == "pcm_32bit" || value == "pcm32" ||
      value == "xr_secure_mr_audio_format_pcm_32bit_pico") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_32BIT_PICO;
  }
  if (value == "float" || value == "float32" || value == "pcm_float" || value == "pcmfloat" ||
      value == "xr_secure_mr_audio_format_pcm_float_pico") {
    return XR_SECURE_MR_AUDIO_FORMAT_PCM_FLOAT_PICO;
  }
  throw std::runtime_error(Fmt("unsupported pcm_type '%s'", pcmType.c_str()));
}

RenderCommand_UpdateMaterial::MaterialAttribute ParseMaterialAttribute(const std::string& attribute) {
  const std::string value = ToLower(attribute);
  if (value == "material_metallic_factor" || value == "metallic" || value == "metallic_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_METALLIC;
  }
  if (value == "material_roughness_factor" || value == "roughness" || value == "roughness_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_ROUGHNESS;
  }
  if (value == "material_emissive_strength" || value == "emissive_strength") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_EMISSIVE_STRENGTH;
  }
  if (value == "material_base_color_factor" || value == "base_color" || value == "base_color_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_BASE_COLOR;
  }
  if (value == "material_emissive_factor" || value == "emissive" || value == "emissive_factor") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_EMISSIVE;
  }
  if (value == "material_occlusion_map_texture" || value == "occlusion_map_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_OCCLUSION_MAP;
  }
  if (value == "material_emissive_texture" || value == "emissive_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_EMISSIVE;
  }
  if (value == "material_base_color_texture" || value == "texture" || value == "base_color_texture" ||
      value == "material_base_color_map_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
  }
  if (value == "material_normal_map_texture" || value == "normal_map_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_NORMAL_MAP;
  }
  if (value == "material_metallic_roughness_texture" || value == "metallic_roughness_texture") {
    return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_METALLIC_ROUGHNESS;
  }
  return RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
}

std::vector<uint16_t> JsonToUInt16Vector(const Json& value) {
  std::vector<uint16_t> out;
  if (value.is_number_integer()) {
    out.push_back(static_cast<uint16_t>(value.get<int>()));
    return out;
  }
  if (!value.is_array()) {
    return out;
  }
  out.reserve(value.size());
  for (const auto& item : value) {
    if (!item.is_number_integer()) {
      throw std::runtime_error("expected uint16 array");
    }
    out.push_back(static_cast<uint16_t>(item.get<int>()));
  }
  return out;
}

std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>> ParseTensorOrUInt16Vector(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  return JsonToUInt16Vector(value);
}

std::variant<std::shared_ptr<PipelineTensor>, uint16_t> ParseTensorOrUInt16(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  if (!value.is_number_integer()) {
    throw std::runtime_error("expected uint16 value or tensor name");
  }
  return static_cast<uint16_t>(value.get<int>());
}

std::variant<std::shared_ptr<PipelineTensor>, float> ParseTensorOrFloat(
    const Json& value, const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }
  if (!value.is_number()) {
    throw std::runtime_error("expected float value or tensor name");
  }
  return value.get<float>();
}

std::variant<std::shared_ptr<PipelineTensor>, std::vector<float>, std::vector<std::uint16_t>,
             std::vector<std::array<uint8_t, 4>>>
ParseMaterialValues(const Json& value, const RenderCommand_UpdateMaterial::MaterialAttribute attribute,
                    const std::function<std::shared_ptr<PipelineTensor>(const std::string&)>& requireTensor) {
  if (value.is_string()) {
    return requireTensor(value.get<std::string>());
  }

  switch (attribute) {
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_METALLIC:
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_ROUGHNESS:
    case RenderCommand_UpdateMaterial::MaterialAttribute::FLOAT_EMISSIVE_STRENGTH: {
      if (value.is_number()) {
        return std::vector<float>{value.get<float>()};
      }
      if (!value.is_array()) {
        throw std::runtime_error("expected material float value array");
      }
      std::vector<float> out;
      out.reserve(value.size());
      for (const auto& item : value) {
        if (!item.is_number()) {
          throw std::runtime_error("expected numeric material value");
        }
        out.push_back(item.get<float>());
      }
      return out;
    }
    case RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_BASE_COLOR:
    case RenderCommand_UpdateMaterial::MaterialAttribute::RGBA_EMISSIVE: {
      if (!value.is_array()) {
        throw std::runtime_error("expected material RGBA value array");
      }
      std::vector<std::array<uint8_t, 4>> out;
      if (value.size() == 4 && value[0].is_number_integer()) {
        out.push_back({static_cast<uint8_t>(value[0].get<int>()), static_cast<uint8_t>(value[1].get<int>()),
                       static_cast<uint8_t>(value[2].get<int>()), static_cast<uint8_t>(value[3].get<int>())});
        return out;
      }
      out.reserve(value.size());
      for (const auto& color : value) {
        if (!color.is_array() || color.size() != 4) {
          throw std::runtime_error("expected material RGBA arrays");
        }
        out.push_back({static_cast<uint8_t>(color[0].get<int>()), static_cast<uint8_t>(color[1].get<int>()),
                       static_cast<uint8_t>(color[2].get<int>()), static_cast<uint8_t>(color[3].get<int>())});
      }
      return out;
    }
    default:
      return JsonToUInt16Vector(value);
  }
}

RenderCommand_DrawText::TypeFaceTypes ParseTypeFace(const std::string& value) {
  const std::string typeface = ToLower(value);
  if (typeface == "sans_serif" || typeface == "sans-serif") return RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF;
  if (typeface == "serif") return RenderCommand_DrawText::TypeFaceTypes::SERIF;
  if (typeface == "monospace") return RenderCommand_DrawText::TypeFaceTypes::MONOSPACE;
  if (typeface == "bold") return RenderCommand_DrawText::TypeFaceTypes::BOLD;
  if (typeface == "italic") return RenderCommand_DrawText::TypeFaceTypes::ITALIC;
  return RenderCommand_DrawText::TypeFaceTypes::DEFAULT;
}

#endif  // SECUREMR_SERIALIZATION_PARSE_ONLY

}  // namespace

Json TensorAttributeToJson(const TensorAttribute& attr) {
  Json j;
  j["dimensions"] = attr.dimensions;
  j["channels"] = attr.channels;
  j["usage"] = static_cast<int>(attr.usage);
  j["data_type"] = static_cast<int>(attr.dataType);
  return j;
}

Json TensorAttributeVariantToJson(const std::variant<std::monostate, TensorAttribute>& attr) {
  Json j;
  if (std::holds_alternative<TensorAttribute>(attr)) {
    j = TensorAttributeToJson(std::get<TensorAttribute>(attr));
  } else {
    j["is_gltf"] = true;
  }
  return j;
}

Json TensorListToJson(const std::vector<std::string>& tensors) {
  Json arr = Json::array();
  for (const auto& name : tensors) {
    arr.push_back(name);
  }
  return arr;
}

Json MappedTensorListToJson(const std::vector<std::pair<std::string, std::string>>& mapping) {
  Json arr = Json::array();
  for (const auto& [alias, tensor] : mapping) {
    Json entry;
    entry["name"] = alias;
    entry["tensor"] = tensor;
    arr.push_back(entry);
  }
  return arr;
}

void SetInputs(Json& spec, const std::vector<std::string>& inputs) {
  spec["inputs"] = TensorListToJson(inputs);
}

void SetOutputs(Json& spec, const std::vector<std::string>& outputs) {
  spec["outputs"] = TensorListToJson(outputs);
}

bool WriteJsonToFile(const std::filesystem::path& filePath, const Json& spec) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)filePath;
  (void)spec;
  return false;
#else
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "WriteJsonToFile failed: writable path unavailable");
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(filePath.parent_path(), ec);
  std::ofstream ofs(filePath);
  if (!ofs) {
    Log::Write(Log::Level::Error,
               Fmt("WriteJsonToFile failed: cannot open %s", filePath.string().c_str()));
    return false;
  }
  ofs << spec.dump(2);
  return true;
#endif
}

bool JsonToTensorAttribute(const Json& j, TensorAttribute& out) {
  if (JsonToSpecialTensorAttribute(j, out)) {
    return true;
  }
  if (j.find("dimensions") == j.end()) {
    return false;
  }
  if (!j["dimensions"].is_array()) {
    return false;
  }
  out.dimensions.clear();
  for (const auto& dim : j["dimensions"]) {
    if (!dim.is_number_integer()) {
      return false;
    }
    out.dimensions.push_back(dim.get<int>());
  }
  out.channels = static_cast<int8_t>(j.value("channels", 1));
  out.usage = j.contains("usage") ? ParseTensorUsage(j["usage"]) : XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  out.dataType = j.contains("data_type") ? ParseDataType(j["data_type"]) : XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
  return true;
}

std::vector<std::string> ParseTensorList(const Json& arr) {
  std::vector<std::string> tensors;
  if (!arr.is_array()) {
    return tensors;
  }
  tensors.reserve(arr.size());
  for (const auto& each : arr) {
    if (each.is_string()) {
      tensors.push_back(each.get<std::string>());
    } else if (each.is_object()) {
      if (auto tensorIt = each.find("tensor"); tensorIt != each.end() && tensorIt->is_string()) {
        tensors.push_back(tensorIt->get<std::string>());
      }
    }
  }
  return tensors;
}

std::vector<std::pair<std::string, std::string>> ParseMappedTensorList(const Json& arr) {
  std::vector<std::pair<std::string, std::string>> mapping;
  if (!arr.is_array()) {
    return mapping;
  }
  mapping.reserve(arr.size());
  for (const auto& each : arr) {
    std::string tensorName;
    std::string alias;
    if (each.is_object()) {
      if (auto aliasIt = each.find("name"); aliasIt != each.end() && aliasIt->is_string()) {
        alias = aliasIt->get<std::string>();
      }
      if (auto tensorIt = each.find("tensor"); tensorIt != each.end() && tensorIt->is_string()) {
        tensorName = tensorIt->get<std::string>();
      }
    } else if (each.is_string()) {
      tensorName = each.get<std::string>();
      alias = tensorName;
    }
    if (!tensorName.empty()) {
      if (alias.empty()) {
        alias = tensorName;
      }
      mapping.emplace_back(alias, tensorName);
    }
  }
  return mapping;
}

bool JsonToFloatArray(const Json& arr, std::array<float, 6>& dest) {
  if (!arr.is_array() || arr.size() != dest.size()) {
    return false;
  }
  for (size_t i = 0; i < dest.size(); ++i) {
    if (!arr[i].is_number()) {
      return false;
    }
    dest[i] = arr[i].get<float>();
  }
  return true;
}

std::vector<float> JsonToFloatVector(const Json& arr, const char* what) {
  if (!arr.is_array() || arr.empty()) {
    throw std::runtime_error(Fmt("%s requires a non-empty float array", what));
  }
  std::vector<float> values;
  values.reserve(arr.size());
  for (const auto& value : arr) {
    if (!value.is_number()) {
      throw std::runtime_error(Fmt("%s contains a non-numeric entry", what));
    }
    values.push_back(value.get<float>());
  }
  return values;
}

Json LoadJsonFromFile(const std::filesystem::path& filePath) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)filePath;
  return Json();
#else
  Json parsed;
  if (filePath.empty()) {
    Log::Write(Log::Level::Error, "LoadJsonFromFile failed: path empty");
    return parsed;
  }
  std::ifstream ifs(filePath);
  if (!ifs) {
    Log::Write(Log::Level::Error,
               Fmt("LoadJsonFromFile failed: cannot open %s", filePath.string().c_str()));
    return parsed;
  }
  try {
    ifs >> parsed;
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error, Fmt("LoadJsonFromFile failed: %s", e.what()));
    parsed = Json();
  }
  return parsed;
#endif
}

std::string FormatOperatorType(const std::string& typeName) {
  if (typeName.empty()) {
    return typeName;
  }
  static const std::unordered_map<std::string, std::string> kAliases = {
      {"camera_access", "camera_access"},
      {"RECTIFIED_VST_ACCESS", "camera_access"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO", "camera_access"},
      {"get_affine", "get_affine"},
      {"GET_AFFINE", "get_affine"},
      {"XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO", "get_affine"},
      {"apply_affine", "apply_affine"},
      {"APPLY_AFFINE", "apply_affine"},
      {"XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO", "apply_affine"},
      {"apply_affine_point", "apply_affine_point"},
      {"APPLY_AFFINE_POINT", "apply_affine_point"},
      {"XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO", "apply_affine_point"},
      {"cvt_color", "cvt_color"},
      {"CONVERT_COLOR", "cvt_color"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", "cvt_color"},
      {"assignment", "assignment"},
      {"ASSIGNMENT", "assignment"},
      {"type_convert", "type_convert"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", "assignment"},
      {"arithmetic", "arithmetic"},
      {"ARITHMETIC_COMPOSE", "arithmetic"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO", "arithmetic"},
      {"run_algorithm", "run_algorithm"},
      {"RUN_MODEL_INFERENCE", "run_algorithm"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO", "run_algorithm"},
      {"javascript", "javascript"},
      {"run_javascript", "javascript"},
      {"JS_SCRIPTING", "javascript"},
      {"XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO", "javascript"},
      {"elementwise", "elementwise"},
      {"ELEMENTWISE_MIN", "elementwise"},
      {"ELEMENTWISE_MAX", "elementwise"},
      {"ELEMENTWISE_MULTIPLY", "elementwise"},
      {"ELEMENTWISE_OR", "elementwise"},
      {"ELEMENTWISE_AND", "elementwise"},
      {"all", "all"},
      {"ALL", "all"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO", "all"},
      {"any", "any"},
      {"ANY", "any"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO", "any"},
      {"nms", "nms"},
      {"NMS", "nms"},
      {"non_maximum_suppression", "nms"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO", "nms"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NON_MAXIMUM_SUPPRESSION_PICO", "nms"},
      {"solve_pnp", "solve_pnp"},
      {"SOLVE_P_N_P", "solve_pnp"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO", "solve_pnp"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SOLVE_PNP_PICO", "solve_pnp"},
      {"uv2_cam", "uv2_cam"},
      {"uv_to_3d", "uv2_cam"},
      {"UV_TO_3D_IN_CAM_SPACE", "uv2_cam"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAMERA_SPACE_PICO", "uv2_cam"},
      {"transform", "transform"},
      {"get_transform_matrix", "transform"},
      {"MAKE_TRANSFORM_MAT", "transform"},
      {"XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MAT_PICO", "transform"},
      {"XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MATRIX_PICO", "transform"},
      {"normalize", "normalize"},
      {"NORMALIZE", "normalize"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO", "normalize"},
      {"cam_space_to_xr_local", "cam_space_to_xr_local"},
      {"camera_space_to_world", "cam_space_to_xr_local"},
      {"CAMERA_SPACE_TO_WORLD", "cam_space_to_xr_local"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO", "cam_space_to_xr_local"},
      {"compare_to", "compare_to"},
      {"CUSTOMIZED_COMPARE", "compare_to"},
      {"XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO", "compare_to"},
      {"argmax", "argmax"},
      {"ARGMAX", "argmax"},
      {"XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO", "argmax"},
      {"sort_vector", "sort_vector"},
      {"sort_vec", "sort_vector"},
      {"SORT_VEC", "sort_vector"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO", "sort_vector"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_VECTOR_PICO", "sort_vector"},
      {"inversion", "inversion"},
      {"inverse", "inversion"},
      {"INVERSION", "inversion"},
      {"XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO", "inversion"},
      {"sort_matrix", "sort_matrix"},
      {"sort_mat", "sort_matrix"},
      {"SORT_MAT", "sort_matrix"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO", "sort_matrix"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SORT_MATRIX_PICO", "sort_matrix"},
      {"svd", "svd"},
      {"SVD", "svd"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO", "svd"},
      {"norm", "norm"},
      {"NORM", "norm"},
      {"XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO", "norm"},
      {"swap_hwc_chw", "swap_hwc_chw"},
      {"chw_hwc", "swap_hwc_chw"},
      {"CHW_HWC", "swap_hwc_chw"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SWAP_HWC_CHW_PICO", "swap_hwc_chw"},
      {"draw_text", "draw_text"},
      {"RENDER_TEXT", "draw_text"},
      {"XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO", "draw_text"},
      {"load_texture", "load_texture"},
      {"UPLOAD_TEXTURE_TO_GLTF", "load_texture"},
      {"XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO", "load_texture"},
      {"update_gltf", "update_gltf"},
      {"UPDATE_GLTF", "update_gltf"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO", "update_gltf"},
      {"update_material", "update_material"},
      {"update_gltf_texture", "update_gltf_texture"},
      {"render_gltf", "render_gltf"},
      {"SWITCH_GLTF_RENDER_STATUS", "render_gltf"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO", "render_gltf"},
      {"scenegraph_visibility", "scenegraph_visibility"},
      {"scenegraph_query", "scenegraph_visibility"},
      {"SSMR_SWITCH_VISIBILITY", "scenegraph_visibility"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO", "scenegraph_visibility"},
      {"update_component", "update_component"},
      {"SSMR_UPDATE_COMPONENT", "update_component"},
      {"XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO", "update_component"},
      {"microphone", "microphone"},
      {"MICROPHONE", "microphone"},
      {"XR_SECURE_MR_OPERATOR_TYPE_AUDIO_MICROPHONE_PICO", "microphone"},
      {"XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO", "microphone"},
      {"speaker", "speaker"},
      {"SPEAKER", "speaker"},
      {"XR_SECURE_MR_OPERATOR_TYPE_AUDIO_SPEAKER_PICO", "speaker"},
      {"XR_SECURE_MR_OPERATOR_TYPE_SPEAKER_PICO", "speaker"},
      {"depth", "depth"},
      {"DEPTH", "depth"},
      {"XR_SECURE_MR_OPERATOR_TYPE_DEPTH_PICO", "depth"},
  };

  if (auto it = kAliases.find(typeName); it != kAliases.end()) {
    return it->second;
  }
  return typeName;
}

bool DeserializePipelineFromJson(const Json& spec,
                                 const std::shared_ptr<FrameworkSession>& session,
                                 PipelineDeserializationResult& outResult,
                                 std::string& outError,
                                 const PipelineDeserializationOptions& options) {
#ifdef SECUREMR_SERIALIZATION_PARSE_ONLY
  (void)spec;
  (void)session;
  (void)options;
  outResult = {};
  outError = "DeserializePipelineFromJson is unavailable in parser-only test builds";
  return false;
#else
  outResult = {};
  outError.clear();
  if (!spec.is_object()) {
    outError = "JSON is not an object";
    return false;
  }

  const auto tensorsIt = spec.find("tensors");
  if (tensorsIt == spec.end() || !tensorsIt->is_object()) {
    outError = "tensors section missing or invalid";
    return false;
  }

  auto pipeline = std::make_shared<Pipeline>(session);
  for (auto it = tensorsIt->begin(); it != tensorsIt->end(); ++it) {
    const std::string tensorName = it.key();
    const Json& tensorSpec = *it;
    const bool isPlaceholder = tensorSpec.value("is_placeholder", false);
    const bool isGltf = tensorSpec.value("is_gltf", false) ||
                        (tensorSpec.contains("tensor_type") && tensorSpec["tensor_type"].is_string() &&
                         ToLower(tensorSpec["tensor_type"].get<std::string>()) == "gltf");
    std::shared_ptr<PipelineTensor> tensor;
    TensorAttribute attr{};
    try {
      if (isGltf) {
        if (!isPlaceholder) {
          outError = Fmt("glTF tensor '%s' requires is_placeholder=true", tensorName.c_str());
          return false;
        }
        tensor = PipelineTensor::PipelineGLTFPlaceholder(pipeline);
      } else {
        if (!JsonToTensorAttribute(tensorSpec, attr)) {
          outError = Fmt("invalid tensor attribute for %s", tensorName.c_str());
          return false;
        }
        tensor = std::make_shared<PipelineTensor>(pipeline, attr, isPlaceholder);
        if (!isPlaceholder && !isGltf) {
          auto valueIt = tensorSpec.find("value");
          if (valueIt != tensorSpec.end() && !valueIt->is_null()) {
            if (!valueIt->is_array()) {
              outError = Fmt("invalid tensor value for %s: expected array", tensorName.c_str());
              return false;
            }
            if (attr.channels <= 0) {
              outError = Fmt("invalid tensor attribute for %s: channels must be positive", tensorName.c_str());
              return false;
            }
            size_t elementCount = 1;
            for (int dim : attr.dimensions) {
              if (dim <= 0) {
                outError = Fmt("invalid tensor attribute for %s: non-positive dimension", tensorName.c_str());
                return false;
              }
              elementCount *= static_cast<size_t>(dim);
            }
            const size_t expectedValues = elementCount * static_cast<size_t>(attr.channels);
            if (valueIt->size() != expectedValues) {
              outError = Fmt("invalid tensor value for %s: expected %zu entries but found %zu", tensorName.c_str(),
                             expectedValues, valueIt->size());
              return false;
            }

            auto ensureInteger = [&](const Json& value, const char* what, int64_t minValue,
                                     int64_t maxValue) -> std::optional<int64_t> {
              if (!value.is_number_integer()) {
                outError = Fmt("invalid tensor value for %s: expected integer for %s", tensorName.c_str(), what);
                return std::nullopt;
              }
              const int64_t numeric = value.get<int64_t>();
              if (numeric < minValue || numeric > maxValue) {
                outError = Fmt("invalid tensor value for %s: %s out of range [%lld, %lld]", tensorName.c_str(), what,
                               static_cast<long long>(minValue), static_cast<long long>(maxValue));
                return std::nullopt;
              }
              return numeric;
            };

            switch (attr.dataType) {
              case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO: {
                std::vector<float> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  const auto& value = (*valueIt)[idx];
                  if (!value.is_number()) {
                    outError = Fmt("invalid tensor value for %s: non-numeric entry at index %zu", tensorName.c_str(),
                                   idx);
                    return false;
                  }
                  buffer[idx] = static_cast<float>(value.get<double>());
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(float));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO: {
                std::vector<double> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  const auto& value = (*valueIt)[idx];
                  if (!value.is_number()) {
                    outError = Fmt("invalid tensor value for %s: non-numeric entry at index %zu", tensorName.c_str(),
                                   idx);
                    return false;
                  }
                  buffer[idx] = value.get<double>();
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(double));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO: {
                std::vector<int32_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT32", std::numeric_limits<int32_t>::min(),
                                               std::numeric_limits<int32_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int32_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int32_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO: {
                std::vector<int16_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT16", std::numeric_limits<int16_t>::min(),
                                               std::numeric_limits<int16_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int16_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int16_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO: {
                std::vector<int8_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "INT8", std::numeric_limits<int8_t>::min(),
                                               std::numeric_limits<int8_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<int8_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(int8_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO: {
                std::vector<uint16_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "UINT16", 0, std::numeric_limits<uint16_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<uint16_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(uint16_t));
                break;
              }
              case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO: {
                std::vector<uint8_t> buffer(expectedValues);
                for (size_t idx = 0; idx < expectedValues; ++idx) {
                  auto numeric = ensureInteger((*valueIt)[idx], "UINT8", 0, std::numeric_limits<uint8_t>::max());
                  if (!numeric.has_value()) {
                    return false;
                  }
                  buffer[idx] = static_cast<uint8_t>(*numeric);
                }
                tensor->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size() * sizeof(uint8_t));
                break;
              }
              default:
                Log::Write(Log::Level::Warning,
                           Fmt("DeserializePipelineFromJson: unsupported data_type %d for tensor %s initial value",
                               static_cast<int>(attr.dataType), tensorName.c_str()));
                break;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      outError = Fmt("failed to create tensor '%s': %s", tensorName.c_str(), e.what());
      return false;
    }
    outResult.tensorMap.emplace(tensorName, std::move(tensor));
  }

  const auto requireTensor = [&](const std::string& name) -> std::shared_ptr<PipelineTensor> {
    auto it = outResult.tensorMap.find(name);
    if (it == outResult.tensorMap.end()) {
      throw std::runtime_error(Fmt("tensor '%s' not found", name.c_str()));
    }
    return it->second;
  };

  const auto operatorsIt = spec.find("operators");
  if (operatorsIt == spec.end() || !operatorsIt->is_array()) {
    outError = "operators section missing or invalid";
    return false;
  }

  try {
    for (const auto& opSpec : *operatorsIt) {
      const std::string rawType = opSpec.value("type", "");
      const std::string type = FormatOperatorType(rawType);
      const auto inputs = ParseTensorList(opSpec.value("inputs", Json::array()));
      const auto outputs = ParseTensorList(opSpec.value("outputs", Json::array()));

      auto requireByIndex = [&](const std::vector<std::string>& container, size_t index,
                                const char* what) -> std::shared_ptr<PipelineTensor> {
        if (index >= container.size()) {
          throw std::runtime_error(Fmt("%s index %zu out of range", what, index));
        }
        return requireTensor(container[index]);
      };

      auto tensorNameFromFieldOrIndex = [&](const char* field, const std::vector<std::string>& container,
                                            size_t index) -> std::string {
        if (auto it = opSpec.find(field); it != opSpec.end() && it->is_string()) {
          return it->get<std::string>();
        }
        return index < container.size() ? container[index] : std::string{};
      };

      auto requireFieldOrIndex = [&](const char* field, const std::vector<std::string>& container, size_t index,
                                     const char* what) -> std::shared_ptr<PipelineTensor> {
        const std::string tensorName = tensorNameFromFieldOrIndex(field, container, index);
        if (tensorName.empty()) {
          throw std::runtime_error(Fmt("%s requires '%s' or tensor index %zu", what, field, index));
        }
        return requireTensor(tensorName);
      };

      auto makePoint2Tensor = [&](const Json& value, const char* what) -> std::shared_ptr<PipelineTensor> {
        auto points = JsonToFloatVector(value, what);
        if (points.size() % 2 != 0) {
          throw std::runtime_error(Fmt("%s requires an even number of XY values", what));
        }
        auto tensor = std::make_shared<PipelineTensor>(
            pipeline, TensorAttribute_Point2Array{.size = points.size() / 2,
                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
        tensor->setData(reinterpret_cast<int8_t*>(points.data()), points.size() * sizeof(float));
        return tensor;
      };

      auto makeBoolTensor = [&](bool value) -> std::shared_ptr<PipelineTensor> {
        auto tensor = std::make_shared<PipelineTensor>(
            pipeline, TensorAttribute{.dimensions = {1},
                                      .channels = 1,
                                      .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
        uint8_t raw = value ? 1u : 0u;
        tensor->setData(reinterpret_cast<int8_t*>(&raw), sizeof(raw));
        return tensor;
      };

      if (type == "camera_access") {
        if (outputs.size() != 4) {
          throw std::runtime_error("camera_access outputs malformed");
        }
        pipeline->cameraAccess(requireByIndex(outputs, 0, "camera_access output"),
                               requireByIndex(outputs, 1, "camera_access output"),
                               requireByIndex(outputs, 2, "camera_access output"),
                               requireByIndex(outputs, 3, "camera_access output"));
      } else if (type == "get_affine") {
        if (outputs.empty()) {
          throw std::runtime_error("get_affine requires output tensor");
        }

        if (inputs.size() >= 2 || opSpec.contains("src") || opSpec.contains("dst")) {
          pipeline->getAffine(requireFieldOrIndex("src", inputs, 0, "get_affine"),
                              requireFieldOrIndex("dst", inputs, 1, "get_affine"), requireTensor(outputs.front()));
        } else if (opSpec.contains("src_points") && opSpec.contains("dst_points")) {
          pipeline->getAffine(makePoint2Tensor(opSpec["src_points"], "get_affine src_points"),
                              makePoint2Tensor(opSpec["dst_points"], "get_affine dst_points"),
                              requireTensor(outputs.front()));
        } else if (inputs.size() >= 2) {
          pipeline->getAffine(requireByIndex(inputs, 0, "get_affine input"),
                              requireByIndex(inputs, 1, "get_affine input"), requireTensor(outputs.front()));
        } else {
          throw std::runtime_error("get_affine requires src/dst points or two input tensors");
        }
      } else if (type == "apply_affine") {
        if (inputs.size() < 2 || outputs.empty()) {
          throw std::runtime_error("apply_affine requires two inputs and one output");
        }
        pipeline->applyAffine(requireByIndex(inputs, 0, "apply_affine input"),
                              requireByIndex(inputs, 1, "apply_affine input"),
                              requireByIndex(outputs, 0, "apply_affine output"));
      } else if (type == "apply_affine_point") {
        if (inputs.size() < 2 || outputs.empty()) {
          throw std::runtime_error("apply_affine_point requires affine, points and one output");
        }
        pipeline->applyAffinePoint(requireByIndex(inputs, 0, "apply_affine_point input"),
                                   requireByIndex(inputs, 1, "apply_affine_point input"),
                                   requireByIndex(outputs, 0, "apply_affine_point output"));
      } else if (type == "assignment") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("assignment requires input and output tensors");
        }
        const auto srcRef = ResolveTensorReference(inputs[0], requireTensor);
        const auto dstRef = ResolveTensorReference(outputs[0], requireTensor);
        if (srcRef.slice.has_value() && dstRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, *dstRef.slice);
        } else if (srcRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, dstRef.tensor);
        } else if (dstRef.slice.has_value()) {
          pipeline->assignment(srcRef.tensor, *dstRef.slice);
        } else {
          pipeline->assignment(srcRef.tensor, dstRef.tensor);
        }
      } else if (type == "cvt_color") {
        const int flag = opSpec.value("flag", 0);
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("cvt_color requires input and output tensors");
        }
        pipeline->cvtColor(flag, requireByIndex(inputs, 0, "cvt_color input"),
                           requireByIndex(outputs, 0, "cvt_color output"));
      } else if (type == "type_convert") {
        const std::string srcName = tensorNameFromFieldOrIndex("src", inputs, 0);
        const std::string dstName = tensorNameFromFieldOrIndex("dst", outputs, 0);
        if (srcName.empty() || dstName.empty()) {
          throw std::runtime_error("type_convert requires src/input and dst/output tensors");
        }
        const auto srcRef = ResolveTensorReference(srcName, requireTensor);
        const auto dstRef = ResolveTensorReference(dstName, requireTensor);
        if (srcRef.slice.has_value() && dstRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, *dstRef.slice);
        } else if (srcRef.slice.has_value()) {
          pipeline->assignment(*srcRef.slice, dstRef.tensor);
        } else if (dstRef.slice.has_value()) {
          pipeline->assignment(srcRef.tensor, *dstRef.slice);
        } else {
          pipeline->typeConvert(srcRef.tensor, dstRef.tensor);
        }
      } else if (type == "arithmetic") {
        const std::string expression = opSpec.value("expression", "");
        std::vector<std::shared_ptr<PipelineTensor>> operands;
        operands.reserve(inputs.size());
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
          operands.push_back(requireByIndex(inputs, idx, "arithmetic input"));
        }
        if (outputs.empty()) {
          throw std::runtime_error("arithmetic requires output tensor");
        }
        pipeline->arithmetic(expression, operands, requireByIndex(outputs, 0, "arithmetic output"));
      } else if (type == "elementwise") {
        if (inputs.size() < 2 || outputs.empty()) {
          throw std::runtime_error("elementwise requires two inputs and one output");
        }
        std::string elementwiseOp = opSpec.value("op", "");
        if (elementwiseOp.empty()) {
          const std::string rawLower = ToLower(rawType);
          if (rawLower.find("min") != std::string::npos) elementwiseOp = "min";
          if (rawLower.find("max") != std::string::npos) elementwiseOp = "max";
          if (rawLower.find("multiply") != std::string::npos) elementwiseOp = "multiply";
          if (rawLower.find("or") != std::string::npos) elementwiseOp = "or";
          if (rawLower.find("and") != std::string::npos) elementwiseOp = "and";
        }
        pipeline->elementwise(ParseElementwiseOp(elementwiseOp),
                              {requireByIndex(inputs, 0, "elementwise input"),
                               requireByIndex(inputs, 1, "elementwise input")},
                              requireByIndex(outputs, 0, "elementwise output"));
      } else if (type == "all") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("all requires input and output tensors");
        }
        pipeline->all(requireByIndex(inputs, 0, "all input"), requireByIndex(outputs, 0, "all output"));
      } else if (type == "any") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("any requires input and output tensors");
        }
        pipeline->any(requireByIndex(inputs, 0, "any input"), requireByIndex(outputs, 0, "any output"));
      } else if (type == "nms") {
        if (inputs.size() < 2 || outputs.empty()) {
          throw std::runtime_error("nms requires scores, boxes and at least one output");
        }
        auto boxes = requireByIndex(inputs, 1, "nms input");
        const float threshold = opSpec.value("threshold", opSpec.value("iou", rawType == "NMS" ? 0.95f : 0.5f));
        pipeline->nms(requireByIndex(inputs, 0, "nms input"), boxes,
                      outputs.size() > 0 ? requireByIndex(outputs, 0, "nms output") : nullptr,
                      outputs.size() > 1 ? requireByIndex(outputs, 1, "nms output") : nullptr,
                      outputs.size() > 2 ? requireByIndex(outputs, 2, "nms output") : nullptr, threshold);
      } else if (type == "solve_pnp") {
        if (inputs.size() < 3 || outputs.empty()) {
          throw std::runtime_error("solve_pnp requires object points, image points, camera matrix and output(s)");
        }
        pipeline->solvePnP(requireByIndex(inputs, 0, "solve_pnp input"),
                           requireByIndex(inputs, 1, "solve_pnp input"),
                           requireByIndex(inputs, 2, "solve_pnp input"),
                           outputs.size() > 0 ? requireByIndex(outputs, 0, "solve_pnp output") : nullptr,
                           outputs.size() > 1 ? requireByIndex(outputs, 1, "solve_pnp output") : nullptr);
      } else if (type == "uv2_cam") {
        if (inputs.size() < 5 || outputs.empty()) {
          throw std::runtime_error("uv2_cam requires uv, timestamp, camera matrix, left image, right image and output");
        }
        pipeline->uv2Cam(requireByIndex(inputs, 0, "uv2_cam input"), requireByIndex(inputs, 1, "uv2_cam input"),
                         requireByIndex(inputs, 2, "uv2_cam input"), requireByIndex(inputs, 3, "uv2_cam input"),
                         requireByIndex(inputs, 4, "uv2_cam input"), requireByIndex(outputs, 0, "uv2_cam output"));
      } else if (type == "transform") {
        if (outputs.empty()) {
          throw std::runtime_error("transform requires output");
        }
        const std::string scaleName = tensorNameFromFieldOrIndex("scale", inputs, 2);
        pipeline->transform(requireFieldOrIndex("rotation", inputs, 0, "transform"),
                            requireFieldOrIndex("translation", inputs, 1, "transform"),
                            scaleName.empty() ? nullptr : requireTensor(scaleName),
                            requireByIndex(outputs, 0, "transform output"));
      } else if (type == "cam_space_to_xr_local") {
        if (outputs.empty()) {
          throw std::runtime_error("cam_space_to_xr_local requires at least one output");
        }
        const std::string eye = ToLower(opSpec.value("eye", "left"));
        if (eye == "right") {
          pipeline->camSpace2XrLocal(requireFieldOrIndex("timestamp", inputs, 0, "cam_space_to_xr_local"),
                                     requireByIndex(outputs, 0, "cam_space_to_xr_local output"), nullptr);
        } else {
          pipeline->camSpace2XrLocal(requireFieldOrIndex("timestamp", inputs, 0, "cam_space_to_xr_local"), nullptr,
                                     requireByIndex(outputs, 0, "cam_space_to_xr_local output"));
        }
      } else if (type == "compare_to") {
        if (outputs.empty()) {
          throw std::runtime_error("compare_to requires one output");
        }
        PipelineTensor::Compare compare;
        compare.left = requireFieldOrIndex("left", inputs, 0, "compare_to");
        compare.right = requireFieldOrIndex("right", inputs, 1, "compare_to");
        compare.comparison = ParseComparison(opSpec.value("compare", ""));
        pipeline->compareTo(compare, requireByIndex(outputs, 0, "compare_to output"));
      } else if (type == "normalize") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("normalize requires input and output tensors");
        }
        pipeline->normalize(requireByIndex(inputs, 0, "normalize input"), requireByIndex(outputs, 0, "normalize output"),
                            ParseNormalizeType(opSpec.value("normalize_type", "l2")));
      } else if (type == "argmax") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("argmax requires input and output tensors");
        }
        pipeline->argMax(requireByIndex(inputs, 0, "argmax input"), requireByIndex(outputs, 0, "argmax output"));
      } else if (type == "sort_vector") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("sort_vector requires input and at least one output");
        }
        pipeline->sortVec(requireByIndex(inputs, 0, "sort_vector input"),
                          outputs.size() > 0 ? requireByIndex(outputs, 0, "sort_vector output") : nullptr,
                          outputs.size() > 1 ? requireByIndex(outputs, 1, "sort_vector output") : nullptr);
      } else if (type == "inversion") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("inversion requires input and output tensors");
        }
        pipeline->inversion(requireByIndex(inputs, 0, "inversion input"),
                            requireByIndex(outputs, 0, "inversion output"));
      } else if (type == "sort_matrix") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("sort_matrix requires input and at least one output");
        }
        const auto sortType = ParseMatrixSortType(opSpec.value("sort_type", "row"));
        if (sortType == XR_SECURE_MR_MATRIX_SORT_TYPE_COLUMN_PICO) {
          pipeline->sortMatByColumn(requireByIndex(inputs, 0, "sort_matrix input"),
                                    outputs.size() > 0 ? requireByIndex(outputs, 0, "sort_matrix output") : nullptr,
                                    outputs.size() > 1 ? requireByIndex(outputs, 1, "sort_matrix output") : nullptr);
        } else {
          pipeline->sortMatByRow(requireByIndex(inputs, 0, "sort_matrix input"),
                                 outputs.size() > 0 ? requireByIndex(outputs, 0, "sort_matrix output") : nullptr,
                                 outputs.size() > 1 ? requireByIndex(outputs, 1, "sort_matrix output") : nullptr);
        }
      } else if (type == "svd") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("svd requires input and output tensor(s)");
        }
        pipeline->singularValueDecomposition(
            requireByIndex(inputs, 0, "svd input"), outputs.size() > 0 ? requireByIndex(outputs, 0, "svd output") : nullptr,
            outputs.size() > 1 ? requireByIndex(outputs, 1, "svd output") : nullptr,
            outputs.size() > 2 ? requireByIndex(outputs, 2, "svd output") : nullptr);
      } else if (type == "norm") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("norm requires input and output tensors");
        }
        pipeline->norm(requireByIndex(inputs, 0, "norm input"), requireByIndex(outputs, 0, "norm output"));
      } else if (type == "swap_hwc_chw") {
        if (inputs.empty() || outputs.empty()) {
          throw std::runtime_error("swap_hwc_chw requires input and output tensors");
        }
        pipeline->convertHWC_CHW(requireByIndex(inputs, 0, "swap_hwc_chw input"),
                                 requireByIndex(outputs, 0, "swap_hwc_chw output"));
      } else if (type == "microphone") {
        if (outputs.empty()) {
          throw std::runtime_error("microphone requires output tensor(s)");
        }
        std::vector<std::shared_ptr<PipelineTensor>> results;
        results.reserve(outputs.size());
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
          results.push_back(requireByIndex(outputs, idx, "microphone output"));
        }
        pipeline->microphone(results, ParseAudioPcmFormat(opSpec.value("pcm_type", "int16")),
                             opSpec.value("sample_rate", 16000));
      } else if (type == "speaker") {
        const std::string src = opSpec.value("src", "");
        if (src.empty() && inputs.empty()) {
          throw std::runtime_error("speaker requires src or input tensor");
        }
        pipeline->speaker(src.empty() ? requireByIndex(inputs, 0, "speaker input") : requireTensor(src),
                          opSpec.value("sample_rate", 16000));
      } else if (type == "depth") {
        if (outputs.empty()) {
          throw std::runtime_error("depth requires output tensor");
        }
        pipeline->depth(requireByIndex(outputs, 0, "depth output"));
      } else if (type == "run_algorithm") {
        // Parse mapped inputs/outputs
        auto mappedInputs = ParseMappedTensorList(opSpec.value("inputs", Json::array()));
        auto mappedOutputs = ParseMappedTensorList(opSpec.value("outputs", Json::array()));
        if (mappedInputs.empty() || mappedOutputs.empty()) {
          throw std::runtime_error("run_algorithm inputs/outputs malformed");
        }

        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> inputMap;
        for (const auto& [alias, tensorName] : mappedInputs) {
          inputMap.emplace(alias, requireTensor(tensorName));
        }
        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> outputMap;
        for (const auto& [alias, tensorName] : mappedOutputs) {
          outputMap.emplace(alias, requireTensor(tensorName));
        }

        const std::string modelName = opSpec.value("model_name", "");
        if (modelName.empty()) {
          throw std::runtime_error("run_algorithm requires 'model_name'");
        }

        // Load model from asset (Android) or file (if provided)
        std::vector<char> modelBuffer;
        if (auto assetIt = opSpec.find("model_asset"); assetIt != opSpec.end() && assetIt->is_string()) {
          const std::string assetName = assetIt->get<std::string>();
#ifdef XR_USE_PLATFORM_ANDROID
          if (g_assetManager == nullptr) {
            throw std::runtime_error("run_algorithm: AssetManager not available for 'model_asset'");
          }
          AAsset* asset = AAssetManager_open(g_assetManager, assetName.c_str(), AASSET_MODE_BUFFER);
          if (asset == nullptr) {
            throw std::runtime_error(Fmt("run_algorithm: unable to open asset '%s'", assetName.c_str()));
          }
          const off_t length = AAsset_getLength(asset);
          modelBuffer.resize(static_cast<size_t>(length));
          const int64_t read = AAsset_read(asset, modelBuffer.data(), length);
          AAsset_close(asset);
          if (read != length) {
            modelBuffer.clear();
            throw std::runtime_error(Fmt("run_algorithm: read %ld of %ld bytes from asset '%s'",
                                         static_cast<long>(read), static_cast<long>(length), assetName.c_str()));
          }
#else
          throw std::runtime_error("run_algorithm: 'model_asset' only supported on Android builds");
#endif
        } else if (auto fileIt = opSpec.find("model_file"); fileIt != opSpec.end() && fileIt->is_string()) {
          const std::string filePath = fileIt->get<std::string>();
          std::ifstream ifs(filePath, std::ios::binary);
          if (!ifs) {
            throw std::runtime_error(Fmt("run_algorithm: cannot open file '%s'", filePath.c_str()));
          }
          modelBuffer.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
          if (modelBuffer.empty()) {
            throw std::runtime_error(Fmt("run_algorithm: file '%s' is empty or read failed", filePath.c_str()));
          }
        } else {
          throw std::runtime_error("run_algorithm requires 'model_asset' (Android) or 'model_file'");
        }

        std::unordered_map<std::string, std::string> operandAliasing;
        std::unordered_map<std::string, std::string> resultAliasing;
        if (auto inAlias = opSpec.find("input_aliasing"); inAlias != opSpec.end() && inAlias->is_object()) {
          for (auto it = inAlias->begin(); it != inAlias->end(); ++it) {
            if (it.value().is_string()) {
              operandAliasing.emplace(it.key(), it.value().get<std::string>());
            }
          }
        }
        if (auto outAlias = opSpec.find("output_aliasing"); outAlias != opSpec.end() && outAlias->is_object()) {
          for (auto it = outAlias->begin(); it != outAlias->end(); ++it) {
            if (it.value().is_string()) {
              resultAliasing.emplace(it.key(), it.value().get<std::string>());
            }
          }
        }

        const XrSecureMrModelTypePICO modelType = ParseModelType(opSpec.value("model_type", "qnn"));
        const XrSecureMrModelTargetPICO modelTarget = ParseModelTarget(opSpec.value("model_target", "npu"));
        const int32_t cpuTargetNumThreads = opSpec.value("cpu_target_num_threads", 1);
        pipeline->runAlgorithm(modelBuffer.data(), modelBuffer.size(), inputMap, operandAliasing, outputMap,
                               resultAliasing, modelName, modelType, modelTarget, cpuTargetNumThreads);
      } else if (type == "javascript") {
        auto mappedInputs = ParseMappedTensorList(opSpec.value("inputs", Json::array()));
        auto mappedOutputs = ParseMappedTensorList(opSpec.value("outputs", Json::array()));
        if (mappedOutputs.empty()) {
          throw std::runtime_error("javascript outputs malformed");
        }

        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> scriptOperands;
        for (const auto& [alias, tensorName] : mappedInputs) {
          scriptOperands.emplace(alias, requireTensor(tensorName));
        }
        std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> scriptResults;
        for (const auto& [alias, tensorName] : mappedOutputs) {
          scriptResults.emplace(alias, requireTensor(tensorName));
        }

        std::string script = opSpec.value("script", "");
        if (script.empty()) {
          auto attrsIt = opSpec.find("attrs");
          if (attrsIt != opSpec.end() && attrsIt->is_array() && !attrsIt->empty() && (*attrsIt)[0].is_string()) {
            script = (*attrsIt)[0].get<std::string>();
          }
        }
        if (script.empty()) {
          throw std::runtime_error("javascript requires 'script'");
        }
        pipeline->runJavascript(script.data(), script.size(), scriptOperands, scriptResults);
      } else if (type == "draw_text") {
        const std::string text = tensorNameFromFieldOrIndex("text", inputs, 0);
        const std::string start = tensorNameFromFieldOrIndex("start", inputs, 1);
        const std::string colors = tensorNameFromFieldOrIndex("colors", inputs, 2);
        const std::string textureId = tensorNameFromFieldOrIndex("texture_id", inputs, 3);
        const std::string fontSize = tensorNameFromFieldOrIndex("font_size", inputs, 4);
        const std::string gltf = tensorNameFromFieldOrIndex("gltf", inputs, 5);
        if (gltf.empty() || text.empty() || start.empty() || fontSize.empty() || colors.empty() || textureId.empty()) {
          throw std::runtime_error(
              "draw_text requires gltf, text, start, font_size, colors and texture_id fields or positional inputs");
        }
        pipeline->execRenderCommand(std::make_shared<RenderCommand_DrawText>(
            requireTensor(gltf), opSpec.value("language_and_locale", "en-US"),
            ParseTypeFace(opSpec.value("typeface", "default")), opSpec.value("canvas_width", 256),
            opSpec.value("canvas_height", 64), requireTensor(text), requireTensor(start), requireTensor(fontSize),
            requireTensor(colors), requireTensor(textureId)));
      } else if (type == "load_texture") {
        const std::string gltf = opSpec.value("gltf", "");
        const std::string textureSrc = opSpec.value("rgb_image", inputs.empty() ? "" : inputs[0]);
        if (gltf.empty() || textureSrc.empty() || outputs.empty()) {
          throw std::runtime_error("load_texture requires gltf, rgb_image/input and texture_id output");
        }
        pipeline->newTextureToGLTF(requireTensor(gltf), requireTensor(textureSrc),
                                   requireByIndex(outputs, 0, "load_texture output"));
      } else if (type == "update_gltf_texture" || type == "update_material" || type == "update_gltf") {
        const std::string gltf = opSpec.value("gltf", "");
        if (gltf.empty()) {
          throw std::runtime_error(Fmt("%s requires gltf", type.c_str()));
        }

        const std::string attribute = ToLower(opSpec.value("attribute", type == "update_material" ? "base_color" : ""));
        if (type == "update_gltf_texture" || attribute == "texture" || attribute == "gltf_texture") {
          const auto textureIdIt = opSpec.find("texture_id");
          const std::string textureSrc = opSpec.value("texture_src", opSpec.value("rgb_image", inputs.empty() ? "" : inputs[0]));
          if (textureIdIt == opSpec.end() || textureSrc.empty()) {
            throw std::runtime_error("update_gltf_texture requires texture_id and texture_src/rgb_image");
          }
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateTextures>(
              requireTensor(gltf), ParseTensorOrUInt16Vector(*textureIdIt, requireTensor), requireTensor(textureSrc)));
        } else if (attribute == "animation") {
          const auto animationIdIt = opSpec.find("animation_id");
          if (animationIdIt == opSpec.end()) {
            throw std::runtime_error("update_gltf animation requires animation_id");
          }
          auto timerIt = opSpec.find("animation_timer");
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateAnimation>(
              requireTensor(gltf), ParseTensorOrUInt16(*animationIdIt, requireTensor),
              timerIt == opSpec.end() ? std::variant<std::shared_ptr<PipelineTensor>, float>{RenderCommand_UpdateAnimation::STOP_TO_PLAY}
                                      : ParseTensorOrFloat(*timerIt, requireTensor)));
        } else if (attribute == "world_pose" || attribute == "pose") {
          const std::string pose = opSpec.value("pose", opSpec.value("value", inputs.empty() ? "" : inputs[0]));
          if (pose.empty()) {
            throw std::runtime_error("update_gltf world_pose requires pose/value tensor");
          }
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdatePose>(requireTensor(gltf), requireTensor(pose)));
        } else if (attribute == "local_transform" || attribute == "local_pose") {
          const auto nodeIt = opSpec.find("node_id");
          const std::string transform = opSpec.value("transform", opSpec.value("value", inputs.empty() ? "" : inputs[0]));
          if (nodeIt == opSpec.end() || transform.empty()) {
            throw std::runtime_error("update_gltf local_transform requires node_id and transform/value tensor");
          }
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateNodesLocalPoses>(
              requireTensor(gltf), ParseTensorOrUInt16Vector(*nodeIt, requireTensor), requireTensor(transform)));
        } else {
          auto materialIt = opSpec.find("material");
          if (materialIt == opSpec.end()) {
            materialIt = opSpec.find("material_id");
          }
          auto valuesIt = opSpec.find("values");
          if (valuesIt == opSpec.end()) {
            valuesIt = opSpec.find("value");
          }
          if (materialIt == opSpec.end() || valuesIt == opSpec.end()) {
            throw std::runtime_error("update_material requires material/material_id and values/value");
          }
          const auto materialAttribute = ParseMaterialAttribute(opSpec.value("attribute", "base_color"));
          pipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateMaterial>(
              requireTensor(gltf), ParseTensorOrUInt16Vector(*materialIt, requireTensor), materialAttribute,
              ParseMaterialValues(*valuesIt, materialAttribute, requireTensor)));
        }
      } else if (type == "render_gltf") {
        const std::string gltf = opSpec.value("gltf", "");
        const std::string pose = opSpec.value("pose", "");
        if (gltf.empty() || pose.empty()) {
          throw std::runtime_error("render_gltf requires gltf and pose");
        }
        auto command = std::make_shared<RenderCommand_Render>();
        command->gltfTensor = requireTensor(gltf);
        command->pose = requireTensor(pose);
        command->viewLocked = opSpec.value("view_locked", false);
        if (auto visibleIt = opSpec.find("visible"); visibleIt != opSpec.end() && visibleIt->is_string()) {
          command->visible = requireTensor(visibleIt->get<std::string>());
        }
        pipeline->execRenderCommand(command);
      } else if (type == "scenegraph_visibility") {
        const std::string scenegraph = opSpec.value("scenegraph", opSpec.value("gltf", inputs.empty() ? "" : inputs[0]));
        if (scenegraph.empty()) {
          throw std::runtime_error("scenegraph_visibility requires scenegraph/gltf tensor");
        }
        std::shared_ptr<PipelineTensor> visible;
        if (auto visibleIt = opSpec.find("visible"); visibleIt != opSpec.end()) {
          if (visibleIt->is_string()) {
            visible = requireTensor(visibleIt->get<std::string>());
          } else if (visibleIt->is_boolean()) {
            visible = makeBoolTensor(visibleIt->get<bool>());
          }
        } else if (inputs.size() > 1) {
          visible = requireByIndex(inputs, 1, "scenegraph_visibility input");
        } else {
          visible = makeBoolTensor(true);
        }
        pipeline->scenegraphVisibility(requireTensor(scenegraph), visible);
      } else if (type == "update_component") {
        const std::string scenegraph = opSpec.value("scenegraph", opSpec.value("gltf", inputs.empty() ? "" : inputs[0]));
        if (scenegraph.empty()) {
          throw std::runtime_error("update_component requires scenegraph/gltf tensor");
        }
        std::shared_ptr<PipelineTensor> data;
        if (auto dataIt = opSpec.find("data"); dataIt != opSpec.end() && dataIt->is_string()) {
          data = requireTensor(dataIt->get<std::string>());
        } else if (auto valueIt = opSpec.find("value"); valueIt != opSpec.end() && valueIt->is_string()) {
          data = requireTensor(valueIt->get<std::string>());
        } else if (auto enabledIt = opSpec.find("enabled"); enabledIt != opSpec.end()) {
          if (enabledIt->is_string()) {
            data = requireTensor(enabledIt->get<std::string>());
          } else if (enabledIt->is_boolean()) {
            data = makeBoolTensor(enabledIt->get<bool>());
          }
        } else if (inputs.size() > 1) {
          data = requireByIndex(inputs, 1, "update_component input");
        }
        if (data == nullptr) {
          throw std::runtime_error("update_component requires data/value/enabled tensor or boolean");
        }
        pipeline->updateComponent(requireTensor(scenegraph), data);
      } else {
        bool handled = false;
        if (options.customOperatorHandler) {
          handled = options.customOperatorHandler(opSpec, requireTensor, pipeline, outError);
        }
        if (!handled) {
          throw std::runtime_error(Fmt("unsupported operator type '%s'",
                                       rawType.empty() ? type.c_str() : rawType.c_str()));
        }
      }
    }
  } catch (const std::exception& e) {
    if (outError.empty()) {
      outError = e.what();
    }
    return false;
  }

  outResult.pipeline = std::move(pipeline);
  return true;
#endif
}

}  // namespace SecureMR
