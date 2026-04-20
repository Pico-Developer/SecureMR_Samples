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
#include <memory>
#include <optional>
#include <string>
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

class SecureMrUtils {
 public:
  static size_t BytesPerElement(XrSecureMrTensorDataTypePICO dataType);
  static size_t ElementCount(const TensorAttribute& attr);
  static std::optional<Json> LoadModelJson(const std::filesystem::path& jsonPath);
  static bool PrepareBindings(const Json& jsonSpec,
                              std::vector<TensorBinding>& inputBindings,
                              std::vector<TensorBinding>& outputBindings,
                              std::string& modelName);
};

}  // namespace SecureMR

#endif  // SECUREMR_UTILS_UTILS_H_
