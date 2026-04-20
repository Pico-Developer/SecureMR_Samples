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

// #include "rendercommand.h"
#include "pipeline.h"
#include "tensor.h"
#include "oxr_utils/logger.h"

#include <utility>

namespace SecureMR {
GlobalTensor::GlobalTensor(const std::shared_ptr<FrameworkSession>& session, TensorAttribute attribute)
    : m_session(session), m_attribute(attribute) {
  xrCreateSecureMrTensorPICO =
      session->getAPIFromXrInstance<PFN_xrCreateSecureMrTensorPICO>("xrCreateSecureMrTensorPICO");
  xrDestroySecureMrTensorPICO =
      session->getAPIFromXrInstance<PFN_xrDestroySecureMrTensorPICO>("xrDestroySecureMrTensorPICO");
  xrResetSecureMrTensorPICO = session->getAPIFromXrInstance<PFN_xrResetSecureMrTensorPICO>("xrResetSecureMrTensorPICO");

  CHECK_MSG(xrCreateSecureMrTensorPICO != nullptr, "xrCreateSecureMrTensorPICO is null")
  CHECK_MSG(xrDestroySecureMrTensorPICO != nullptr, "xrDestroySecureMrTensorPICO is null")
  CHECK_MSG(xrResetSecureMrTensorPICO != nullptr, "xrResetSecureMrTensorPICO is null")

  XrSecureMrTensorFormatPICO format = {
      .dataType = attribute.dataType, .channel = attribute.channels, .tensorType = attribute.usage};
  XrSecureMrTensorCreateInfoShapePICO createInfo = {
      .type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
      .placeHolder = false,
      .dimensionsCount = static_cast<uint32_t>(attribute.dimensions.size()),
      .dimensions = attribute.dimensions.data(),
      .format = &format};
  auto result =
      xrCreateSecureMrTensorPICO(m_session->getFrameworkPICO(),
                                 reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &m_handle);
  CHECK_XRRESULT(
      result,
      Fmt("xrCreateSecureMrTensorPICO(dimensionsCount = %d, format = {datatype = %d, channel = %d, tensorType = %d})",
          createInfo.dimensionsCount, format.dataType, format.channel, format.tensorType)
          .c_str())
}

GlobalTensor::GlobalTensor(const std::shared_ptr<FrameworkSession>& session, TensorAttribute attribute, int8_t* data,
                           size_t size)
    : GlobalTensor(session, std::move(attribute)) {
  setData(data, size);
}

GlobalTensor::GlobalTensor(const std::shared_ptr<FrameworkSession>& session, char* const gltfContent, size_t size)
    : m_session(session), m_attribute(std::monostate()) {
  xrCreateSecureMrTensorPICO =
      session->getAPIFromXrInstance<PFN_xrCreateSecureMrTensorPICO>("xrCreateSecureMrTensorPICO");
  xrDestroySecureMrTensorPICO =
      session->getAPIFromXrInstance<PFN_xrDestroySecureMrTensorPICO>("xrDestroySecureMrTensorPICO");
  xrResetSecureMrTensorPICO = session->getAPIFromXrInstance<PFN_xrResetSecureMrTensorPICO>("xrResetSecureMrTensorPICO");

  CHECK_MSG(xrCreateSecureMrTensorPICO != nullptr, "xrCreateSecureMrTensorPICO is null")
  CHECK_MSG(xrDestroySecureMrTensorPICO != nullptr, "xrDestroySecureMrTensorPICO is null")
  CHECK_MSG(xrResetSecureMrTensorPICO != nullptr, "xrResetSecureMrTensorPICO is null")

  XrSecureMrTensorCreateInfoGltfPICO createInfo = {.type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_GLTF_PICO,
                                                   .placeHolder = false,
                                                   .bufferSize = static_cast<uint32_t>(size),
                                                   .buffer = gltfContent};

  auto result =
      xrCreateSecureMrTensorPICO(m_session->getFrameworkPICO(),
                                 reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &m_handle);
  CHECK_XRRESULT(result, Fmt("xrCreateSecureMrTensorPICO(gltf[%d])", size).c_str())
}

GlobalTensor::GlobalTensor(const GlobalTensor& other)
    : XrHandleAdapter(other),
      m_session(other.m_session),
      m_attribute(other.m_attribute),
      xrCreateSecureMrTensorPICO(other.xrCreateSecureMrTensorPICO),
      xrDestroySecureMrTensorPICO(other.xrDestroySecureMrTensorPICO),
      xrResetSecureMrTensorPICO(other.xrResetSecureMrTensorPICO) {
  CHECK_MSG(std::holds_alternative<TensorAttribute>(other.m_attribute),
            "GlobalTensor(GlobalTensor&) can only copy non-glTF global tensor")
  auto& attr = std::get<TensorAttribute>(m_attribute);
  XrSecureMrTensorFormatPICO format = {.dataType = attr.dataType, .channel = attr.channels, .tensorType = attr.usage};
  XrSecureMrTensorCreateInfoShapePICO createInfo = {.type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                    .placeHolder = false,
                                                    .dimensionsCount = static_cast<uint32_t>(attr.dimensions.size()),
                                                    .dimensions = attr.dimensions.data(),
                                                    .format = &format};
  auto result =
      xrCreateSecureMrTensorPICO(m_session->getFrameworkPICO(),
                                 reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &m_handle);
  CHECK_XRRESULT(
      result,
      Fmt("xrCreateSecureMrTensorPICO(dimensionsCount = %d, format = {datatype = %d, channel = %d, tensorType = %d})",
          createInfo.dimensionsCount, format.dataType, format.channel, format.tensorType)
          .c_str())
}

GlobalTensor::~GlobalTensor() {
  if (xrCreateSecureMrTensorPICO != nullptr) {
    xrDestroySecureMrTensorPICO(m_handle);
  }
}

void GlobalTensor::setData(int8_t* data, size_t size) const {
  CHECK_MSG(std::holds_alternative<TensorAttribute>(m_attribute),
            "GlobalTensor::setData(...) only for non-glTF global tensor")
  XrSecureMrTensorBufferPICO buffer{
      .type = XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO, .bufferSize = static_cast<uint32_t>(size), .buffer = data};
  const auto result = xrResetSecureMrTensorPICO(m_handle, &buffer);
  CHECK_XRRESULT(result, Fmt("xrResetSecureMrTensorPICO(%p, %zu)", data, size).c_str());
}

PipelineTensor::Slice::Slice(const std::shared_ptr<PipelineTensor>& tensor,
                             const std::shared_ptr<PipelineTensor>& slices)
    : m_tensor(tensor), m_slices(slices), m_channelSlice(nullptr) {}

PipelineTensor::Slice& PipelineTensor::Slice::operator[](const std::shared_ptr<PipelineTensor>& channelSlice) {
  m_channelSlice = channelSlice;
  return *this;
}

PipelineTensor::Slice& PipelineTensor::Slice::operator[](std::array<int, 3> channelSliceStatic) {
  m_channelSlice = std::make_shared<PipelineTensor>(
      m_tensor->m_pipeline, TensorAttribute{.dimensions = {1},
                                            .channels = 3,
                                            .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  m_channelSlice->setData(reinterpret_cast<int8_t*>(channelSliceStatic.data()),
                          channelSliceStatic.size() * sizeof(int32_t));
  return *this;
}

PipelineTensor::Slice& PipelineTensor::Slice::operator[](std::array<int, 2> channelSliceStatic) {
  m_channelSlice = std::make_shared<PipelineTensor>(
      m_tensor->m_pipeline, TensorAttribute{.dimensions = {1},
                                            .channels = 2,
                                            .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  m_channelSlice->setData(reinterpret_cast<int8_t*>(channelSliceStatic.data()),
                          channelSliceStatic.size() * sizeof(int32_t));
  return *this;
}

PipelineTensor::Slice& PipelineTensor::Slice::operator[](int index) {
  m_channelSlice = std::make_shared<PipelineTensor>(
      m_tensor->m_pipeline, TensorAttribute{.dimensions = {1},
                                            .channels = 2,
                                            .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  int channelSliceData[]{static_cast<int>(index), static_cast<int>(index + 1)};
  m_channelSlice->setData(reinterpret_cast<int8_t*>(channelSliceData), 2 * sizeof(int32_t));
  return *this;
}

PipelineTensor::PipelineTensor(std::shared_ptr<Pipeline> pipeline)
    : m_pipeline(std::move(pipeline)), m_attribute(std::monostate()), isPlaceholder(true) {
  xrCreateSecureMrPipelineTensorPICO =
      m_pipeline->getRootSession()->getAPIFromXrInstance<PFN_xrCreateSecureMrPipelineTensorPICO>(
          "xrCreateSecureMrPipelineTensorPICO");
  xrResetSecureMrPipelineTensorPICO =
      m_pipeline->getRootSession()->getAPIFromXrInstance<PFN_xrResetSecureMrPipelineTensorPICO>(
          "xrResetSecureMrPipelineTensorPICO");
}

PipelineTensor::PipelineTensor(std::shared_ptr<Pipeline> pipeline, TensorAttribute attribute, bool isPlaceholder)
    : m_pipeline(std::move(pipeline)), m_attribute(attribute), isPlaceholder(isPlaceholder) {
  xrCreateSecureMrPipelineTensorPICO =
      m_pipeline->getRootSession()->getAPIFromXrInstance<PFN_xrCreateSecureMrPipelineTensorPICO>(
          "xrCreateSecureMrPipelineTensorPICO");
  xrResetSecureMrPipelineTensorPICO =
      m_pipeline->getRootSession()->getAPIFromXrInstance<PFN_xrResetSecureMrPipelineTensorPICO>(
          "xrResetSecureMrPipelineTensorPICO");

  CHECK_MSG(xrCreateSecureMrPipelineTensorPICO != nullptr, "xrCreateSecureMrPipelineTensorPICO is null")
  CHECK_MSG(xrResetSecureMrPipelineTensorPICO != nullptr, "xrResetSecureMrPipelineTensorPICO is null")

  XrSecureMrTensorFormatPICO format = {
      .dataType = attribute.dataType, .channel = attribute.channels, .tensorType = attribute.usage};

  XrSecureMrTensorCreateInfoShapePICO createInfo = {
      .type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
      .placeHolder = isPlaceholder,
      .dimensionsCount = static_cast<uint32_t>(attribute.dimensions.size()),
      .dimensions = attribute.dimensions.data(),
      .format = &format};

  auto result = xrCreateSecureMrPipelineTensorPICO(
      static_cast<XrSecureMrPipelinePICO>(*m_pipeline),
      reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &m_handle);
  CHECK_XRRESULT(
      result,
      Fmt("xrCreateSecureMrPipelineTensorPICO(isPlaceholder = %s, dimensionsCount = %d, format = {datatype = %d, "
          "channel = %d, tensorType = %d})",
          isPlaceholder ? "true" : "false", createInfo.dimensionsCount, format.dataType, format.channel,
          format.tensorType)
          .c_str())
}

PipelineTensor::PipelineTensor(std::shared_ptr<Pipeline> pipeline, TensorAttribute attribute, int8_t* data, size_t size)
    : PipelineTensor(std::move(pipeline), std::move(attribute), false) {
  setData(data, size);
}

std::shared_ptr<PipelineTensor> PipelineTensor::PipelineGLTFPlaceholder(const std::shared_ptr<Pipeline>& root) {
  auto pt = std::make_shared<PipelineTensor>(root);
  XrSecureMrTensorCreateInfoGltfPICO createInfo = {
      .type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_GLTF_PICO,
      .placeHolder = true,
  };
  const auto result = pt->xrCreateSecureMrPipelineTensorPICO(
      static_cast<XrSecureMrPipelinePICO>(*pt->m_pipeline),
      reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &pt->m_handle);
  CHECK_XRRESULT(result, "xrCreateSecureMrPipelineTensorPICO(...GLTF placeholder...)")
  return pt;
}

std::shared_ptr<PipelineTensor> PipelineTensor::PipelinePlaceholderLike(const std::shared_ptr<Pipeline>& root,
                                                                        const std::shared_ptr<GlobalTensor>& like) {
  auto attr = like->getAttribute();
  if (!std::holds_alternative<TensorAttribute>(attr)) {
    return PipelineGLTFPlaceholder(root);
  }
  auto& tensorAttr = std::get<TensorAttribute>(attr);

  Log::Write(Log::Level::Verbose, Fmt("tensorAttr.dataType: %d", tensorAttr.dataType));
  return std::make_shared<PipelineTensor>(root, tensorAttr, true);
}

PipelineTensor::PipelineTensor(const PipelineTensor& other)
    : XrHandleAdapter(other),
      enable_shared_from_this(other),
      m_pipeline(other.m_pipeline),
      m_attribute(other.m_attribute),
      isPlaceholder(other.isPlaceholder),
      xrCreateSecureMrPipelineTensorPICO(other.xrCreateSecureMrPipelineTensorPICO),
      xrResetSecureMrPipelineTensorPICO(other.xrResetSecureMrPipelineTensorPICO) {
  CHECK_MSG(std::holds_alternative<TensorAttribute>(other.m_attribute),
            "PipelineTensor(PipelineTensor&) can only copy non-glTF pipeline tensor")

  auto& attr = std::get<TensorAttribute>(m_attribute);
  XrSecureMrTensorFormatPICO format = {.dataType = attr.dataType, .channel = attr.channels, .tensorType = attr.usage};
  XrSecureMrTensorCreateInfoShapePICO createInfo = {.type = XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_SHAPE_PICO,
                                                    .placeHolder = isPlaceholder,
                                                    .dimensionsCount = static_cast<uint32_t>(attr.dimensions.size()),
                                                    .dimensions = attr.dimensions.data(),
                                                    .format = &format};

  auto result = xrCreateSecureMrPipelineTensorPICO(
      static_cast<XrSecureMrPipelinePICO>(*m_pipeline),
      reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO*>(&createInfo), &m_handle);
  CHECK_XRRESULT(
      result,
      Fmt("xrCreateSecureMrPipelineTensorPICO(isPlaceholder = %s, dimensionsCount = %d, format = {datatype = %d, "
          "channel = %d, tensorType = %d})",
          isPlaceholder ? "true" : "false", createInfo.dimensionsCount, format.dataType, format.channel,
          format.tensorType)
          .c_str())
}

void PipelineTensor::setData(int8_t* const data, const size_t size) const {
  CHECK_MSG(!isPlaceholder, "setData(...) not for pipeline placeholder")
  CHECK_MSG(std::holds_alternative<TensorAttribute>(m_attribute), "setData(...) not for glTF tensor")

  XrSecureMrTensorBufferPICO buffer{.type = XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO,
                                    .bufferSize = static_cast<uint32_t>(size),
                                    .buffer = reinterpret_cast<int8_t*>(data)};
  const auto result =
      xrResetSecureMrPipelineTensorPICO(static_cast<XrSecureMrPipelinePICO>(*m_pipeline), m_handle, &buffer);
  CHECK_XRRESULT(result, Fmt("xrResetSecureMrPipelineTensorPICO(%p, %zu)", data, size).c_str())
}

PipelineTensor::Slice PipelineTensor::operator[](const std::vector<std::vector<int>>& slices) {
  CHECK_MSG(!slices.empty(), "operator[]: empty slice")
  CHECK_MSG(std::holds_alternative<TensorAttribute>(m_attribute), "operator[](...) slices not for glTF tensor")
  std::vector<int> allSliceData;  // flatten slices
  CHECK_MSG(slices.size() == std::get<TensorAttribute>(m_attribute).dimensions.size(),
            "operator[](...) slices not matching target tensor dimensions")
  auto channelCnt = slices[0].size();
  CHECK_MSG(channelCnt == 2 || channelCnt == 3, "operator[]: each slice must have either 2 or 3 values")

  for (auto& eachSlice : slices) {
    CHECK_MSG(eachSlice.size() == channelCnt, "operator[]: slices must be of the same size")
    for (auto& element : eachSlice) allSliceData.push_back(element);
  }
  auto slicesTensor = std::make_shared<PipelineTensor>(
      m_pipeline, TensorAttribute{.dimensions = {static_cast<int>(slices.size())},
                                  .channels = static_cast<int8_t>(channelCnt),
                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  slicesTensor->setData(reinterpret_cast<int8_t*>(allSliceData.data()), allSliceData.size() * sizeof(int));

  return {shared_from_this(), slicesTensor};
}

PipelineTensor::Slice PipelineTensor::operator[](const std::vector<int>& slices) {
  CHECK_MSG(!slices.empty(), "operator[]: empty slice")
  CHECK_MSG(std::holds_alternative<TensorAttribute>(m_attribute), "operator[](...) slices not for glTF tensor")
  CHECK_MSG(slices.size() == std::get<TensorAttribute>(m_attribute).dimensions.size(),
            "operator[](...) slices not matching target tensor dimensions")
  std::vector<int> allSliceData;
  for (auto& eachDimSlice : slices) {
    allSliceData.push_back(eachDimSlice);
    allSliceData.push_back(eachDimSlice + 1);
  }
  auto slicesTensor = std::make_shared<PipelineTensor>(
      m_pipeline, TensorAttribute{.dimensions = {static_cast<int>(slices.size())},
                                  .channels = 2,
                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  slicesTensor->setData(reinterpret_cast<int8_t*>(allSliceData.data()), allSliceData.size() * sizeof(int));

  return {shared_from_this(), slicesTensor};
}

PipelineTensor::Slice PipelineTensor::operator[](const std::shared_ptr<PipelineTensor>& sliceTensor) {
  return {shared_from_this(), sliceTensor};
}

PipelineTensor::Slice PipelineTensor::operator[](int index) {
  CHECK_MSG(std::holds_alternative<TensorAttribute>(m_attribute), "operator[](...) slices not for glTF tensor")
  auto& attr = std::get<TensorAttribute>(m_attribute);
  CHECK_MSG(index >= 0 && index < attr.dimensions[0], "operator[]: index out of bounds")

  std::vector<int> sliceData = {index, index + 1};
  auto sliceTensor = std::make_shared<PipelineTensor>(
      m_pipeline, TensorAttribute{.dimensions = {1},
                                  .channels = 2,
                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  sliceTensor->setData(reinterpret_cast<int8_t*>(sliceData.data()), sliceData.size() * sizeof(int));

  return {shared_from_this(), sliceTensor};
}

PipelineTensor::Compare PipelineTensor::operator>(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{.left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_LARGER_THAN_PICO};
}
PipelineTensor::Compare PipelineTensor::operator<(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{.left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_SMALLER_THAN_PICO};
}

PipelineTensor::Compare PipelineTensor::operator>=(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{
      .left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_LARGER_OR_EQUAL_PICO};
}

PipelineTensor::Compare PipelineTensor::operator<=(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{
      .left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_SMALLER_OR_EQUAL_PICO};
}

PipelineTensor::Compare PipelineTensor::operator==(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{.left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_EQUAL_TO_PICO};
}

PipelineTensor::Compare PipelineTensor::operator!=(const std::shared_ptr<PipelineTensor>& other) const {
  return Compare{.left = shared_from_this(), .right = other, .comparison = XR_SECURE_MR_COMPARISON_NOT_EQUAL_PICO};
}

}  // namespace SecureMR
