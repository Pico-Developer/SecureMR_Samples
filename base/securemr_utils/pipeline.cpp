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

#include "rendercommand.h"
#include "tensor.h"
#include "pipeline.h"

#include <variant>

namespace SecureMR {
Pipeline::Pipeline(std::shared_ptr<FrameworkSession> root) : m_rootSession(std::move(root)) {
  if (m_rootSession) {
    Log::Write(Log::Level::Info, "Attempting to get xrCreateSecureMrPipelinePICO");

    xrCreateSecureMrPipelinePICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrCreateSecureMrPipelinePICO>("xrCreateSecureMrPipelinePICO");
    xrDestroySecureMrPipelinePICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrDestroySecureMrPipelinePICO>("xrDestroySecureMrPipelinePICO");
    xrCreateSecureMrOperatorPICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrCreateSecureMrOperatorPICO>("xrCreateSecureMrOperatorPICO");
    xrSetSecureMrOperatorOperandByNamePICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrSetSecureMrOperatorOperandByNamePICO>(
            "xrSetSecureMrOperatorOperandByNamePICO");
    xrSetSecureMrOperatorOperandByIndexPICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrSetSecureMrOperatorOperandByIndexPICO>(
            "xrSetSecureMrOperatorOperandByIndexPICO");
    xrSetSecureMrOperatorResultByNamePICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrSetSecureMrOperatorResultByNamePICO>(
            "xrSetSecureMrOperatorResultByNamePICO");
    xrExecuteSecureMrPipelinePICO =
        m_rootSession->getAPIFromXrInstance<PFN_xrExecuteSecureMrPipelinePICO>("xrExecuteSecureMrPipelinePICO");
  }
  CHECK_MSG(xrCreateSecureMrPipelinePICO != nullptr, "xrCreateSecureMrPipelinePICO failed");
  CHECK_MSG(xrDestroySecureMrPipelinePICO != nullptr, "xrDestroySecureMrPipelinePICO failed");
  CHECK_MSG(xrCreateSecureMrOperatorPICO != nullptr, "xrCreateSecureMrOperatorPICO failed");
  CHECK_MSG(xrSetSecureMrOperatorOperandByNamePICO != nullptr, "xrSetSecureMrOperatorOperandByNamePICO failed");
  CHECK_MSG(xrSetSecureMrOperatorOperandByIndexPICO != nullptr, "xrSetSecureMrOperatorOperandByIndexPICO failed");
  CHECK_MSG(xrSetSecureMrOperatorResultByNamePICO != nullptr, "xrSetSecureMrOperatorResultByNamePICO failed");
  CHECK_MSG(xrExecuteSecureMrPipelinePICO != nullptr, "xrExecuteSecureMrPipelinePICO failed");

  constexpr XrSecureMrPipelineCreateInfoPICO createInfo = {XR_TYPE_SECURE_MR_PIPELINE_CREATE_INFO_PICO};
  CHECK_XRCMD(xrCreateSecureMrPipelinePICO(m_rootSession->getFrameworkPICO(), &createInfo, &m_handle))
}

bool Pipeline::verifyPipelineTensor(const std::shared_ptr<PipelineTensor>& candidateTensor) const {
  return candidateTensor != nullptr && candidateTensor->getPipeline().get() == this;
}

Pipeline::~Pipeline(){CHECK_XRCMD(xrDestroySecureMrPipelinePICO(m_handle))}

Pipeline& Pipeline::typeConvert(const std::shared_ptr<PipelineTensor>& src,
                                const std::shared_ptr<PipelineTensor>& dst) {
  return assignment(src, dst);
}

Pipeline& Pipeline::assignment(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& dst) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle,
                                                     static_cast<XrSecureMrPipelineTensorPICO>(*src), "src"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*dst), "dst"))
  return *this;
}

Pipeline& Pipeline::assignment(const std::shared_ptr<PipelineTensor>& src, const PipelineTensor::Slice& dstSlice) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = nullptr,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle,
                                                     static_cast<XrSecureMrPipelineTensorPICO>(*src), "src"))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, dstSlice.sliceTensor(), "dst slices"))
  if (dstSlice.hasChannelSlice()) {
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, dstSlice.channelSliceTensor(), "dst channel slice"))
  }

  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, dstSlice.targetTensor(), "dst"))
  return *this;
}

Pipeline& Pipeline::assignment(const PipelineTensor::Slice& srcSlice, const std::shared_ptr<PipelineTensor>& dst) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = nullptr,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.targetTensor(), "src"))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.sliceTensor(), "src slices"))
  if (srcSlice.hasChannelSlice()) {
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.channelSliceTensor(), "src channel slice"))
  }

  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*dst), "dst"))
  return *this;
}

Pipeline& Pipeline::assignment(const PipelineTensor::Slice& srcSlice, const PipelineTensor::Slice& dstSlice) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = nullptr,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.targetTensor(), "src"))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.sliceTensor(), "src slices"))
  if (srcSlice.hasChannelSlice()) {
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, srcSlice.channelSliceTensor(), "src channel slice"))
  }
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, dstSlice.sliceTensor(), "dst slices"))
  if (dstSlice.hasChannelSlice()) {
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, dstSlice.channelSliceTensor(), "dst channel slice"))
  }

  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, dstSlice.targetTensor(), "dst"))
  return *this;
}

Pipeline& Pipeline::compareTo(const PipelineTensor::Compare& compare, const std::shared_ptr<PipelineTensor>& dst) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorComparisonPICO comparisonConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_COMPARISON_PICO,
                                                    .comparison = compare.comparison};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&comparisonConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*compare.left),
                                         "operand0");
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*compare.right),
                                         "operand1");

  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*dst),
                                                    "result"))
  return *this;
}

Pipeline& Pipeline::arithmetic(const std::string& expression, const std::vector<std::shared_ptr<PipelineTensor>>& ops,
                               const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorArithmeticComposePICO arithmeticConfig{XR_TYPE_SECURE_MR_OPERATOR_ARITHMETIC_COMPOSE_PICO};
  std::strcpy(arithmeticConfig.configText, expression.c_str());

  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&arithmeticConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  int operandIndex = 0;
  for (auto& operand : ops) {
    xrSetSecureMrOperatorOperandByIndexPICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*operand),
                                            operandIndex);
    operandIndex++;
  }
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result");
  return *this;
}

Pipeline& Pipeline::elementwise(const Pipeline::ElementwiseOp operation,
                                const std::array<std::shared_ptr<PipelineTensor>, 2>& ops,
                                const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;

  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO};
  switch (operation) {
    case ElementwiseOp::MIN:
      operatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO;
      break;
    case ElementwiseOp::MAX:
      operatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MAX_PICO;
      break;
    case ElementwiseOp::MULTIPLY:
      operatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO;
      break;
    case ElementwiseOp::OR:
      operatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_OR_PICO;
      break;
    case ElementwiseOp::AND:
      operatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_AND_PICO;
      break;
  }
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, ops[0]->operator XrSecureMrPipelineTensorPICO_T*(),
                                         "operand0");
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, ops[1]->operator XrSecureMrPipelineTensorPICO_T*(),
                                         "operand1");
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result");
  return *this;
}

Pipeline& Pipeline::all(const std::shared_ptr<PipelineTensor>& op, const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
                                                      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, op->operator XrSecureMrPipelineTensorPICO_T*(),
                                                     "operand"))
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result");
  return *this;
}

Pipeline& Pipeline::any(const std::shared_ptr<PipelineTensor>& op, const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
                                                      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, op->operator XrSecureMrPipelineTensorPICO_T*(),
                                                     "operand"))
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result");
  return *this;
}

Pipeline& Pipeline::nms(const std::shared_ptr<PipelineTensor>& scores, std::shared_ptr<PipelineTensor>& boxes,
                        const std::shared_ptr<PipelineTensor>& result_scores,
                        const std::shared_ptr<PipelineTensor>& result_boxes,
                        const std::shared_ptr<PipelineTensor>& result_indices, float threshold) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorNonMaximumSuppressionPICO nmsConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_NON_MAXIMUM_SUPPRESSION_PICO,
                                                        .threshold = threshold};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&nmsConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*scores, "scores"))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*boxes, "boxes"))
  if (result_scores != nullptr) {
    CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_scores,
                                                      "scores"))
  }
  if (result_boxes != nullptr) {
    CHECK_XRCMD(
        xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_boxes, "boxes"))
  }
  if (result_indices != nullptr) {
    CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_indices,
                                                      "indices"))
  }
  return *this;
}

Pipeline& Pipeline::solvePnP(const std::shared_ptr<PipelineTensor>& objectPoints,
                             const std::shared_ptr<PipelineTensor>& imgPoints,
                             const std::shared_ptr<PipelineTensor>& cameraMatrix,
                             const std::shared_ptr<PipelineTensor>& result_rotation,
                             const std::shared_ptr<PipelineTensor>& result_translation) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
                                                      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*objectPoints,
                                         "object points");
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*imgPoints,
                                                     "image points"))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*cameraMatrix,
                                         "camera matrix");
  if (result_rotation != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_rotation,
                                          "rotation");
  }
  if (result_translation != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_translation,
                                          "translation");
  }
  return *this;
}

Pipeline& Pipeline::getAffine(const AffinePoints& srcPoints, const AffinePoints& dstPoints,
                              const std::shared_ptr<PipelineTensor>& result) {
  constexpr TensorAttribute_Point2Array POINT2F_ARRAY3{.size = 3,
                                                       .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO};

  auto operandFromArray2_3 = [&POINT2F_ARRAY3](const std::shared_ptr<Pipeline>& pipeline,
                                               std::array<float, 6> rawPoint2) {
    const auto srcTensorPtr = std::make_shared<PipelineTensor>(pipeline, POINT2F_ARRAY3);
    srcTensorPtr->setData(reinterpret_cast<int8_t*>(rawPoint2.data()), sizeof(float) * rawPoint2.size());
    return static_cast<XrSecureMrPipelineTensorPICO>(*srcTensorPtr);
  };

  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
                                                      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  if (std::holds_alternative<std::shared_ptr<PipelineTensor>>(srcPoints)) {
    const auto srcTensorPtr = std::get<std::shared_ptr<PipelineTensor>>(srcPoints);
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*srcTensorPtr, "src"))
  } else {
    CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(
        m_handle, opHandle, operandFromArray2_3(shared_from_this(), std::get<std::array<float, 6>>(srcPoints)), "src"))
  }
  if (std::holds_alternative<std::shared_ptr<PipelineTensor>>(dstPoints)) {
    const auto dstTensorPtr = std::get<std::shared_ptr<PipelineTensor>>(dstPoints);
    CHECK_XRCMD(
        xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*dstTensorPtr, "dst"))
  } else {
    CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(
        m_handle, opHandle, operandFromArray2_3(shared_from_this(), std::get<std::array<float, 6>>(dstPoints)), "dst"))
  }
  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result, "result"))

  return *this;
}

Pipeline& Pipeline::applyAffine(const std::shared_ptr<PipelineTensor>& affine,
                                const std::shared_ptr<PipelineTensor>& img,
                                const std::shared_ptr<PipelineTensor>& result_img) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
                                                      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*affine, "affine"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*img, "src image"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_img, "dst image"))
  return *this;
}

Pipeline& Pipeline::applyAffinePoint(const std::shared_ptr<PipelineTensor>& affine,
                                     const std::shared_ptr<PipelineTensor>& points,
                                     const std::shared_ptr<PipelineTensor>& result_points) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*affine, "affine"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*points, "src points"))
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_points,
                                                    "dst points"))
  return *this;
}

Pipeline& Pipeline::uv2Cam(const std::shared_ptr<PipelineTensor>& uv, const std::shared_ptr<PipelineTensor>& timestamp,
                           const std::shared_ptr<PipelineTensor>& cameraMatrix,
                           const std::shared_ptr<PipelineTensor>& leftImg,
                           const std::shared_ptr<PipelineTensor>& rightImg,
                           const std::shared_ptr<PipelineTensor>& result) {
  // Check for null pointers
  CHECK_MSG(uv != nullptr, "uv2Cam uvPlaceholder1 is null")
  CHECK_MSG(timestamp != nullptr, "uv2Cam timestampPlaceholder1 is null")
  CHECK_MSG(cameraMatrix != nullptr, "uv2Cam cameraMatrixPlaceholder1 is null")
  CHECK_MSG(leftImg != nullptr, "uv2Cam leftImagePlaceholder is null")
  CHECK_MSG(rightImg != nullptr, "uv2Cam rightImagePlaceholder is null")
  CHECK_MSG(result != nullptr, "uv2Cam pointXYZ is null")

  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorUVTo3DPICO uvTo3DOperatorPico{XR_TYPE_SECURE_MR_OPERATOR_UV_TO_3D_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO uvTo3DCreateInfoPico{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&uvTo3DOperatorPico),
      XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &uvTo3DCreateInfoPico, &opHandle))

  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*uv, "uv"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*timestamp, "timestamp"))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*cameraMatrix,
                                         "camera intrinsic");
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*leftImg, "left image"))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*rightImg,
                                                     "right image"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result, "point_xyz"))
  return *this;
}

Pipeline& Pipeline::normalize(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& result,
                              const Pipeline::NormalizeType type) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorNormalizePICO normalizeConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_NORMALIZE_PICO,
                                                  .normalizeType = static_cast<XrSecureMrNormalizeTypePICO>(type)};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&normalizeConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*src, "operand0"))
  CHECK_XRCMD(
      xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result, "result"))
  return *this;
}

Pipeline& Pipeline::camSpace2XrLocal(const std::shared_ptr<PipelineTensor>& timestamp,
                                     const std::shared_ptr<PipelineTensor>& result_rightEyeTransform,
                                     const std::shared_ptr<PipelineTensor>& result_leftEyeTransform) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(
      xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*timestamp, "timestamp"))
  if (result_leftEyeTransform != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_leftEyeTransform,
                                          "left");
  }
  if (result_rightEyeTransform != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_rightEyeTransform,
                                          "right");
  }
  return *this;
}

Pipeline& Pipeline::cameraAccess(const std::shared_ptr<PipelineTensor>& result_rightEye,
                                 const std::shared_ptr<PipelineTensor>& result_leftEye,
                                 const std::shared_ptr<PipelineTensor>& result_timeStamp,
                                 const std::shared_ptr<PipelineTensor>& result_camMatrix) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO};

  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  if (result_leftEye != nullptr) {
    CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_leftEye,
                                                      "left image"))
  }
  if (result_rightEye != nullptr) {
    CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                                      (XrSecureMrPipelineTensorPICO)*result_rightEye, "right image"))
  }
  if (result_timeStamp != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_timeStamp,
                                          "timestamp");
  }
  if (result_camMatrix != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, (XrSecureMrPipelineTensorPICO)*result_camMatrix,
                                          "camera matrix");
  }
  return *this;
}

Pipeline& Pipeline::argMax(const std::shared_ptr<PipelineTensor>& src,
                           const std::shared_ptr<PipelineTensor>& result_indexPerChannel) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;

  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*src),
                                         "operand");
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                        static_cast<XrSecureMrPipelineTensorPICO>(*result_indexPerChannel), "result");
  return *this;
}

Pipeline& Pipeline::cvtColor(const int convertFlag, const std::shared_ptr<PipelineTensor>& image,
                             const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorColorConvertPICO convertConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_COLOR_CONVERT_PICO,
                                                   .convert = convertFlag};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&convertConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle,
                                                     static_cast<XrSecureMrPipelineTensorPICO>(*image), "src"))
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                                    static_cast<XrSecureMrPipelineTensorPICO>(*result), "dst"))
  return *this;
}

Pipeline& Pipeline::sortVec(const std::shared_ptr<PipelineTensor>& srcVec,
                            const std::shared_ptr<PipelineTensor>& result_sortedVec,
                            const std::shared_ptr<PipelineTensor>& result_indices) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;

  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*srcVec),
                                         "input");
  if (result_sortedVec != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                          static_cast<XrSecureMrPipelineTensorPICO>(*result_sortedVec), "sorted");
  }
  if (result_indices != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                          static_cast<XrSecureMrPipelineTensorPICO>(*result_indices), "indices");
  }
  return *this;
}

Pipeline& Pipeline::sortMatByRow(const std::shared_ptr<PipelineTensor>& srcMat,
                                 const std::shared_ptr<PipelineTensor>& result_sortedMat,
                                 const std::shared_ptr<PipelineTensor>& result_indicesPerRow) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorSortMatrixPICO sortType{.type = XR_TYPE_SECURE_MR_OPERATOR_SORT_MATRIX_PICO,
                                            .sortType = XR_SECURE_MR_MATRIX_SORT_TYPE_ROW_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&sortType),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*srcMat),
                                         "input");
  if (result_sortedMat != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                          static_cast<XrSecureMrPipelineTensorPICO>(*result_sortedMat), "sorted");
  }
  if (result_indicesPerRow != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                          static_cast<XrSecureMrPipelineTensorPICO>(*result_indicesPerRow), "indices");
  }
  return *this;
}

Pipeline& Pipeline::sortMatByColumn(const std::shared_ptr<PipelineTensor>& srcMat,
                                    const std::shared_ptr<PipelineTensor>& result_sortedMat,
                                    const std::shared_ptr<PipelineTensor>& result_indicesPerColumn) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorSortMatrixPICO sortType{.type = XR_TYPE_SECURE_MR_OPERATOR_SORT_MATRIX_PICO,
                                            .sortType = XR_SECURE_MR_MATRIX_SORT_TYPE_COLUMN_PICO};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&sortType),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*srcMat),
                                         "operand0");
  if (result_sortedMat != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                          static_cast<XrSecureMrPipelineTensorPICO>(*result_sortedMat), "sorted");
  }
  if (result_indicesPerColumn != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(
        m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_indicesPerColumn), "indices");
  }
  return *this;
}

Pipeline& Pipeline::singularValueDecomposition(const std::shared_ptr<PipelineTensor>& src,
                                               const std::shared_ptr<PipelineTensor>& result_w,
                                               const std::shared_ptr<PipelineTensor>& result_u,
                                               const std::shared_ptr<PipelineTensor>& result_vt) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*src), "src");
  if (result_w != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_w),
                                          "w");
  }
  if (result_u != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_u),
                                          "u");
  }
  if (result_vt != nullptr) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_vt),
                                          "vt");
  }
  return *this;
}

Pipeline& Pipeline::norm(const std::shared_ptr<PipelineTensor>& src,
                         const std::shared_ptr<PipelineTensor>& result_norm) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*src),
                                         "operand0");
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_norm),
                                        "result0");
  return *this;
}

Pipeline& Pipeline::convertHWC_CHW(const std::shared_ptr<PipelineTensor>& src,
                                   const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_SWAP_HWC_CHW_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*src),
                                         "operand0");
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result0");
  return *this;
}

Pipeline& Pipeline::inversion(const std::shared_ptr<PipelineTensor>& srcMat,
                              const std::shared_ptr<PipelineTensor>& result_inverted) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*srcMat),
                                         "operand");
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result_inverted),
                                        "result");
  return *this;
}

Pipeline& Pipeline::transform(const std::shared_ptr<PipelineTensor>& rotation,
                              const std::shared_ptr<PipelineTensor>& translation,
                              const std::shared_ptr<PipelineTensor>& scale,
                              const std::shared_ptr<PipelineTensor>& result) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MAT_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*rotation),
                                         "rotation");
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*translation),
                                         "translation");
  if (scale != nullptr) {
    xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*scale),
                                           "scale");
  }
  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result),
                                        "result");
  return *this;
}

Pipeline& Pipeline::newTextureToGLTF(const std::shared_ptr<PipelineTensor>& gltfPlaceholder,
                                     const std::shared_ptr<PipelineTensor>& textureSrc,
                                     const std::shared_ptr<PipelineTensor>& result_newTextureId) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle,
                                         static_cast<XrSecureMrPipelineTensorPICO>(*gltfPlaceholder), "gltf");
  xrSetSecureMrOperatorOperandByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*textureSrc),
                                         "rgb image");

  xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle,
                                        static_cast<XrSecureMrPipelineTensorPICO>(*result_newTextureId), "texture ID");
  return *this;
}

Pipeline& Pipeline::execRenderCommand(const std::shared_ptr<RenderCommand>& command) {
  if (command != nullptr) {
    command->execute();
  }
  return *this;
}

static std::vector<XrSecureMrOperatorIOMapPICO> prepareIoMap(
    const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& tensors,
    const std::unordered_map<std::string, std::string>& aliasing) {
  std::vector<XrSecureMrOperatorIOMapPICO> ioMaps;

  for (auto& tensorPair : tensors) {
    auto attribute = tensorPair.second->getAttribute();
    CHECK_MSG(!std::holds_alternative<std::monostate>(attribute), "Customized algorithm operator not for GLTF tensors")
    const auto& tensorAttribute = std::get<TensorAttribute>(attribute);

    XrSecureMrModelEncodingPICO ecd;
    switch (tensorAttribute.dataType) {
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
        ecd = XR_SECURE_MR_MODEL_ENCODING_UFIXED_POINT8_PICO;
        break;
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
        ecd = XR_SECURE_MR_MODEL_ENCODING_SFIXED_POINT8_PICO;
        break;
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
        Write(Log::Level::Warning, "INT16 will be interpreted as unsigned 16-bit fixed point");
      // fall-through
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
        ecd = XR_SECURE_MR_MODEL_ENCODING_UFIXED_POINT16_PICO;
        break;
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
        ecd = XR_SECURE_MR_MODEL_ENCODING_INT32_PICO;
        break;
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
        ecd = XR_SECURE_MR_MODEL_ENCODING_FLOAT_32_PICO;
        break;
      default:
        THROW("float64 is not supported as a customized algorithm operator's operand");
    }
    ioMaps.emplace_back(XrSecureMrOperatorIOMapPICO{
        .type = XR_TYPE_SECURE_MR_OPERATOR_IO_MAP_PICO,
        .encodingType = ecd,
    });
    auto aliasLookup = aliasing.find(tensorPair.first);
    std::strcpy(ioMaps.back().nodeName,
                aliasLookup == aliasing.end() ? tensorPair.first.c_str() : aliasLookup->second.c_str());
    std::strcpy(ioMaps.back().operatorIOName, tensorPair.first.c_str());
  }
  return ioMaps;
}

Pipeline& Pipeline::runAlgorithm(char* algPackageBuf, size_t algPackageSize,
                                 const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& algOps,
                                 const std::unordered_map<std::string, std::string>& operandAliasing,
                                 const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& algResults,
                                 const std::unordered_map<std::string, std::string>& resultAliasing,
                                 const std::string& modelName) {
  XrSecureMrOperatorPICO opHandle = XR_NULL_HANDLE;
  std::vector<XrSecureMrOperatorIOMapPICO> inputConfigs = prepareIoMap(algOps, operandAliasing);
  std::vector<XrSecureMrOperatorIOMapPICO> outputConfigs = prepareIoMap(algResults, resultAliasing);
  XrSecureMrOperatorModelPICO algConfig{.type = XR_TYPE_SECURE_MR_OPERATOR_MODEL_PICO,
                                        .modelInputCount = static_cast<uint32_t>(inputConfigs.size()),
                                        .modelInputs = inputConfigs.data(),
                                        .modelOutputCount = static_cast<uint32_t>(outputConfigs.size()),
                                        .modelOutputs = outputConfigs.data(),
                                        .bufferSize = static_cast<uint32_t>(algPackageSize),
                                        .buffer = algPackageBuf,
                                        .modelType = XR_SECURE_MR_MODEL_TYPE_QNN_CONTEXT_BINARY_PICO,
                                        .modelName = modelName.c_str()};
  XrSecureMrOperatorCreateInfoPICO operatorCreateInfo{
      .type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO,
      .operatorInfo = reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO*>(&algConfig),
      .operatorType = XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO,
  };
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_handle, &operatorCreateInfo, &opHandle))
  for (auto& operand : algOps) {
    xrSetSecureMrOperatorOperandByNamePICO(
        m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*operand.second), operand.first.c_str());
  }
  for (auto& result : algResults) {
    xrSetSecureMrOperatorResultByNamePICO(m_handle, opHandle, static_cast<XrSecureMrPipelineTensorPICO>(*result.second),
                                          result.first.c_str());
  }
  return *this;
}

XrSecureMrPipelineRunPICO Pipeline::submit(
    const std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>>& argumentMap,
    XrSecureMrPipelineRunPICO waitFor, const std::shared_ptr<GlobalTensor>& condition) {
  std::vector<XrSecureMrPipelineIOPairPICO> pairs;
  pairs.reserve(argumentMap.size());
  for (auto& eachPair : argumentMap) {
    pairs.emplace_back(XrSecureMrPipelineIOPairPICO{
        .type = XR_TYPE_SECURE_MR_PIPELINE_IO_PAIR_PICO,
        .localPlaceHolderTensor = static_cast<XrSecureMrPipelineTensorPICO>(*eachPair.first),
        .globalTensor = static_cast<XrSecureMrTensorPICO>(*eachPair.second)});
  }
  XrSecureMrPipelineExecuteParameterPICO runParam{
      .type = XR_TYPE_SECURE_MR_PIPELINE_EXECUTE_PARAMETER_PICO,
      .pipelineRunToBeWaited = waitFor,
      .conditionTensor = condition != nullptr ? static_cast<XrSecureMrTensorPICO>(*condition) : XR_NULL_HANDLE,
      .pairCount = static_cast<uint32_t>(pairs.size()),
      .pipelineIOPair = pairs.data()};
  XrSecureMrPipelineRunPICO runHandle;
  CHECK_XRCMD(xrExecuteSecureMrPipelinePICO(m_handle, &runParam, &runHandle))
  return runHandle;
}
}  // namespace SecureMR
