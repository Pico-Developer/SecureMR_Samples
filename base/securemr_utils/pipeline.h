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

#ifndef PIPELINE_H
#define PIPELINE_H

#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <array>

#include "tensor.h"

namespace SecureMR {

class PipelineTensor;
struct RenderCommand;

/**
 * Pipeline, an adapter for <code>XrSecureMrPipelinePICO</code> handle. By using the class:
 *
 * <ol>
 * <li> the SecureMR pipeline's lifespan is connected with the lifespan of the <code>Pipeline</code> object. The
 * pipeline will be automatically destroyed once the object is no longer valid. </li> <li> all pipeline operators are
 * encapsulated as methods of the <code>Pipeline</code> object, so that operators can be added simply in one line, such
 * as : <code>myPipeline.assignment(...).arithmetic(...);</code> </li> <li> pipeline methods allows static and literal
 * values as inputs, to avoid explicitly creating extra pipeline tensors to store those values.</li>
 * </ol>
 * <br/>
 * <b>Note</b> A pipeline is one computation graph, composed by operators and local tensors. The methods that
 * encapsulates operators <i>only</i> add the corresponding operators to the pipeline. The operators will not be
 * really executed until the pipeline object is submitted.
 */
class Pipeline final : public XrHandleAdapter<XrSecureMrPipelinePICO>, public std::enable_shared_from_this<Pipeline> {
  const std::shared_ptr<FrameworkSession> m_rootSession;

 protected:
  PFN_xrCreateSecureMrPipelinePICO xrCreateSecureMrPipelinePICO = nullptr;
  PFN_xrDestroySecureMrPipelinePICO xrDestroySecureMrPipelinePICO = nullptr;
  PFN_xrSetSecureMrOperatorOperandByNamePICO xrSetSecureMrOperatorOperandByNamePICO = nullptr;
  PFN_xrSetSecureMrOperatorOperandByIndexPICO xrSetSecureMrOperatorOperandByIndexPICO = nullptr;
  PFN_xrSetSecureMrOperatorResultByNamePICO xrSetSecureMrOperatorResultByNamePICO = nullptr;
  PFN_xrCreateSecureMrOperatorPICO xrCreateSecureMrOperatorPICO = nullptr;
  PFN_xrExecuteSecureMrPipelinePICO xrExecuteSecureMrPipelinePICO = nullptr;

  /**
   * Verify provided pipeline tensor is from the same pipeline before attaching it
   * to operator's operands or results <br/>
   * <b>NOTE</b>: only the pipeline tensors (or placeholders) from the same pipeline can be used as operators'
   * operands or results <br/>
   * @param candidateTensor the pipeline tensor to be verified
   * @return <code>true</code> if the pipeline tensor can be used as operands or results
   *
   * <b>Also note</b> The utility classes do not verify the run-time compatibility of operands or results, such as
   * whether the data types or the dimensions are supported by certain operators. We suggest to refer to the OpenXR
   * specifications for extension XR_PICO_secure_mixed_reality regarding each operators' input/output requirements.
   */
  [[nodiscard]] bool verifyPipelineTensor(const std::shared_ptr<PipelineTensor>& candidateTensor) const;

 public:
  friend struct RenderCommand;

  /**
   * Pipeline must be constructed in association with a FrameworkSession, which performs as the camera provider and
   * the manager for resources.
   * @param root The root framework session to be associated with
   */
  explicit Pipeline(std::shared_ptr<FrameworkSession> root);

  Pipeline(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = default;

  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  ~Pipeline() override;

  [[nodiscard]] std::shared_ptr<FrameworkSession> getRootSession() const { return m_rootSession; }

  // ------------------ The following methods each encapsulate one operator --------------------------- //
  // --- They add the encapsulated operators to the pipeline, but they are not executed until the ----- //
  // ----------------------------- pipeline is submitted for execution -------------------------------- //

  /**
   * Add an operator to conduct copy-by-value with auto type conversion from <code>src</code> tensor to
   * <code>dst</code> tensor to the pipeline.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO</code>
   * @param src The copy source, with no restriction on its attributes
   * @param dst The copy destination, no restriction on attributes but its channel and dimensions
   *            must match those of the <code>src</code>
   * @return Reference to this pipeline
   */
  Pipeline& typeConvert(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& dst);

  /**
   * Add to the pipeline an operator to perform
   * copy-by-value from <code>src</code> tensor to <code>dst</code> tensor.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO</code>
   * @param src Copy src
   * @param dst Copy destination
   * @return Reference to this pipeline
   */
  Pipeline& assignment(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& dst);

  /**
   * Add to the pipeline an operator to
   * copy all values from <code>src</code> tensor to a slice of <code>dst</code> tensor. For example, you can
   * use this method to extract the R-channel from an RGB image.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO</code>
   * @param src The source of the copy
   * @param dstSlice The destination tensor slice of the copy
   * @return Reference to this pipeline
   */
  Pipeline& assignment(const std::shared_ptr<PipelineTensor>& src, const PipelineTensor::Slice& dstSlice);
  /**
   * Add to the pipeline an operator to
   * copy all values from a slice <code>src</code> tensor to a <code>dst</code> tensor. For example, you
   * can use this method to assign a 3x1 translation vector to a 4x4 transform matrix.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO</code>
   * @param srcSlice The tensor slice as the copy source
   * @param dst The destination tensor
   * @return Reference to this pipeline
   */
  Pipeline& assignment(const PipelineTensor::Slice& srcSlice, const std::shared_ptr<PipelineTensor>& dst);
  /**
   * Add to the pipeline an operator to
   * copy values from one tensor slice to another.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO</code>
   * @param srcSlice The tensor slice as the copy source
   * @param dstSlice The tensor slice as the copy destination
   * @return Reference to this pipeline
   */
  Pipeline& assignment(const PipelineTensor::Slice& srcSlice, const PipelineTensor::Slice& dstSlice);
  /**
   * Add to the pipeline an operator to
   * conduct an elementwise comparison of two tensors, and write the compare result to the destination tensor.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO</code>
   * @param compare A comparison of two tensors, such as <code>myPipeline.compareTo(tensor1 > tensor2, result);</code>.
   *                The two tensors to form the comparison must have the same channel and the same size along each
   *                dimension.
   * @param dst The tensor to store the compare result, which must have the same size along each dimension as the
   *            tensors forming the comparison. The tensor must be of an integral datatype. The number of channel
   *            must be the same as the tensors forming the comparison.
   * @return Reference to this pipeline
   */
  Pipeline& compareTo(const PipelineTensor::Compare& compare, const std::shared_ptr<PipelineTensor>& dst);
  /**
   * Add to the pipeline an operator to
   * evaluate an arithmetic expression, such as <code>{0} + {1} / 2</code>.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO</code>
   * @param expression The arithmetic expression, where you can use <code>{IDX}</code> to refer to the No. IDX
   *                   operands. We support +, -, *, / and ().
   * @param ops Operands to the arithmetic expression. The usage type of all the operands must be declared as
   *            <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result The result from the arithmetic expression, whose usage type must also be
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& arithmetic(const std::string& expression, const std::vector<std::shared_ptr<PipelineTensor>>& ops,
                       const std::shared_ptr<PipelineTensor>& result);
  enum class ElementwiseOp { MIN, MAX, MULTIPLY, OR, AND };
  /**
   * Add to the pipeline an operator to
   * perform elementwise operation on tensors.
   * Encapsulating operators of type:
   * <ul>
   * <li> <code>XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO</code> </li>
   * <li> <code>XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MAX_PICO</code> </li>
   * <li> <code>XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO</code> </li>
   * <li> <code>XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_OR_PICO</code> </li>
   * <li> <code>XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_AND_PICO</code> </li>
   * </ul>
   * @param operation An enum, to determine what elementwise operator to be performed
   * @param ops Two operand tensors to the elementwise operation. They must be of the same dimensions and channels
   * @param result The result tensor, which must be of the same dimensions and channels as each of the ops.
   * @return Reference to this pipeline
   */
  Pipeline& elementwise(ElementwiseOp operation, const std::array<std::shared_ptr<PipelineTensor>, 2>& ops,
                        const std::shared_ptr<PipelineTensor>& result);
  /**
   * Add to the pipeline an operator to perform ALL on given tensor: TRUE (non-zero) if all values in the tensor
   * is non-zero, FALSE otherwise.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO</code>
   * @param op input tensor
   * @param result the result tensor, must be a tensor of single 1-channel integral value,
   *               with usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& all(const std::shared_ptr<PipelineTensor>& op, const std::shared_ptr<PipelineTensor>& result);
  /**
   * Add to the pipeline an operator to perform ANY on given tensor: TRUE (non-zero) if any values in the
   * tensor is non-zero, FALSE otherwise.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO</code>
   * @param op input tensor
   * @param result the result tensor, must be a tensor of single 1-channel integral value,
   *               with usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& any(const std::shared_ptr<PipelineTensor>& op, const std::shared_ptr<PipelineTensor>& result);
  /**
   * Add to the pipeline an operator to compute NMS on N bounding boxes, producing the top-M bounding boxes sorted
   * by their scores.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO</code>
   * @param scores a (N, 1) or (1, N) tensor of 1-channel floating point values for bounding box
   *               scores, requiring usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param boxes a (N, 1) or (1, N) tensor of 4-channel floating point values, or a (N, 4) tensors of 1-channel
   *              floating point values, describing the source bounding boxes in XYXY format,
   *              requiring usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_scores a (M, 1) or (1, M) tensor of 1-channel floating point values for scores of the bounding box
   *               passed the NMS filtering, requiring usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>,
   *               sorted by scores from highest to lowest.
   * @param result_boxes a (M, 1) or (1, M) tensor of 4-channel floating point values, or a (M, 4) tensors of 1-channel
   *              floating point values, to store the result bounding boxes passed in XYXY format,
   *              requiring usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>, corresponding to the order
   *              in parameter <code>result_scores</code>
   * @param result_indices a (M, 1) or (1, M) tensor of 1-channel integral values for the indices of the bounding box
   *               passed the NMS filtering in the input <code>boxes</code> tensor, requiring usage flag =
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>, corresponding to the order
   *               in parameter <code>result_scores</code>
   * @param threshold the IOU threshold
   * @return Reference to this pipeline
   */
  Pipeline& nms(const std::shared_ptr<PipelineTensor>& scores, std::shared_ptr<PipelineTensor>& boxes,
                const std::shared_ptr<PipelineTensor>& result_scores,
                const std::shared_ptr<PipelineTensor>& result_boxes,
                const std::shared_ptr<PipelineTensor>& result_indices, float threshold);
  /**
   * Add to the pipeline an operator to solve PnP. You may refer to
   * <a href="https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html">this OpenCV page</a>
   * regarding the principle and usage of solve PNP:
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO</code>
   * @param objectPoints 3D points, must be a 1D, (N, 1) or (1, N) tensor of 3-channel floating points
   * @param imgPoints 2D points, must be a 1D, (N, 1) or (1, N) tensor of 2-channel floating points
   * @param cameraMatrix camera matrix making the project from 3D to 2D, must be a (3, 3) tensor of 1-channel
   *                     floating points, with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_rotation the rotation vector from solve pnp, relative to the camera, must be a (3, 1) or
   *                        (1, 3) tensor of 1-channel floating points, with usage flag
   *                        <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_translation the translation vector from solve pnp, relative to the camera, must be a (3, 1) or
   *                        (1, 3) tensor of 1-channel floating points, with usage flag
   *                        <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& solvePnP(const std::shared_ptr<PipelineTensor>& objectPoints,
                     const std::shared_ptr<PipelineTensor>& imgPoints,
                     const std::shared_ptr<PipelineTensor>& cameraMatrix,
                     const std::shared_ptr<PipelineTensor>& result_rotation,
                     const std::shared_ptr<PipelineTensor>& result_translation);

  typedef std::variant<std::shared_ptr<PipelineTensor>, std::array<float, 6>> AffinePoints;

  /**
   * Add to the pipeline an operator to compute the affine matrix. You may refer to
   * <a href="https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html">this OpenCV page</a>
   * regarding the affine transform.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO</code>
   * @param srcPoints 3 source points, must be a tensor containing 3 2-channel floating point values
   * @param dstPoints 3 source points, must be a tensor containing 3 2-channel floating point values
   * @param result the affine transform, must be (2, 3) tensor of 1-channel floating mat, with usage flag =
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& getAffine(const AffinePoints& srcPoints, const AffinePoints& dstPoints,
                      const std::shared_ptr<PipelineTensor>& result);

  /**
   * Add to the pipeline an operator to apply affine on 2D image tensor
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO</code>
   * @param affine the affine transform matrix, must be (2, 3) tensor of 1-channel floating mat, with usage flag =
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param img the source image, must be a 2D tensor with usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_img the result image, must be a 2D tensor with usage flag =
   *                   <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& applyAffine(const std::shared_ptr<PipelineTensor>& affine, const std::shared_ptr<PipelineTensor>& img,
                        const std::shared_ptr<PipelineTensor>& result_img);
  /**
   * Add to the pipeline an operator to 2D points
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO</code>
   * @param affine the affine transform matrix, must be (2, 3) tensor of 1-channel floating mat, with usage flag =
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param points the source points, must contain N  2-channel floating point values, i.e., 1D tensor of
   *               2-channel values, or (N, 1) or (1, N) tensor of 2-channel values
   * @param result_points the result points after affine, must containing N 2-chanel floating point values
   * @return Reference to this pipeline
   */
  Pipeline& applyAffinePoint(const std::shared_ptr<PipelineTensor>& affine,
                             const std::shared_ptr<PipelineTensor>& points,
                             const std::shared_ptr<PipelineTensor>& result_points);
  /**
   * Add to the pipeline an operator to use the on-device depth sensor to get the 3D coordinates relative to
   * the <b>left-eye</b> camera from pixel coordinates on the left-eye image.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO</code>
   * @param uv The 2D points to be queried. Must be a 1D tensor of N 2-channel INT32 points. The usage flag
   *           must be <code>XR_SECURE_MR_TENSOR_TYPE_POINT_PICO</code>.
   *           Note, the point (0, 0) is the image's top-left corner. The first channel is the column ID of the
   *           queried pixels and the second channel is the row ID.
   * @param timestamp the camera timestamp when the image is taken, must come from <code>Pipeline::cameraAccess</code>
   * @param cameraMatrix the camera matrix when the image is taken, must come from <code>Pipeline::cameraAccess</code>
   * @param leftImg the left-eye image, must come from <code>Pipeline::cameraAccess</code>
   * @param rightImg the right-eye image, must come from <code>Pipeline::cameraAccess</code>
   * @param result The 3D coordinates of the queried pixels in left-eye camera space, can be 1D 3-channel tensor,
   *               a 2D (N, 1) or (1, N) 3-channel tensor, or a 2D (N, 3) tensor. The datatype must be either
   *               float or double.
   * @return Reference to this pipeline
   */
  Pipeline& uv2Cam(const std::shared_ptr<PipelineTensor>& uv, const std::shared_ptr<PipelineTensor>& timestamp,
                   const std::shared_ptr<PipelineTensor>& cameraMatrix, const std::shared_ptr<PipelineTensor>& leftImg,
                   const std::shared_ptr<PipelineTensor>& rightImg, const std::shared_ptr<PipelineTensor>& result);
  enum class NormalizeType {
    L1 = XR_SECURE_MR_NORMALIZE_TYPE_L1_PICO,
    L2 = XR_SECURE_MR_NORMALIZE_TYPE_L2_PICO,
    MINMAX = XR_SECURE_MR_NORMALIZE_TYPE_MINMAX_PICO,
    INF = XR_SECURE_MR_NORMALIZE_TYPE_INF_PICO
  };
  /**
   * Add to the pipeline an operator to normalize a tensor
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO</code>
   * @param src the source tensor to be normalized
   * @param result the normalization result
   * @param type the normalize type
   * @return Reference to this pipeline
   */
  Pipeline& normalize(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& result,
                      NormalizeType type = NormalizeType::L2);
  /**
   * Add to the pipeline an operator to query the transform matrix for left/right-eye camera to
   * OpenXR's <code>XR_REFERENCE_SPACE_TYPE_LOCAL</code> at the time when the image is taken
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO</code>
   * @param timestamp the camera timestamp when the image is taken, must come from <code>Pipeline::cameraAccess</code>
   * @param result_rightEyeTransform The transform matrix from right-eye camera to
   *                                 <code>XR_REFERENCE_SPACE_TYPE_LOCAL</code>. Must be a (4, 4) matrix of 1-channel
   *                                 floating point values, usage flag = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_leftEyeTransform The transform matrix from left-eye camera to
   *                                 <code>XR_REFERENCE_SPACE_TYPE_LOCAL</code>. Must be a (4, 4) matrix of 1-channel
   *                                 floating point values, usage flat = <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& camSpace2XrLocal(const std::shared_ptr<PipelineTensor>& timestamp,
                             const std::shared_ptr<PipelineTensor>& result_rightEyeTransform,
                             const std::shared_ptr<PipelineTensor>& result_leftEyeTransform);
  /**
   * Add to the pipeline an operator to get the latest camera image for both eyes
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO</code>
   * @param result_rightEye the image for right eye, must be a 2D tensor of the same dimensions as the framework
   *                        session's width and height, 3-channel UINT8, usage flag =
   *                        <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_leftEye the image for left eye, must be a 2D tensor of the same dimensions as the framework
   *                       session's width and height, 3-channel UINT8, usage flag =
   *                       <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result_timeStamp timestamp when the image is taken, must be a 1D tensor of single 4-channel INT32
   *                         values, of usage flag <code>XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO</code>
   * @param result_camMatrix camera intrinsic matrix when the image is taken, must be a 2D (3, 3) tensor of 1-channel
   *                         floating point values, of usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& cameraAccess(const std::shared_ptr<PipelineTensor>& result_rightEye,
                         const std::shared_ptr<PipelineTensor>& result_leftEye,
                         const std::shared_ptr<PipelineTensor>& result_timeStamp,
                         const std::shared_ptr<PipelineTensor>& result_camMatrix);
  /**
   * Add to the pipeline an operator to conduct channel-wise arg max
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO</code>
   * @param src input tensor
   * @param result_indexPerChannel indices to the max on each channel. It must be a tensor of C values, where
   *                               C matches the input tensor's number of channels. The tensor itself must have
   *                               D channels, where D matches the number of dimensions of the input tensor.
   * @return Reference to this pipeline
   */
  Pipeline& argMax(const std::shared_ptr<PipelineTensor>& src,
                   const std::shared_ptr<PipelineTensor>& result_indexPerChannel);
  /**
   * Add to the pipeline an operator to convert color space
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO</code>
   * @param convertFlag The OpenCV color conversion flag
   * @param image the source image tensor, must be a 2D tensor with usage flag
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * @param result the result image tensor, must be a 2D tensor with usage flag
   *               <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>, and of the same sizes along dimensions as the
   *               source image.
   * @return Reference to this pipeline
   */
  Pipeline& cvtColor(int convertFlag, const std::shared_ptr<PipelineTensor>& image,
                     const std::shared_ptr<PipelineTensor>& result);
  /**
   * Add to the pipeline an operator to sort a 1D 1-channel tensor
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO</code>
   * @param srcVec the 1D tensor to be sorted
   * @param result_sortedVec the result tensor, must be of the same size as the srcVec
   * @param result_indices the indices in srcVec of each value in the sorted tensor, must be of the same size as
   *                       the srcVec, and the datatype must be integer
   * @return Reference to this pipeline
   */
  Pipeline& sortVec(const std::shared_ptr<PipelineTensor>& srcVec,
                    const std::shared_ptr<PipelineTensor>& result_sortedVec,
                    const std::shared_ptr<PipelineTensor>& result_indices);
  /**
   * Add to the pipeline an operator to sort 2D 1-channel tensor (matrix) row-by-row.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO</code>
   * with config <code>XR_SECURE_MR_MATRIX_SORT_TYPE_ROW_PICO</code>
   * @param srcMat the sorting source, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param result_sortedMat the sorted result, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param result_indicesPerRow the column indices in srcMat of each value in the sorted tensor per row
   * @return Reference to this pipeline
   */
  Pipeline& sortMatByRow(const std::shared_ptr<PipelineTensor>& srcMat,
                         const std::shared_ptr<PipelineTensor>& result_sortedMat,
                         const std::shared_ptr<PipelineTensor>& result_indicesPerRow);
  /**
   * Add to the pipeline an operator to sort 2D 1-channel tensor (matrix) column-by-column.
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO</code>
   * with config <code>XR_SECURE_MR_MATRIX_SORT_TYPE_ROW_PICO</code>
   * @param srcMat the sorting source, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param result_sortedMat the sorted result, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param result_indicesPerColumn the row indices in srcMat of each value in the sorted tensor per column
   * @return Reference to this pipeline
   */
  Pipeline& sortMatByColumn(const std::shared_ptr<PipelineTensor>& srcMat,
                            const std::shared_ptr<PipelineTensor>& result_sortedMat,
                            const std::shared_ptr<PipelineTensor>& result_indicesPerColumn);

  /**
   *
   * Perform the singularity value decomposition (SVD) of a matrix
   *
   * @param src required, the matrix to be decomposed, must be a multi-dimensional tensor with 2 dimensions, therefore
   *    usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required銆?The shapes along the 2 dimensions
   *    must be same, i.e., must be a square matrix.
   * @param result_w An optional result, the w result of the decomposition, usage flag
   *    <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required, and the tensor must have only two dimensions.
   * @param result_u An optional result, the u result of the decomposition, usage flag
   *    <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required, and the tensor must have only two dimensions.
   * @param result_vt An optional result, the vt result of the decomposition, usage flag
   *    <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required, and the tensor must have only two dimensions.
   * @return Reference to this pipeline
   */
  Pipeline& singularValueDecomposition(const std::shared_ptr<PipelineTensor>& src,
                                       const std::shared_ptr<PipelineTensor>& result_w,
                                       const std::shared_ptr<PipelineTensor>& result_u,
                                       const std::shared_ptr<PipelineTensor>& result_vt);

  /**
   * Compute the norm of a tensor (by default, L2 norm)
   *
   * @param src required, the tensor whose norm will be computed銆?It can be any tensor but the glTF tensor, i.e.,
   *    the src tensor must not have <code>XR_SECURE_MR_TENSOR_TYPE_GLTF_PICO</code> usage flag.
   * @param result_norm required, the tensor to store the norm of <code>src</code>. It must be of only 1 channel, and
   *    contains only one element. Hence, the tensor can only be: <ol>
   *    <li>A scalar array tensor with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>, of size 1, or</li>
   *    <li>A scalar array tensor with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>, with
   *        dimensions = <code>(1, 1)</code></li>
   *    </ol>
   * @return Reference to this pipeline
   */
  Pipeline& norm(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& result_norm);

  /**
   * Convert the HWC and CHW tensors. An HWC tensor is a 2-dimension tensor, with dimensions = <code>(H, W)</code> and
   * <code>C</code> channels. A CHW tensor is a 3-dimension tensor, with dimensions = <code>(C, H, W)</code> and
   * only 1 channel. If the input is an HWC tensor, the method converts it to a CHW one; otherwise, converting the
   * CHW input to HWC output.
   *
   * @param src required, the input tensor to be converted. Must either be a CHW or an HWC tensor, usage flag
   *    <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required, and number of dimensions must be either 2 or 3.
   * @param result required, the output of the conversion. If the <code>src</code> is a CHW tensor, the
   * <code>result</code> must be an HWC tensor. If the <code>src</code> is an HWC tensor, the <code>result</code> must
   * be a CHW tensor. usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required, and number of dimensions must
   * be either 2 or 3. The last 2 dimensions of <code>result</code> must be the same as those of <code>src</code>
   * @return Reference to this pipeline
   */
  Pipeline& convertHWC_CHW(const std::shared_ptr<PipelineTensor>& src, const std::shared_ptr<PipelineTensor>& result);

  /**
   * Add to the pipeline an operator to conduct matrix inversion
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO</code>
   * @param srcMat a tensor 2D 1-channel square matrix, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * required
   * @param result_inverted the result matrix, must have the same attribute as the srcMat
   * @return Reference to this pipeline
   */
  Pipeline& inversion(const std::shared_ptr<PipelineTensor>& srcMat,
                      const std::shared_ptr<PipelineTensor>& result_inverted);
  /**
   * Add to the pipeline an operator to make a 4x4 transform
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MAT_PICO</code>
   * @param rotation a (3, 1) or (1, 3) tensor of 1-channel floating points describing the rotation vector,
   *                 usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param translation a (3, 1) or (1, 3) tensor of 1-channel floating points describing the translation vector,
   *                    usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param scale a (3, 1) or (1, 3) tensor of 1-channel floating points describing the scale vector,
   *              usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @param result The result transform matrix, a (4, 4) tensor of 1-channel floating points,
   *               usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code> required
   * @return Reference to this pipeline
   */
  Pipeline& transform(const std::shared_ptr<PipelineTensor>& rotation,
                      const std::shared_ptr<PipelineTensor>& translation, const std::shared_ptr<PipelineTensor>& scale,
                      const std::shared_ptr<PipelineTensor>& result);
  /**
   * Add to the pipeline an operator to create a new texture to a glTF object from a tensor
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO</code>
   * @param gltfPlaceholder the target glTF object
   * @param textureSrc the tensor as the texture src, usage flag <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   *                   required. Must be a 2D tensor of 3-/4-channel UINT8 values, for RGB and RGBA texture
   *                   respectively.
   * @param result_newTextureId The texture ID to the newly created texture, must be a 1D tensor of a single 1-channel
   *                            UINT16 value, with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   * @return Reference to this pipeline
   */
  Pipeline& newTextureToGLTF(const std::shared_ptr<PipelineTensor>& gltfPlaceholder,
                             const std::shared_ptr<PipelineTensor>& textureSrc,
                             const std::shared_ptr<PipelineTensor>& result_newTextureId);
  /**
   * Add a render command to the pipeline.
   *
   * Encapsulating render-related operator of different types, determined by parameter <code>command</code>
   * @param command The render command to be executed
   * @return Reference to this pipeline
   */
  Pipeline& execRenderCommand(const std::shared_ptr<RenderCommand>& command);
  /**
   * Add a customized algorithm operator the pipeline from a loaded binary algorithm package.
   *
   * Encapsulating operator of type <code>XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO</code>
   * @param algPackageBuf the memory buffer to the loaded binary algorithm package
   * @param algPackageSize the size of the memory buffer of the binary algorithm package
   * @param algOps A mapping from operand name to pipeline tensor to be used as the operand
   * @param operandAliasing A mapping from the operand name to the internal node ID inside the algorithm package. If
   *                        an operand in parameter <code>algOps</code> is not given here, the operand name will be used
   *                        directly as the internal node ID.
   * @param algResults A mapping from result name to pipeline tensor to be used as the result
   * @param resultAliasing A mapping from the result name to the internal node ID inside the algorithm package. If
   *                       a result in parameter <code>algResults</code> is not given here, the result name will be used
   *                       directly as the internal node ID.
   * @param modelName An identifier to the operator
   * @return Reference to this pipeline
   */
  Pipeline& runAlgorithm(char* algPackageBuf, size_t algPackageSize,
                         const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& algOps,
                         const std::unordered_map<std::string, std::string>& operandAliasing,
                         const std::unordered_map<std::string, std::shared_ptr<PipelineTensor>>& algResults,
                         const std::unordered_map<std::string, std::string>& resultAliasing,
                         const std::string& modelName);

  /**
   * Submit the pipeline to be executed. The architecture (pipeline tensors, operators) of the pipeline will be frozen
   * until the execution is finished. Executions submitted from the same pipeline will be executed in the submission
   * order. Executions submitted from different pipelines may be executed in parallel if they are not competing on the
   * same global tensor.
   * @param argumentMap A mapping from pipeline local placeholders to the referred global tensors. Note the mapping
   *                    is only valid for one submission. Each submission can use different tensor mappings.
   * @param waitFor If set, the submission will not be executed until the execution of specified pipeline submission
   *                is completed. Note the executions from the same pipeline will be executed in the submission order,
   *                so there is no need to wait for the execution from the same pipeline.
   * @param condition If set, the execution will be aborted if the condition tensor is all zero when the execution
   *                  happens.
   * @return The run handle to track this execution of this submission
   */
  XrSecureMrPipelineRunPICO submit(
      const std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>>& argumentMap,
      XrSecureMrPipelineRunPICO waitFor, const std::shared_ptr<GlobalTensor>& condition);
};

}  // namespace SecureMR

#endif  // PIPELINE_H
