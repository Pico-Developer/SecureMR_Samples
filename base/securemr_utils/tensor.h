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

#ifndef ATENSOR_H
#define ATENSOR_H
#include <array>
#include <utility>
#include <variant>
#include <vector>

#include "adapter.hpp"
#include "check.h"

namespace SecureMR {
class Pipeline;

/**
 * Describing a tensor (for both global and local tensors)
 */
struct TensorAttribute {
  /**
   * Size along each dimension of the tensor
   */
  std::vector<int> dimensions{};
  /**
   * Number of channels. Observing the OpenCV's conversion, channel is not regarded as one dimension, but
   * as a part of the datatype. For example, a 768x1024 R8G8B8 image is regarded as a tensor:
   *
   * <ul>
   * <li> of <b>two</b> dimensions: 768 along the first dimension, and 1024 along the second</li>
   * <li> whose datatype is 3-channel UINT8</li>
   * </ul>
   *
   * which means, the image has 768x1024 = 786432 elements, each of which consists of 3 UINT8 values.
   */
  int8_t channels = 1;
  /**
   * Usage flag of the tensor, which determines how the values will be interpreted by operators during run time.
   * You will find the default MAT usage, i.e., <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   * is compatible for most scenarios. As the name suggests, this default usage type requires the tensor has
   * at least 2 dimension. If you want to create a vector, you need to specify the <code>dimensions</code> to be
   * (N, 1) or (1, N) to distinguish row v.s. column vectors.
   * <br/>
   * Yet, the other usage types can be applied for cases (you may find the
   * following <code>TensorAttribute_xxx</code> structures useful to create tensors for these cases):
   *
   * <ul>
   *
   * <li> <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code> for a simple array of values, which requires channel to be
   *      1 and there must be one dimension only (different from a single-dimensional vector of MAT usage). </li>
   * <li> <code>XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO</code> to describe a python-style slice, such as [BEING:END] or
   *      [BEING:END:SKIP]. Hence, the channel must either be 2 or 3, and there must be one dimension only.</li>
   * <li> <code>XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO</code> to describe a <code>longlong</code> timestamp. The
   *      channel must be 4 and field <code>dimensions</code> must be <code>{1}</code>. </li>
   * <li> <code>XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO</code> to describe an array of RGBA or RGB colors. Hence,
   *       intuitively, the channel must be 3 or 4, and the number of dimensions must be 1</li>
   * <li> <code>XR_SECURE_MR_TENSOR_TYPE_POINT_PICO</code> to describe an array of 2D points (XY, XY, ...), or 3D
   *      points, (XYZ, XYZ, ...). Hence, intuitively, the channel must be 2 or 3, and the number of dimensions
   *      must be 1</li>
   *
   * </ul>
   */
  XrSecureMrTensorTypePICO usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO;
  /**
   * The basic datatype for each value
   */
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;
};

struct TensorAttribute_ScalarArray {
  size_t size = 1;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = 1,
            .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
            .dataType = dataType};
  }
};

struct TensorAttribute_Point2Array {
  size_t size = 1;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = 2,
            .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
            .dataType = dataType};
  }
};

struct TensorAttribute_Point3Array {
  size_t size = 1;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = 3,
            .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
            .dataType = dataType};
  }
};

struct TensorAttribute_RGB_Array {
  size_t size = 1;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = 3,
            .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO};
  }
};

struct TensorAttribute_RGBA_Array {
  size_t size = 1;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = 4,
            .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO};
  }
};

struct TensorAttribute_TimeStamp {
  operator TensorAttribute() const {
    return {.dimensions = {1},
            .channels = 4,
            .usage = XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO};
  }
};

struct TensorAttribute_SliceArray {
  size_t size = 1;
  bool hasSkip = false;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO;

  operator TensorAttribute() const {
    return {.dimensions = {static_cast<int>(size)},
            .channels = static_cast<int8_t>(hasSkip ? 3 : 2),
            .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
            .dataType = dataType};
  }
};

/**
 * Adapter of <code>XrSecureMrTensorPICO</code>, which represents a global tensor.
 * A global tensor is a tensor that is shared between pipelines. As pipelines are executed in different threads
 * in parallel, global tensors are used for inter-thread data transmission.
 * <br/>
 * <b>NOTE</b> to avoid parallel read/write to the same global tensor, methods in pipeline cannot directly use
 * global tensors as inputs or outputs. A pipeline-local <i>placeholder</i> must be declared as reference to
 * global tensors and pipeline method can only use placeholders for indirect access.
 * <br/>
 * The same placeholder can refer to different global tensor in different pipeline submission, to allow the same
 * pipeline to be applied on different data.
 */
class GlobalTensor : public XrHandleAdapter<XrSecureMrTensorPICO> {
 private:
  std::shared_ptr<FrameworkSession> m_session = nullptr;
  std::variant<std::monostate, TensorAttribute> m_attribute{};

 protected:
  PFN_xrCreateSecureMrTensorPICO xrCreateSecureMrTensorPICO = nullptr;
  PFN_xrDestroySecureMrTensorPICO xrDestroySecureMrTensorPICO = nullptr;
  PFN_xrResetSecureMrTensorPICO xrResetSecureMrTensorPICO = nullptr;

 public:
  /**
   * Create a global tensor
   * @param session The framework session to which the global tensor's lifespan will be associated to
   * @param attribute The tensor's attribute
   */
  GlobalTensor(const std::shared_ptr<FrameworkSession>& session, TensorAttribute attribute);

  /**
   * Create a global tensor while set the initial data for it at the mean time.
   * <br/>
   * <b>NOTE</b> this only works for non-glTF tensors.
   * @param session The framework session to which the global tensor's lifespan will be associated to
   * @param attribute The tensor's attribute
   * @param data The start address of the data to be written.
   * @param size The size of the data. If the data size is smaller than the tensor's size, the data will be duplicated
   *             to fill the entire tensor. Hence, the tensor's size must be divisible by the data size
   */
  GlobalTensor(const std::shared_ptr<FrameworkSession>& session, TensorAttribute attribute, int8_t* data, size_t size);

  /**
   * Create a glTF tensor. In the implementation of SecureMR, a glTF object is treated as a tensor as well,
   * but of 0 dimension, 0 channel, none datatype and special GLTF usage flag.
   * <br/>
   * The glTF object can thereafter be rendered and updated using <code>RenderCommand</code> in pipelines,
   * to perform MR effects and data-driven animations.
   * <br/>
   * The glTF file describing the object to be rendered must be preloaded into memory. Additionally, the
   * file must be an embedded glTF. Binary glTF are not supported yet.
   *
   * @param session The framework session to which the global tensor's lifespan will be associated to
   * @param gltfContent The start of the memory containing the loaded glTF file
   * @param size The size of the memory containing the loaded glTF file
   */
  GlobalTensor(const std::shared_ptr<FrameworkSession>& session, char* gltfContent, size_t size);
  /**
   * Make a copy of the tensor. This will create a new global tensor of the same attributes. But the
   * content will not be copied unless you use <code>Pipeline::assignment</code>.
   *
   * A glTF tensor cannot be copied.
   *
   * @param other The tensor to be copied.
   */
  GlobalTensor(const GlobalTensor& other);
  GlobalTensor(GlobalTensor&& other) = default;
  ~GlobalTensor() override;

  GlobalTensor& operator=(GlobalTensor& other) = delete;
  GlobalTensor& operator=(GlobalTensor&& other) = delete;

  /**
   * Write values to the tensor.
   * @param data The start address of the data to be written.
   * @param size The size of the data. If the data size is smaller than the tensor's size, the data will be duplicated
   *             to fill the entire tensor. Hence, the tensor's size must be divisible by the data size
   */
  void setData(int8_t* data, size_t size) const;

  [[nodiscard]] std::variant<std::monostate, TensorAttribute> getAttribute() const { return m_attribute; }

  /**
   * A syntax sugar for <code>setData</code>
   */
  template <typename T>
  GlobalTensor& operator=(std::vector<T> input) {
    setData(reinterpret_cast<int8_t*>(input.data()), input.size() * sizeof(T));
    return *this;
  }
};

/**
 * Adapter for <code>XrSecureMrPipelineTensorPICO</code>, representation a tensor local to a pipeline.
 * <br/>
 * A pipeline tensor is the data carrier between operators, where operators are regarded as the pipeline's
 * nodes, and the pipeline tensors are the edges between nodes.
 * <br/>
 * If a pipeline tensor is backed by no underlying storage, it is a placeholder tensor. A placeholder must
 * refer to some global tensors (of the same attribute) when its pipeline is submitted for executed. Reading
 * from or writing to a placeholder in face read from or write to the referred global tensor.
 */
class PipelineTensor : public XrHandleAdapter<XrSecureMrPipelineTensorPICO>,
                       public std::enable_shared_from_this<PipelineTensor> {
 private:
  std::shared_ptr<Pipeline> m_pipeline = nullptr;
  /**
   * If m_attribute is a std::monostate, the tensor is representing a glTF object.
   */
  std::variant<std::monostate, TensorAttribute> m_attribute{};
  bool isPlaceholder = false;

 protected:
  PFN_xrCreateSecureMrPipelineTensorPICO xrCreateSecureMrPipelineTensorPICO;
  PFN_xrResetSecureMrPipelineTensorPICO xrResetSecureMrPipelineTensorPICO;

 public:
  /**
   * Describe a comparison of two pipeline tensors, to be used by
   * <code>Pipeline::compareTo</code>.
   * <br/>
   * A <code>Compare</code> object can be constructed by simply using the override comparator between two pipeline
   * tensors, such as <code>Compare largeThan = pipelineTensor1 > pipelineTensor2</code>
   */
  struct Compare {
    std::shared_ptr<const PipelineTensor> left;
    std::shared_ptr<const PipelineTensor> right;

    XrSecureMrComparisonPICO comparison = XR_SECURE_MR_COMPARISON_EQUAL_TO_PICO;
  };

  /**
   * Describe a slice on a pipeline tensor, to be used by <code>Pipeline::assignment</code>.
   * <br/>
   * A <code>Slice</code> object can be constructed by simply using the override [..] on a pipeline tensor, for example
   * <ul>
   * <li> <code>pipelineTensor1D[2]</code> select the third element from the tensor </li>
   * <li> <code>pipelineTensor2D[{2, 0}]</code> select the element at row 2, column 0 </li>
   * <li> <code>pipelineTensor2D[{{0,5}, {0,2}}]</code> select the rectangle region of the first 5 rows and the first 2
   *      columns from the tensor </li>
   * <li> <code>pipelineTensor1D[{{10,-1,-1}}]</code> performs a backward selection from the element indexing 10 to
   *      the start of the tensor </li>
   * </ul>
   * <br/>
   * You can refer to the OpenXR specification for <code>XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO</code> for more details
   * about the slicing on tensors.
   *
   * You can apply a secondary <code>[IDX...]</code> to further slicing on tensor's channels, such as:
   * <code> rgbTensor[{{0,256}, {0,256}}][1] </code> gives a slice on the G channel of the top left 256x256 subregion
   * of the RGB tensor.
   */
  class Slice {
   private:
    std::shared_ptr<const PipelineTensor> m_tensor;
    std::shared_ptr<const PipelineTensor> m_slices;
    std::shared_ptr<const PipelineTensor> m_channelSlice;

   public:
    Slice(const std::shared_ptr<PipelineTensor>& tensor, const std::shared_ptr<PipelineTensor>& slices);

    Slice(const Slice& other) = default;
    Slice(Slice&&) = default;
    Slice& operator=(const Slice& other) = default;
    Slice& operator=(Slice&& other) = default;

    /**
     * Take a channel slice on a slice
     * @param channelSlice the tensor used as the channel slice, which must be a 1D tensor of 2 or 3 channels,
     *        of one element only, with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO</code>
     * @return Reference to this
     */
    Slice& operator[](const std::shared_ptr<PipelineTensor>& channelSlice);
    /**
     * Take a channel slice on a slice
     * @param channelSliceStatic a syntax sugar to use a static channel slice. The 3 values shall be [BEGIN, END, SKIP]
     *                           of the channel slice.
     * @return Reference to this
     */
    Slice& operator[](std::array<int, 3> channelSliceStatic);
    /**
     * Take a channel slice on a slice
     * @param channelSliceStatic a syntax sugar to use a static channel slice. The 2 values shall be [BEGIN, END]
     *                           of the channel slice.
     * @return Reference to this
     */
    Slice& operator[](std::array<int, 2> channelSliceStatic);
    /**
     * Take a channel slice on a slice
     * @param index a syntax sugar to use a static channel slice. The index shall is the index of the desired channel
     *               slice to be sliced.
     * @return Reference to this
     */
    Slice& operator[](int index);

    [[nodiscard]] XrSecureMrPipelineTensorPICO targetTensor() const {
      return static_cast<XrSecureMrPipelineTensorPICO>(*m_tensor);
    }

    [[nodiscard]] XrSecureMrPipelineTensorPICO sliceTensor() const {
      return static_cast<XrSecureMrPipelineTensorPICO>(*m_slices);
    }

    [[nodiscard]] bool hasChannelSlice() const { return m_channelSlice != nullptr; }

    [[nodiscard]] XrSecureMrPipelineTensorPICO channelSliceTensor() const {
      return static_cast<XrSecureMrPipelineTensorPICO>(*m_channelSlice);
    }
  };

  /**
   * Create a pipeline tensor according to tensor attributes.
   * <br/>
   * <b>NOTE</b> this constructor cannot be called to create a local placeholder for glTF tensor. To represent a
   * glTF object, you need to create tensor using the static function <code>PipelineGLTFPlaceholder</code> defined
   * below.
   * @param pipeline The SecureMR pipeline in which the pipeline tensor is local to
   * @param attribute Describing the desired tensor attributes
   * @param isPlaceholder Whether the pipeline tensor should have underlying storage, if <code>true</code>, the
   *                      result tensor will be a pipeline placeholder, with no storage and requiring to refer to
   *                      some global tensors before executing the pipeline.
   */
  PipelineTensor(std::shared_ptr<Pipeline> pipeline, TensorAttribute attribute, bool isPlaceholder = false);

  /**
   * Create a pipeline tensor while set data
   * <b>NOTE</b> since the data is provided, the pipeline tensor being created can not be a placeholder
   * @param pipeline The SecureMR pipeline in which the pipeline tensor is local to
   * @param attribute Describing the desired tensor attributes
   * @param data The start address of the data to be written.
   * @param size The size of the data. If the data size is smaller than the tensor's size, the data will be duplicated
   *             to fill the entire tensor. Hence, the tensor's size must be divisible by the data size
   */
  PipelineTensor(std::shared_ptr<Pipeline> pipeline, TensorAttribute attribute, int8_t* data, size_t size);

  /**
   * Create a pipeline placeholder to refer to a glTF tensor
   * @param root The SecureMR pipeline in which the pipeline tensor is local to
   * @return The pipeline placeholder tensor for a glTF tensor reference
   */
  static std::shared_ptr<PipelineTensor> PipelineGLTFPlaceholder(const std::shared_ptr<Pipeline>& root);
  /**
   * A helper function to create a placeholder easily, without needing to know the attributes of the global tensor
   * to be referred. The function will copy the attributes from the parameter <code>like</code> and use to it
   * build a placeholder.
   *
   * @param root The SecureMR pipeline in which the pipeline tensor is local to
   * @param like The global tensor providing target attributes
   * @return The placeholder tensor created using the attributes from <code>like</code>
   */
  static std::shared_ptr<PipelineTensor> PipelinePlaceholderLike(const std::shared_ptr<Pipeline>& root,
                                                                 const std::shared_ptr<GlobalTensor>& like);

  PipelineTensor(const PipelineTensor& other);
  PipelineTensor(PipelineTensor&& other) = default;

  PipelineTensor& operator=(PipelineTensor& other) = delete;
  PipelineTensor& operator=(PipelineTensor&& other) = delete;

  [[nodiscard]] std::shared_ptr<Pipeline> getPipeline() const { return m_pipeline; }
  [[nodiscard]] std::variant<std::monostate, TensorAttribute> getAttribute() const { return m_attribute; }

  /**
   * Write values to the tensor.
   * @param data The start address of the data to be written.
   * @param size The size of the data. If the data size is smaller than the tensor's size, the data will be duplicated
   *             to fill the entire tensor. Hence, the tensor's size must be divisible by the data size
   */
  void setData(int8_t* data, size_t size) const;

  /**
   * A syntax sugar to <code>setData</code>
   */
  template <typename T>
  PipelineTensor& operator=(std::vector<T> input) {
    setData(reinterpret_cast<int8_t*>(input.data()), input.size() * sizeof(T));
    return *this;
  }

  /**
   * Syntax sugar to quickly create a slice on <b>dimensions</b>
   * @param slices a vector of tuples. The tuples must all be of format {START, END} or {START, END, SKIP}, which
   *               describes the slice along each dimension. Hence, the size of parameter <code>slices</code> must
   *               match the number of dimensions of this tensor. For example, a valid usage for a 2D tensor can be:
   *               <code>tensor2D[{{0, 10}, {2, 4}}]</code> which gives the slice of the tensor's first 10 rows and
   *               the 2~3 columns.
   * @return The slice on the tensor's dimensions. You can thereafter apply <code>[IDX...]</code> on the returned slice
   *         object to further slice the channel of the tensor.
   */
  Slice operator[](const std::vector<std::vector<int>>& slices);
  /**
   * Syntax sugar to quickly create a single-element slice on <b>dimensions</b>
   * @param slices a vector of indices along each dimension. For example, a valid usage for a 2D tensor can be:
   *               <code>tensor2D[{0, 0}]</code> which gives the first element.
   * @return The slice on the tensor's dimensions. You can thereafter apply <code>[IDX...]</code> on the returned slice
   *         object to further slice the channels the tensor.
   */
  Slice operator[](const std::vector<int>& slices);
  /**
   * Syntax sugar to quickly create a single-element slice on <b>dimensions</b>
   * @param sliceTensor A tensor used as the tensor's slice, which must be a 1D tensor of 2 or 3 channels,
   *               of one element only, with usage flag <code>XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO</code>
   * @return The slice on the tensor's dimensions. You can thereafter apply <code>[IDX...]</code> on the returned slice
   *         object to further slice the channels the tensor.
   */
  Slice operator[](const std::shared_ptr<PipelineTensor>& sliceTensor);
  /**
   * Syntax sugar to quickly create a single-element slice on a 1D tensor
   * @param index The index along the <i>only</i> dimension of the tensor.
   * @return The slice on the tensor's dimensions. You can thereafter apply <code>[IDX...]</code> on the returned slice
   *         object to further slice the channels the tensor.
   */
  Slice operator[](int index);

  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator>(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator>(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator>(other);
  }
  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator<(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator<(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator<(other);
  }
  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator>=(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator>=(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator>=(other);
  }
  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator<=(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator<=(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator<=(other);
  }
  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator==(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator==(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator==(other);
  }
  /**
   * A syntax sugar to quickly create a compare between two tensors, to be used by <code>Pipeline::compareTo</code>.
   */
  Compare operator!=(const std::shared_ptr<PipelineTensor>& other) const;
  /**
   * Compare directly to literal values. This pipeline tensor will be compare
   * against a implicitly-constructed PipelineTensor having the same attribute
   * as this one and containing the given literal values as its data.
   * @param compareBase the literal values this tensor is to be compared with
   */
  template <typename T>
  Compare operator!=(std::vector<T> compareBase) const {
    auto other = std::make_shared<PipelineTensor>(*this);
    *other = compareBase;
    return operator!=(other);
  }

  /**
   * Create an empty pipeline tensor. It is strongly recommended to call this constructor directly, as the result
   * object is lacking essential attributes. We suggest you to use the static method defined above.
   * @param pipeline The SecureMr pipeline to which the pipeline tensor is associated to
   */
  explicit PipelineTensor(std::shared_ptr<Pipeline> pipeline);
};
}  // namespace SecureMR

#endif  // ATENSOR_H
