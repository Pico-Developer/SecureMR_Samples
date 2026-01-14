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

#ifndef HELLO_XR_RENDERCOMMAND_H
#define HELLO_XR_RENDERCOMMAND_H

#include <utility>

#include "pipeline.h"
#include "tensor.h"
#include "check.h"

namespace SecureMR {

class PipelineTensor;
struct TensorAttribute;

/**
 * A base encapsulation of render-related SecureMR operators, to be used
 * by <code>Pipeline::execRenderCommand</code>.
 *
 * Developers are discouraged from directly using this structure. Instead, its derived types
 * may be found useful.
 */
struct RenderCommand {
  virtual ~RenderCommand() = default;

  /**
   * The render target, i.e., the glTF object to be rendered.
   */
  std::shared_ptr<PipelineTensor> gltfTensor;

  XrSecureMrOperatorPICO createOperator(const XrSecureMrOperatorCreateInfoPICO* config) const;
  void setOperandByName(XrSecureMrOperatorPICO opHandle, const std::shared_ptr<PipelineTensor>& tensor,
                        const std::string& opName) const;

  template <typename... VARIANT_T>
  void setOperandByName(XrSecureMrOperatorPICO opHandle, std::variant<VARIANT_T...> variant,
                        const std::string& opName) const {
    std::shared_ptr<PipelineTensor> opTensor = nullptr;
    std::visit(
        [&](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::shared_ptr<PipelineTensor>>) {
            opTensor = arg;
          } else if constexpr (std::is_same_v<T, bool>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {1},
                                                           .channels = 1,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
            uint8_t rawBoolValue = arg ? 1u : 0u;
            opTensor->setData(reinterpret_cast<int8_t*>(&rawBoolValue), sizeof(uint8_t));
          } else if constexpr (std::is_same_v<T, uint16_t>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {1},
                                                           .channels = 1,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
            opTensor->setData(reinterpret_cast<int8_t*>(&arg), sizeof(uint16_t));
          } else if constexpr (std::is_same_v<T, float>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {1},
                                                           .channels = 1,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
            opTensor->setData(reinterpret_cast<int8_t*>(&arg), sizeof(float));
          } else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {static_cast<int>(arg.size())},
                                                           .channels = 1,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
            opTensor->setData(reinterpret_cast<int8_t*>(arg.data()), sizeof(uint16_t) * arg.size());
          } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {static_cast<int>(arg.size())},
                                                           .channels = 1,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
            opTensor->setData(reinterpret_cast<int8_t*>(arg.data()), sizeof(float) * arg.size());
          } else if constexpr (std::is_same_v<T, std::vector<std::array<uint8_t, 4>>>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {static_cast<int>(arg.size())},
                                                           .channels = 4,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
            std::vector<uint8_t> flattedArg;
            for (auto& eachColor : arg) {
              flattedArg.push_back(eachColor[0]);
              flattedArg.push_back(eachColor[1]);
              flattedArg.push_back(eachColor[2]);
              flattedArg.push_back(eachColor[3]);
            }
            opTensor->setData(reinterpret_cast<int8_t*>(flattedArg.data()), sizeof(uint8_t) * flattedArg.size());
          } else if constexpr (std::is_same_v<T, std::array<std::array<uint8_t, 4>, 2>>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {2},
                                                           .channels = 4,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
            uint8_t flattedArg[]{arg[0][0], arg[0][1], arg[0][2], arg[0][3],
                                 arg[1][0], arg[1][1], arg[1][2], arg[1][3]};
            opTensor->setData(reinterpret_cast<int8_t*>(flattedArg), sizeof(flattedArg));
          } else if constexpr (std::is_same_v<T, std::tuple<float, float>>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {1},
                                                           .channels = 2,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
            float point2f[]{std::get<0>(arg), std::get<1>(arg)};
            opTensor->setData(reinterpret_cast<int8_t*>(point2f), sizeof(point2f));
          } else if constexpr (std::is_same_v<T, std::string>) {
            opTensor = std::make_shared<PipelineTensor>(
                gltfTensor->getPipeline(), TensorAttribute{.dimensions = {static_cast<int>(arg.size())},
                                                           .channels = 2,
                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
            opTensor->setData(reinterpret_cast<int8_t*>(arg.data()), arg.size());
          } else {
            THROW(
                "Cannot create a static tensor in-line; only bool, uint16, float, vector<uint16>, vector<float>, "
                "vector<array<uint8, 4>>, array<array<uint8, 4>, 2>, tuple<float, float> and string are supported");
          }

          if (opTensor != nullptr) {
            CHECK_MSG(gltfTensor->getPipeline()->verifyPipelineTensor(opTensor),
                      "operand tensors for render command are not associated with the same pipeline of the target glTF "
                      "placeholder tensor");
            auto result = gltfTensor->getPipeline()->xrSetSecureMrOperatorOperandByNamePICO(
                static_cast<XrSecureMrPipelinePICO>(*gltfTensor->getPipeline()), opHandle,
                static_cast<XrSecureMrPipelineTensorPICO>(*opTensor), opName.c_str());
            CHECK_XRRESULT(result, Fmt("xrSetSecureMrOperatorOperandByNamePICO(..., %s)", opName.c_str()).c_str())
          }
        },
        variant);
  }

  virtual void execute() = 0;
};

/**
 * An encapsulation of SecureMR operator <code>XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO</code>,
 * which toggles the render status (visibility, view-or-world-locked, and init pose) of a glTF object.
 * <p/>
 * Note a newly created glTF object will by default be world-locked and invisible, if the RenderCommand_Render
 * is never executed.
 * <p/>
 * Also note that if the glTF object is view locked, results from <code>Pipeline::camSpace2XrLocal</code> shall
 * not be applied onto the glTF object's pose, because the results, as is suggested by the method name, is relative to
 * the OpenXR's <code>XR_REFERENCE_SPACE_TYPE_LOCAL</code>, but if view-locked, the world space of the glTF object
 * is OpenXR's <code>XR_REFERENCE_SPACE_TYPE_VIEW</code>.
 */
struct RenderCommand_Render : public RenderCommand {
  /**
   * The initial pose, must be a 4x4 tensor with 1-channel, and a floating-point datatype, with usage =
   * <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   */
  std::shared_ptr<PipelineTensor> pose;
  /**
   * A flag determining whether the glTF object will be view-locked or not (world-locked). You can provide a boolean or
   * a tensor which will regarded as <code>true</code> if non-zero, and <code>false</code> otherwise.
   * <p/>
   * If true, then the world space of the glTF object is OpenXR's <code>XR_REFERENCE_SPACE_TYPE_VIEW</code>.
   * Otherwise, the object's world space is OpenXR's <code>XR_REFERENCE_SPACE_TYPE_LOCAL</code>
   */
  std::variant<bool, std::shared_ptr<PipelineTensor>> viewLocked = false;
  std::shared_ptr<PipelineTensor> visible = nullptr;

  RenderCommand_Render() = default;
  RenderCommand_Render(const RenderCommand_Render&) = default;
  RenderCommand_Render(RenderCommand_Render&&) = default;
  RenderCommand_Render& operator=(const RenderCommand_Render&) = default;
  RenderCommand_Render& operator=(RenderCommand_Render&&) = default;
  RenderCommand_Render(const std::shared_ptr<PipelineTensor>& gltfTensor, const std::shared_ptr<PipelineTensor>& pose,
                       const std::variant<bool, std::shared_ptr<PipelineTensor>>& viewLocked = false,
                       const std::shared_ptr<PipelineTensor>& visible = nullptr)
      : pose(pose), viewLocked(viewLocked), visible(visible) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * A virtual encapsulation for different configurations of operator
 * <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code>
 */
struct RenderCommand_Update : public RenderCommand {
  void execute() override = 0;
};

/**
 * An encapsulation of operator <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code> specifically for
 * the texture attributes of a glTF object
 */
struct RenderCommand_UpdateTextures : public RenderCommand_Update {
  /**
   * IDs of textures to be updated. It can be a 1D vector of UINT16 values, or a tensor containing such values.
   * Note, the tensor must be declared with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>> textureIds;
  /**
   * The RGB or RGBA tensor to update the texture. If the field <code>textureIds</code> has only one
   * index, the pipeline tensor can be a 2D (HEIGHT, WIDTH) tensor, matching the shape of the target texture.
   * Otherwise, the pipeline tensor must be 3D (N, HEIGHT, WIDTH) tensor, where the N matches the number of
   * texture indices in field <code>textureIds</code>, and HEIGHT and WIDTH matches those of the target texture.
   * <p/>
   * Note the tensor must be declared with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   */
  std::shared_ptr<PipelineTensor> textureNewContents;

  RenderCommand_UpdateTextures() = default;
  RenderCommand_UpdateTextures(const RenderCommand_UpdateTextures&) = default;
  RenderCommand_UpdateTextures(RenderCommand_UpdateTextures&&) = default;
  RenderCommand_UpdateTextures& operator=(const RenderCommand_UpdateTextures&) = default;
  RenderCommand_UpdateTextures& operator=(RenderCommand_UpdateTextures&&) = default;
  RenderCommand_UpdateTextures(const std::shared_ptr<PipelineTensor>& gltfTensor,
                               const std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>>& textureIds,
                               const std::shared_ptr<PipelineTensor>& textureNewContents)
      : textureIds(textureIds), textureNewContents(textureNewContents) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * An encapsulation of operator <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code> specifically for
 * the animation a glTF object
 */
struct RenderCommand_UpdateAnimation : public RenderCommand_Update {
  /**
   * Default timer value to stop the replay of animation
   */
  constexpr static float STOP_TO_PLAY = -1.0f;

  /**
   * The ID of the animation trace to be replayed. It can be a single UINT16 value, or a tensor containing such
   * data, with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>.
   */
  std::variant<std::shared_ptr<PipelineTensor>, uint16_t> animationId;
  /**
   * Staring at which time point, the animation should be replayed. It can be a single float value, or a tensor
   * containing such data with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>.
   * <p/>
   * If the timer value is larger than the animation's total frame count, the timer's mod to animation frame count
   * will be used as the starting time point
   */
  std::variant<std::shared_ptr<PipelineTensor>, float> animationTimer = STOP_TO_PLAY;

  RenderCommand_UpdateAnimation() = default;
  RenderCommand_UpdateAnimation(const RenderCommand_UpdateAnimation&) = default;
  RenderCommand_UpdateAnimation(RenderCommand_UpdateAnimation&&) = default;
  RenderCommand_UpdateAnimation& operator=(const RenderCommand_UpdateAnimation&) = default;
  RenderCommand_UpdateAnimation& operator=(RenderCommand_UpdateAnimation&&) = default;
  RenderCommand_UpdateAnimation(
      const std::shared_ptr<PipelineTensor>& gltfTensor,
      const std::variant<std::shared_ptr<PipelineTensor>, uint16_t>& animationId,
      const std::variant<std::shared_ptr<PipelineTensor>, float>& animationTimer = STOP_TO_PLAY)
      : animationId(animationId), animationTimer(animationTimer) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * An encapsulation of operator <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code> specifically for
 * the world pose of the glTF tensor.
 * <p/>
 * Note the render command will have no effect if a glTF object's status is still invisible.
 */
struct RenderCommand_UpdatePose : public RenderCommand_Update {
  /**
   * The new pose, must be a 4x4 tensor with 1-channel, and a floating-point datatype, with usage =
   * <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>
   */
  std::shared_ptr<PipelineTensor> newPose;

  RenderCommand_UpdatePose() = default;
  RenderCommand_UpdatePose(const RenderCommand_UpdatePose&) = default;
  RenderCommand_UpdatePose(RenderCommand_UpdatePose&&) = default;
  RenderCommand_UpdatePose& operator=(const RenderCommand_UpdatePose&) = default;
  RenderCommand_UpdatePose& operator=(RenderCommand_UpdatePose&&) = default;
  RenderCommand_UpdatePose(const std::shared_ptr<PipelineTensor>& gltfTensor,
                           const std::shared_ptr<PipelineTensor>& newPose)
      : newPose(newPose) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * An encapsulation of operator <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code> specifically for
 * the node's local pose of the glTF tensor.
 */
struct RenderCommand_UpdateNodesLocalPoses : public RenderCommand_Update {
  /**
   * IDs of nodes to be updated. It can be a 1D vector of UINT16 values, or a tensor containing such values.
   * Note, the tensor must be declared with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>> nodeIds;
  /**
   * The new poses, must be a 3-dimensional tensor like (N, 4, 4), of 1-channel floating-point datatype, with usage =
   * <code>XR_SECURE_MR_TENSOR_TYPE_MAT_PICO</code>. N is the number of values in field <code>nodeIds</code>.
   * <p/>
   * If N = 1, the tensor can be a 2-dimensional one like (4, 4).
   */
  std::shared_ptr<PipelineTensor> nodeNewLocalPoses;

  RenderCommand_UpdateNodesLocalPoses() = default;
  RenderCommand_UpdateNodesLocalPoses(const RenderCommand_UpdateNodesLocalPoses&) = default;
  RenderCommand_UpdateNodesLocalPoses(RenderCommand_UpdateNodesLocalPoses&&) = default;
  RenderCommand_UpdateNodesLocalPoses& operator=(const RenderCommand_UpdateNodesLocalPoses&) = default;
  RenderCommand_UpdateNodesLocalPoses& operator=(RenderCommand_UpdateNodesLocalPoses&&) = default;
  RenderCommand_UpdateNodesLocalPoses(
      const std::shared_ptr<PipelineTensor>& gltfTensor,
      const std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>>& nodeIds,
      const std::shared_ptr<PipelineTensor>& nodeNewLocalPoses)
      : nodeIds(nodeIds), nodeNewLocalPoses(nodeNewLocalPoses) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * An encapsulation of operator <code>XR_TYPE_SECURE_MR_OPERATOR_UPDATE_GLTF_PICO</code> specifically for
 * the material attribute of the glTF tensor.
 */
struct RenderCommand_UpdateMaterial : public RenderCommand_Update {
  /**
   * IDs of materials to be updated. It can be a 1D vector of UINT16 values, or a tensor containing such values.
   * Note, the tensor must be declared with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>> materialIds;
  /**
   * Which attribute of the material to be updated.
   */
  enum class MaterialAttribute {
    FLOAT_METALLIC = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_METALLIC_FACTOR_PICO,
    FLOAT_ROUGHNESS = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_ROUGHNESS_FACTOR_PICO,
    FLOAT_EMISSIVE_STRENGTH = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_EMISSIVE_STRENGTH_PICO,

    RGBA_BASE_COLOR = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_BASE_COLOR_FACTOR_PICO,
    RGBA_EMISSIVE = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_EMISSIVE_FACTOR_PICO,

    TEXTURE_OCCLUSION_MAP = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_OCCLUSION_MAP_TEXTURE_PICO,
    TEXTURE_EMISSIVE = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_EMISSIVE_TEXTURE_PICO,
    TEXTURE_BASE_COLOR = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_BASE_COLOR_TEXTURE_PICO,
    TEXTURE_NORMAL_MAP = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_NORMAL_MAP_TEXTURE_PICO,
    TEXTURE_METALLIC_ROUGHNESS = XR_SECURE_MR_GLTF_OPERATOR_ATTRIBUTE_MATERIAL_METALLIC_ROUGHNESS_TEXTURE_PICO,
  } attribute = MaterialAttribute::RGBA_BASE_COLOR;
  /**
   * The new value(s) for the specified attribute of the materials(s) being affected.
   * <p/>
   * The value type must match the declared attribute type:
   * <ul>
   *    <li>
   *        For <code>FLOAT_METALLIC</code>, <code>FLOAT_ROUGHNESS</code>, <code>FLOAT_EMISSIVE_STRENGTH</code>: the
   *        materialValue must be a 1D vector of floating point values, of a pipeline tensor containing such values
   *        with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>
   *    </li>
   *    <li>
   *        For <code>RGBA_BASE_COLOR</code>, <code>RGBA_EMISSIVE</code>: the materialValue must be a 1D vector of
   *        4-channel UINT8 values as the R8G8B8A8 color representation, of a pipeline tensor containing such values
   *        with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO</code>
   *    </li>
   *    <li>
   *        For all others (<code>TEXTURE_....P</code>) the materialValue must be a 1D vector of UINT16 values
   *        as the texture index to be used as the corresponding map, of a pipeline tensor containing such values
   *        with usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO</code>
   *    </li>
   * </ul>
   *
   * The number of elements in field <code>materialValues</code> must match that of field <code>materialId</code>
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::vector<float>, std::vector<std::uint16_t>,
               std::vector<std::array<uint8_t, 4>>>
      materialValues;

  RenderCommand_UpdateMaterial() = default;
  RenderCommand_UpdateMaterial(const RenderCommand_UpdateMaterial&) = default;
  RenderCommand_UpdateMaterial(RenderCommand_UpdateMaterial&&) = default;
  RenderCommand_UpdateMaterial& operator=(const RenderCommand_UpdateMaterial&) = default;
  RenderCommand_UpdateMaterial& operator=(RenderCommand_UpdateMaterial&&) = default;
  RenderCommand_UpdateMaterial(
      const std::shared_ptr<PipelineTensor>& gltfTensor,
      const std::variant<std::shared_ptr<PipelineTensor>, std::vector<uint16_t>>& materialIds,
      const MaterialAttribute attribute,
      const std::variant<std::shared_ptr<PipelineTensor>, std::vector<float>, std::vector<std::uint16_t>,
                         std::vector<std::array<uint8_t, 4>>>& materialValues)
      : materialIds(materialIds), attribute(attribute), materialValues(materialValues) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

/**
 * An encapsulation of operator <code>XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO</code> to draw text on a canvas,
 * and the content on the canvas will update a specified texture of the glTF object.
 */
struct RenderCommand_DrawText : public RenderCommand {
  /**
   * Language and locale config, such as "en-us"
   */
  std::string languageAndLocale;
  /**
   * Type face of the text
   */
  enum class TypeFaceTypes {
    DEFAULT = XR_SECURE_MR_FONT_TYPEFACE_DEFAULT_PICO,
    SANS_SERIF = XR_SECURE_MR_FONT_TYPEFACE_SANS_SERIF_PICO,
    SERIF = XR_SECURE_MR_FONT_TYPEFACE_SERIF_PICO,
    MONOSPACE = XR_SECURE_MR_FONT_TYPEFACE_MONOSPACE_PICO,
    BOLD = XR_SECURE_MR_FONT_TYPEFACE_BOLD_PICO,
    ITALIC = XR_SECURE_MR_FONT_TYPEFACE_ITALIC_PICO
  } typeFace = TypeFaceTypes::DEFAULT;
  /**
   * The width of the canvas used to draw the text. It must match the target glTF object texture's width in pixel
   */
  int canvasWidth = 256;
  /**
   * The height of the canvas used to draw the text. It must match the target glTF object texture's height in pixel
   */
  int canvasHeight = 64;
  /**
   * The text to be drawn on the canvas. It can a UTF-8 string, or a tensor of any type.
   * <p/>
   * If the tensor has usage flag: <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code> and its datatype is a char-type,
   * i.e., INT8 or UINT8, the tensor content will be interpreted as a UTF-8 string.
   * <p/>
   * Otherwise, the raw values of the tensor will be printed to the canvas.
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::string> text;
  /**
   * Staring point of the text's left-bottom corner. It must be a 2-channel floats or a tensor containing
   * such value, of usage flag <code>XR_SECURE_MR_TENSOR_TYPE_POINT_PICO</code>, describing the X coordinate
   * and Y coordinate of the starting point. The X and Y shall be normalized to the canvas's width and height, i.e.
   * {0.5, 0.5} means {0.5 * canvasWidth, 0.5 * canvasHeight} in pixel.
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::tuple<float, float>> startPosition;
  /**
   * The font size of the text draw. It must be a single float or a tensor containing
   * such value, of usage flag <code>XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO</code>.
   */
  std::variant<std::shared_ptr<PipelineTensor>, float> fontSize;
  /**
   * The text color and the background color of the text draw. It must be two R8G8B8A8 color values, or a tensor
   * containing such, of usage flag <code>XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO</code>. The first color value
   * is for the text and second for the background.
   */
  std::variant<std::shared_ptr<PipelineTensor>, std::array<std::array<uint8_t, 4>, 2>> colors;
  /**
   * The ID of the target glTF texture, which the content of the canvas will be updated to after the text drawing is
   * completed.
   */
  std::variant<std::shared_ptr<PipelineTensor>, uint16_t> textureId;

  RenderCommand_DrawText() = default;
  RenderCommand_DrawText(const RenderCommand_DrawText&) = default;
  RenderCommand_DrawText(RenderCommand_DrawText&&) = default;
  RenderCommand_DrawText& operator=(const RenderCommand_DrawText&) = default;
  RenderCommand_DrawText& operator=(RenderCommand_DrawText&&) = default;
  RenderCommand_DrawText(
      const std::shared_ptr<PipelineTensor>& gltfTensor, std::string languageAndLocale, const TypeFaceTypes typeFace,
      const int canvasWidth, const int canvasHeight,
      const std::variant<std::shared_ptr<PipelineTensor>, std::string>& text,
      const std::variant<std::shared_ptr<PipelineTensor>, std::tuple<float, float>>& startPosition,
      const std::variant<std::shared_ptr<PipelineTensor>, float>& fontSize,
      const std::variant<std::shared_ptr<PipelineTensor>, std::array<std::array<uint8_t, 4>, 2>>& colors,
      const std::variant<std::shared_ptr<PipelineTensor>, uint16_t>& textureId)
      : languageAndLocale(std::move(languageAndLocale)),
        typeFace(typeFace),
        canvasWidth(canvasWidth),
        canvasHeight(canvasHeight),
        text(text),
        startPosition(startPosition),
        fontSize(fontSize),
        colors(colors),
        textureId(textureId) {
    this->gltfTensor = gltfTensor;
  }

  void execute() override;
};

}  // namespace SecureMR

#endif  // HELLO_XR_RENDERCOMMAND_H
