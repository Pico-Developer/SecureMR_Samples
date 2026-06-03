// Copyright (c) 2017-2024, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications"). All
// Bytedance's Modifications are Copyright (2025) Bytedance Ltd. and/or its affiliates.

#include "pch.h"
#include "openxr/openxr.h"
#include "common.h"
#include "options.h"
#include "platformplugin.h"
#include "graphicsplugin.h"
#include "oxr_utils/geometry.h"
#include "openxr_program.h"
#include "xr_linear.h"
#include <array>
#include <cmath>
#include <set>
#include <fstream>
#include <vector>
#include "securemr_base.h"

namespace {

#if !defined(XR_USE_PLATFORM_WIN32)
#define strcpy_s(dest, source) strncpy((dest), (source), sizeof(dest))
#endif

namespace Side {
const int LEFT = 0;
const int RIGHT = 1;
const int COUNT = 2;
}  // namespace Side

inline std::string GetXrVersionString(XrVersion ver) {
  return Fmt("%d.%d.%d", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
}

namespace Math {
namespace Pose {
XrPosef Identity() {
  XrPosef t{};
  t.orientation.w = 1;
  return t;
}

XrPosef Translation(const XrVector3f& translation) {
  XrPosef t = Identity();
  t.position = translation;
  return t;
}

XrPosef RotateCCWAboutYAxis(float radians, XrVector3f translation) {
  XrPosef t = Identity();
  t.orientation.x = 0.f;
  t.orientation.y = std::sin(radians * 0.5f);
  t.orientation.z = 0.f;
  t.orientation.w = std::cos(radians * 0.5f);
  t.position = translation;
  return t;
}
}  // namespace Pose
}  // namespace Math

inline XrReferenceSpaceCreateInfo GetXrReferenceSpaceCreateInfo(const std::string& referenceSpaceTypeStr) {
  XrReferenceSpaceCreateInfo referenceSpaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Identity();
  if (EqualsIgnoreCase(referenceSpaceTypeStr, "View")) {
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "ViewFront")) {
    // Render head-locked 2m in front of device.
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Translation({0.f, 0.f, -2.f}),
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Local")) {
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Stage")) {
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeft")) {
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {-2.f, 0.f, -2.f});
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRight")) {
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {2.f, 0.f, -2.f});
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeftRotated")) {
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(3.14f / 3.f, {-2.f, 0.5f, -2.f});
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRightRotated")) {
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(-3.14f / 3.f, {2.f, 0.5f, -2.f});
    referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  } else {
    throw std::invalid_argument(Fmt("Unknown reference space type '%s'", referenceSpaceTypeStr.c_str()));
  }
  return referenceSpaceCreateInfo;
}

struct OpenXrProgram : IOpenXrProgram {
  OpenXrProgram(const std::shared_ptr<Options>& options, const std::shared_ptr<IPlatformPlugin>& platformPlugin,
                const std::shared_ptr<IGraphicsPlugin>& graphicsPlugin)
      : m_options(options),
        m_platformPlugin(platformPlugin),
        m_graphicsPlugin(graphicsPlugin),
        m_acceptableBlendModes{XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND} {}

  ~OpenXrProgram() override {
    if (m_graphicsPlugin != nullptr) {
      m_graphicsPlugin->ReleaseSwapchainImageResources();
    }

    if (m_overlaySwapchain.handle != XR_NULL_HANDLE) {
      xrDestroySwapchain(m_overlaySwapchain.handle);
      m_overlaySwapchain.handle = XR_NULL_HANDLE;
    }

    if (m_input.actionSet != XR_NULL_HANDLE) {
      for (auto hand : {Side::LEFT, Side::RIGHT}) {
        xrDestroySpace(m_input.handSpace[hand]);
      }
      xrDestroyActionSet(m_input.actionSet);
    }

    for (Swapchain swapchain : m_swapchains) {
      xrDestroySwapchain(swapchain.handle);
    }

    xrDestroySpace(m_spaceForLoading);

    if (m_appSpace != XR_NULL_HANDLE) {
      xrDestroySpace(m_appSpace);
    }

    if (m_session != XR_NULL_HANDLE) {
      xrDestroySession(m_session);
    }

    if (m_instance != XR_NULL_HANDLE) {
      xrDestroyInstance(m_instance);
    }
  }

  static void LogLayersAndExtensions() {
    // Write out extension properties for a given layer.
    const auto logExtensions = [](const char* layerName, int indent = 0) {
      uint32_t instanceExtensionCount;
      CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(layerName, 0, &instanceExtensionCount, nullptr));
      std::vector<XrExtensionProperties> extensions(instanceExtensionCount, {XR_TYPE_EXTENSION_PROPERTIES});
      CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(layerName, (uint32_t)extensions.size(),
                                                         &instanceExtensionCount, extensions.data()));
      for (auto property : extensions) {
        Log::Write(Log::Level::Verbose, Fmt("extension = %s", property.extensionName));
      }

      const std::string indentStr(indent, ' ');
      Log::Write(Log::Level::Verbose, Fmt("%sAvailable Extensions: (%d)", indentStr.c_str(), instanceExtensionCount));
      for (const XrExtensionProperties& extension : extensions) {
        Log::Write(Log::Level::Verbose, Fmt("%s  Name=%s SpecVersion=%d", indentStr.c_str(), extension.extensionName,
                                            extension.extensionVersion));
      }
    };

    // Log non-layer extensions (layerName==nullptr).
    logExtensions(nullptr);

    // Log layers and any of their extensions.
    {
      uint32_t layerCount;
      CHECK_XRCMD(xrEnumerateApiLayerProperties(0, &layerCount, nullptr));
      std::vector<XrApiLayerProperties> layers(layerCount, {XR_TYPE_API_LAYER_PROPERTIES});
      CHECK_XRCMD(xrEnumerateApiLayerProperties((uint32_t)layers.size(), &layerCount, layers.data()));

      Log::Write(Log::Level::Info, Fmt("Available Layers: (%d)", layerCount));
      for (const XrApiLayerProperties& layer : layers) {
        Log::Write(Log::Level::Verbose,
                   Fmt("  Name=%s SpecVersion=%s LayerVersion=%d Description=%s", layer.layerName,
                       GetXrVersionString(layer.specVersion).c_str(), layer.layerVersion, layer.description));
        logExtensions(layer.layerName, 4);
      }
    }
  }

  void LogInstanceInfo() {
    CHECK(m_instance != XR_NULL_HANDLE);

    XrInstanceProperties instanceProperties{XR_TYPE_INSTANCE_PROPERTIES};
    CHECK_XRCMD(xrGetInstanceProperties(m_instance, &instanceProperties));

    Log::Write(Log::Level::Info, Fmt("Instance RuntimeName=%s RuntimeVersion=%s", instanceProperties.runtimeName,
                                     GetXrVersionString(instanceProperties.runtimeVersion).c_str()));
  }

  void CreateInstanceInternal() {
    CHECK(m_instance == XR_NULL_HANDLE);

    // Create union of extensions required by platform and graphics plugins.
    std::vector<const char*> extensions;

    // Transform platform and graphics extension std::strings to C strings.
    const std::vector<std::string> platformExtensions = m_platformPlugin->GetInstanceExtensions();
    std::transform(platformExtensions.begin(), platformExtensions.end(), std::back_inserter(extensions),
                   [](const std::string& ext) { return ext.c_str(); });
    const std::vector<std::string> graphicsExtensions = m_graphicsPlugin->GetInstanceExtensions();
    std::transform(graphicsExtensions.begin(), graphicsExtensions.end(), std::back_inserter(extensions),
                   [](const std::string& ext) { return ext.c_str(); });

    extensions.push_back(XR_PICO_SECURE_MIXED_REALITY_EXTENSION_NAME);
    extensions.push_back(XR_PICO_READBACK_TENSOR_EXTENSION_NAME);
#ifdef XR_USE_GRAPHICS_API_VULKAN
    extensions.push_back(XR_PICO_READBACK_TENSOR_VULKAN_EXTENSION_NAME);
#endif
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
    extensions.push_back(XR_PICO_READBACK_TENSOR_OPENGLES_EXTENSION_NAME);
#endif
    extensions.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);

    XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    createInfo.next = m_platformPlugin->GetInstanceCreateExtension();
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.enabledExtensionNames = extensions.data();

    strcpy(createInfo.applicationInfo.applicationName, "HelloXR");

    // Current version is 1.1.x, but hello_xr only requires 1.0.x
    createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;

    CHECK_XRCMD(xrCreateInstance(&createInfo, &m_instance));
  }

  void CreateInstance() override {
    LogLayersAndExtensions();

    CreateInstanceInternal();

    LogInstanceInfo();
  }

  void LogViewConfigurations() {
    CHECK(m_instance != XR_NULL_HANDLE);
    CHECK(m_systemId != XR_NULL_SYSTEM_ID);

    uint32_t viewConfigTypeCount;
    CHECK_XRCMD(xrEnumerateViewConfigurations(m_instance, m_systemId, 0, &viewConfigTypeCount, nullptr));
    std::vector<XrViewConfigurationType> viewConfigTypes(viewConfigTypeCount);
    CHECK_XRCMD(xrEnumerateViewConfigurations(m_instance, m_systemId, viewConfigTypeCount, &viewConfigTypeCount,
                                              viewConfigTypes.data()));
    CHECK((uint32_t)viewConfigTypes.size() == viewConfigTypeCount);

    Log::Write(Log::Level::Info, Fmt("Available View Configuration Types: (%d)", viewConfigTypeCount));
    for (XrViewConfigurationType viewConfigType : viewConfigTypes) {
      Log::Write(Log::Level::Verbose, Fmt("  View Configuration Type: %s %s", to_string(viewConfigType),
                                          viewConfigType == m_options->Parsed.ViewConfigType ? "(Selected)" : ""));

      XrViewConfigurationProperties viewConfigProperties{XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
      CHECK_XRCMD(xrGetViewConfigurationProperties(m_instance, m_systemId, viewConfigType, &viewConfigProperties));

      Log::Write(Log::Level::Verbose, Fmt("  View configuration FovMutable=%s",
                                          viewConfigProperties.fovMutable == XR_TRUE ? "True" : "False"));

      uint32_t viewCount;
      CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfigType, 0, &viewCount, nullptr));
      if (viewCount > 0) {
        std::vector<XrViewConfigurationView> views(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfigType, viewCount, &viewCount,
                                                      views.data()));

        for (uint32_t i = 0; i < views.size(); i++) {
          const XrViewConfigurationView& view = views[i];

          Log::Write(Log::Level::Verbose, Fmt("    View [%d]: Recommended Width=%d Height=%d SampleCount=%d", i,
                                              view.recommendedImageRectWidth, view.recommendedImageRectHeight,
                                              view.recommendedSwapchainSampleCount));
          Log::Write(Log::Level::Verbose,
                     Fmt("    View [%d]:     Maximum Width=%d Height=%d SampleCount=%d", i, view.maxImageRectWidth,
                         view.maxImageRectHeight, view.maxSwapchainSampleCount));
        }
      } else {
        Log::Write(Log::Level::Error, Fmt("Empty view configuration type"));
      }

      LogEnvironmentBlendMode(viewConfigType);
    }
  }

  void LogEnvironmentBlendMode(XrViewConfigurationType type) {
    CHECK(m_instance != XR_NULL_HANDLE);
    CHECK(m_systemId != 0);

    uint32_t count;
    CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, type, 0, &count, nullptr));
    CHECK(count > 0);

    Log::Write(Log::Level::Info, Fmt("Available Environment Blend Mode count : (%d)", count));

    std::vector<XrEnvironmentBlendMode> blendModes(count);
    CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, type, count, &count, blendModes.data()));

    bool blendModeFound = false;
    for (XrEnvironmentBlendMode mode : blendModes) {
      const bool blendModeMatch = (mode == m_options->Parsed.EnvironmentBlendMode);
      Log::Write(Log::Level::Info,
                 Fmt("Environment Blend Mode (%s) : %s", to_string(mode), blendModeMatch ? "(Selected)" : ""));
      blendModeFound |= blendModeMatch;
    }
    CHECK(blendModeFound);
  }

  XrEnvironmentBlendMode GetPreferredBlendMode() const override {
    uint32_t count;
    CHECK_XRCMD(
        xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, m_options->Parsed.ViewConfigType, 0, &count, nullptr));
    CHECK(count > 0);

    std::vector<XrEnvironmentBlendMode> blendModes(count);
    CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, m_options->Parsed.ViewConfigType, count,
                                                 &count, blendModes.data()));
    // for (const auto& blendMode : blendModes) {
    //     if (m_acceptableBlendModes.count(blendMode)) return blendMode;
    // }
    // THROW("No acceptable blend mode returned from the xrEnumerateEnvironmentBlendModes");

    // Turn on Vst background.
    return XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
  }

  void InitializeSystem() override {
    CHECK(m_instance != XR_NULL_HANDLE);
    CHECK(m_systemId == XR_NULL_SYSTEM_ID);

    XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
    systemInfo.formFactor = m_options->Parsed.FormFactor;
    CHECK_XRCMD(xrGetSystem(m_instance, &systemInfo, &m_systemId));

    Log::Write(Log::Level::Verbose,
               Fmt("Using system %d for form factor %s", m_systemId, to_string(m_options->Parsed.FormFactor)));
    CHECK(m_instance != XR_NULL_HANDLE);
    CHECK(m_systemId != XR_NULL_SYSTEM_ID);
  }

  void InitializeDevice() override {
    LogViewConfigurations();

    // The graphics API can initialize the graphics device now that the systemId and instance
    // handle are available.
    m_graphicsPlugin->InitializeDevice(m_instance, m_systemId);
  }

  void LogReferenceSpaces() {
    CHECK(m_session != XR_NULL_HANDLE);

    uint32_t spaceCount;
    CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, 0, &spaceCount, nullptr));
    std::vector<XrReferenceSpaceType> spaces(spaceCount);
    CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, spaceCount, &spaceCount, spaces.data()));

    Log::Write(Log::Level::Info, Fmt("Available reference spaces: %d", spaceCount));
    for (XrReferenceSpaceType space : spaces) {
      Log::Write(Log::Level::Verbose, Fmt("  Name: %s", to_string(space)));
    }
  }

  struct InputState {
    XrActionSet actionSet{XR_NULL_HANDLE};
    XrAction grabAction{XR_NULL_HANDLE};
    XrAction poseAction{XR_NULL_HANDLE};
    XrAction vibrateAction{XR_NULL_HANDLE};
    XrAction quitAction{XR_NULL_HANDLE};
    std::array<XrPath, Side::COUNT> handSubactionPath;
    std::array<XrSpace, Side::COUNT> handSpace;
    std::array<float, Side::COUNT> handScale = {{1.0f, 1.0f}};
    std::array<XrBool32, Side::COUNT> handActive;

    enum ToggleStatus {
      PRESS_DOWN = 0,
      RELEASE = 1,
    };

    std::array<ToggleStatus, Side::COUNT> handToggle;
    std::array<std::optional<XrVector3f>, Side::COUNT> toggleStartPosition{std::nullopt, std::nullopt};
  };

  void InitializeActions() {
    // Create an action set.
    {
      XrActionSetCreateInfo actionSetInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
      strcpy_s(actionSetInfo.actionSetName, "gameplay");
      strcpy_s(actionSetInfo.localizedActionSetName, "Gameplay");
      actionSetInfo.priority = 0;
      CHECK_XRCMD(xrCreateActionSet(m_instance, &actionSetInfo, &m_input.actionSet));
    }

    // Get the XrPath for the left and right hands - we will use them as subaction paths.
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left", &m_input.handSubactionPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right", &m_input.handSubactionPath[Side::RIGHT]));

    // Create actions.
    {
      // Create an input action for grabbing objects with the left and right hands.
      XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
      actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
      strcpy_s(actionInfo.actionName, "grab_object");
      strcpy_s(actionInfo.localizedActionName, "Grab Object");
      actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
      actionInfo.subactionPaths = m_input.handSubactionPath.data();
      CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.grabAction));

      // Create an input action getting the left and right hand poses.
      actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
      strcpy_s(actionInfo.actionName, "hand_pose");
      strcpy_s(actionInfo.localizedActionName, "Hand Pose");
      actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
      actionInfo.subactionPaths = m_input.handSubactionPath.data();
      CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.poseAction));

      // Create output actions for vibrating the left and right controller.
      actionInfo.actionType = XR_ACTION_TYPE_VIBRATION_OUTPUT;
      strcpy_s(actionInfo.actionName, "vibrate_hand");
      strcpy_s(actionInfo.localizedActionName, "Vibrate Hand");
      actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
      actionInfo.subactionPaths = m_input.handSubactionPath.data();
      CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.vibrateAction));

      // Create input actions for quitting the session using the left and right controller.
      // Since it doesn't matter which hand did this, we do not specify subaction paths for it.
      // We will just suggest bindings for both hands, where possible.
      actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
      strcpy_s(actionInfo.actionName, "quit_session");
      strcpy_s(actionInfo.localizedActionName, "Quit Session");
      actionInfo.countSubactionPaths = 0;
      actionInfo.subactionPaths = nullptr;
      CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.quitAction));
    }

    std::array<XrPath, Side::COUNT> selectPath;
    std::array<XrPath, Side::COUNT> squeezeValuePath;
    std::array<XrPath, Side::COUNT> squeezeForcePath;
    std::array<XrPath, Side::COUNT> squeezeClickPath;
    std::array<XrPath, Side::COUNT> posePath;
    std::array<XrPath, Side::COUNT> hapticPath;
    std::array<XrPath, Side::COUNT> menuClickPath;
    std::array<XrPath, Side::COUNT> bClickPath;
    std::array<XrPath, Side::COUNT> triggerValuePath;
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/select/click", &selectPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/select/click", &selectPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/value", &squeezeValuePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/value", &squeezeValuePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/force", &squeezeForcePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/force", &squeezeForcePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/click", &squeezeClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/click", &squeezeClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/grip/pose", &posePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/grip/pose", &posePath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/output/haptic", &hapticPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/output/haptic", &hapticPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/menu/click", &menuClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/menu/click", &menuClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/b/click", &bClickPath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/b/click", &bClickPath[Side::RIGHT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/trigger/value", &triggerValuePath[Side::LEFT]));
    CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/trigger/value", &triggerValuePath[Side::RIGHT]));
    // Suggest bindings for KHR Simple.
    {
      XrPath khrSimpleInteractionProfilePath;
      CHECK_XRCMD(
          xrStringToPath(m_instance, "/interaction_profiles/khr/simple_controller", &khrSimpleInteractionProfilePath));
      std::vector<XrActionSuggestedBinding> bindings{{// Fall back to a click input for the grab action.
                                                      {m_input.grabAction, selectPath[Side::LEFT]},
                                                      {m_input.grabAction, selectPath[Side::RIGHT]},
                                                      {m_input.poseAction, posePath[Side::LEFT]},
                                                      {m_input.poseAction, posePath[Side::RIGHT]},
                                                      {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                      {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                      {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
      XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
      suggestedBindings.interactionProfile = khrSimpleInteractionProfilePath;
      suggestedBindings.suggestedBindings = bindings.data();
      suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
      CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
    }
    // Suggest bindings for the Oculus Touch.
    {
      XrPath oculusTouchInteractionProfilePath;
      CHECK_XRCMD(xrStringToPath(m_instance, "/interaction_profiles/oculus/touch_controller",
                                 &oculusTouchInteractionProfilePath));
      std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeValuePath[Side::LEFT]},
                                                      {m_input.grabAction, squeezeValuePath[Side::RIGHT]},
                                                      {m_input.poseAction, posePath[Side::LEFT]},
                                                      {m_input.poseAction, posePath[Side::RIGHT]},
                                                      {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
      XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
      suggestedBindings.interactionProfile = oculusTouchInteractionProfilePath;
      suggestedBindings.suggestedBindings = bindings.data();
      suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
      CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
    }
    // Suggest bindings for the Vive Controller.
    {
      XrPath viveControllerInteractionProfilePath;
      CHECK_XRCMD(xrStringToPath(m_instance, "/interaction_profiles/htc/vive_controller",
                                 &viveControllerInteractionProfilePath));
      std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, triggerValuePath[Side::LEFT]},
                                                      {m_input.grabAction, triggerValuePath[Side::RIGHT]},
                                                      {m_input.poseAction, posePath[Side::LEFT]},
                                                      {m_input.poseAction, posePath[Side::RIGHT]},
                                                      {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                      {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                      {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
      XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
      suggestedBindings.interactionProfile = viveControllerInteractionProfilePath;
      suggestedBindings.suggestedBindings = bindings.data();
      suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
      CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
    }

    // Suggest bindings for the Valve Index Controller.
    {
      XrPath indexControllerInteractionProfilePath;
      CHECK_XRCMD(xrStringToPath(m_instance, "/interaction_profiles/valve/index_controller",
                                 &indexControllerInteractionProfilePath));
      std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeForcePath[Side::LEFT]},
                                                      {m_input.grabAction, squeezeForcePath[Side::RIGHT]},
                                                      {m_input.poseAction, posePath[Side::LEFT]},
                                                      {m_input.poseAction, posePath[Side::RIGHT]},
                                                      {m_input.quitAction, bClickPath[Side::LEFT]},
                                                      {m_input.quitAction, bClickPath[Side::RIGHT]},
                                                      {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
      XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
      suggestedBindings.interactionProfile = indexControllerInteractionProfilePath;
      suggestedBindings.suggestedBindings = bindings.data();
      suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
      CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
    }

    // Suggest bindings for the Microsoft Mixed Reality Motion Controller.
    {
      XrPath microsoftMixedRealityInteractionProfilePath;
      CHECK_XRCMD(xrStringToPath(m_instance, "/interaction_profiles/microsoft/motion_controller",
                                 &microsoftMixedRealityInteractionProfilePath));
      std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeClickPath[Side::LEFT]},
                                                      {m_input.grabAction, squeezeClickPath[Side::RIGHT]},
                                                      {m_input.poseAction, posePath[Side::LEFT]},
                                                      {m_input.poseAction, posePath[Side::RIGHT]},
                                                      {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                      {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                      {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                      {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
      XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
      suggestedBindings.interactionProfile = microsoftMixedRealityInteractionProfilePath;
      suggestedBindings.suggestedBindings = bindings.data();
      suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
      CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
    }
    XrActionSpaceCreateInfo actionSpaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    actionSpaceInfo.action = m_input.poseAction;
    actionSpaceInfo.poseInActionSpace.orientation.w = 1.f;
    actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::LEFT];
    CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::LEFT]));
    actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::RIGHT];
    CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::RIGHT]));

    XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attachInfo.countActionSets = 1;
    attachInfo.actionSets = &m_input.actionSet;
    CHECK_XRCMD(xrAttachSessionActionSets(m_session, &attachInfo));
  }

  void CreateVisualizedSpaces() {
    CHECK(m_session != XR_NULL_HANDLE);

    XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo("ViewFront");
    CHECK_XRCMD(xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &m_spaceForLoading));
  }

  void InitializeSession() override {
    CHECK(m_instance != XR_NULL_HANDLE);
    CHECK(m_session == XR_NULL_HANDLE);

    {
      Log::Write(Log::Level::Verbose, Fmt("Creating session..."));

      XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
      createInfo.next = m_graphicsPlugin->GetGraphicsBinding();
      createInfo.systemId = m_systemId;
      CHECK_XRCMD(xrCreateSession(m_instance, &createInfo, &m_session));
    }

    LogReferenceSpaces();
    InitializeActions();
    CreateVisualizedSpaces();
    // Create a pure view space for overlays
    {
      XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo("View");
      XrSpace viewSpace{XR_NULL_HANDLE};
      CHECK_XRCMD(xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &viewSpace));
      m_viewSpace = viewSpace;
    }

    {
      XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo(m_options->AppSpace);
      CHECK_XRCMD(xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &m_appSpace));
    }
  }

  void CreateSwapchains() override {
    CHECK(m_session != XR_NULL_HANDLE);
    CHECK(m_swapchains.empty());
    CHECK(m_configViews.empty());

    // Read graphics properties for preferred swapchain length and logging.
    XrSystemProperties systemProperties{XR_TYPE_SYSTEM_PROPERTIES};
    CHECK_XRCMD(xrGetSystemProperties(m_instance, m_systemId, &systemProperties));

    // Log system properties.
    Log::Write(Log::Level::Info,
               Fmt("System Properties: Name=%s VendorId=%d", systemProperties.systemName, systemProperties.vendorId));
    Log::Write(Log::Level::Info, Fmt("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxLayers=%d",
                                     systemProperties.graphicsProperties.maxSwapchainImageWidth,
                                     systemProperties.graphicsProperties.maxSwapchainImageHeight,
                                     systemProperties.graphicsProperties.maxLayerCount));
    Log::Write(Log::Level::Info,
               Fmt("System Tracking Properties: OrientationTracking=%s PositionTracking=%s",
                   systemProperties.trackingProperties.orientationTracking == XR_TRUE ? "True" : "False",
                   systemProperties.trackingProperties.positionTracking == XR_TRUE ? "True" : "False"));

    // Note: No other view configurations exist at the time this code was written. If this
    // condition is not met, the project will need to be audited to see how support should be
    // added.
    CHECK_MSG(m_options->Parsed.ViewConfigType == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
              "Unsupported view configuration type");

    // Query and cache view configuration views.
    uint32_t viewCount;
    CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_options->Parsed.ViewConfigType, 0,
                                                  &viewCount, nullptr));
    m_configViews.resize(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_options->Parsed.ViewConfigType, viewCount,
                                                  &viewCount, m_configViews.data()));

    // Create and cache view buffer for xrLocateViews later.
    m_views.resize(viewCount, {XR_TYPE_VIEW});

    // Create the swapchain and get the images.
    if (viewCount > 0) {
      // Select a swapchain format.
      uint32_t swapchainFormatCount;
      CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, 0, &swapchainFormatCount, nullptr));
      std::vector<int64_t> swapchainFormats(swapchainFormatCount);
      CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, (uint32_t)swapchainFormats.size(), &swapchainFormatCount,
                                              swapchainFormats.data()));
      CHECK(swapchainFormatCount == swapchainFormats.size());
      m_colorSwapchainFormat = m_graphicsPlugin->SelectColorSwapchainFormat(swapchainFormats);

      // Print swapchain formats and the selected one.
      {
        std::string swapchainFormatsString;
        for (int64_t format : swapchainFormats) {
          const bool selected = format == m_colorSwapchainFormat;
          swapchainFormatsString += " ";
          if (selected) {
            swapchainFormatsString += "[";
          }
          swapchainFormatsString += std::to_string(format);
          if (selected) {
            swapchainFormatsString += "]";
          }
        }
        Log::Write(Log::Level::Verbose, Fmt("Swapchain Formats: %s", swapchainFormatsString.c_str()));
      }

      // Create a swapchain for each view.
      for (uint32_t i = 0; i < viewCount; i++) {
        const XrViewConfigurationView& vp = m_configViews[i];
        Log::Write(
            Log::Level::Info,
            Fmt("Creating swapchain for view %d with dimensions Width=%d Height=%d SampleCount=%d", i,
                vp.recommendedImageRectWidth, vp.recommendedImageRectHeight, vp.recommendedSwapchainSampleCount));

        // Create the swapchain.
        XrSwapchainCreateInfo swapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        swapchainCreateInfo.arraySize = 1;
        swapchainCreateInfo.format = m_colorSwapchainFormat;
        swapchainCreateInfo.width = vp.recommendedImageRectWidth;
        swapchainCreateInfo.height = vp.recommendedImageRectHeight;
        swapchainCreateInfo.mipCount = 1;
        swapchainCreateInfo.faceCount = 1;
        swapchainCreateInfo.sampleCount = m_graphicsPlugin->GetSupportedSwapchainSampleCount(vp);
        swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        Swapchain swapchain;
        swapchain.width = swapchainCreateInfo.width;
        swapchain.height = swapchainCreateInfo.height;
        CHECK_XRCMD(xrCreateSwapchain(m_session, &swapchainCreateInfo, &swapchain.handle));

        m_swapchains.push_back(swapchain);

        uint32_t imageCount;
        CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr));
        // XXX This should really just return XrSwapchainImageBaseHeader*
        std::vector<XrSwapchainImageBaseHeader*> swapchainImages =
            m_graphicsPlugin->AllocateSwapchainImageStructs(imageCount, swapchainCreateInfo);
        CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount, swapchainImages[0]));

        m_swapchainImages.insert(std::make_pair(swapchain.handle, std::move(swapchainImages)));
      }
      // Create overlay quad swapchain after main swapchains are ready
      CreateOverlaySwapchain();
    }
  }

  // Return event if one is available, otherwise return null.
  const XrEventDataBaseHeader* TryReadNextEvent() {
    // It is sufficient to clear the just the XrEventDataBuffer header to
    // XR_TYPE_EVENT_DATA_BUFFER
    XrEventDataBaseHeader* baseHeader = reinterpret_cast<XrEventDataBaseHeader*>(&m_eventDataBuffer);
    *baseHeader = {XR_TYPE_EVENT_DATA_BUFFER};
    const XrResult xr = xrPollEvent(m_instance, &m_eventDataBuffer);
    if (xr == XR_SUCCESS) {
      if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST) {
        const XrEventDataEventsLost* const eventsLost = reinterpret_cast<const XrEventDataEventsLost*>(baseHeader);
        Log::Write(Log::Level::Warning, Fmt("%d events lost", eventsLost->lostEventCount));
      }

      return baseHeader;
    }
    if (xr == XR_EVENT_UNAVAILABLE) {
      return nullptr;
    }
    THROW_XR(xr, "xrPollEvent");
  }

  void PollEvents(bool* exitRenderLoop, bool* requestRestart) override {
    *exitRenderLoop = *requestRestart = false;
    // Process all pending messages.
    while (const XrEventDataBaseHeader* event = TryReadNextEvent()) {
      switch (event->type) {
        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
          const auto& instanceLossPending = *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
          Log::Write(Log::Level::Warning, Fmt("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime));
          *exitRenderLoop = true;
          *requestRestart = true;
          return;
        }
        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
          auto sessionStateChangedEvent = *reinterpret_cast<const XrEventDataSessionStateChanged*>(event);
          HandleSessionStateChangedEvent(sessionStateChangedEvent, exitRenderLoop, requestRestart);
          break;
        }
        case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
          LogActionSourceName(m_input.grabAction, "Grab");
          LogActionSourceName(m_input.quitAction, "Quit");
          LogActionSourceName(m_input.poseAction, "Pose");
          LogActionSourceName(m_input.vibrateAction, "Vibrate");
          break;
        case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
        default: {
          Log::Write(Log::Level::Verbose, Fmt("Ignoring event type %d", event->type));
          break;
        }
      }
    }
  }

  void HandleSessionStateChangedEvent(const XrEventDataSessionStateChanged& stateChangedEvent, bool* exitRenderLoop,
                                      bool* requestRestart) {
    const XrSessionState oldState = m_sessionState;
    m_sessionState = stateChangedEvent.state;

    Log::Write(Log::Level::Info,
               Fmt("XrEventDataSessionStateChanged: state %s->%s session=%lld time=%lld", to_string(oldState),
                   to_string(m_sessionState), stateChangedEvent.session, stateChangedEvent.time));

    if ((stateChangedEvent.session != XR_NULL_HANDLE) && (stateChangedEvent.session != m_session)) {
      Log::Write(Log::Level::Error, "XrEventDataSessionStateChanged for unknown session");
      return;
    }

    switch (m_sessionState) {
      case XR_SESSION_STATE_READY: {
        CHECK(m_session != XR_NULL_HANDLE);
        XrSessionBeginInfo sessionBeginInfo{XR_TYPE_SESSION_BEGIN_INFO};
        sessionBeginInfo.primaryViewConfigurationType = m_options->Parsed.ViewConfigType;
        CHECK_XRCMD(xrBeginSession(m_session, &sessionBeginInfo));
        m_sessionRunning = true;
        break;
      }
      case XR_SESSION_STATE_STOPPING: {
        CHECK(m_session != XR_NULL_HANDLE);
        m_sessionRunning = false;
        CHECK_XRCMD(xrEndSession(m_session))
        break;
      }
      case XR_SESSION_STATE_EXITING: {
        *exitRenderLoop = true;
        // Do not attempt to restart because user closed this session.
        *requestRestart = false;
        break;
      }
      case XR_SESSION_STATE_LOSS_PENDING: {
        *exitRenderLoop = true;
        // Poll for a new instance.
        *requestRestart = true;
        break;
      }
      default:
        break;
    }
  }

  void LogActionSourceName(XrAction action, const std::string& actionName) const {
    XrBoundSourcesForActionEnumerateInfo getInfo = {XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO};
    getInfo.action = action;
    uint32_t pathCount = 0;
    CHECK_XRCMD(xrEnumerateBoundSourcesForAction(m_session, &getInfo, 0, &pathCount, nullptr));
    std::vector<XrPath> paths(pathCount);
    CHECK_XRCMD(
        xrEnumerateBoundSourcesForAction(m_session, &getInfo, uint32_t(paths.size()), &pathCount, paths.data()));

    std::string sourceName;
    for (uint32_t i = 0; i < pathCount; ++i) {
      constexpr XrInputSourceLocalizedNameFlags all = XR_INPUT_SOURCE_LOCALIZED_NAME_USER_PATH_BIT |
                                                      XR_INPUT_SOURCE_LOCALIZED_NAME_INTERACTION_PROFILE_BIT |
                                                      XR_INPUT_SOURCE_LOCALIZED_NAME_COMPONENT_BIT;

      XrInputSourceLocalizedNameGetInfo nameInfo = {XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO};
      nameInfo.sourcePath = paths[i];
      nameInfo.whichComponents = all;

      uint32_t size = 0;
      CHECK_XRCMD(xrGetInputSourceLocalizedName(m_session, &nameInfo, 0, &size, nullptr));
      if (size < 1) {
        continue;
      }
      std::vector<char> grabSource(size);
      CHECK_XRCMD(
          xrGetInputSourceLocalizedName(m_session, &nameInfo, uint32_t(grabSource.size()), &size, grabSource.data()));
      if (!sourceName.empty()) {
        sourceName += " and ";
      }
      sourceName += "'";
      sourceName += std::string(grabSource.data(), size - 1);
      sourceName += "'";
    }

    Log::Write(Log::Level::Info, Fmt("%s action is bound to %s", actionName.c_str(),
                                     ((!sourceName.empty()) ? sourceName.c_str() : "nothing")));
  }

  bool IsSessionRunning() const override { return m_sessionRunning; }

  bool IsSessionFocused() const override { return m_sessionState == XR_SESSION_STATE_FOCUSED; }

  void PollActions() override {
    m_input.handActive = {XR_FALSE, XR_FALSE};

    // Sync actions
    const XrActiveActionSet activeActionSet{m_input.actionSet, XR_NULL_PATH};
    XrActionsSyncInfo syncInfo{XR_TYPE_ACTIONS_SYNC_INFO};
    syncInfo.countActiveActionSets = 1;
    syncInfo.activeActionSets = &activeActionSet;
    CHECK_XRCMD(xrSyncActions(m_session, &syncInfo));

    // Get pose and grab action state and start haptic vibrate when hand is 90% squeezed.
    for (auto hand : {Side::LEFT, Side::RIGHT}) {
      XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
      getInfo.action = m_input.grabAction;
      getInfo.subactionPath = m_input.handSubactionPath[hand];

      XrActionStateFloat grabValue{XR_TYPE_ACTION_STATE_FLOAT};
      CHECK_XRCMD(xrGetActionStateFloat(m_session, &getInfo, &grabValue));
      if (grabValue.isActive == XR_TRUE) {
        // Scale the rendered hand by 1.0f (open) to 0.5f (fully squeezed).
        m_input.handScale[hand] = 1.0f - 0.5f * grabValue.currentState;
        if (grabValue.currentState > 0.9f) {
          XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
          vibration.amplitude = 0.5;
          vibration.duration = XR_MIN_HAPTIC_DURATION;
          vibration.frequency = XR_FREQUENCY_UNSPECIFIED;

          XrHapticActionInfo hapticActionInfo{XR_TYPE_HAPTIC_ACTION_INFO};
          hapticActionInfo.action = m_input.vibrateAction;
          hapticActionInfo.subactionPath = m_input.handSubactionPath[hand];
          CHECK_XRCMD(xrApplyHapticFeedback(m_session, &hapticActionInfo, (XrHapticBaseHeader*)&vibration));
          m_input.handToggle[hand] = InputState::PRESS_DOWN;
        } else {
          m_input.handToggle[hand] = InputState::RELEASE;
        }
      }

      getInfo.action = m_input.poseAction;
      XrActionStatePose poseState{XR_TYPE_ACTION_STATE_POSE};
      CHECK_XRCMD(xrGetActionStatePose(m_session, &getInfo, &poseState));
      m_input.handActive[hand] = poseState.isActive;
    }

    // There were no subaction paths specified for the quit action, because we don't care which hand did it.
    // XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, m_input.quitAction, XR_NULL_PATH};
    // XrActionStateBoolean quitValue{XR_TYPE_ACTION_STATE_BOOLEAN};
    // CHECK_XRCMD(xrGetActionStateBoolean(m_session, &getInfo, &quitValue));
    // if ((quitValue.isActive == XR_TRUE) && (quitValue.changedSinceLastSync == XR_TRUE) &&
    //     (quitValue.currentState == XR_TRUE)) {
    //   CHECK_XRCMD(xrRequestExitSession(m_session));
    // }
  }

  void RenderFrame() override {
    CHECK(m_session != XR_NULL_HANDLE);

    XrFrameWaitInfo frameWaitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    CHECK_XRCMD(xrWaitFrame(m_session, &frameWaitInfo, &frameState));

    XrFrameBeginInfo frameBeginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    CHECK_XRCMD(xrBeginFrame(m_session, &frameBeginInfo));

    std::vector<XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    std::vector<XrCompositionLayerProjectionView> projectionLayerViews;
    if (frameState.shouldRender == XR_TRUE) {
      if (RenderLayer(frameState.predictedDisplayTime, projectionLayerViews, layer)) {
        layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
      }
      if (m_overlayInitialized && m_secureMrProgram && m_secureMrProgram->WantsScanOverlay()) {
        // Ensure overlay swapchain has a recently released image for the runtime
        XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        uint32_t overlayImageIndex = 0;
        CHECK_XRCMD(xrAcquireSwapchainImage(m_overlaySwapchain.handle, &acquireInfo, &overlayImageIndex));
        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        CHECK_XRCMD(xrWaitSwapchainImage(m_overlaySwapchain.handle, &waitInfo));

        // Allow the app to update overlay RGBA content dynamically.
        const int W = m_overlaySwapchain.width;
        const int H = m_overlaySwapchain.height;
        if (m_secureMrProgram->UpdateOverlayRgba(W, H, m_overlayRgba)) {
          if (m_overlayRgba.size() != static_cast<size_t>(W) * H * 4) {
            // Ensure buffer size correctness if the implementation changed it.
            m_overlayRgba.resize(static_cast<size_t>(W) * H * 4, 0);
          }
          m_graphicsPlugin->UploadRgbaToSwapchainImages(m_overlayImages, W, H, m_overlayRgba);
          Log::Write(Log::Level::Debug,
                     Fmt("OpenXR|Overlay: Uploaded dynamic RGBA (%dx%d, bytes=%zu)", W, H, m_overlayRgba.size()));
        }

        XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        CHECK_XRCMD(xrReleaseSwapchainImage(m_overlaySwapchain.handle, &releaseInfo));

        // Submit two quads with different poses for left/right eyes
        XrCompositionLayerQuad quadL{XR_TYPE_COMPOSITION_LAYER_QUAD};
        quadL.space = m_viewSpace == XR_NULL_HANDLE ? m_appSpace : m_viewSpace;
        quadL.eyeVisibility = XR_EYE_VISIBILITY_LEFT;
        quadL.layerFlags =
            m_options->Parsed.EnvironmentBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND
                ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT
                : 0;
        quadL.subImage.swapchain = m_overlaySwapchain.handle;
        quadL.subImage.imageArrayIndex = 0;
        quadL.subImage.imageRect.offset = {0, 0};
        quadL.subImage.imageRect.extent = {m_overlaySwapchain.width, m_overlaySwapchain.height};
        quadL.pose = Math::Pose::Translation({0.0f, 0.f, -0.35f});
        quadL.size = {0.3f, 0.3f};
        layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&quadL));

        XrCompositionLayerQuad quadR{XR_TYPE_COMPOSITION_LAYER_QUAD};
        quadR.space = m_viewSpace == XR_NULL_HANDLE ? m_appSpace : m_viewSpace;
        quadR.eyeVisibility = XR_EYE_VISIBILITY_RIGHT;
        quadR.layerFlags = quadL.layerFlags;
        quadR.subImage.swapchain = m_overlaySwapchain.handle;
        quadR.subImage.imageArrayIndex = 0;
        quadR.subImage.imageRect.offset = {0, 0};
        quadR.subImage.imageRect.extent = {m_overlaySwapchain.width, m_overlaySwapchain.height};
        quadR.pose = Math::Pose::Translation({0.0f, 0.f, -0.35f});
        quadR.size = quadL.size;
        layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&quadR));
      }
    }

    // End the frame without standard layers
    XrFrameEndInfo frameEndInfo{XR_TYPE_FRAME_END_INFO};
    frameEndInfo.displayTime = frameState.predictedDisplayTime;
    frameEndInfo.environmentBlendMode = m_options->Parsed.EnvironmentBlendMode;
    frameEndInfo.layerCount = (uint32_t)layers.size();
    frameEndInfo.layers = layers.data();
    CHECK_XRCMD(xrEndFrame(m_session, &frameEndInfo));
  }

  void CreateOverlaySwapchain() {
    if (m_overlayInitialized) return;
    // Choose overlay size and format
    m_overlaySwapchain.width = 1024;
    m_overlaySwapchain.height = 1024;

    XrSwapchainCreateInfo swapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    swapchainCreateInfo.arraySize = 1;
    swapchainCreateInfo.format = m_colorSwapchainFormat;
    swapchainCreateInfo.width = m_overlaySwapchain.width;
    swapchainCreateInfo.height = m_overlaySwapchain.height;
    swapchainCreateInfo.mipCount = 1;
    swapchainCreateInfo.faceCount = 1;
    swapchainCreateInfo.sampleCount = 1;
    swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    CHECK_XRCMD(xrCreateSwapchain(m_session, &swapchainCreateInfo, &m_overlaySwapchain.handle));

    uint32_t imageCount = 0;
    CHECK_XRCMD(xrEnumerateSwapchainImages(m_overlaySwapchain.handle, 0, &imageCount, nullptr));
    m_overlayImages = m_graphicsPlugin->AllocateSwapchainImageStructs(imageCount, swapchainCreateInfo);
    CHECK_XRCMD(xrEnumerateSwapchainImages(m_overlaySwapchain.handle, imageCount, &imageCount, m_overlayImages[0]));

    // Generate an RGBA image with transparent background and a 500x500 red border
    const int W = m_overlaySwapchain.width;
    const int H = m_overlaySwapchain.height;
    m_overlayRgba.assign(static_cast<size_t>(W) * H * 4, 0);
    const int rectW = 500;
    const int rectH = 500;
    const int thickness = 12; // increased line width for red border
    const int left = (W - rectW) / 2;
    const int top = (H - rectH) / 2 - 100;
    const int right = left + rectW - 1;
    const int bottom = top + rectH - 1;
    auto setPixel = [&](int x, int y) {
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      size_t idx = (static_cast<size_t>(y) * W + x) * 4;
      m_overlayRgba[idx + 0] = 255;
      m_overlayRgba[idx + 1] = 0;
      m_overlayRgba[idx + 2] = 0;
      m_overlayRgba[idx + 3] = 255;
    };
    for (int t = 0; t < thickness; ++t) {
      int yTop = top + t;
      int yBot = bottom - t;
      for (int x = left; x <= right; ++x) {
        setPixel(x, yTop);
        setPixel(x, yBot);
      }
    }
    for (int t = 0; t < thickness; ++t) {
      int xL = left + t;
      int xR = right - t;
      for (int y = top; y <= bottom; ++y) {
        setPixel(xL, y);
        setPixel(xR, y);
      }
    }

    m_graphicsPlugin->UploadRgbaToSwapchainImages(m_overlayImages, W, H, m_overlayRgba);
    m_overlayInitialized = true;
    Log::Write(Log::Level::Info, Fmt("OpenXR|Overlay: Created overlay swapchain %dx%d and uploaded initial content", W, H));
  }

  bool RenderLayer(XrTime predictedDisplayTime, std::vector<XrCompositionLayerProjectionView>& projectionLayerViews,
                   XrCompositionLayerProjection& layer) {
    XrResult res;

    XrViewState viewState{XR_TYPE_VIEW_STATE};
    uint32_t viewCapacityInput = (uint32_t)m_views.size();
    uint32_t viewCountOutput;

    XrViewLocateInfo viewLocateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    viewLocateInfo.viewConfigurationType = m_options->Parsed.ViewConfigType;
    viewLocateInfo.displayTime = predictedDisplayTime;
    viewLocateInfo.space = m_appSpace;

    res = xrLocateViews(m_session, &viewLocateInfo, &viewState, viewCapacityInput, &viewCountOutput, m_views.data());
    CHECK_XRRESULT(res, "xrLocateViews");
    if ((viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0 ||
        (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0) {
      return false;  // There is no valid tracking poses for the views.
    }

    CHECK(viewCountOutput == viewCapacityInput);
    CHECK(viewCountOutput == m_configViews.size());
    CHECK(viewCountOutput == m_swapchains.size());

    projectionLayerViews.resize(viewCountOutput);

    // For each locatable space that we want to visualize, render a 25cm cube.
    std::vector<Cube> cubes;
    const bool renderControllerCubes =
        (m_secureMrProgram == nullptr) || m_secureMrProgram->WantsControllerVisualization();
    static auto firstLoadingTime = std::chrono::steady_clock::now();

    if (!m_secureMrProgram->LoadingFinished()) {
      XrSpaceLocation spaceLocation{XR_TYPE_SPACE_LOCATION};
      CHECK_XRCMD(xrLocateSpace(m_spaceForLoading, m_appSpace, predictedDisplayTime, &spaceLocation));
      if (XR_UNQUALIFIED_SUCCESS(res)) {
        if ((spaceLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
            (spaceLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
          auto loadingTimeInSec = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                         std::chrono::steady_clock::now() - firstLoadingTime)
                                                         .count()) /
                                  1000.0f;
          auto x = fmod(loadingTimeInSec, 2.0f);
          auto totalAngle = x * (2.0f - x) * 0.22f + loadingTimeInSec * 0.78539815f;
          constexpr static XrVector3f Y_AXIS{0.0f, 1.0f, 0.0f};
          XrQuaternionf loadingRotation;
          XrQuaternionf_CreateFromAxisAngle(&loadingRotation, &Y_AXIS, totalAngle);
          XrQuaternionf_Multiply(&spaceLocation.pose.orientation, &spaceLocation.pose.orientation, &loadingRotation);
          cubes.push_back(Cube{spaceLocation.pose, {0.25f, 0.25f, 0.25f}});
        }
      } else {
        Log::Write(Log::Level::Verbose, Fmt("Unable to locate a visualized reference space in app space: %d", res));
      }
    }

    // Optionally render a 10cm cube scaled by grabAction for each hand. Note renderHand will only be
    // true when the application has focus.
    std::array<XrVector3f*, 2> handDeltas{};
    std::array<std::optional<XrPosef>, 2> handPoses{};
    bool buttonPressed = false;
    for (auto hand : {Side::LEFT, Side::RIGHT}) {
      XrSpaceLocation spaceLocation{XR_TYPE_SPACE_LOCATION};
      res = xrLocateSpace(m_input.handSpace[hand], m_appSpace, predictedDisplayTime, &spaceLocation);
      CHECK_XRRESULT(res, "xrLocateSpace");
      if (XR_UNQUALIFIED_SUCCESS(res)) {
        if ((spaceLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
            (spaceLocation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
          float scale = 0.1f * m_input.handScale[hand];
          if (renderControllerCubes) {
            cubes.push_back(Cube{spaceLocation.pose, {scale, scale, scale}});
          }
          handPoses[hand] = spaceLocation.pose;

          if (m_input.handToggle[hand] == InputState::PRESS_DOWN) {
            if (!m_input.toggleStartPosition[hand]) {
              // no recorded pose --- first pressed down
              m_input.toggleStartPosition[hand] = spaceLocation.pose.position;
              buttonPressed = true;
            }
          } else if (m_input.handToggle[hand] == InputState::RELEASE) {
            if (m_input.toggleStartPosition[hand]) {
              // has recorded pose --- first release
              auto& [x, y, z] = *m_input.toggleStartPosition[hand];
              handDeltas[hand] = new XrVector3f{.x = spaceLocation.pose.position.x - x,
                                                .y = spaceLocation.pose.position.y - y,
                                                .z = spaceLocation.pose.position.z - z};

              m_input.toggleStartPosition[hand] = std::nullopt;
            }
          }  // else: unknown status
        }
      } else {
        // Tracking loss is expected when the hand is not active so only log a message
        // if the hand is active.
        if (m_input.handActive[hand] == XR_TRUE) {
          const char* handName[] = {"left", "right"};
          Log::Write(Log::Level::Verbose,
                     Fmt("Unable to locate %s hand action space in app space: %d", handName[hand], res));
        }
      }
    }
    if (m_secureMrProgram) {
      const XrPosef* leftPosePtr = handPoses[Side::LEFT] ? &(*handPoses[Side::LEFT]) : nullptr;
      const XrPosef* rightPosePtr = handPoses[Side::RIGHT] ? &(*handPoses[Side::RIGHT]) : nullptr;
      m_secureMrProgram->UpdateControllerPose(leftPosePtr, rightPosePtr, m_views.data(), (uint32_t)m_views.size());
      m_secureMrProgram->UpdateHandPose(handDeltas[0], handDeltas[1]);
      if (buttonPressed) {
        for (auto hand : {Side::LEFT, Side::RIGHT}) {
          if (m_input.handToggle[hand] == InputState::PRESS_DOWN && m_input.toggleStartPosition[hand]) {
            // Check if it was JUST pressed by seeing if toggleStartPosition was set in this frame
            // The loop above (lines 981-986) sets toggleStartPosition ONLY if it wasn't set.
            // So if it is set now, and we are in the frame where buttonPressed is true, 
            // it means it was just pressed.
            m_secureMrProgram->HandleButtonPress(static_cast<int>(hand));
          }
        }
      }
      // Head motion: compare current view[0] pose with previous
      static bool hasPrevHead = false;
      static XrPosef prevHeadPose{};
      float linDelta = 0.f;
      float angDelta = 0.f;
      if (!m_views.empty()) {
        const XrPosef& cur = m_views[0].pose;
        m_secureMrProgram->UpdateHeadPose(cur);
        if (hasPrevHead) {
          const float dx = cur.position.x - prevHeadPose.position.x;
          const float dy = cur.position.y - prevHeadPose.position.y;
          const float dz = cur.position.z - prevHeadPose.position.z;
          linDelta = std::sqrt(dx * dx + dy * dy + dz * dz);
          // Quaternion delta angle = 2*acos(|dot(q1,q2)|) when normalized
          const float dot = std::abs(cur.orientation.x * prevHeadPose.orientation.x +
                                     cur.orientation.y * prevHeadPose.orientation.y +
                                     cur.orientation.z * prevHeadPose.orientation.z +
                                     cur.orientation.w * prevHeadPose.orientation.w);
          float clamped = std::min(1.0f, std::max(0.0f, dot));
          angDelta = 2.0f * std::acos(clamped);
        }
        prevHeadPose = cur;
        hasPrevHead = true;
      }
      m_secureMrProgram->UpdateHeadMotion(linDelta, angDelta);
    }
    for (const auto& handPtr : handDeltas) {
      delete handPtr;
    }

    // Render view to the appropriate part of the swapchain image.
    for (uint32_t i = 0; i < viewCountOutput; i++) {
      // Each view has a separate swapchain which is acquired, rendered to, and released.
      const Swapchain viewSwapchain = m_swapchains[i];

      XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};

      uint32_t swapchainImageIndex;
      CHECK_XRCMD(xrAcquireSwapchainImage(viewSwapchain.handle, &acquireInfo, &swapchainImageIndex));

      XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
      waitInfo.timeout = XR_INFINITE_DURATION;
      CHECK_XRCMD(xrWaitSwapchainImage(viewSwapchain.handle, &waitInfo));

      projectionLayerViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
      projectionLayerViews[i].pose = m_views[i].pose;
      projectionLayerViews[i].fov = m_views[i].fov;
      projectionLayerViews[i].subImage.swapchain = viewSwapchain.handle;
      projectionLayerViews[i].subImage.imageRect.offset = {0, 0};
      projectionLayerViews[i].subImage.imageRect.extent = {viewSwapchain.width, viewSwapchain.height};

      const XrSwapchainImageBaseHeader* const swapchainImage =
          m_swapchainImages[viewSwapchain.handle][swapchainImageIndex];
      m_graphicsPlugin->RenderView(projectionLayerViews[i], swapchainImage, m_colorSwapchainFormat, cubes);

      // Optional: render user-provided procedural mesh (e.g., native Rubik's cube)
      if (m_secureMrProgram && m_secureMrProgram->WantsUserMesh()) {
        std::vector<Geometry::Vertex> verts;
        std::vector<uint16_t> idx;
        XrPosef worldPose{};
        worldPose.orientation.w = 1.0f;
        if (m_secureMrProgram->UpdateUserMesh(verts, idx, worldPose) && !verts.empty() && !idx.empty()) {
          m_graphicsPlugin->RenderUserMesh(projectionLayerViews[i], swapchainImage, m_colorSwapchainFormat,
                                          verts.data(), static_cast<uint32_t>(verts.size()), idx.data(),
                                          static_cast<uint32_t>(idx.size()), worldPose);
        }
      }

      XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      CHECK_XRCMD(xrReleaseSwapchainImage(viewSwapchain.handle, &releaseInfo));
    }

    layer.space = m_appSpace;
    layer.layerFlags =
        m_options->Parsed.EnvironmentBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND
            ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT
            : 0;
    layer.viewCount = (uint32_t)projectionLayerViews.size();
    layer.views = projectionLayerViews.data();
    return true;
  }

  void InitializeSecureMrProgram() override {
    m_secureMrProgram = SecureMR::CreateSecureMrProgram(m_instance, m_session);
    // Provide overlay size before framework/pipeline creation so app can align image sizes.
    if (m_secureMrProgram) {
      m_secureMrProgram->SetOverlaySize(m_overlaySwapchain.width, m_overlaySwapchain.height);
    }
    m_secureMrProgram->CreateFramework();
    m_secureMrProgram->RequestPermission(IOpenXrProgram::gapp);
    m_secureMrProgram->CreatePipelines();
  }

  void RunSecureMr() override {
    if (m_secureMrProgram == nullptr) {
      Log::Write(Log::Level::Error, "m_secureMrProgram is nullptr, skip RunSecureMrVSTImagePipeline");
      return;
    }
    m_secureMrProgram->RunPipelines();
  }

  void TickSecureMr() override {
    if (m_secureMrProgram == nullptr) {
      Log::Write(Log::Level::Error, "m_secureMrProgram is nullptr, skip RunSecureMrVSTImagePipeline");
      return;
    }
    m_secureMrProgram->Tick();
  }

  void DestroySecureMr() override {
    if (m_secureMrProgram != nullptr) {
      m_secureMrProgram = nullptr;
    }
  }

 private:
  const std::shared_ptr<const Options> m_options;
  std::shared_ptr<IPlatformPlugin> m_platformPlugin;
  std::shared_ptr<IGraphicsPlugin> m_graphicsPlugin;
  XrInstance m_instance{XR_NULL_HANDLE};
  XrSession m_session{XR_NULL_HANDLE};
  XrSpace m_appSpace{XR_NULL_HANDLE};
  XrSystemId m_systemId{XR_NULL_SYSTEM_ID};

  std::vector<XrViewConfigurationView> m_configViews;
  std::vector<Swapchain> m_swapchains;
  std::map<XrSwapchain, std::vector<XrSwapchainImageBaseHeader*>> m_swapchainImages;
  std::vector<XrView> m_views;
  int64_t m_colorSwapchainFormat{-1};

  XrSpace m_spaceForLoading;

  // Application's current lifecycle state according to the runtime
  XrSessionState m_sessionState{XR_SESSION_STATE_UNKNOWN};
  bool m_sessionRunning{false};

  XrEventDataBuffer m_eventDataBuffer;
  InputState m_input;

  const std::set<XrEnvironmentBlendMode> m_acceptableBlendModes;

  std::shared_ptr<SecureMR::ISecureMR> m_secureMrProgram = nullptr;
  // Overlay quad swapchain and space
  struct OverlaySwapchain {
    XrSwapchain handle{XR_NULL_HANDLE};
    int width{0};
    int height{0};
  } m_overlaySwapchain;
  bool m_overlayInitialized{false};
  std::vector<XrSwapchainImageBaseHeader*> m_overlayImages;  // Overlay swapchain images
  std::vector<uint8_t> m_overlayRgba;                        // Staging RGBA buffer for overlay updates
  XrSpace m_viewSpace{XR_NULL_HANDLE};
};

}  // namespace

std::shared_ptr<IOpenXrProgram> CreateOpenXrProgram(const std::shared_ptr<Options>& options,
                                                    const std::shared_ptr<IPlatformPlugin>& platformPlugin,
                                                    const std::shared_ptr<IGraphicsPlugin>& graphicsPlugin) {
  return std::make_shared<OpenXrProgram>(options, platformPlugin, graphicsPlugin);
}
