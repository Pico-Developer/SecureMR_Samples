// Copyright (c) 2017-2024, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications"). All
// Bytedance's Modifications are Copyright (2025) Bytedance Ltd. and/or its affiliates.

#include "pch.h"
#include "common.h"
#include "options.h"
#include "platformdata.h"
#include "platformplugin.h"
#include "graphicsplugin.h"
#include "openxr_program.h"

AAssetManager* g_assetManager;

namespace {

#ifdef XR_USE_PLATFORM_ANDROID
void ShowHelp() {
  Log::Write(Log::Level::Info, "adb shell setprop debug.xr.graphicsPlugin OpenGLES|Vulkan");
  Log::Write(Log::Level::Info, "adb shell setprop debug.xr.formFactor Hmd|Handheld");
  Log::Write(Log::Level::Info, "adb shell setprop debug.xr.viewConfiguration Stereo|Mono");
  Log::Write(Log::Level::Info, "adb shell setprop debug.xr.blendMode Opaque|Additive|AlphaBlend");
}

bool UpdateOptionsFromSystemProperties(Options& options) {
#if defined(DEFAULT_GRAPHICS_PLUGIN_OPENGLES)
  options.GraphicsPlugin = "OpenGLES";
#elif defined(DEFAULT_GRAPHICS_PLUGIN_VULKAN)
  options.GraphicsPlugin = "Vulkan";
#endif

  char value[PROP_VALUE_MAX] = {};
  if (__system_property_get("debug.xr.graphicsPlugin", value) != 0) {
    options.GraphicsPlugin = value;
  }

  if (__system_property_get("debug.xr.formFactor", value) != 0) {
    options.FormFactor = value;
  }

  if (__system_property_get("debug.xr.viewConfiguration", value) != 0) {
    options.ViewConfiguration = value;
  }

  if (__system_property_get("debug.xr.blendMode", value) != 0) {
    options.EnvironmentBlendMode = value;
  }

  try {
    options.ParseStrings();
  } catch (std::invalid_argument& ia) {
    Log::Write(Log::Level::Error, ia.what());
    ShowHelp();
    return false;
  }
  return true;
}
#else
void ShowHelp() {
  // TODO: Improve/update when things are more settled.
  Log::Write(Log::Level::Info,
             "HelloXr --graphics|-g <Graphics API> [--formfactor|-ff <Form factor>] [--viewconfig|-vc <View config>] "
             "[--blendmode|-bm <Blend mode>] [--space|-s <Space>] [--verbose|-v]");
  Log::Write(Log::Level::Info, "Graphics APIs:            D3D11, D3D12, OpenGLES, OpenGL, Vulkan2, Vulkan, Metal");
  Log::Write(Log::Level::Info, "Form factors:             Hmd, Handheld");
  Log::Write(Log::Level::Info, "View configurations:      Mono, Stereo");
  Log::Write(Log::Level::Info, "Environment blend modes:  Opaque, Additive, AlphaBlend");
  Log::Write(Log::Level::Info, "Spaces:                   View, Local, Stage");
}

bool UpdateOptionsFromCommandLine(Options& options, int argc, char* argv[]) {
  int i = 1;  // Index 0 is the program name and is skipped.

  auto getNextArg = [&] {
    if (i >= argc) {
      throw std::invalid_argument("Argument parameter missing");
    }

    return std::string(argv[i++]);
  };

  while (i < argc) {
    const std::string arg = getNextArg();
    if (EqualsIgnoreCase(arg, "--graphics") || EqualsIgnoreCase(arg, "-g")) {
      options.GraphicsPlugin = getNextArg();
    } else if (EqualsIgnoreCase(arg, "--formfactor") || EqualsIgnoreCase(arg, "-ff")) {
      options.FormFactor = getNextArg();
    } else if (EqualsIgnoreCase(arg, "--viewconfig") || EqualsIgnoreCase(arg, "-vc")) {
      options.ViewConfiguration = getNextArg();
    } else if (EqualsIgnoreCase(arg, "--blendmode") || EqualsIgnoreCase(arg, "-bm")) {
      options.EnvironmentBlendMode = getNextArg();
    } else if (EqualsIgnoreCase(arg, "--space") || EqualsIgnoreCase(arg, "-s")) {
      options.AppSpace = getNextArg();
    } else if (EqualsIgnoreCase(arg, "--verbose") || EqualsIgnoreCase(arg, "-v")) {
      Log::SetLevel(Log::Level::Verbose);
    } else if (EqualsIgnoreCase(arg, "--help") || EqualsIgnoreCase(arg, "-h")) {
      ShowHelp();
      return false;
    } else {
      throw std::invalid_argument(Fmt("Unknown argument: %s", arg.c_str()));
    }
  }

  // Check for required parameters.
  if (options.GraphicsPlugin.empty()) {
    Log::Write(Log::Level::Error, "GraphicsPlugin parameter is required");
    ShowHelp();
    return false;
  }

  try {
    options.ParseStrings();
  } catch (std::invalid_argument& ia) {
    Log::Write(Log::Level::Error, ia.what());
    ShowHelp();
    return false;
  }
  return true;
}
#endif
}  // namespace

struct AndroidAppState {
  ANativeWindow* NativeWindow = nullptr;
  bool Resumed = false;
};

/**
 * Process the next main command.
 */
static void app_handle_cmd(struct android_app* app, int32_t cmd) {
  AndroidAppState* appState = (AndroidAppState*)app->userData;

  switch (cmd) {
    // There is no APP_CMD_CREATE. The ANativeActivity creates the
    // application thread from onCreate(). The application thread
    // then calls android_main().
    case APP_CMD_START: {
      Log::Write(Log::Level::Info, "    APP_CMD_START");
      Log::Write(Log::Level::Info, "onStart()");
      break;
    }
    case APP_CMD_RESUME: {
      Log::Write(Log::Level::Info, "onResume()");
      Log::Write(Log::Level::Info, "    APP_CMD_RESUME");
      appState->Resumed = true;
      break;
    }
    case APP_CMD_PAUSE: {
      Log::Write(Log::Level::Info, "onPause()");
      Log::Write(Log::Level::Info, "    APP_CMD_PAUSE");
      appState->Resumed = false;
      break;
    }
    case APP_CMD_STOP: {
      Log::Write(Log::Level::Info, "onStop()");
      Log::Write(Log::Level::Info, "    APP_CMD_STOP");
      break;
    }
    case APP_CMD_DESTROY: {
      Log::Write(Log::Level::Info, "onDestroy()");
      Log::Write(Log::Level::Info, "    APP_CMD_DESTROY");
      appState->NativeWindow = NULL;
      break;
    }
    case APP_CMD_INIT_WINDOW: {
      Log::Write(Log::Level::Info, "surfaceCreated()");
      Log::Write(Log::Level::Info, "    APP_CMD_INIT_WINDOW");
      appState->NativeWindow = app->window;
      break;
    }
    case APP_CMD_TERM_WINDOW: {
      Log::Write(Log::Level::Info, "surfaceDestroyed()");
      Log::Write(Log::Level::Info, "    APP_CMD_TERM_WINDOW");
      appState->NativeWindow = NULL;
      break;
    }
  }
}

void android_main(struct android_app* app) {
  Log::Write(Log::Level::Error, "=========== main ===========");
  try {
    JNIEnv* Env;
    app->activity->vm->AttachCurrentThread(&Env, nullptr);

    AAssetManager* assetManager = app->activity->assetManager;
    g_assetManager = assetManager;

    AndroidAppState appState = {};

    app->userData = &appState;
    app->onAppCmd = app_handle_cmd;

    std::shared_ptr<Options> options = std::make_shared<Options>();
    if (!UpdateOptionsFromSystemProperties(*options)) {
      return;
    }

    std::shared_ptr<PlatformData> data = std::make_shared<PlatformData>();
    data->applicationVM = app->activity->vm;
    data->applicationActivity = app->activity->clazz;

    bool requestRestart = false;
    bool exitRenderLoop = false;

    // Create platform-specific implementation.
    std::shared_ptr<IPlatformPlugin> platformPlugin = CreatePlatformPlugin(options, data);
    // Create graphics API implementation.
    std::shared_ptr<IGraphicsPlugin> graphicsPlugin = CreateGraphicsPlugin(options, platformPlugin);

    // Initialize the OpenXR program.
    std::shared_ptr<IOpenXrProgram> program = CreateOpenXrProgram(options, platformPlugin, graphicsPlugin);

    // Initialize the loader for this platform
    PFN_xrInitializeLoaderKHR initializeLoader = nullptr;
    if (XR_SUCCEEDED(
            xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction*)(&initializeLoader)))) {
      XrLoaderInitInfoAndroidKHR loaderInitInfoAndroid = {XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
      loaderInitInfoAndroid.applicationVM = app->activity->vm;
      loaderInitInfoAndroid.applicationContext = app->activity->clazz;
      initializeLoader((const XrLoaderInitInfoBaseHeaderKHR*)&loaderInitInfoAndroid);
    }

    program->CreateInstance();
    program->InitializeSystem();

    options->SetEnvironmentBlendMode(program->GetPreferredBlendMode());
    UpdateOptionsFromSystemProperties(*options);
    platformPlugin->UpdateOptions(options);
    graphicsPlugin->UpdateOptions(options);

    program->InitializeDevice();
    program->InitializeSession();
    program->CreateSwapchains();
    program->InitializeSecureMrProgram();
    program->RunSecureMr();

    while (app->destroyRequested == 0) {
      // Read all pending events.
      for (;;) {
        int events;
        struct android_poll_source* source;
        // If the timeout is zero, returns immediately without blocking.
        // If the timeout is negative, waits indefinitely until an event appears.
        const int timeoutMilliseconds =
            (!appState.Resumed && !program->IsSessionRunning() && app->destroyRequested == 0) ? -1 : 0;
        if (ALooper_pollAll(timeoutMilliseconds, nullptr, &events, (void**)&source) < 0) {
          break;
        }

        // Process this event.
        if (source != nullptr) {
          source->process(app, source);
        }
      }

      program->PollEvents(&exitRenderLoop, &requestRestart);
      if (exitRenderLoop) {
        ANativeActivity_finish(app->activity);
        continue;
      }

      if (!program->IsSessionRunning()) {
        // Throttle loop since xrWaitFrame won't be called.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        continue;
      }

      program->PollActions();
      program->RenderFrame();
    }
    program->DestroySecureMr();
    app->activity->vm->DetachCurrentThread();
  } catch (const std::exception& ex) {
    Log::Write(Log::Level::Error, ex.what());
  } catch (...) {
    Log::Write(Log::Level::Error, "Unknown Error");
  }
  Log::Write(Log::Level::Error, "=========== exit ===========");
}
