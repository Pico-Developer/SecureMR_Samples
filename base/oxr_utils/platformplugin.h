// Copyright (c) 2017-2024, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Wraps platform-specific implementation so the main openxr program can be platform-independent.
struct IPlatformPlugin {
  virtual ~IPlatformPlugin() = default;

  // Provide extension to XrInstanceCreateInfo for xrCreateInstance.
  virtual XrBaseInStructure* GetInstanceCreateExtension() const = 0;

  // OpenXR instance-level extensions required by this platform.
  virtual std::vector<std::string> GetInstanceExtensions() const = 0;

  // Perform required steps after updating Options
  virtual void UpdateOptions(const std::shared_ptr<struct Options>& options) = 0;
};

// Create a platform plugin for the platform specified at compile time.
std::shared_ptr<IPlatformPlugin> CreatePlatformPlugin(const std::shared_ptr<struct Options>& options,
                                                      const std::shared_ptr<struct PlatformData>& data);
