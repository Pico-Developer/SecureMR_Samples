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

#ifndef SESSION_H
#define SESSION_H
#include <optional>
#include <string>
#include "openxr/openxr.h"

namespace SecureMR {

/**
 * Adapter of OpenXR <code>XrSecureMrFrameworkPICO</code> handle. Enabling auto boxing/unboxing and
 * auto release.
 *
 * The framework session is the root to all SecureMR operations. It serves as a camera provider, and
 * also a resource manager. Pipelines and global tensors must be associated with a framework session.
 * The destroying of a framework session will automatically release all associated objects. As a
 * camera provider, the framework session shall be initialized with the desired image width and height, and
 * can only be used within an MR application where pass-through is enabled.
 *
 * Only one framework session can be alive in one application process.
 *
 * The destroy of the framework session will also yield the access to camera.
 */
class FrameworkSession {
 private:
  XrInstance m_instance = XR_NULL_HANDLE;
  XrSession m_session = XR_NULL_HANDLE;
  XrSecureMrFrameworkPICO m_frameworkSession = XR_NULL_HANDLE;

 public:
  static PFN_xrCreateSecureMrFrameworkPICO xrCreateSecureMrFrameworkPICO;
  static PFN_xrDestroySecureMrFrameworkPICO xrDestroySecureMrFrameworkPICO;
  static PFN_xrDestroySecureMrTensorPICO xrDestroySecureMrTensorPICO;

  template <typename PFN_T>
  PFN_T getAPIFromXrInstance(const std::string& name) {
    PFN_T func = nullptr;
    if (m_instance != XR_NULL_HANDLE) {
      xrGetInstanceProcAddr(m_instance, name.c_str(), reinterpret_cast<PFN_xrVoidFunction*>(&func));
    }
    return func;
  }

  [[nodiscard]] XrSecureMrFrameworkPICO getFrameworkPICO() const { return m_frameworkSession; }

  /**
   * Create a framework session
   * @param instance The OpenXR instance
   * @param rootSession The OpenXR session
   * @param width Width in pixel of the desired camera images to be accessed throughout this framework session
   * @param height Height in pixel of the desired camera images to be accessed throughout this framework session
   */
  FrameworkSession(const XrInstance& instance, const XrSession& rootSession, int width, int height);

  FrameworkSession() = default;
  FrameworkSession(const FrameworkSession& other) = delete;
  FrameworkSession(FrameworkSession&& other) = default;
  FrameworkSession& operator=(const FrameworkSession& other) = default;
  FrameworkSession& operator=(FrameworkSession&& other) = default;

  ~FrameworkSession();
};

}  // namespace SecureMr

#endif  // SESSION_H
