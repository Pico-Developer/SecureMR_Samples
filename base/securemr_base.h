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

#ifndef SECURE_MR_DEMOS_SECUREMR_BASE_H
#define SECURE_MR_DEMOS_SECUREMR_BASE_H

#include "pch.h"

namespace Geometry {
struct Vertex;
}
#include <fstream>
#include <random>
#include "logger.h"
#include "common.h"

namespace SecureMR {

/**
 * Interface for SecureMR logic in each demo app. Each app <b>must</b> implement
 * this interface, which will be called from <code>base/openxr_program.cpp</code>.
 */
class ISecureMR {
 public:
  /**
   * Optional per-frame hand motion input expressed as deltas since the last frame.
   * Use this for gesture-driven interactions that do not require absolute poses.
   */
  virtual void UpdateHandPose(const XrVector3f* leftHandDelta, const XrVector3f* rightHandDelta) {}
  /**
   * Optional per-frame controller pose input in OpenXR space.
   * Unlike UpdateHandPose, this provides absolute poses for left/right controllers
   * plus the current views, enabling world-anchored interactions such as placing
   * or moving content relative to the user’s view.
   */
  virtual void UpdateControllerPose(const XrPosef* leftPose, const XrPosef* rightPose, const XrView* views, uint32_t viewCount) {}
  /**
   * Optional button input callback. The side parameter (left==0, right==1) identifies which controller
   * triggered the action; implementations may use -1 to mean "both" or "unspecified"
   * depending on their input layer.
   */
  virtual void HandleButtonPress(int side = -1) {}
  // Optional: per-frame head pose for applications that anchor content in front of the user.
  virtual void UpdateHeadPose(const XrPosef& /*pose*/) {}
  // Optional: head motion hint between frames; deltas are per-frame magnitudes in meters and radians
  virtual void UpdateHeadMotion(float /*linearDeltaM*/, float /*angularDeltaRad*/) {}

  virtual ~ISecureMR() = default;

  /**
   * This method will be called first, after the OpenXR instance and session are ready.
   *
   * The method's implementation is expected to complete the following tasks:
   * <ol>
   * <li>A Secure MR Framework handle which holds the MR resources and serves as a camera provider will be created
   * and</li> <li>The camera resolution shall be determined</li>
   * </ol>
   *
   * The calling of this method starts the lifecycle of a camera client.
   */
  virtual void CreateFramework() = 0;

  /**
   * This method will be called after <code>CreateFramework</code>.
   *
   * The method's implementation is expected to complete the following tasks:
   * <ol>
   * <li>Load assets and contents</li>
   * <li>Create global tensors and set content for them</li>
   * <li>Create pipelines where you may: </li>
   * <ol>
   * <li>Declare pipeline local tensors or placeholders inside pipelines</li>
   * <li>Add operators to pipelines to assemble MR logics</li>
   * </ol>
   * </ol>
   *
   * NOTE: As the creation of tensors/pipelines may be time consuming, you are
   * suggested to launch a secondary thread in this method to complete the
   * setup, avoiding blocking the caller thread (the OpenXR program's main thread)
   *
   * If you choose to hand over the burden of setting up to other threads, you
   * <b>must</b> maintain a cross-thread signal mechanism, so that any pipeline shall
   * not be executed unless its creation is finished by another thread.
   */
  virtual void CreatePipelines() = 0;

  /**
   * This method will be called <i>before</i> the OpenXR app's main loop, which starts
   * the submission of Secure MR pipelines created in <code>CreatePipelines</code>
   *
   * Note this method will only be called once. Hence, if you are considering running
   * some MR pipelines continuously, you <i>may</i> consider to start a new thread
   * in this method's implementation, which submits the desired pipelines timely.
   */
  virtual void RunPipelines() = 0;
  /**
   * This method will be called every frame, after the OpenXR instance and session are ready.
   *
   * The method's implementation is expected to complete the following tasks:
   * <ol>
   * <li>A Secure MR Framework handle which holds the MR resources and serves as a camera provider will be created
   * and</li> <li>The camera resolution shall be determined</li>
   * </ol>
   *
   * The calling of this method starts the lifecycle of a camera client.
   */
  virtual void RequestPermission(struct android_app* app) {}
  
  // Optional: inform app of overlay swapchain size so it can align camera/readback sizes.
  virtual void SetOverlaySize(int /*width*/, int /*height*/) {}

  /**
   * This method is called once on every iteration of the OpenXR app's main loop.
   *
   * Implementations should perform only lightweight, non-blocking per-frame work here, such as:
   * <ul>
   * <li>Polling results produced asynchronously by pipelines started in <code>RunPipelines</code> and
   *     updating any shared state used by rendering.</li>
   * <li>Uploading per-frame inputs (for example, time values, poses, hand deltas) into shared buffers/tensors or
   *     placeholders to prepare for the next pipeline execution.</li>
   * <li>Advancing app-side animations or housekeeping logic that interacts with Secure MR content.</li>
   * </ul>
   *
   * <b>Notes</b>:
   * <ul>
   * <li>This method runs on the OpenXR program's main thread before events are polled and before a frame is
   *     rendered, so avoid heavy computations or long blocking operations.</li>
   * <li>If interacting with worker threads created in <code>CreatePipelines</code> or <code>RunPipelines</code>,
   *     ensure proper synchronization and keep critical sections short.</li>
   * <li>Do not submit pipelines in this method; use <code>RunPipelines</code> for pipeline submission.</li>
   * </ul>
   */
  virtual void Tick() {}

  /**
   * This method is designed to indicate the OpenXR app's main loop whether the Secure MR
   * resources (framework, tensors, pipelines) are all ready. In our demos, a loading animation
   * is displayed until this method returns <code>true</code>.
   *
   * The method can be useful if you create a secondary thread in <code>CreatePipeline</code>
   * to take over the initialization task.
   *
   * @return True if all Secure MR resources are ready.
   */
  [[nodiscard]] virtual bool LoadingFinished() const = 0;

  // Optional: request a head-locked 2D overlay (quad) during rendering.
  // Default: no overlay.
  [[nodiscard]] virtual bool WantsScanOverlay() const { return false; }

  // Optional: provide dynamic RGBA data for the head-locked overlay.
  // Return true if outRgba is filled/updated and needs uploading to the overlay swapchain.
  // The buffer must have size width*height*4 in RGBA (8-bit) format.
  virtual bool UpdateOverlayRgba(int /*width*/, int /*height*/, std::vector<uint8_t>& /*outRgba*/) {
    return false;
  }

  // Optional: user 3D mesh (for native procedural content). Default: disabled.
  // If enabled, UpdateUserMesh should fill outVerts/outIdx and set outWorldPose.
  // Geometry::Vertex uses Position+Color, and indices are 16-bit triangle indices.
  [[nodiscard]] virtual bool WantsUserMesh() const { return false; }
  virtual bool UpdateUserMesh(std::vector<struct Geometry::Vertex>& /*outVerts*/,
                              std::vector<uint16_t>& /*outIdx*/, XrPosef& /*outWorldPose*/) {
    return false;
  }

  // Optional: allow apps to suppress the default controller visualization cubes.
  [[nodiscard]] virtual bool WantsControllerVisualization() const { return true; }
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session);

}  // namespace SecureMR

#endif  // SECURE_MR_DEMOS_SECUREMR_BASE_H
