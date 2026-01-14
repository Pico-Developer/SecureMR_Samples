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

#pragma once

#include "pch.h"

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include "securemr_base.h"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/readback_async.h"
#include "securemr_utils/session.h"
#include "securemr_utils/tensor.h"

// Local modules
#include "cube_types.h"
#include "geometry.h"
#include "scan_ui.h"
#include "virtual_cube.h"

struct android_app;

namespace SecureMR {

class RubicsCube : public ISecureMR {
 public:
  static struct android_app* gapp;

  RubicsCube(const XrInstance& instance, const XrSession& session);
  ~RubicsCube() override;

  void CreateFramework() override;
  void CreatePipelines() override;
  void RunPipelines() override {}
  void Tick() override;
  void RequestPermission(struct android_app* app) override;

  [[nodiscard]] bool LoadingFinished() const override { return pipelinesInitialized_; }
  [[nodiscard]] bool WantsScanOverlay() const override {
    return stage_ == AppStage::Scan || stage_ == AppStage::Validate || stage_ == AppStage::Solve;
  }
  bool UpdateOverlayRgba(int width, int height, std::vector<uint8_t>& outRgba) override;
  void SetOverlaySize(int width, int height) override;

  // Native mesh (Vulkan overlay)
  [[nodiscard]] bool WantsUserMesh() const override { return true; }
  bool UpdateUserMesh(std::vector<struct Geometry::Vertex>& outVerts, std::vector<uint16_t>& outIdx,
                      XrPosef& outWorldPose) override;
  [[nodiscard]] bool WantsControllerVisualization() const override { return false; }

  void UpdateHandPose(const XrVector3f*, const XrVector3f*) override {}
  void UpdateHeadPose(const XrPosef& pose) override;
  void UpdateHeadMotion(float linearDeltaM, float angularDeltaRad) override;
  void QueueWarmupAnimation();

 private:
  void CreateGlobalTensors();
  void CreateRelaxMrReadbackPipeline();
  void RunRelaxMrReadbackPipeline();
  void HandleReadbackResult(TensorReadbackResult&& result);
  void ResetVirtualCubeAnchorToHead();
  [[nodiscard]] std::optional<Face> GetFacingFaceFromHeadPose() const;
  std::optional<std::string> BuildFaceletsFromVirtualCube();
  void DebugFacesColor(const std::array<std::array<XrVector3f, 9>, 6>& faces);
  bool ValidateFacelets(const std::string& code) const;
  bool UpdateCubeModelFromFacelets(const std::string& facelets);
  void StartArrowHold(ScanUiState::Arrow arrow, const std::chrono::steady_clock::time_point& now);
  void ClearArrowHold();

  XrInstance instance_;
  XrSession session_;
  std::shared_ptr<FrameworkSession> frameworkSession_;
  std::shared_ptr<Pipeline> relaxMrReadbackPipeline_;
  std::shared_ptr<GlobalTensor> vstOutputLeftUint8Global_;
  std::shared_ptr<PipelineTensor> vstOutputLeftUint8Placeholder_;
  std::unique_ptr<TensorReadback> readback_;

  std::chrono::steady_clock::time_point lastRelaxReadbackRunTime_{};
  bool pipelinesInitialized_ = false;

  // Managers
  std::unique_ptr<class CameraManager> cameraManager_;
  std::shared_ptr<class ColorClassifier> colorClassifier_;
  std::unique_ptr<class ScanManager> scanManager_;
  std::unique_ptr<class CubeModel> cubeModel_;
  std::unique_ptr<class Solver> solver_;
  std::unique_ptr<class VirtualCube> virtualCube_;
  std::unique_ptr<class ScanUi> scanUi_;

  enum class AppStage { Idle, Scan, Validate, Solve, Animate, Done };
  AppStage stage_{AppStage::Idle};
  std::array<Face, 6> scanOrder_{Face::F, Face::U, Face::L, Face::B, Face::D, Face::R};
  int scanIndex_{0};
  bool overlayHookLogged_{false};
  std::chrono::steady_clock::time_point lastUiTick_{};
  std::chrono::steady_clock::time_point lastTickLog_{};

  int overlayW_{0};
  int overlayH_{0};

  // Track last accepted face center color to avoid advancing when scene is static
  std::array<uint8_t, 3> lastAcceptedCenterU8_{0, 0, 0};
  bool hasLastAcceptedCenter_{false};

  std::string solutionText_;
  std::vector<std::string> solutionSteps_;
  std::optional<std::string> cachedFacelets_;

  // Motion-based reset controls
  std::chrono::steady_clock::time_point lastMotionReset_{};
  XrPosef lastHeadPose_{};
  bool hasHeadPose_{false};

  // Delay before leaving Scan stage
  bool awaitingValidate_{false};
  std::chrono::steady_clock::time_point scanCompleteTime_{};
  bool arrowHoldActive_{false};
  ScanUiState::Arrow arrowHoldArrow_{ScanUiState::Arrow::None};
  std::chrono::steady_clock::time_point arrowHoldStart_{};
  std::chrono::milliseconds arrowHoldDuration_{1000};

};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session);

}  // namespace SecureMR
