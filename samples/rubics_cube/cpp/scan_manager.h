// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <optional>
#include <array>
#include <vector>
#include <chrono>

#include "camera_manager.h"
#include "color_classifier.h"
#include "cube_types.h"

namespace SecureMR {

class ScanManager {
 public:
  explicit ScanManager(std::shared_ptr<ColorClassifier> classifier) : classifier_(std::move(classifier)) {}

  // Returns true if the target face is locked; when locked, call GetLockedFace
  bool TryLockFace(Face target, const Frame& frame);
  std::optional<ScannedFace> GetLockedFace();
  // UI helper: last computed center color (boosted) in 8-bit RGB
  std::array<uint8_t, 3> GetCenterColorU8() const { return lastCenterColorU8_; }
  // UI helper: last computed 3x3 patch colors in 8-bit RGB (row-major)
  std::array<std::array<uint8_t, 3>, 9> GetPatchColors9U8() const { return lastPatchColors9U8_; }
  // UI helper: stabilization progress in [0,1]
  float GetStabilizeProgress01() const {
    const float denom = stableThresholdSec_ > 0.f ? stableThresholdSec_ : 1.f;
    float p = stableAccumSec_ / denom;
    if (p < 0.f) p = 0.f;
    if (p > 1.f) p = 1.f;
    return p;
  }
 
  static XrRect2Di GetRoiRectPx(int W, int H);
  void ResetStabilizer();
  void SetPauseUntil(std::chrono::steady_clock::time_point t) { pauseUntil_ = t; }
  void SetPauseForMs(int ms) { pauseUntil_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms); }

 private:
  std::shared_ptr<ColorClassifier> classifier_;
  std::optional<ScannedFace> locked_;
  std::array<uint8_t, 3> lastCenterColorU8_{255, 0, 0};
  std::array<std::array<uint8_t, 3>, 9> lastPatchColors9U8_{};
  std::array<std::array<uint8_t, 3>, 9> prevPatchColors9U8_{};
  bool hasPrev_{false};
  std::chrono::steady_clock::time_point lastSampleTime_{};
  float stableAccumSec_{0.f};
  float stableThresholdSec_{1.0f};
  // Cache sampled 9-patch colors during stabilization window (200ms interval)
  std::vector<std::array<std::array<uint8_t, 3>, 9>> sampleHistory_;
  int sampleSeq_{0};
  std::chrono::steady_clock::time_point pauseUntil_{};
};

}  // namespace SecureMR
