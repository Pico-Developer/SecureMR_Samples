// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <array>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>

#include "cube_types.h"

namespace SecureMR {

class ScanManager;

struct ScanUiState {
  enum class Arrow { None, Right, Down };
  enum class Phase { Intro, Aligning, Stabilizing, Locked, PromptRotate, Validate, Solve };
  Phase phase{Phase::Intro};
  float stableSec{0.f};
  Arrow nextArrow{Arrow::None};
  bool showText{false};
  int scannedCount{0};
  std::array<std::array<uint8_t, 3>, 9> hintColors9{};  // for Validate/Solve top grid
  std::string bottomText;
  std::vector<std::string> solveSteps;
  int activeSolveIndex{-1};
};

class ScanUi {
 public:
  ScanUi() = default;
  ~ScanUi() { StopThread(); }

  // Update UI model and sample center pixel color from frame (if available)
  void Update(const Frame* latestFrame, const ScanUiState& s, float dt);

  // Inject ScanManager to fetch center color from it
  void SetScanManager(ScanManager* mgr) { manager_ = mgr; }

  // Compose RGBA for overlay. Return true when outRgba changed and needs upload.
  bool ComposeOverlayRgba(int W, int H, std::vector<uint8_t>& outRgba);

  // Background thread control for UI composition
  void StartThread();
  void StopThread();
  void SetOverlaySize(int W, int H) {
    std::lock_guard<std::mutex> lk(mtx_);
    overlayW_ = W;
    overlayH_ = H;
  }
  bool TryConsumeOverlayRgba(std::vector<uint8_t>& outRgba) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!dirty_) return false;
    outRgba = composed_;
    dirty_ = false;
    return true;
  }

 private:
  std::array<uint8_t, 3> boxColor_{255, 0, 0};
  // Latest 3x3 patch colors in 8-bit RGB, row-major order.
  std::array<std::array<uint8_t, 3>, 9> patchColors9_{};
  std::array<std::array<uint8_t, 3>, 9> hintColors9_{};
  float t_{0.f};
  ScanUiState state_{};
  std::vector<uint8_t> last_;  // last composed rgba for change detection
  ScanManager* manager_{nullptr};

  // Threading and buffer
  std::thread worker_;
  std::atomic<bool> running_{false};
  int overlayW_{1024};
  int overlayH_{1024};
  std::mutex mtx_;
  std::vector<uint8_t> composed_;
  bool dirty_{false};
};

}  // namespace SecureMR
