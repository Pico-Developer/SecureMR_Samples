// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <mutex>
#include <optional>

#include "cube_types.h"
#include "securemr_utils/readback_async.h"

namespace SecureMR {

// Lightweight manager that accepts TensorReadbackResult and exposes a CPU-side Frame
class CameraManager {
 public:
  CameraManager() = default;
  ~CameraManager() = default;

  void OnReadbackResult(TensorReadbackResult&& result);

  // Thread-safe snapshot of latest frame
  std::optional<Frame> GetLatestFrame() const;

 private:
  mutable std::mutex mtx_;
  Frame latest_{};
  bool hasFrame_{false};
};

}  // namespace SecureMR

