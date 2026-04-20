// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "camera_manager.h"
#include "logger.h"
#include "common.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace SecureMR {

void CameraManager::OnReadbackResult(TensorReadbackResult&& result) {
  if (result.data.empty() || result.dimensions.empty()) {
    Log::Write(Log::Level::Debug, "RubicsCube|CameraManager: empty readback or no dimensions");
    return;
  }
  Frame frame;
  // Conservatively interpret dims: if two dims, treat as HxW; otherwise 1D
  if (result.dimensions.size() >= 2) {
    frame.height = result.dimensions[0];
    frame.width = result.dimensions[1];
  } else {
    frame.height = 1;
    frame.width = result.dimensions[0];
  }
  const int srcChannels = result.channels;
  const int w = frame.width, h = frame.height;
  const size_t bytes = result.data.size();

  // Construct src Mat that views the incoming buffer, then convert to BGR and own it.
  if (srcChannels == 3) {
    cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(result.data.data()));
    cv::cvtColor(src, frame.mat, cv::COLOR_RGB2BGR);
  } else if (srcChannels == 4) {
    cv::Mat src(h, w, CV_8UC4, const_cast<uint8_t*>(result.data.data()));
    cv::cvtColor(src, frame.mat, cv::COLOR_RGBA2BGR);
  } else {
    // Fallback: treat as single-channel and expand to BGR by replication
    cv::Mat src(h, w, CV_8UC1, const_cast<uint8_t*>(result.data.data()));
    cv::cvtColor(src, frame.mat, cv::COLOR_GRAY2BGR);
  }
  frame.channels = frame.mat.channels();  // should be 3
  frame.timestamp = std::chrono::steady_clock::now();

  {
    std::scoped_lock lk(mtx_);
    latest_ = std::move(frame);
    hasFrame_ = true;
  }

  Log::Write(Log::Level::Debug, Fmt("RubicsCube|CameraManager: frame updated %dx%dx%d (BGR), bytes=%zu", w, h, 3, bytes));
}

std::optional<Frame> CameraManager::GetLatestFrame() const {
  std::scoped_lock lk(mtx_);
  if (!hasFrame_) return std::nullopt;
  return latest_;
}

}  // namespace SecureMR
