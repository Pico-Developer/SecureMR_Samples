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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "tensor.h"

namespace SecureMR {

class FrameworkSession;

struct TensorReadbackResult {
  std::string_view name;
  std::shared_ptr<GlobalTensor> tensor;
  std::vector<uint8_t> data;
  std::vector<int> dimensions;
  int channels = 0;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO;
  XrResult futureResult = XR_SUCCESS;
};

class TensorReadback {
 public:
  using Callback = std::function<void(TensorReadbackResult&&)>;

  struct Target {
    std::shared_ptr<GlobalTensor> tensor;
    Callback callback;
    std::string name;
  };

  struct Config {
    std::chrono::milliseconds pollingInterval{33};
  };

  TensorReadback(std::shared_ptr<FrameworkSession> session, std::vector<Target> targets);
  TensorReadback(std::shared_ptr<FrameworkSession> session, std::vector<Target> targets, Config config);
  TensorReadback(const TensorReadback&) = delete;
  TensorReadback& operator=(const TensorReadback&) = delete;
  TensorReadback(TensorReadback&&) = delete;
  TensorReadback& operator=(TensorReadback&&) = delete;
  ~TensorReadback();

  void Start();
  void Stop();
  [[nodiscard]] bool IsRunning() const { return running_.load(std::memory_order_acquire); }

 private:
  struct TargetState {
    Target target;
    bool inFlight = false;
  };

  struct PendingFuture {
    TargetState* state = nullptr;
    XrFutureEXT future = XR_NULL_HANDLE;
  };

  void Loop();
  void ScheduleFutures();
  bool EnqueueFuture(TargetState& state);
  bool ProcessSingleFuture();
  bool WaitCompletion(const TargetState& state, XrSecureMrTensorPICO tensorHandle, XrFutureEXT future,
                       XrReadbackTensorBufferPICO& buffer, XrCreateBufferFromGlobalTensorCompletionPICO& completion);
  void ProcessFuture(TargetState& state, XrFutureEXT future);

  std::shared_ptr<FrameworkSession> session_;
  std::vector<TargetState> targets_;
  Config config_;
  PFN_xrCreateBufferFromGlobalTensorAsyncPICO createAsync_ = nullptr;
  PFN_xrCreateBufferFromGlobalTensorCompletePICO complete_ = nullptr;

  std::atomic<bool> running_{false};
  std::thread worker_;
  std::deque<PendingFuture> pendingFutures_;
  static constexpr std::size_t kMaxQueueDepth = 100;
  std::mutex stateMutex_;
  std::condition_variable stateCv_;
};

}  // namespace SecureMR
