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

#include "readback_async.h"

#include <utility>

#include "common.h"
#include "logger.h"
#include "check.h"
#include "session.h"

namespace SecureMR {

TensorReadback::TensorReadback(std::shared_ptr<FrameworkSession> session, std::vector<Target> targets)
    : TensorReadback(std::move(session), std::move(targets), Config{}) {}

TensorReadback::TensorReadback(std::shared_ptr<FrameworkSession> session, std::vector<Target> targets, Config config)
    : session_(std::move(session)), config_(config) {
  CHECK_MSG(session_ != nullptr, "TensorReadback requires a valid FrameworkSession");

  createAsync_ =
      session_->getAPIFromXrInstance<PFN_xrCreateBufferFromGlobalTensorAsyncPICO>("xrCreateBufferFromGlobalTensorAsyncPICO");
  complete_ =
      session_->getAPIFromXrInstance<PFN_xrCreateBufferFromGlobalTensorCompletePICO>("xrCreateBufferFromGlobalTensorCompletePICO");
  CHECK_MSG(createAsync_ != nullptr, "Failed to resolve xrCreateBufferFromGlobalTensorAsyncPICO");
  CHECK_MSG(complete_ != nullptr, "Failed to resolve xrCreateBufferFromGlobalTensorCompletePICO");

  targets_.reserve(targets.size());
  for (auto& target : targets) {
    CHECK_MSG(target.tensor != nullptr, "TensorReadback target tensor must not be null");
    targets_.emplace_back(TargetState{.target = std::move(target)});
  }
}

TensorReadback::~TensorReadback() {
  Stop();
}

void TensorReadback::Start() {
  if (targets_.empty()) {
    Log::Write(Log::Level::Warning, "TensorReadback::Start called with no targets; skipping thread creation.");
    return;
  }
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  worker_ = std::thread(&TensorReadback::Loop, this);
}

void TensorReadback::Stop() {
  bool wasRunning = running_.exchange(false);
  if (!wasRunning) {
    return;
  }
  stateCv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  pendingFutures_.clear();
  for (auto& state : targets_) {
    state.inFlight = false;
  }
}

void TensorReadback::Loop() {
  auto nextSchedule = std::chrono::steady_clock::now();
  while (running_.load(std::memory_order_acquire)) {
    if (pendingFutures_.empty()) {
      auto now = std::chrono::steady_clock::now();
      if (now < nextSchedule) {
        std::unique_lock<std::mutex> lock(stateMutex_);
        stateCv_.wait_until(lock, nextSchedule, [this]() { return !running_.load(std::memory_order_acquire); });
        continue;
      }
    }

    if (!running_.load(std::memory_order_acquire)) {
      break;
    }

    auto now = std::chrono::steady_clock::now();
    if (now >= nextSchedule) {
      ScheduleFutures();
      nextSchedule = std::chrono::steady_clock::now() + config_.pollingInterval;
    }

    // Process queue
    while (running_.load(std::memory_order_acquire) && ProcessSingleFuture()) {}
  }

  pendingFutures_.clear();
  for (auto& state : targets_) {
    state.inFlight = false;
  }
}

void TensorReadback::ScheduleFutures() {
  for (auto& state : targets_) {
    if (!running_.load(std::memory_order_acquire)) {
      return;
    }
    if (state.inFlight) {
      continue;
    }

    while (pendingFutures_.size() >= kMaxQueueDepth && running_.load(std::memory_order_acquire)) {
      if (!ProcessSingleFuture()) {
        break;
      }
    }

    if (pendingFutures_.size() >= kMaxQueueDepth) {
      break;
    }

    EnqueueFuture(state);
  }
}

bool TensorReadback::EnqueueFuture(TargetState& state) {
  if (state.inFlight || state.target.tensor == nullptr) {
    return false;
  }

  auto tensorHandle = static_cast<XrSecureMrTensorPICO>(*state.target.tensor);
  XrFutureEXT future = XR_NULL_HANDLE;
  XrResult result = createAsync_(tensorHandle, &future);
  if (XR_FAILED(result) || future == XR_NULL_HANDLE) {
    Log::Write(Log::Level::Error,
               Fmt("xrCreateBufferFromGlobalTensorAsyncPICO failed for %s (result=%d)",
                   state.target.name.c_str(), result));
    return false;
  }

  pendingFutures_.push_back(PendingFuture{.state = &state, .future = future});
  state.inFlight = true;
  return true;
}


bool TensorReadback::ProcessSingleFuture() {
  if (pendingFutures_.empty()) {
    return false;
  }

  PendingFuture pending = std::move(pendingFutures_.front());
  pendingFutures_.pop_front();

  if (!running_.load(std::memory_order_acquire)) {
    if (pending.state) {
      pending.state->inFlight = false;
    }
    return false;
  }

  if (pending.state == nullptr || pending.state->target.tensor == nullptr || pending.future == XR_NULL_HANDLE) {
    if (pending.state) {
      pending.state->inFlight = false;
    }
    return true;
  }

  ProcessFuture(*pending.state, pending.future);
  pending.state->inFlight = false;
  return true;
}

bool TensorReadback::WaitCompletion(const TargetState& state, XrSecureMrTensorPICO tensorHandle, XrFutureEXT future,
                                     XrReadbackTensorBufferPICO& buffer,
                                     XrCreateBufferFromGlobalTensorCompletionPICO& completion) {
  XrResult result;
  while (true) {
    completion = {.type = XR_TYPE_CREATE_BUFFER_FROM_GLOBAL_TENSOR_COMPLETION_PICO,
                  .next = nullptr,
                  .futureResult = XR_SUCCESS,
                  .tensorBuffer = &buffer};
    while (running_.load(std::memory_order_acquire)) {
      result = complete_(tensorHandle, future, &completion);
      if (result == XR_SUCCESS) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Log::Write(Log::Level::Warning,
               Fmt("xrCreateBufferFromGlobalTensorCompletePICO failed for %s (result=%d)",
                 state.target.name.c_str(), result));
  }
  return false;
}

void TensorReadback::ProcessFuture(TargetState& state, XrFutureEXT future) {
  auto tensorHandle = static_cast<XrSecureMrTensorPICO>(*state.target.tensor);

  XrReadbackTensorBufferPICO buffer{
      .type = XR_TYPE_READBACK_TENSOR_BUFFER_PICO,
      .next = nullptr,
      .bufferCapacityInput = 0,
      .bufferCountOutput = 0,
      .buffer = nullptr};

  XrCreateBufferFromGlobalTensorCompletionPICO completion{
      .type = XR_TYPE_CREATE_BUFFER_FROM_GLOBAL_TENSOR_COMPLETION_PICO,
      .next = nullptr,
      .futureResult = XR_SUCCESS,
      .tensorBuffer = &buffer};


  Log::Write(Log::Level::Verbose, Fmt("WaitCompletion_1 start"));
  if (!WaitCompletion(state, tensorHandle, future, buffer, completion)) {
    return;
  }
  Log::Write(Log::Level::Verbose, Fmt("WaitCompletion_1 end"));

  buffer.bufferCapacityInput = buffer.bufferCountOutput;
  std::vector<uint8_t> payload(buffer.bufferCapacityInput);
  if (!payload.empty()) {
    buffer.buffer = payload.data();
  }

  Log::Write(Log::Level::Verbose, Fmt("WaitCompletion_2 start"));
  if (!WaitCompletion(state, tensorHandle, future, buffer, completion)) {
    return;
  }
  Log::Write(Log::Level::Verbose, Fmt("WaitCompletion_2 end"));

  auto attrVariant = state.target.tensor->getAttribute();
  std::vector<int> dimensions;
  int channels = 0;
  XrSecureMrTensorDataTypePICO dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_MAX_ENUM_PICO;
  if (auto* attr = std::get_if<TensorAttribute>(&attrVariant)) {
    dimensions = attr->dimensions;
    channels = attr->channels;
    dataType = attr->dataType;
  }

  if (state.target.callback) {
    TensorReadbackResult resultPayload{.name = state.target.name,
                                       .tensor = state.target.tensor,
                                       .data = std::move(payload),
                                       .dimensions = std::move(dimensions),
                                       .channels = channels,
                                       .dataType = dataType,
                                       .futureResult = completion.futureResult};
    state.target.callback(std::move(resultPayload));
  }
}


}  // namespace SecureMR
