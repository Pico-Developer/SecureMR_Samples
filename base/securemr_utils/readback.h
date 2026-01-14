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
#include <openxr/openxr.h>
#include <map>

#include "tensor.h"
#include "pch.h"


#define LOG_TAG "Readback"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

#include "graphicsplugin.h"

namespace SecureMR {

class FrameworkSession;

class ReadbackController {
 public:

  typedef uint64_t ReadbackRequest;

  /**
   * Constructor: initializes the readback controller.
   * @param frameworkSession OpenXR framework session object
   * @param readbackTensor global tensor to read back
   */
  ReadbackController(std::shared_ptr<FrameworkSession> frameworkSession, std::shared_ptr<GlobalTensor> readbackTensor);

  /**
   * Issue an asynchronous buffer readback request.
   * @param request output parameter; returns a request handle used to retrieve the result later
   * @return true if the request is issued successfully; false otherwise
   */
  bool RequestReadbackBuffer(ReadbackRequest* &request);

  /**
   * Attempt to complete buffer readback and retrieve the result (non-blocking).
   * @param req request handle
   * @param out output parameter; pointer to the readback tensor buffer
   * @return true if acquired; false if not ready or failed
   */
  bool TryAcquireReadbackBuffer(const ReadbackRequest& req, XrReadbackTensorBufferPICO* &out);

  /**
   * Issue an asynchronous texture readback/creation request.
   * @param request output parameter; returns a request handle
   * @return true on success; false otherwise
   */
  bool RequestReadbackTexture(ReadbackRequest* &request);

  /**
   * Attempt to complete texture readback/creation and return the texture object (non-blocking).
   * @param req request handle
   * @param out output parameter; the resulting `XrReadbackTexturePICO`
   * @return true if acquired; false if not ready or failed
   */
  bool TryAcquireReadbackTexture(const ReadbackRequest& req, XrReadbackTexturePICO &out);

  /**
   * Retrieve the image content of the readback texture to a file.
   * @param texture readback texture handle
   * @param textureRes output parameter; returns the real texture resource based on graphics platform.
   * @return true on success; false otherwise
   */
  bool RetrieveTexture(XrReadbackTexturePICO texture, XrReadbackTextureImageBaseHeaderPICO* textureRes);

  /**
   * Release texture resources produced by readback.
   * @param texture texture handle to release
   */
  void ReleaseReadbackTexture(XrReadbackTexturePICO texture);

  /**
   * Reset pending requests.
   */
  void ResetPendingRequests();

 private:
  void initializeGraphicsContext();

  std::shared_ptr<GlobalTensor> mReadbackTensor;
  std::map<uint64_t, XrFutureEXT*> mPendingFutures;
  // buffer openxr interfaces
  PFN_xrCreateBufferFromGlobalTensorAsyncPICO xrCreateBufferFromGlobalTensorAsyncPICO;
  PFN_xrCreateBufferFromGlobalTensorCompletePICO xrCreateBufferFromGlobalTensorCompletePICO;

  // texture openxr interfaces
  PFN_xrCreateTextureFromGlobalTensorCompletePICO xrCreateTextureFromGlobalTensorCompletePICO;
  PFN_xrCreateTextureFromGlobalTensorAsyncPICO xrCreateTextureFromGlobalTensorAsyncPICO;
  PFN_xrGetReadbackTextureImagePICO xrGetReadbackTextureImagePICO;
  PFN_xrReleaseReadbackTexturePICO xrReleaseReadbackTexturePICO;

};
}
