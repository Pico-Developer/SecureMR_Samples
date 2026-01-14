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

#include "readback.h"

namespace SecureMR  {
ReadbackController::ReadbackController(std::shared_ptr<FrameworkSession> frameworkSession, std::shared_ptr<GlobalTensor> readbackTensor)
{
    mReadbackTensor = readbackTensor;

    xrCreateBufferFromGlobalTensorAsyncPICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrCreateBufferFromGlobalTensorAsyncPICO>(
            "xrCreateBufferFromGlobalTensorAsyncPICO");
    xrCreateBufferFromGlobalTensorCompletePICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrCreateBufferFromGlobalTensorCompletePICO>(
            "xrCreateBufferFromGlobalTensorCompletePICO");
    xrCreateTextureFromGlobalTensorAsyncPICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrCreateTextureFromGlobalTensorAsyncPICO>(
            "xrCreateTextureFromGlobalTensorAsyncPICO");
    xrCreateTextureFromGlobalTensorCompletePICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrCreateTextureFromGlobalTensorCompletePICO>(
            "xrCreateTextureFromGlobalTensorCompletePICO");
    xrGetReadbackTextureImagePICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrGetReadbackTextureImagePICO>("xrGetReadbackTextureImagePICO");
    xrReleaseReadbackTexturePICO =
        frameworkSession->getAPIFromXrInstance<PFN_xrReleaseReadbackTexturePICO>("xrReleaseReadbackTexturePICO");
  }


  bool ReadbackController::RequestReadbackBuffer(ReadbackRequest* &request) {
    auto new_future = new XrFutureEXT();
    auto res = xrCreateBufferFromGlobalTensorAsyncPICO(static_cast<XrSecureMrTensorPICO>(*mReadbackTensor), new_future);
    if (res != XR_SUCCESS)
    {
      LOGE("Create Readback Buffer failed! res = %d", res);
      return false;
    }
    mPendingFutures.emplace((uint64_t)new_future, new_future);
    request = new ReadbackRequest((ReadbackRequest)new_future);
    return true;
  }

  bool ReadbackController::RequestReadbackTexture(ReadbackRequest * &request) {
    auto new_future = new XrFutureEXT();

    auto res = xrCreateTextureFromGlobalTensorAsyncPICO(static_cast<XrSecureMrTensorPICO>(*mReadbackTensor), new_future);
    if (res != XR_SUCCESS)
    {
      LOGE("Create Readback Texture failed! res = %d", res);
      return false;
    }
    mPendingFutures.emplace((uint64_t)new_future, new_future);
    request = new ReadbackRequest((ReadbackRequest)new_future);
    return true;
  }

  bool ReadbackController::TryAcquireReadbackTexture(const SecureMR::ReadbackController::ReadbackRequest &req, XrReadbackTexturePICO &out) {
    auto key = (uint64_t)(req);
    if (mPendingFutures.find(key) == mPendingFutures.end())
    {
      LOGE("Readback Request invalid!");
      return false;
    }
    XrCreateTextureFromGlobalTensorCompletionPICO completion;
    auto ret = xrCreateTextureFromGlobalTensorCompletePICO(
        XrSecureMrTensorPICO(*mReadbackTensor),
        *mPendingFutures[key],
        &completion);
    if (ret != XR_SUCCESS) {
      return false;
    }
    mPendingFutures.erase(key);
    out = completion.texture;
    return true;
  }

  bool ReadbackController::TryAcquireReadbackBuffer(const SecureMR::ReadbackController::ReadbackRequest &req, XrReadbackTensorBufferPICO* &out) {
    auto key = (uint64_t)(req);
    if (mPendingFutures.find(key) == mPendingFutures.end())
    {
      LOGE("Readback Request invalid!");
      return false;
    }

    XrCreateBufferFromGlobalTensorCompletionPICO completion;
    completion.tensorBuffer = new XrReadbackTensorBufferPICO();

    completion.tensorBuffer->bufferCapacityInput = 0;
    auto ret =
        xrCreateBufferFromGlobalTensorCompletePICO(static_cast<XrSecureMrTensorPICO>(*mReadbackTensor), *mPendingFutures[key], &completion);
    if (ret != XR_SUCCESS) {
      if (ret != XR_ERROR_FUTURE_PENDING_EXT)
      {
        LOGE("TryAcquireReadbackBuffer failed! ret = %d", ret);
      }
      return false;
    }
    completion.tensorBuffer->bufferCapacityInput = completion.tensorBuffer->bufferCountOutput;
    completion.tensorBuffer->buffer = new char[completion.tensorBuffer->bufferCapacityInput];
    ret = xrCreateBufferFromGlobalTensorCompletePICO(static_cast<XrSecureMrTensorPICO>(*mReadbackTensor), *mPendingFutures[key], &completion);
    if (ret != XR_SUCCESS)
    {
      LOGE("TryAcquireReadbackBuffer failed! ret = %d", ret);
      return false;
    }
    mPendingFutures.erase(key);
    out = completion.tensorBuffer;
    return true;
  }

  bool ReadbackController::RetrieveTexture(XrReadbackTexturePICO texture, XrReadbackTextureImageBaseHeaderPICO* textureRes)
  {
    XrResult ret;
    ret = xrGetReadbackTextureImagePICO(texture, textureRes);
    if (ret != XR_SUCCESS)
    {
      LOGE("RetrieveTexture failed! ret = %d", ret);
      return false;
    }
    return true;
  }

  void ReadbackController::ReleaseReadbackTexture(XrReadbackTexturePICO texture)
  {
    xrReleaseReadbackTexturePICO(texture);
  }

  void ReadbackController::ResetPendingRequests() {
    mPendingFutures.clear();
  }
}