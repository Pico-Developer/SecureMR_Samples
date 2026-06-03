#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "securemr_base.h"
#include "securemr_utils/session.h"
#include "securemr_utils/utils.h"

namespace SecureMR {

class FaceMediaPipePipelineProgram final : public ISecureMR {
 public:
  FaceMediaPipePipelineProgram(const XrInstance& instance, const XrSession& session);
  ~FaceMediaPipePipelineProgram() override;

  void CreateFramework() override;
  void CreatePipelines() override;
  void RunPipelines() override;
  [[nodiscard]] bool LoadingFinished() const override;

 private:
  XrInstance xr_instance;
  XrSession xr_session;

  std::shared_ptr<FrameworkSession> frameworkSession;
  ModelPackagePipelineBundle modelPackageBundle;

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;

  std::atomic<bool> keepRunning{true};
  bool pipelineAllInitialized = false;
};

}  // namespace SecureMR
