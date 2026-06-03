#include "face_mediapipe_pipeline.h"

#include "pch.h"
#include "securemr_utils/pipeline.h"

#include <chrono>
#include <string>

namespace {
constexpr int kCameraWidth = 580;
constexpr int kCameraHeight = 326;
constexpr const char* kModelPackageAssetPath = "face-mediapipe-pipeline";
constexpr auto kPipelineSubmitInterval = std::chrono::milliseconds(50);

}  // namespace

namespace SecureMR {

FaceMediaPipePipelineProgram::FaceMediaPipePipelineProgram(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

FaceMediaPipePipelineProgram::~FaceMediaPipePipelineProgram() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& t : pipelineRunners) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void FaceMediaPipePipelineProgram::CreateFramework() {
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, kCameraWidth, kCameraHeight);
}

void FaceMediaPipePipelineProgram::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    std::string loadError;
    const bool packageLoaded = SecureMrUtils::LoadModelPackagePipelinesFromAssets(
        kModelPackageAssetPath, frameworkSession, {}, modelPackageBundle, loadError);
    if (!packageLoaded) {
      Log::Write(Log::Level::Error, Fmt("FaceMediaPipePipeline: model package load failed: %s", loadError.c_str()));
    }

    const auto detectionPipelineIt = modelPackageBundle.pipelines.find("detection");
    const auto displayPipelineIt = modelPackageBundle.pipelines.find("display");
    const bool packageReady = packageLoaded && detectionPipelineIt != modelPackageBundle.pipelines.end() &&
                              displayPipelineIt != modelPackageBundle.pipelines.end() &&
                              detectionPipelineIt->second.pipeline != nullptr && displayPipelineIt->second.pipeline != nullptr;
    if (packageLoaded && !packageReady) {
      Log::Write(Log::Level::Error, "FaceMediaPipePipeline: package pipelines missing detection/display entries");
    }

    keepRunning = packageReady;
    pipelineAllInitialized = true;
    initialized.notify_all();
  });
}

void FaceMediaPipePipelineProgram::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard, [this]() { return pipelineAllInitialized; });
    }
    if (!keepRunning) {
      return;
    }

    const auto& modelPackage = modelPackageBundle.pipelines.at("detection");
    const auto& displayPackage = modelPackageBundle.pipelines.at("display");

    while (keepRunning) {
      const XrSecureMrPipelineRunPICO inferenceRun =
          modelPackage.pipeline->submit(modelPackage.submitBindings, XR_NULL_HANDLE, nullptr);
      displayPackage.pipeline->submit(displayPackage.submitBindings, inferenceRun, nullptr);

      std::this_thread::sleep_for(kPipelineSubmitInterval);
    }
  });
}

bool FaceMediaPipePipelineProgram::LoadingFinished() const { return pipelineAllInitialized; }

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<FaceMediaPipePipelineProgram>(instance, session);
}

}  // namespace SecureMR
