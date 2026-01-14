// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "virtual_cube.h"

#include "common.h"
#include "cube_model.h"
#include "logger.h"
#include "xr_linear.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <random>
#include <string>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

namespace SecureMR {

namespace {
constexpr float kDefaultCubeDistance = -1.2f;
struct FaceAxes {
  Face face;
  XrVector3f normal;
  XrVector3f up;
  XrVector3f right;
};
constexpr std::array<FaceAxes, 6> kFaceAxes = {{
    {Face::U, {0.f, 1.f, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}},
    {Face::R, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}},
    {Face::F, {0.f, 0.f, -1.f}, {0.f, 1.f, 0.f}, {1.f, 0.f, 0.f}},
    {Face::D, {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}},
    {Face::L, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, -1.f}},
    {Face::B, {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f}, {-1.f, 0.f, 0.f}},
}};

const FaceAxes& AxesForFace(Face face) {
  for (const auto& axes : kFaceAxes) {
    if (axes.face == face) return axes;
  }
  return kFaceAxes[0];
}
}  // namespace

VirtualCube::VirtualCube() {
  downQuat_.w = 1.f;
  rightQuat_.w = 1.f;
  leftQuat_.w = 1.f;
  upQuat_.w = 1.f;
  anchorPosition_ = {0.f, 0.f, kDefaultCubeDistance};
  SetDefaultOrientation();
}

void VirtualCube::SetCubeModel(const CubeModel* model) {
  cubeModel_ = model;
  initialized_ = false;
}

void VirtualCube::PreviewFace(Face face, const std::array<Color, 9>& colors,
                              const std::array<XrVector3f, 9>& displayColors) {
  previewFace_ = face;
  previewColors_ = colors;
  previewDisplayColors_ = displayColors;
  initialized_ = false;
}

void VirtualCube::ClearPreviewFace(Face face) {
  if (previewFace_ && *previewFace_ == face) {
    previewFace_.reset();
    initialized_ = false;
  }
}

void VirtualCube::ClearPreview() {
  if (previewFace_) {
    previewFace_.reset();
    initialized_ = false;
  }
}

void VirtualCube::QueueFaceMoves(const std::vector<Move>& moves) {
  ClearFaceMoves();
  for (const auto& m : moves) {
    faceQueue_.push_back(m);
  }
  totalMoves_ = static_cast<int>(faceQueue_.size());
  Log::Write(Log::Level::Info,
             Fmt("VirtualCube: queued %zu face moves (pending=%zu)", moves.size(), faceQueue_.size()));
}

void VirtualCube::ClearFaceMoves() {
  faceQueue_.clear();
  faceMoveActive_ = false;
  faceMoveTimer_ = 0.f;
  pauseActive_ = false;
  pauseTimer_ = 0.f;
  totalMoves_ = 0;
  movesStarted_ = 0;
  currentMoveIdx_.reset();
}

void VirtualCube::QueueOrientationOp(CubeTurnOp op) {
  EnsureTurnQuats();
  OrientationCommand cmd{};
  cmd.op = op;
  cmd.delta = DeltaForOp(op);
  orientationOps_.push_back(cmd);
  StartNextOrientationOp();
}

void VirtualCube::QueueOrientationOpRelative(CubeTurnOp op, const XrQuaternionf& viewerOrientation) {
  auto normals = ComputeFaceWorldNormals();
  auto info = ComputeViewerFacingInfo(viewerOrientation, normals);
  if (!info) {
    QueueOrientationOp(op);
    return;
  }
  Face targetFace = info->front;
  switch (op) {
    case CubeTurnOp::Down: targetFace = info->up; break;
    case CubeTurnOp::Up: targetFace = info->down; break;
    case CubeTurnOp::Right: targetFace = info->left; break;
    case CubeTurnOp::Left: targetFace = info->right; break;
  }
  if (targetFace == info->front) {
    return;
  }
  XrQuaternionf delta{};
  if (!BuildDeltaBetweenFaces(info->front, targetFace, normals, delta)) {
    QueueOrientationOp(op);
    return;
  }
  OrientationCommand cmd{};
  cmd.op = op;
  cmd.delta = delta;
  orientationOps_.push_back(cmd);
  StartNextOrientationOp();
}

void VirtualCube::ResetOrientation() {
  SetDefaultOrientation();
  orientationTimer_ = 0.f;
  orientationAnimating_ = false;
  orientationOps_.clear();
  didWarmup_ = false;
  // Default face turn duration for animations
  faceMoveDuration_ = 2.0f;
}

void VirtualCube::QueueWarmupAnimation() {
  if (didWarmup_) return;
  QueueOrientationOp(CubeTurnOp::Down);
  QueueOrientationOp(CubeTurnOp::Right);
  didWarmup_ = true;
}

void VirtualCube::SetAnchorPosition(const XrVector3f& position) {
  anchorPosition_ = position;
}

std::optional<Face> VirtualCube::GetFacingFace(const XrQuaternionf& viewerOrientation) const {
  XrVector3f viewDirLocal{0.f, 0.f, 1.f};
  XrVector3f viewDir{};
  XrQuaternionf_RotateVector3f(&viewDir, &viewerOrientation, &viewDirLocal);

  float bestDot = -2.f;
  std::optional<Face> bestFace;
  for (const auto& entry : kFaceAxes) {
    XrVector3f worldNormal{};
    XrQuaternionf_RotateVector3f(&worldNormal, &cubeOrientation_, &entry.normal);
    const float dot = XrVector3f_Dot(&worldNormal, &viewDir);
    if (dot > bestDot) {
      bestDot = dot;
      bestFace = entry.face;
    }
  }
  return bestFace;
}

std::optional<std::array<XrVector3f, 9>> VirtualCube::GetFaceDisplayColors(Face face) const {
  if (cubies_.empty()) return std::nullopt;
  std::array<XrVector3f, 9> out{};
  bool any = false;
  const int faceIdx = static_cast<int>(face);
  for (const auto& c : cubies_) {
    bool boundary = false;
    int row = 0;
    int col = 0;
    switch (face) {
      case Face::U:
        boundary = (c.y == +1);
        row = (c.z + 1);
        col = (c.x + 1);
        break;
      case Face::D:
        boundary = (c.y == -1);
        row = (-c.z + 1);
        col = (c.x + 1);
        break;
      case Face::F:
        boundary = (c.z == -1);
        row = (-c.y + 1);
        col = (c.x + 1);
        break;
      case Face::B:
        boundary = (c.z == +1);
        row = (-c.y + 1);
        col = (-c.x + 1);
        break;
      case Face::R:
        boundary = (c.x == +1);
        row = (-c.y + 1);
        col = (c.z + 1);
        break;
      case Face::L:
        boundary = (c.x == -1);
        row = (-c.y + 1);
        col = (-c.z + 1);
        break;
    }
    if (!boundary) continue;
    int idx = row * 3 + col;
    if (idx >= 0 && idx < 9) {
      out[idx] = c.displayColors[faceIdx];
      any = true;
    }
  }
  if (!any) return std::nullopt;
  return out;
}

bool VirtualCube::BuildMesh(std::vector<Geometry::Vertex>& outVerts, std::vector<uint16_t>& outIdx,
                            XrPosef& outPose) {
  outVerts.clear();
  outIdx.clear();
  outPose = {};
  outPose.orientation = cubeOrientation_;
  outPose.position = anchorPosition_;

  EnsureInitialized();

  const float half = (cubeSize_ / 3.0f - gap_) * 0.5f;
  const float spacing = (cubeSize_ / 3.0f);

  const bool rotatingFace = faceMoveActive_;
  const float angle = CurrentFaceAngleRad();
  char rotAxis = 0;
  int layerVal = 0;
  if (rotatingFace) {
    switch (currentFaceMove_.type) {
      case MoveType::U: rotAxis = 'y'; layerVal = +1; break;
      case MoveType::D: rotAxis = 'y'; layerVal = -1; break;
      case MoveType::R: rotAxis = 'x'; layerVal = +1; break;
      case MoveType::L: rotAxis = 'x'; layerVal = -1; break;
      case MoveType::F: rotAxis = 'z'; layerVal = -1; break;
      case MoveType::B: rotAxis = 'z'; layerVal = +1; break;
    }
  }

  for (const auto& c : cubies_) {
    XrVector3f center = {c.x * spacing, c.y * spacing, c.z * spacing};
    bool inLayer = false;
    if (rotatingFace) {
      if (rotAxis == 'x') inLayer = (c.x == layerVal);
      if (rotAxis == 'y') inLayer = (c.y == layerVal);
      if (rotAxis == 'z') inLayer = (c.z == layerVal);
    }
    PushBox(outVerts, outIdx, center, half, rotatingFace && inLayer, rotAxis, angle);

    auto addStickerIf = [&](char faceChar, bool cond, int faceIdx) {
      if (!cond) return;
      XrVector3f rgb = c.displayColors[faceIdx];
      const bool hasDisplay = (rgb.x != 0.f || rgb.y != 0.f || rgb.z != 0.f);
      if (!hasDisplay && c.faces[faceIdx] != Color::Unknown) {
        // Regenerate display color if the sticker lost its cached RGB value
        rgb = ToDisplayColor(c.faces[faceIdx]);
      } else if (!hasDisplay && c.faces[faceIdx] == Color::Unknown) {
        rgb = {0.3f, 0.3f, 0.3f};
      }
      PushSticker(outVerts, outIdx, center, half, faceChar, rgb, rotatingFace && inLayer, rotAxis, angle);
    };

    addStickerIf('U', c.y == +1, 0);
    addStickerIf('R', c.x == +1, 1);
    addStickerIf('F', c.z == -1, 2);
    addStickerIf('D', c.y == -1, 3);
    addStickerIf('L', c.x == -1, 4);
    addStickerIf('B', c.z == +1, 5);
  }

  return (!outVerts.empty() && !outIdx.empty());
}

void VirtualCube::Solving() {
  if (!faceMoveActive_ && !pauseActive_ && faceQueue_.empty()) {
    currentMoveIdx_.reset();
  }
  if (faceMoveActive_) {
    faceMoveTimer_ += 0.016f;  // 60 FPS
    if (faceMoveTimer_ >= faceMoveDuration_) {
      // Commit the just-finished face move to sticker positions/colors
      ApplyFaceMoveToCubies(currentFaceMove_);
      faceMoveActive_ = false;
      faceMoveTimer_ = 0.f;
      // Start pause after move completes
      pauseActive_ = true;
      pauseTimer_ = 0.f;
    }
  }
  
  if (pauseActive_) {
    pauseTimer_ += 0.016f;  // 60 FPS
    if (pauseTimer_ >= pauseAfterMove_) {
      pauseActive_ = false;
      pauseTimer_ = 0.f;
    }
  }
  
  if (!faceMoveActive_ && !pauseActive_) {
    BeginNextFaceMove();
  }
}

void VirtualCube::Update(float dt) {
  EnsureInitialized();
  UpdateOrientation(dt);
}

void VirtualCube::EnsureInitialized() {
  if (initialized_) return;

  if (const char* s = std::getenv("RUBICS_SIZE_M")) {
    float v = std::strtof(s, nullptr);
    if (v > 0.05f && v < 1.0f) cubeSize_ = v;
  }
  if (const char* s = std::getenv("RUBICS_GAP_M")) {
    float v = std::strtof(s, nullptr);
    if (v >= 0.0f && v < 0.02f) gap_ = v;
  }
  if (const char* s = std::getenv("RUBICS_EPS_M")) {
    float v = std::strtof(s, nullptr);
    if (v >= 0.0f && v < 0.01f) stickerEps_ = v;
  }
  if (const char* s = std::getenv("RUBICS_SWAP_RB")) {
    swapRB_ = (std::atoi(s) != 0);
  }
  if (const char* s = std::getenv("RUBICS_TURN_MS")) {
    int ms = std::atoi(s);
    if (ms >= 50 && ms <= 10000) faceMoveDuration_ = ms / 1000.0f;
  }
#ifdef __ANDROID__
  char prop[PROP_VALUE_MAX] = {0};
  if (__system_property_get("rubics_size_m", prop) > 0) {
    float v = std::strtof(prop, nullptr);
    if (v > 0.05f && v < 1.0f) cubeSize_ = v;
  }
  if (__system_property_get("rubics_gap_m", prop) > 0) {
    float v = std::strtof(prop, nullptr);
    if (v >= 0.0f && v < 0.02f) gap_ = v;
  }
  if (__system_property_get("rubics_eps_m", prop) > 0) {
    float v = std::strtof(prop, nullptr);
    if (v >= 0.0f && v < 0.01f) stickerEps_ = v;
  }
  if (__system_property_get("rubics_swap_rb", prop) > 0) {
    swapRB_ = (std::atoi(prop) != 0);
  }
  if (__system_property_get("rubics_turn_ms", prop) > 0) {
    int ms = std::atoi(prop);
    if (ms >= 50 && ms <= 10000) faceMoveDuration_ = ms / 1000.0f;
  }
#endif

  cubies_.clear();
  cubies_.reserve(27);
  for (int iy = -1; iy <= 1; ++iy) {
    for (int iz = -1; iz <= 1; ++iz) {
      for (int ix = -1; ix <= 1; ++ix) {
        CubieState c{};
        c.x = ix;
        c.y = iy;
        c.z = iz;
        for (auto& dc : c.displayColors) {
          dc = {0.3f, 0.3f, 0.3f};
        }
        cubies_.push_back(c);
      }
    }
  }

  if (cubeModel_) {
    InitializeFromCubeModel();
  } else {
    InitializeDefaultColors();
  }
  initialized_ = true;
}

void VirtualCube::InitializeDefaultColors() {
  std::array<Color, 9> fill{};
  auto apply = [&](Face face, Color color) {
    fill.fill(color);
    std::array<XrVector3f, 9> display{};
    display.fill(ToRgb(color));
    AssignFaceColors(face, fill, &display);
  };
  apply(Face::F, Color::G);
  apply(Face::U, Color::W);
  apply(Face::L, Color::O);
  apply(Face::B, Color::B);
  apply(Face::D, Color::Y);
  apply(Face::R, Color::R);
  ApplyPreviewOverride();
}

void VirtualCube::InitializeFromCubeModel() {
  for (auto& c : cubies_) {
    for (int k = 0; k < 6; ++k) c.faces[k] = Color::Unknown;
  }
  if (cubeModel_) {
    auto apply = [&](Face face) {
      auto opt = cubeModel_->GetFaceColors(face);
      if (!opt) return;
      auto raw = cubeModel_->GetFaceRawColors(face);
      std::array<XrVector3f, 9> display{};
      const std::array<XrVector3f, 9>* displayPtr = nullptr;
      if (raw) {
        for (int i = 0; i < 9; ++i) {
          display[i] = {(*raw)[i].r, (*raw)[i].g, (*raw)[i].b};
        }
        displayPtr = &display;
      }
      AssignFaceColors(face, *opt, displayPtr);
    };
    apply(Face::U);
    apply(Face::D);
    apply(Face::F);
    apply(Face::B);
    apply(Face::R);
    apply(Face::L);
  }
  ApplyPreviewOverride();

  // Debug: verify that every boundary sticker now has a resolved color
  int boundaryUnknown = 0;
  std::array<int, 6> missingPerFace{};
  for (const auto& c : cubies_) {
    for (int faceIdx = 0; faceIdx < 6; ++faceIdx) {
      bool boundary = false;
      switch (static_cast<Face>(faceIdx)) {
        case Face::U: boundary = (c.y == +1); break;
        case Face::D: boundary = (c.y == -1); break;
        case Face::F: boundary = (c.z == -1); break;
        case Face::B: boundary = (c.z == +1); break;
        case Face::R: boundary = (c.x == +1); break;
        case Face::L: boundary = (c.x == -1); break;
      }
      if (!boundary) continue;
      if (c.faces[faceIdx] == Color::Unknown) {
        ++boundaryUnknown;
        missingPerFace[faceIdx]++;
      }
    }
  }
  if (boundaryUnknown > 0) {
    Log::Write(Log::Level::Warning,
               Fmt("VirtualCube: InitializeFromCubeModel missing %d stickers (perFace=%d,%d,%d,%d,%d,%d)",
                   boundaryUnknown, missingPerFace[0], missingPerFace[1], missingPerFace[2],
                   missingPerFace[3], missingPerFace[4], missingPerFace[5]));
  } else {
    Log::Write(Log::Level::Info, "VirtualCube: InitializeFromCubeModel populated all stickers");
  }
}

void VirtualCube::AssignFaceColors(Face face, const std::array<Color, 9>& colors,
                                   const std::array<XrVector3f, 9>* displayColors) {
  const int faceIdx = static_cast<int>(face);
  for (auto& c : cubies_) {
    bool boundary = false;
    int row = 0;
    int col = 0;
    switch (face) {
      case Face::U:
        boundary = (c.y == +1);
        row = (c.z + 1);
        col = (c.x + 1);
        break;
      case Face::D:
        boundary = (c.y == -1);
        row = (-c.z + 1);
        col = (c.x + 1);
        break;
      case Face::F:
        boundary = (c.z == -1);
        row = (-c.y + 1);
        col = (c.x + 1);
        break;
      case Face::B:
        boundary = (c.z == +1);
        row = (-c.y + 1);
        col = (-c.x + 1);
        break;
      case Face::R:
        boundary = (c.x == +1);
        row = (-c.y + 1);
        col = (c.z + 1);
        break;
      case Face::L:
        boundary = (c.x == -1);
        row = (-c.y + 1);
        col = (-c.z + 1);
        break;
    }
    if (!boundary) continue;
    int idx = row * 3 + col;
    if (idx >= 0 && idx < 9) {
      c.faces[faceIdx] = colors[idx];
      XrVector3f disp{};
      if (displayColors) {
        disp = (*displayColors)[idx];
      } else {
        disp = ToDisplayColor(colors[idx]);
      }
      c.displayColors[faceIdx] = disp;
    }
  }
}

void VirtualCube::ApplyPreviewOverride() {
  if (previewFace_) {
    AssignFaceColors(*previewFace_, previewColors_, &previewDisplayColors_);
  }
}

void VirtualCube::BeginNextFaceMove() {
  if (faceQueue_.empty()) return;
  EnsureInitialized();
  currentFaceMove_ = faceQueue_.front();
  faceQueue_.pop_front();
  currentMoveIdx_ = movesStarted_;
  movesStarted_ = std::min(totalMoves_, movesStarted_ + 1);
  WarnIfMissingStickerColors(currentFaceMove_);
  faceMoveActive_ = true;
  faceMoveTimer_ = 0.f;
}

void VirtualCube::WarnIfMissingStickerColors(const Move& upcomingMove) const {
  if (cubies_.empty()) return;
  auto hasSticker = [](const CubieState& c, Face face) -> bool {
    switch (face) {
      case Face::U: return c.y == +1;
      case Face::D: return c.y == -1;
      case Face::F: return c.z == -1;
      case Face::B: return c.z == +1;
      case Face::R: return c.x == +1;
      case Face::L: return c.x == -1;
    }
    return false;
  };
  int stickersWithIssues = 0;
  int missingRgb = 0;
  int unknownColor = 0;
  std::array<int, 6> missingPerFace{};
  std::vector<std::string> samples;
  for (const auto& c : cubies_) {
    for (int faceIdx = 0; faceIdx < 6; ++faceIdx) {
      Face face = static_cast<Face>(faceIdx);
      if (!hasSticker(c, face)) continue;
      const XrVector3f& rgb = c.displayColors[faceIdx];
      bool issue = false;
      if (rgb.x == 0.f && rgb.y == 0.f && rgb.z == 0.f) {
        ++missingRgb;
        issue = true;
      }
      if (c.faces[faceIdx] == Color::Unknown) {
        ++unknownColor;
        issue = true;
      }
      if (issue) {
        ++stickersWithIssues;
        missingPerFace[faceIdx]++;
        if (samples.size() < 4) {
          samples.push_back(
              Fmt("face=%d xyz=(%d,%d,%d) color=%s", faceIdx, c.x, c.y, c.z, ColorToChar(c.faces[faceIdx]).c_str()));
        }
      }
    }
  }
  if (stickersWithIssues <= 0) return;
  auto moveChar = [](MoveType type) -> char {
    switch (type) {
      case MoveType::U: return 'U';
      case MoveType::R: return 'R';
      case MoveType::F: return 'F';
      case MoveType::D: return 'D';
      case MoveType::L: return 'L';
      case MoveType::B: return 'B';
    }
    return '?';
  };
  const char* suffix = "";
  switch (upcomingMove.turn) {
    case Turn::CW: suffix = ""; break;
    case Turn::CCW: suffix = "'"; break;
    case Turn::Double: suffix = "2"; break;
  }
  Log::Write(Log::Level::Warning,
             Fmt("VirtualCube: %d stickers missing colors before move %c%s (missingRGB=%d unknownColor=%d perFace=%d,%d,%d,%d,%d,%d samples=%s)",
                 stickersWithIssues, moveChar(upcomingMove.type), suffix, missingRgb, unknownColor,
                 missingPerFace[0], missingPerFace[1], missingPerFace[2],
                 missingPerFace[3], missingPerFace[4], missingPerFace[5],
                 samples.empty() ? "none" : samples[0].c_str()));
}

bool VirtualCube::IsCubieInActiveLayer(const CubieState& c) const {
  if (!faceMoveActive_) return false;
  switch (currentFaceMove_.type) {
    case MoveType::U: return c.y == +1;
    case MoveType::D: return c.y == -1;
    case MoveType::R: return c.x == +1;
    case MoveType::L: return c.x == -1;
    case MoveType::F: return c.z == -1;
    case MoveType::B: return c.z == +1;
  }
  return false;
}

std::optional<int> VirtualCube::ActiveMoveIndex() const {
  if ((faceMoveActive_ || pauseActive_) && currentMoveIdx_) return currentMoveIdx_;
  return std::nullopt;
}

int VirtualCube::SignForCW(MoveType t) {
  switch (t) {
    case MoveType::U: return -1;
    case MoveType::D: return +1;
    case MoveType::R: return -1;
    case MoveType::L: return +1;
    case MoveType::F: return -1;
    case MoveType::B: return +1;
  }
  return 1;
}

float VirtualCube::CurrentFaceAngleRad() const {
  if (!faceMoveActive_) return 0.f;
  float base = (currentFaceMove_.turn == Turn::Double) ? static_cast<float>(M_PI)
                                                      : static_cast<float>(M_PI) * 0.5f;
  float progress = std::clamp(faceMoveTimer_ / std::max(0.001f, faceMoveDuration_), 0.f, 1.f);
  float s = (currentFaceMove_.turn == Turn::Double)
                ? 1.f
                : (currentFaceMove_.turn == Turn::CW ? static_cast<float>(SignForCW(currentFaceMove_.type))
                                                     : -static_cast<float>(SignForCW(currentFaceMove_.type)));
  return s * base * progress;
}

void VirtualCube::RotateCubie(CubieState& c, char axis, int signedQuarterTurns) const {
  if (signedQuarterTurns == 0) return;
  // Normalize turns to [0,3] with positive meaning +90deg about the +axis (right-hand rule)
  int turns = signedQuarterTurns % 4;
  if (turns < 0) turns += 4;
  auto rotateOnce = [&](CubieState& cc) {
    // Rotate position
    int x = cc.x, y = cc.y, z = cc.z;
    if (axis == 'x') {
      cc.y = -z;
      cc.z = y;
      // Face orientation cycle: U -> B -> D -> F -> U
      Color u = cc.faces[static_cast<int>(Face::U)];
      Color f = cc.faces[static_cast<int>(Face::F)];
      Color d = cc.faces[static_cast<int>(Face::D)];
      Color b = cc.faces[static_cast<int>(Face::B)];
      XrVector3f ud = cc.displayColors[static_cast<int>(Face::U)];
      XrVector3f fd = cc.displayColors[static_cast<int>(Face::F)];
      XrVector3f dd = cc.displayColors[static_cast<int>(Face::D)];
      XrVector3f bd = cc.displayColors[static_cast<int>(Face::B)];
      cc.faces[static_cast<int>(Face::U)] = f;
      cc.faces[static_cast<int>(Face::B)] = u;
      cc.faces[static_cast<int>(Face::D)] = b;
      cc.faces[static_cast<int>(Face::F)] = d;
      cc.displayColors[static_cast<int>(Face::U)] = fd;
      cc.displayColors[static_cast<int>(Face::B)] = ud;
      cc.displayColors[static_cast<int>(Face::D)] = bd;
      cc.displayColors[static_cast<int>(Face::F)] = dd;
    } else if (axis == 'y') {
      cc.x = z;
      cc.z = -x;
      // Face orientation cycle: R -> F -> L -> B -> R
      Color r = cc.faces[static_cast<int>(Face::R)];
      Color f = cc.faces[static_cast<int>(Face::F)];
      Color l = cc.faces[static_cast<int>(Face::L)];
      Color b = cc.faces[static_cast<int>(Face::B)];
      XrVector3f rd = cc.displayColors[static_cast<int>(Face::R)];
      XrVector3f fd = cc.displayColors[static_cast<int>(Face::F)];
      XrVector3f ld = cc.displayColors[static_cast<int>(Face::L)];
      XrVector3f bd = cc.displayColors[static_cast<int>(Face::B)];
      cc.faces[static_cast<int>(Face::F)] = r;
      cc.faces[static_cast<int>(Face::L)] = f;
      cc.faces[static_cast<int>(Face::B)] = l;
      cc.faces[static_cast<int>(Face::R)] = b;
      cc.displayColors[static_cast<int>(Face::F)] = rd;
      cc.displayColors[static_cast<int>(Face::L)] = fd;
      cc.displayColors[static_cast<int>(Face::B)] = ld;
      cc.displayColors[static_cast<int>(Face::R)] = bd;
    } else if (axis == 'z') {
      cc.x = -y;
      cc.y = x;
      // Face orientation cycle: U -> L -> D -> R -> U
      Color u = cc.faces[static_cast<int>(Face::U)];
      Color l = cc.faces[static_cast<int>(Face::L)];
      Color d = cc.faces[static_cast<int>(Face::D)];
      Color r = cc.faces[static_cast<int>(Face::R)];
      XrVector3f ud = cc.displayColors[static_cast<int>(Face::U)];
      XrVector3f ld = cc.displayColors[static_cast<int>(Face::L)];
      XrVector3f dd = cc.displayColors[static_cast<int>(Face::D)];
      XrVector3f rd = cc.displayColors[static_cast<int>(Face::R)];
      cc.faces[static_cast<int>(Face::L)] = u;
      cc.faces[static_cast<int>(Face::D)] = l;
      cc.faces[static_cast<int>(Face::R)] = d;
      cc.faces[static_cast<int>(Face::U)] = r;
      cc.displayColors[static_cast<int>(Face::L)] = ud;
      cc.displayColors[static_cast<int>(Face::D)] = ld;
      cc.displayColors[static_cast<int>(Face::R)] = dd;
      cc.displayColors[static_cast<int>(Face::U)] = rd;
    }
  };

  for (int i = 0; i < turns; ++i) {
    rotateOnce(c);
  }
}

void VirtualCube::ApplyFaceMoveToCubies(const Move& move) {
  if (cubies_.empty()) return;

  char axis = 'y';
  int layer = 0;
  switch (move.type) {
    case MoveType::U: axis = 'y'; layer = +1; break;
    case MoveType::D: axis = 'y'; layer = -1; break;
    case MoveType::R: axis = 'x'; layer = +1; break;
    case MoveType::L: axis = 'x'; layer = -1; break;
    case MoveType::F: axis = 'z'; layer = -1; break;
    case MoveType::B: axis = 'z'; layer = +1; break;
  }

  int baseTurns = (move.turn == Turn::Double) ? 2 : 1;
  int dir = (move.turn == Turn::Double) ? 1 : (move.turn == Turn::CW ? 1 : -1);
  int signedTurns = baseTurns * dir;
  if (move.turn != Turn::Double) signedTurns *= SignForCW(move.type);
  if ((signedTurns % 4) == 0) return;

  for (auto& c : cubies_) {
    bool inLayer = false;
    switch (axis) {
      case 'x': inLayer = (c.x == layer); break;
      case 'y': inLayer = (c.y == layer); break;
      case 'z': inLayer = (c.z == layer); break;
    }
    if (!inLayer) continue;
    RotateCubie(c, axis, signedTurns);
  }
}

void VirtualCube::PushTri(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
                          const XrVector3f& a, const XrVector3f& b, const XrVector3f& c,
                          const XrVector3f& color) {
  uint16_t base = static_cast<uint16_t>(v.size());
  v.push_back(Geometry::Vertex{a, color});
  v.push_back(Geometry::Vertex{b, color});
  v.push_back(Geometry::Vertex{c, color});
  i.push_back(base + 0);
  i.push_back(base + 1);
  i.push_back(base + 2);
}

XrVector3f VirtualCube::ToRgb(Color c) {
  switch (c) {
    case Color::W: return {1.f, 1.f, 1.f};
    case Color::Y: return {1.f, 1.f, 0.f};
    case Color::R: return {1.f, 0.f, 0.f};
    case Color::O: return {1.f, 0.55f, 0.f};
    case Color::G: return {0.f, 1.f, 0.f};
    case Color::B: return {0.f, 0.f, 1.f};
    default: return {0.3f, 0.3f, 0.3f};
  }
}

XrVector3f VirtualCube::ToDisplayColor(Color c) const {
  XrVector3f rgb = ToRgb(c);
  if (swapRB_) {
    std::swap(rgb.x, rgb.z);
  }
  return rgb;
}

void VirtualCube::PushBox(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
                          const XrVector3f& center, float half, bool rotate, char axis, float angleRad) {
  const XrVector3f color{0.05f, 0.05f, 0.05f};
  XrVector3f LBB{center.x - half, center.y - half, center.z - half};
  XrVector3f LBF{center.x - half, center.y - half, center.z + half};
  XrVector3f LTB{center.x - half, center.y + half, center.z - half};
  XrVector3f LTF{center.x - half, center.y + half, center.z + half};
  XrVector3f RBB{center.x + half, center.y - half, center.z - half};
  XrVector3f RBF{center.x + half, center.y - half, center.z + half};
  XrVector3f RTB{center.x + half, center.y + half, center.z - half};
  XrVector3f RTF{center.x + half, center.y + half, center.z + half};

  auto rotatePoint = [&](XrVector3f& p) {
    if (!rotate || angleRad == 0.f) return;
    XrVector3f out = p;
    float s = std::sin(angleRad);
    float c = std::cos(angleRad);
    switch (axis) {
      case 'x':
        out.y = c * p.y - s * p.z;
        out.z = s * p.y + c * p.z;
        break;
      case 'y':
        out.x = c * p.x + s * p.z;
        out.z = -s * p.x + c * p.z;
        break;
      case 'z':
        out.x = c * p.x - s * p.y;
        out.y = s * p.x + c * p.y;
        break;
    }
    p = out;
  };

  if (rotate) {
    rotatePoint(LBB);
    rotatePoint(LBF);
    rotatePoint(LTB);
    rotatePoint(LTF);
    rotatePoint(RBB);
    rotatePoint(RBF);
    rotatePoint(RTB);
    rotatePoint(RTF);
  }

  PushTri(v, i, LTB, LBF, LBB, color); PushTri(v, i, LTB, LTF, LBF, color);
  PushTri(v, i, RTB, RBB, RBF, color); PushTri(v, i, RTB, RBF, RTF, color);
  PushTri(v, i, LBB, LBF, RBF, color); PushTri(v, i, LBB, RBF, RBB, color);
  PushTri(v, i, LTB, RTB, RTF, color); PushTri(v, i, LTB, RTF, LTF, color);
  PushTri(v, i, LBB, RBB, RTB, color); PushTri(v, i, LBB, RTB, LTB, color);
  PushTri(v, i, LBF, LTF, RTF, color); PushTri(v, i, LBF, RTF, RBF, color);
}

void VirtualCube::PushSticker(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
                              const XrVector3f& center, float half, char face,
                              const XrVector3f& color, bool rotate, char axis, float angleRad) {
  const float margin = gap_ * 0.4f;
  const float offset = stickerEps_;
  XrVector3f a{}, b{}, c{}, d{};
  switch (face) {
    case 'U':
      a = {center.x - half + margin, center.y + half + offset, center.z - half + margin};
      b = {center.x + half - margin, center.y + half + offset, center.z - half + margin};
      c = {center.x + half - margin, center.y + half + offset, center.z + half - margin};
      d = {center.x - half + margin, center.y + half + offset, center.z + half - margin};
      break;
    case 'D':
      a = {center.x - half + margin, center.y - half - offset, center.z + half - margin};
      b = {center.x + half - margin, center.y - half - offset, center.z + half - margin};
      c = {center.x + half - margin, center.y - half - offset, center.z - half + margin};
      d = {center.x - half + margin, center.y - half - offset, center.z - half + margin};
      break;
    case 'R':
      a = {center.x + half + offset, center.y + half - margin, center.z - half + margin};
      b = {center.x + half + offset, center.y + half - margin, center.z + half - margin};
      c = {center.x + half + offset, center.y - half + margin, center.z + half - margin};
      d = {center.x + half + offset, center.y - half + margin, center.z - half + margin};
      break;
    case 'L':
      a = {center.x - half - offset, center.y + half - margin, center.z + half - margin};
      b = {center.x - half - offset, center.y + half - margin, center.z - half + margin};
      c = {center.x - half - offset, center.y - half + margin, center.z - half + margin};
      d = {center.x - half - offset, center.y - half + margin, center.z + half - margin};
      break;
    case 'F':
      a = {center.x - half + margin, center.y + half - margin, center.z - half - offset};
      b = {center.x + half - margin, center.y + half - margin, center.z - half - offset};
      c = {center.x + half - margin, center.y - half + margin, center.z - half - offset};
      d = {center.x - half + margin, center.y - half + margin, center.z - half - offset};
      break;
    case 'B':
      a = {center.x + half - margin, center.y + half - margin, center.z + half + offset};
      b = {center.x - half + margin, center.y + half - margin, center.z + half + offset};
      c = {center.x - half + margin, center.y - half + margin, center.z + half + offset};
      d = {center.x + half - margin, center.y - half + margin, center.z + half + offset};
      break;
    default:
      return;
  }
  auto rotatePoint = [&](XrVector3f& p) {
    if (!rotate || angleRad == 0.f) return;
    XrVector3f out = p;
    float s = std::sin(angleRad);
    float c = std::cos(angleRad);
    switch (axis) {
      case 'x':
        out.y = c * p.y - s * p.z;
        out.z = s * p.y + c * p.z;
        break;
      case 'y':
        out.x = c * p.x + s * p.z;
        out.z = -s * p.x + c * p.z;
        break;
      case 'z':
        out.x = c * p.x - s * p.y;
        out.y = s * p.x + c * p.y;
        break;
    }
    p = out;
  };
  rotatePoint(a);
  rotatePoint(b);
  rotatePoint(c);
  rotatePoint(d);

  PushTri(v, i, a, c, b, color);
  PushTri(v, i, a, d, c, color);
}

void VirtualCube::UpdateOrientation(float dt) {
  if (!orientationAnimating_) StartNextOrientationOp();
  if (!orientationAnimating_) return;

  orientationTimer_ += std::max(0.f, dt);
  float denom = orientationDuration_ > 0.f ? orientationDuration_ : 0.001f;
  float s = orientationTimer_ / denom;
  if (s >= 1.f) {
    cubeOrientation_ = orientationEnd_;
    orientationAnimating_ = false;
    orientationTimer_ = 0.f;
    StartNextOrientationOp();
    return;
  }
  XrQuaternionf_Lerp(&cubeOrientation_, &orientationStart_, &orientationEnd_, s);
  XrQuaternionf_Normalize(&cubeOrientation_);
}

void VirtualCube::StartNextOrientationOp() {
  if (orientationAnimating_ || orientationOps_.empty()) return;
  OrientationCommand cmd = orientationOps_.front();
  orientationOps_.pop_front();
  orientationStart_ = cubeOrientation_;
  XrQuaternionf_Multiply(&orientationEnd_, &cubeOrientation_, &cmd.delta);
  XrQuaternionf_Normalize(&orientationEnd_);
  orientationTimer_ = 0.f;
  orientationAnimating_ = true;
}

XrQuaternionf VirtualCube::DeltaForOp(CubeTurnOp op) {
  EnsureTurnQuats();
  switch (op) {
    case CubeTurnOp::Down: return downQuat_;
    case CubeTurnOp::Right: return rightQuat_;
    case CubeTurnOp::Left: return leftQuat_;
    case CubeTurnOp::Up: return upQuat_;
  }
  return {};
}

void VirtualCube::EnsureTurnQuats() {
  if (turnQuatsInit_) return;
  const float halfPi = static_cast<float>(M_PI) * 0.5f;
  XrVector3f axisX{1.f, 0.f, 0.f};
  XrVector3f axisY{0.f, 1.f, 0.f};
  XrQuaternionf_CreateFromAxisAngle(&upQuat_, &axisX, halfPi);
  XrQuaternionf_CreateFromAxisAngle(&downQuat_, &axisX, -halfPi);
  XrQuaternionf_CreateFromAxisAngle(&leftQuat_, &axisY, -halfPi);
  XrQuaternionf_CreateFromAxisAngle(&rightQuat_, &axisY, halfPi);
  turnQuatsInit_ = true;
}

std::array<XrVector3f, 6> VirtualCube::ComputeFaceWorldNormals() const {
  std::array<XrVector3f, 6> normals{};
  for (const auto& entry : kFaceAxes) {
    XrVector3f world{};
    XrQuaternionf_RotateVector3f(&world, &cubeOrientation_, &entry.normal);
    normals[static_cast<int>(entry.face)] = world;
  }
  return normals;
}

std::optional<VirtualCube::ViewerFacingInfo> VirtualCube::ComputeViewerFacingInfo(
    const XrQuaternionf& viewerOrientation, const std::array<XrVector3f, 6>& normals) const {
  XrVector3f viewForward{};
  XrVector3f viewUp{};
  const XrVector3f localForward{0.f, 0.f, -1.f};
  const XrVector3f localUp{0.f, 1.f, 0.f};
  XrQuaternionf_RotateVector3f(&viewForward, &viewerOrientation, &localForward);
  XrQuaternionf_RotateVector3f(&viewUp, &viewerOrientation, &localUp);
  XrVector3f viewRight{};
  XrVector3f_Cross(&viewRight, &viewForward, &viewUp);
  float rightLen = XrVector3f_Length(&viewRight);
  if (rightLen < 1e-4f) {
    return std::nullopt;
  }
  XrVector3f_Normalize(&viewRight);

  auto selectFace = [&](const XrVector3f& axis) {
    float bestDot = -FLT_MAX;
    Face bestFace = Face::F;
    for (int i = 0; i < 6; ++i) {
      float dot = XrVector3f_Dot(&normals[i], &axis);
      if (dot > bestDot) {
        bestDot = dot;
        bestFace = static_cast<Face>(i);
      }
    }
    return bestFace;
  };

  ViewerFacingInfo info{};
  info.front = selectFace(viewForward);
  XrVector3f negForward{-viewForward.x, -viewForward.y, -viewForward.z};
  info.back = selectFace(negForward);
  info.up = selectFace(viewUp);
  XrVector3f negUp{-viewUp.x, -viewUp.y, -viewUp.z};
  info.down = selectFace(negUp);
  info.right = selectFace(viewRight);
  XrVector3f negRight{-viewRight.x, -viewRight.y, -viewRight.z};
  info.left = selectFace(negRight);
  return info;
}

bool VirtualCube::BuildDeltaBetweenFaces(Face from, Face to, const std::array<XrVector3f, 6>& normals,
                                         XrQuaternionf& outDelta) const {
  if (from == to) return false;
  const XrVector3f& fromDir = normals[static_cast<int>(from)];
  const XrVector3f& toDir = normals[static_cast<int>(to)];
  XrVector3f axis{};
  XrVector3f_Cross(&axis, &fromDir, &toDir);
  float axisLen = XrVector3f_Length(&axis);
  float dot = XrVector3f_Dot(&fromDir, &toDir);
  dot = std::clamp(dot, -1.0f, 1.0f);
  if (axisLen < 1e-4f) {
    // Faces are parallel or opposite; fall back to default behavior
    return false;
  }
  XrVector3f_Normalize(&axis);
  float angle = std::acos(dot);
  XrQuaternionf_CreateFromAxisAngle(&outDelta, &axis, angle);
  return true;
}

bool VirtualCube::MapScreenGridToFaceIndices(Face face, const XrQuaternionf& viewerOrientation,
                                              std::array<int, 9>& outMapping) const {
  const auto& axes = AxesForFace(face);
  XrVector3f normal{};
  XrVector3f faceUp{};
  XrVector3f faceRight{};
  XrQuaternionf_RotateVector3f(&normal, &cubeOrientation_, &axes.normal);
  XrQuaternionf_RotateVector3f(&faceUp, &cubeOrientation_, &axes.up);
  XrQuaternionf_RotateVector3f(&faceRight, &cubeOrientation_, &axes.right);
  const XrVector3f localRight{1.f, 0.f, 0.f};
  const XrVector3f localDown{0.f, -1.f, 0.f};
  XrVector3f viewRight{};
  XrVector3f viewDown{};
  XrQuaternionf_RotateVector3f(&viewRight, &viewerOrientation, &localRight);
  XrQuaternionf_RotateVector3f(&viewDown, &viewerOrientation, &localDown);

  auto projectPlane = [&](XrVector3f& v) {
    float d = XrVector3f_Dot(&v, &normal);
    v.x -= d * normal.x;
    v.y -= d * normal.y;
    v.z -= d * normal.z;
    float len = XrVector3f_Length(&v);
    if (len < 1e-4f) return false;
    XrVector3f_Scale(&v, &v, 1.0f / len);
    return true;
  };
  if (!projectPlane(viewRight)) return false;
  if (!projectPlane(viewDown)) {
    XrVector3f cross{};
    XrVector3f_Cross(&cross, &normal, &viewRight);
    if (!projectPlane(cross)) return false;
    viewDown = cross;
  } else {
    // Orthonormalize
    const float dot = XrVector3f_Dot(&viewDown, &viewRight);
    viewDown.x -= dot * viewRight.x;
    viewDown.y -= dot * viewRight.y;
    viewDown.z -= dot * viewRight.z;
    if (!projectPlane(viewDown)) {
      XrVector3f cross{};
      XrVector3f_Cross(&cross, &normal, &viewRight);
      if (!projectPlane(cross)) return false;
      viewDown = cross;
    }
  }

  auto quantize = [](float v) -> int {
    if (v > 0.5f) return 1;
    if (v < -0.5f) return -1;
    return 0;
  };
  int srR = quantize(XrVector3f_Dot(&viewRight, &faceRight));
  int srU = quantize(XrVector3f_Dot(&viewRight, &faceUp));
  int sdR = quantize(XrVector3f_Dot(&viewDown, &faceRight));
  int sdU = quantize(XrVector3f_Dot(&viewDown, &faceUp));
  if (srR == 0 && srU == 0) return false;
  if (sdR == 0 && sdU == 0) return false;
  if (srR * sdR + srU * sdU != 0) return false;
  int det = srR * sdU - srU * sdR;
  if (det == 0) return false;

  auto clampCoord = [](int v) -> bool { return v >= -1 && v <= 1; };
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      int relCol = col - 1;
      int relRow = row - 1;
      int x = relCol * srR + relRow * sdR;
      int y = relCol * srU + relRow * sdU;
      if (!clampCoord(x) || !clampCoord(y)) return false;
      int colIdx = x + 1;
      int rowIdx = 1 - y;
      if (colIdx < 0 || colIdx > 2 || rowIdx < 0 || rowIdx > 2) return false;
      outMapping[row * 3 + col] = rowIdx * 3 + colIdx;
    }
  }
  return true;
}

void VirtualCube::SetDefaultOrientation() {
  cubeOrientation_ = {};
  orientationStart_ = {};
  orientationEnd_ = {};
  XrVector3f axis{0.f, 1.f, 0.f};
  XrQuaternionf_CreateFromAxisAngle(&cubeOrientation_, &axis, static_cast<float>(M_PI));
  orientationStart_ = cubeOrientation_;
  orientationEnd_ = cubeOrientation_;
}

}  // namespace SecureMR
