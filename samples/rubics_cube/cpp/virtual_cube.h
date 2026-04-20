// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <array>
#include <deque>
#include <optional>
#include <vector>

#include "cube_types.h"
#include "geometry.h"

namespace SecureMR {

class CubeModel;

class VirtualCube {
 public:
  enum class CubeTurnOp { Down, Right, Left, Up };

  VirtualCube();

  void Update(float dt);
  void SetCubeModel(const CubeModel* model);
  void PreviewFace(Face face, const std::array<Color, 9>& colors,
                   const std::array<XrVector3f, 9>& displayColors);
  void ClearPreviewFace(Face face);
  void ClearPreview();
  void QueueFaceMoves(const std::vector<Move>& moves);
  void ClearFaceMoves();
  void QueueOrientationOp(CubeTurnOp op);
  void QueueOrientationOpRelative(CubeTurnOp op, const XrQuaternionf& viewerOrientation);
  void ResetOrientation();
  void QueueWarmupAnimation();
  void SetAnchorPosition(const XrVector3f& position);
  [[nodiscard]] const XrVector3f& AnchorPosition() const { return anchorPosition_; }
  [[nodiscard]] int TotalMoveCount() const { return totalMoves_; }
  [[nodiscard]] std::optional<int> ActiveMoveIndex() const;
  [[nodiscard]] std::optional<Face> GetFacingFace(const XrQuaternionf& viewerOrientation) const;
  [[nodiscard]] bool MapScreenGridToFaceIndices(Face face, const XrQuaternionf& viewerOrientation,
                                                std::array<int, 9>& outMapping) const;
  [[nodiscard]] std::optional<std::array<XrVector3f, 9>> GetFaceDisplayColors(Face face) const;

  bool BuildMesh(std::vector<Geometry::Vertex>& outVerts, std::vector<uint16_t>& outIdx, XrPosef& outPose);
  void Solving();

 private:
  struct CubieState {
    int x{0}, y{0}, z{0};
    Color faces[6] = {Color::Unknown, Color::Unknown, Color::Unknown,
                      Color::Unknown, Color::Unknown, Color::Unknown};
    XrVector3f displayColors[6]{};
  };

  // Cubie/color helpers
  void EnsureInitialized();
  void InitializeFromCubeModel();
  void InitializeDefaultColors();
  void AssignFaceColors(Face face, const std::array<Color, 9>& colors,
                        const std::array<XrVector3f, 9>* displayColors);
  void ApplyPreviewOverride();

  // Face move animation helpers
  void BeginNextFaceMove();
  bool IsCubieInActiveLayer(const CubieState& c) const;
  float CurrentFaceAngleRad() const;
  void WarnIfMissingStickerColors(const Move& upcomingMove) const;
  static int SignForCW(MoveType type);
  void ApplyFaceMoveToCubies(const Move& move);
  void RotateCubie(CubieState& c, char axis, int signedQuarterTurns) const;

  void PushTri(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
               const XrVector3f& a, const XrVector3f& b, const XrVector3f& c,
               const XrVector3f& color);
  void PushBox(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
               const XrVector3f& center, float half, bool rotate, char axis, float angleRad);
  void PushSticker(std::vector<Geometry::Vertex>& v, std::vector<uint16_t>& i,
                   const XrVector3f& center, float half, char face, const XrVector3f& color,
                   bool rotate, char axis, float angleRad);
  static XrVector3f ToRgb(Color c);
  XrVector3f ToDisplayColor(Color c) const;

  // Orientation helpers
  void UpdateOrientation(float dt);
  void StartNextOrientationOp();
  XrQuaternionf DeltaForOp(CubeTurnOp op);
  struct ViewerFacingInfo {
    Face front{Face::F};
    Face back{Face::B};
    Face left{Face::L};
    Face right{Face::R};
    Face up{Face::U};
    Face down{Face::D};
  };
  std::array<XrVector3f, 6> ComputeFaceWorldNormals() const;
  std::optional<ViewerFacingInfo> ComputeViewerFacingInfo(const XrQuaternionf& viewerOrientation,
                                                          const std::array<XrVector3f, 6>& normals) const;
  bool BuildDeltaBetweenFaces(Face from, Face to, const std::array<XrVector3f, 6>& normals, XrQuaternionf& outDelta) const;
  void EnsureTurnQuats();
  void SetDefaultOrientation();

  // Face animation state
  std::deque<Move> faceQueue_;
  bool faceMoveActive_{false};
  Move currentFaceMove_{};
  float faceMoveTimer_{0.f};
  float faceMoveDuration_{0.6f};
  float pauseAfterMove_{1.0f};
  float pauseTimer_{0.f};
  bool pauseActive_{false};

  std::vector<CubieState> cubies_;
  bool initialized_{false};
  const CubeModel* cubeModel_{nullptr};
  float cubeSize_{0.3f};
  float gap_{0.003f};
  float stickerEps_{0.0008f};
  bool swapRB_{true};
  std::optional<Face> previewFace_{};
  std::array<Color, 9> previewColors_{};
  std::array<XrVector3f, 9> previewDisplayColors_{};

  // Orientation state
  struct OrientationCommand {
    CubeTurnOp op;
    XrQuaternionf delta{};
  };
  bool didWarmup_{false};
  std::deque<OrientationCommand> orientationOps_;
  XrQuaternionf cubeOrientation_{};
  XrQuaternionf orientationStart_{};
  XrQuaternionf orientationEnd_{};
  float orientationTimer_{0.f};
  float orientationDuration_{0.6f};
  bool orientationAnimating_{false};
  XrQuaternionf downQuat_{};
  XrQuaternionf rightQuat_{};
  XrQuaternionf leftQuat_{};
  XrQuaternionf upQuat_{};
  bool turnQuatsInit_{false};
  XrVector3f anchorPosition_{};
  int totalMoves_{0};
  int movesStarted_{0};
  std::optional<int> currentMoveIdx_{};
};

}  // namespace SecureMR
