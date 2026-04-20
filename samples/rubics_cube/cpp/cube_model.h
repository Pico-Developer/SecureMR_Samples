// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <array>
#include <optional>

#include "color_classifier.h"
#include "cube_types.h"

namespace SecureMR {

class CubeModel {
 public:
  CubeModel();

  // Set colors for a face directly
  void SetFaceColors(Face face, const std::array<Color, 9>& colors);
  // Convenience: classify from patches via classifier
  void SetFaceFromPatches(const ScannedFace& face, const ColorClassifier& classifier);
  void SetFaceRawColors(Face face, const std::array<PatchColor, 9>& patches);

  bool Validate() const;  // basic count check
  std::string ToKociembaCode() const;
  // Produce facelet string in URFDLB letters for kociemba
  std::string ToKociembaFacelets() const;

  // Reset all faces to Unknown and clear set flags
  void Clear();
  // Count how many faces have been set
  int GetScannedCount() const;
  // Get face colors if set
  std::optional<std::array<Color, 9>> GetFaceColors(Face face) const;
  std::optional<std::array<PatchColor, 9>> GetFaceRawColors(Face face) const;

  void ApplyMove(const Move& /*move*/) {};  // stub for now

 private:
  // Stored as 6 faces x 9 stickers, face index matches Face enum order
  std::array<std::array<Color, 9>, 6> stickers_;
  std::array<bool, 6> faceSet_{};
  std::array<std::array<PatchColor, 9>, 6> faceRawColors_{};
  std::array<bool, 6> faceRawSet_{};
};

}  // namespace SecureMR
