// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <optional>
#include <unordered_map>

#include "cube_types.h"

namespace SecureMR {

class ColorClassifier {
 public:
  ColorClassifier() = default;

  void CalibrateCenter(Face face, const PatchColor& c);
  Color Classify(const PatchColor& c) const;

 private:
  std::unordered_map<Face, PatchColor> centers_;
};

}  // namespace SecureMR

