// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "color_classifier.h"
#include "logger.h"
#include "common.h"

#include <cmath>

namespace SecureMR {

static float Dist2(const PatchColor& a, const PatchColor& b) {
  const float dr = a.r - b.r;
  const float dg = a.g - b.g;
  const float db = a.b - b.b;
  return dr * dr + dg * dg + db * db;
}

void ColorClassifier::CalibrateCenter(Face face, const PatchColor& c) {
  centers_[face] = c;
  Log::Write(Log::Level::Debug,
             Fmt("RubicsCube|ColorClassifier: calibrate face %d center rgb=(%.2f,%.2f,%.2f)", static_cast<int>(face), c.r, c.g,
                 c.b));
}

Color ColorClassifier::Classify(const PatchColor& c) const {
  if (!centers_.empty()) {
    float best = std::numeric_limits<float>::max();
    Face bestFace = Face::U;
    for (const auto& kv : centers_) {
      float d = Dist2(c, kv.second);
      if (d < best) {
        best = d;
        bestFace = kv.first;
      }
    }
    // Map face centers to fixed colors based on standard scheme
    Color out;
    switch (bestFace) {
      case Face::U: out = Color::W; break;
      case Face::D: out = Color::Y; break;
      case Face::F: out = Color::G; break;
      case Face::B: out = Color::B; break;
      case Face::R: out = Color::R; break;
      case Face::L: out = Color::O; break;
    }
    Log::Write(Log::Level::Debug,
               Fmt("RubicsCube|ColorClassifier: classify rgb=(%.2f,%.2f,%.2f) -> %s by nearest face %d", c.r, c.g, c.b,
                   ColorToChar(out).c_str(), static_cast<int>(bestFace)));
    return out;
  }

  // Fallback heuristic: dominant channel
  Color out = Color::Unknown;
  if (c.r >= c.g && c.r >= c.b) out = Color::R;
  else if (c.g >= c.r && c.g >= c.b) out = Color::G;
  else if (c.b >= c.r && c.b >= c.g) out = Color::B;
  Log::Write(Log::Level::Debug,
             Fmt("RubicsCube|ColorClassifier: fallback classify rgb=(%.2f,%.2f,%.2f) -> %s", c.r, c.g, c.b,
                 ColorToChar(out).c_str()));
  return out;
}

}  // namespace SecureMR
