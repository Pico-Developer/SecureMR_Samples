// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "cube_model.h"
#include "logger.h"
#include "common.h"

namespace SecureMR {

CubeModel::CubeModel() {
  for (auto& face : stickers_) {
    face.fill(Color::Unknown);
  }
  faceSet_.fill(false);
  for (auto& rawFace : faceRawColors_) {
    for (auto& patch : rawFace) {
      patch = {};
    }
  }
  faceRawSet_.fill(false);
}

void CubeModel::SetFaceColors(Face face, const std::array<Color, 9>& colors) {
  const int idx = static_cast<int>(face);
  stickers_[idx] = colors;
  faceSet_[idx] = true;
  std::string seq;
  seq.reserve(9);
  for (Color c : colors) seq += ColorToChar(c);
  Log::Write(Log::Level::Info, Fmt("RubicsCube|CubeModel: set face %d colors=%s", idx, seq.c_str()));
}

void CubeModel::SetFaceFromPatches(const ScannedFace& face, const ColorClassifier& classifier) {
  std::array<Color, 9> c{};
  for (size_t i = 0; i < 9; ++i) {
    c[i] = classifier.Classify(face.patches[i]);
  }
  SetFaceColors(face.face, c);
  const int idx = static_cast<int>(face.face);
  faceRawColors_[idx] = face.patches;
  faceRawSet_[idx] = true;
}

void CubeModel::SetFaceRawColors(Face face, const std::array<PatchColor, 9>& patches) {
  const int idx = static_cast<int>(face);
  if (idx < 0 || idx >= 6) return;
  faceRawColors_[idx] = patches;
  faceRawSet_[idx] = true;
}

bool CubeModel::Validate() const {
  // Basic check: every face set, and each color count equals 9
  for (bool s : faceSet_) {
    if (!s) return false;
  }
  std::array<int, 6> cnt{};
  for (const auto& face : stickers_) {
    for (Color c : face) {
      switch (c) {
        case Color::W: cnt[0]++; break;
        case Color::Y: cnt[1]++; break;
        case Color::R: cnt[2]++; break;
        case Color::O: cnt[3]++; break;
        case Color::G: cnt[4]++; break;
        case Color::B: cnt[5]++; break;
        default: return false;
      }
    }
  }
  Log::Write(Log::Level::Info,
             Fmt("RubicsCube|CubeModel: counts W=%d Y=%d R=%d O=%d G=%d B=%d", cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]));
  for (int v : cnt) {
    if (v != 9) return false;
  }
  return true;
}

std::string CubeModel::ToKociembaCode() const {
  // Order: U, R, F, D, L, B; each face left-to-right top-to-bottom
  std::string s;
  s.reserve(54);
  auto appendFace = [&s](const std::array<Color, 9>& f) {
    for (Color c : f) s += ColorToChar(c);
  };
  appendFace(stickers_[static_cast<int>(Face::U)]);
  appendFace(stickers_[static_cast<int>(Face::R)]);
  appendFace(stickers_[static_cast<int>(Face::F)]);
  appendFace(stickers_[static_cast<int>(Face::D)]);
  appendFace(stickers_[static_cast<int>(Face::L)]);
  appendFace(stickers_[static_cast<int>(Face::B)]);
  Log::Write(Log::Level::Info, Fmt("RubicsCube|CubeModel: Kociemba code=%s", s.c_str()));
  return s;
}

std::string CubeModel::ToKociembaFacelets() const {
  // Map each Color to one of URFDLB letters based on center stickers of faces
  // Center index = 4 in each 3x3 face array
  char colorToFace[6];
  for (int i = 0; i < 6; ++i) colorToFace[i] = '?';
  auto colorIndex = [](Color c) -> int {
    switch (c) {
      case Color::W: return 0;
      case Color::Y: return 1;
      case Color::R: return 2;
      case Color::O: return 3;
      case Color::G: return 4;
      case Color::B: return 5;
      default: return -1;
    }
  };
  const char faceLetters[6] = {'U','R','F','D','L','B'};
  for (int f = 0; f < 6; ++f) {
    Color center = stickers_[f][4];
    int idx = colorIndex(center);
    if (idx >= 0) colorToFace[idx] = faceLetters[f];
  }
  // Build facelets string in order U,R,F,D,L,B each 9 stickers
  std::string s;
  s.reserve(54);
  auto appendFace = [&](int fIndex) {
    for (int i = 0; i < 9; ++i) {
      Color c = stickers_[fIndex][i];
      int ci = colorIndex(c);
      char ch = (ci >= 0) ? colorToFace[ci] : '?';
      s += ch;
    }
  };
  appendFace(static_cast<int>(Face::U));
  appendFace(static_cast<int>(Face::R));
  appendFace(static_cast<int>(Face::F));
  appendFace(static_cast<int>(Face::D));
  appendFace(static_cast<int>(Face::L));
  appendFace(static_cast<int>(Face::B));
  Log::Write(Log::Level::Info, Fmt("RubicsCube|CubeModel: facelets=%s", s.c_str()));
  return s;
}

void CubeModel::Clear() {
  for (auto& face : stickers_) {
    face.fill(Color::Unknown);
  }
  faceSet_.fill(false);
  for (auto& rawFace : faceRawColors_) {
    for (auto& patch : rawFace) {
      patch = {};
    }
  }
  faceRawSet_.fill(false);
  Log::Write(Log::Level::Info, "RubicsCube|CubeModel: cleared all faces");
}

int CubeModel::GetScannedCount() const {
  int cnt = 0;
  for (bool s : faceSet_) cnt += s ? 1 : 0;
  return cnt;
}

std::optional<std::array<Color, 9>> CubeModel::GetFaceColors(Face face) const {
  const int idx = static_cast<int>(face);
  if (idx < 0 || idx >= 6) return std::nullopt;
  if (!faceSet_[idx]) return std::nullopt;
  return stickers_[idx];
}

std::optional<std::array<PatchColor, 9>> CubeModel::GetFaceRawColors(Face face) const {
  const int idx = static_cast<int>(face);
  if (idx < 0 || idx >= 6) return std::nullopt;
  if (!faceRawSet_[idx]) return std::nullopt;
  return faceRawColors_[idx];
}

}  // namespace SecureMR
