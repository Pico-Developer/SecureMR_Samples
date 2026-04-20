// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace SecureMR {

// Face order follows design: U, R, F, D, L, B
enum class Face { U = 0, R = 1, F = 2, D = 3, L = 4, B = 5 };

// Discrete cube colors
enum class Color { W = 0, Y, R, O, G, B, Unknown };

// Basic cube move types
enum class MoveType { U, R, F, D, L, B };

// Quarter-turn metric: clockwise, counter-clockwise, double
enum class Turn { CW = 1, CCW = -1, Double = 2 };

struct Move {
  MoveType type{MoveType::U};
  Turn turn{Turn::CW};
};

// Frame captured from camera tensor
struct Frame {
  int width{0};
  int height{0};
  int channels{0};
  // BGR image buffer
  cv::Mat mat;  // CV_8UC3 (BGR)
  std::chrono::steady_clock::time_point timestamp{};
};

// Average color of a grid patch
struct PatchColor {
  float r{0.0f}, g{0.0f}, b{0.0f};
};

struct ScannedFace {
  Face face{Face::U};
  std::array<PatchColor, 9> patches{};  // row-major 3x3
};

inline std::string ColorToChar(Color c) {
  switch (c) {
    case Color::W: return "W";
    case Color::Y: return "Y";
    case Color::R: return "R";
    case Color::O: return "O";
    case Color::G: return "G";
    case Color::B: return "B";
    default: return "?";
  }
}

}  // namespace SecureMR
