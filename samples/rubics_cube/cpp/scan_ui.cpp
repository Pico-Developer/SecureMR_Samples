// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "scan_ui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "logger.h"
#include "scan_manager.h"

#include <filesystem>
#include <chrono>

#if defined(XR_USE_PLATFORM_ANDROID)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
// Provided by base/main.cpp on Android
extern std::string g_internalDataPath;
#endif

namespace SecureMR {

namespace {
inline uint8_t clampU8(int v) { return static_cast<uint8_t>(std::min(std::max(v, 0), 255)); }

static void setPixel(std::vector<uint8_t>& img, int W, int H, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
  img[idx + 0] = r;
  img[idx + 1] = g;
  img[idx + 2] = b;
  img[idx + 3] = a;
}

static void drawDashedHLine(std::vector<uint8_t>& img, int W, int H, int x0, int x1, int y, int thickness,
                            int dash, int gap, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (x0 > x1) std::swap(x0, x1);
  int period = std::max(1, dash + gap);
  for (int x = x0; x <= x1; ++x) {
    int seg = (x - x0) % period;
    if (seg < dash) {
      for (int t = 0; t < thickness; ++t) setPixel(img, W, H, x, y + t, r, g, b, a);
    }
  }
}

static void drawDashedVLine(std::vector<uint8_t>& img, int W, int H, int x, int y0, int y1, int thickness, int dash, int gap,
                            uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (y0 > y1) std::swap(y0, y1);
  int period = std::max(1, dash + gap);
  for (int y = y0; y <= y1; ++y) {
    int seg = (y - y0) % period;
    if (seg < dash) {
      for (int t = 0; t < thickness; ++t) setPixel(img, W, H, x + t, y, r, g, b, a);
    }
  }
}

static void drawLine(std::vector<uint8_t>& img, int W, int H, int x0, int y0, int x1, int y1, int thickness, uint8_t r,
                     uint8_t g, uint8_t b, uint8_t a) {
  // Bresenham-like simple line with thickness
  const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy, e2; /* error value e_xy */
  while (true) {
    for (int tx = -thickness / 2; tx <= thickness / 2; ++tx) {
      for (int ty = -thickness / 2; ty <= thickness / 2; ++ty) {
        setPixel(img, W, H, x0 + tx, y0 + ty, r, g, b, a);
      }
    }
    if (x0 == x1 && y0 == y1) break;
    e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

static void drawRightArrow(std::vector<uint8_t>& img, int W, int H, int cx, int cy, int size, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a) {
  // Horizontal shaft
  int half = size / 2;
  drawLine(img, W, H, cx - half, cy, cx + half - size / 6, cy, size / 8 + 1, r, g, b, a);
  // Arrow head (two lines)
  drawLine(img, W, H, cx + half - size / 6, cy, cx + half - size / 3, cy - size / 4, size / 8 + 1, r, g, b, a);
  drawLine(img, W, H, cx + half - size / 6, cy, cx + half - size / 3, cy + size / 4, size / 8 + 1, r, g, b, a);
}

static void drawDownArrow(std::vector<uint8_t>& img, int W, int H, int cx, int cy, int size, uint8_t r, uint8_t g, uint8_t b,
                          uint8_t a) {
  int half = size / 2;
  drawLine(img, W, H, cx, cy - half, cx, cy + half - size / 6, size / 8 + 1, r, g, b, a);
  drawLine(img, W, H, cx, cy + half - size / 6, cx - size / 4, cy + half - size / 3, size / 8 + 1, r, g, b, a);
  drawLine(img, W, H, cx, cy + half - size / 6, cx + size / 4, cy + half - size / 3, size / 8 + 1, r, g, b, a);
}

static void fillRect(std::vector<uint8_t>& img, int W, int H, int x0, int y0, int x1, int y1,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (x0 > x1) std::swap(x0, x1);
  if (y0 > y1) std::swap(y0, y1);
  const int xs = std::max(0, x0);
  const int xe = std::min(W - 1, x1);
  const int ys = std::max(0, y0);
  const int ye = std::min(H - 1, y1);
  for (int y = ys; y <= ye; ++y) {
    for (int x = xs; x <= xe; ++x) {
      setPixel(img, W, H, x, y, r, g, b, a);
    }
  }
}

static void drawDigit7(std::vector<uint8_t>& img, int W, int H, int x, int y, int dw, int dh, int t,
                       uint8_t r, uint8_t g, uint8_t b, uint8_t a, int digit) {
  // 7-seg bit order: A,B,C,D,E,F,G -> bits 0..6
  static const int segMap[10] = {
      /*0*/ 0b0111111, /*1*/ 0b0000110, /*2*/ 0b1011011, /*3*/ 0b1001111, /*4*/ 0b1100110,
      /*5*/ 0b1101101, /*6*/ 0b1111101, /*7*/ 0b0000111, /*8*/ 0b1111111, /*9*/ 0b1101111};
  if (digit < 0 || digit > 9) return;
  const int mask = segMap[digit];
  const int mid = y + dh / 2;
  // Horizontal segments: A (top), G (middle), D (bottom)
  if (mask & 0b0000001) fillRect(img, W, H, x + t, y, x + dw - t - 1, y + t - 1, r, g, b, a);       // A
  if (mask & 0b1000000) fillRect(img, W, H, x + t, mid - t / 2, x + dw - t - 1, mid + (t - 1) / 2, r, g, b, a);  // G
  if (mask & 0b0001000) fillRect(img, W, H, x + t, y + dh - t, x + dw - t - 1, y + dh - 1, r, g, b, a);          // D
  // Vertical segments: F (top-left), E (bottom-left), B (top-right), C (bottom-right)
  const int vLenTop = dh / 2 - t;
  const int vLenBot = dh / 2 - t;
  if (mask & 0b0100000) fillRect(img, W, H, x, y + t, x + t - 1, y + t + vLenTop - 1, r, g, b, a);           // F
  if (mask & 0b00010000) fillRect(img, W, H, x, mid + t / 2, x + t - 1, mid + t / 2 + vLenBot - 1, r, g, b, a);  // E
  if (mask & 0b0000010) fillRect(img, W, H, x + dw - t, y + t, x + dw - 1, y + t + vLenTop - 1, r, g, b, a);     // B
  if (mask & 0b0000100) fillRect(img, W, H, x + dw - t, mid + t / 2, x + dw - 1, mid + t / 2 + vLenBot - 1, r, g, b, a);  // C
}

// Simple 3x5 bitmap font for digits and '/'
static const uint8_t kDigit3x5[10][5] = {
    {0b111, 0b101, 0b101, 0b101, 0b111},  // 0
    {0b010, 0b110, 0b010, 0b010, 0b111},  // 1
    {0b111, 0b001, 0b111, 0b100, 0b111},  // 2
    {0b111, 0b001, 0b111, 0b001, 0b111},  // 3
    {0b101, 0b101, 0b111, 0b001, 0b001},  // 4
    {0b111, 0b100, 0b111, 0b001, 0b111},  // 5
    {0b111, 0b100, 0b111, 0b101, 0b111},  // 6
    {0b111, 0b001, 0b010, 0b010, 0b010},  // 7
    {0b111, 0b101, 0b111, 0b101, 0b111},  // 8
    {0b111, 0b101, 0b111, 0b001, 0b111},  // 9
};

static const uint8_t kSlash3x5[5] = {0b001, 0b001, 0b010, 0b010, 0b100};

static void drawGlyph3x5(std::vector<uint8_t>& img, int W, int H, int x, int y, int scale,
                          uint8_t r, uint8_t g, uint8_t b, uint8_t a, const uint8_t rows[5]) {
  for (int ry = 0; ry < 5; ++ry) {
    for (int rx = 0; rx < 3; ++rx) {
      if (rows[ry] & (1u << (2 - rx))) {
        const int gx = x + rx * scale;
        const int gy = y + ry * scale;
        fillRect(img, W, H, gx, gy, gx + scale - 1, gy + scale - 1, r, g, b, a);
      }
    }
  }
}
}  // namespace

void ScanUi::Update(const Frame* latestFrame, const ScanUiState& s, float dt) {
  std::lock_guard<std::mutex> lk(mtx_);
  state_ = s;
  t_ += dt;
  if (manager_) {
    boxColor_ = manager_->GetCenterColorU8();
    patchColors9_ = manager_->GetPatchColors9U8();
  } else {
    boxColor_ = {255, 0, 0};
  }
  hintColors9_ = s.hintColors9;
}

bool ScanUi::ComposeOverlayRgba(int W, int H, std::vector<uint8_t>& outRgba) {
  if (W <= 0 || H <= 0) return false;
  outRgba.assign(static_cast<size_t>(W) * H * 4, 0);

  const auto roi = ScanManager::GetRoiRectPx(W, H);
  const int left = roi.offset.x;
  const int top = roi.offset.y;
  const int right = left + roi.extent.width - 1;
  const int bottom = top + roi.extent.height - 1;
  const int thickness = 6;
  const int dash = 12;
  const int gap = 8;

  const uint8_t r = boxColor_[0], g = boxColor_[1], b = boxColor_[2];

  const bool isScanPhase = !(state_.phase == ScanUiState::Phase::Validate || state_.phase == ScanUiState::Phase::Solve);
  if (isScanPhase) {
    for (int t = 0; t < thickness; ++t) {
      drawDashedHLine(outRgba, W, H, left, right, top + t, thickness, dash, gap, r, g, b, 255);
      drawDashedHLine(outRgba, W, H, left, right, bottom - t, thickness, dash, gap, r, g, b, 255);
      drawDashedVLine(outRgba, W, H, left + t, top, bottom, thickness, dash, gap, r, g, b, 255);
      drawDashedVLine(outRgba, W, H, right - t, top, bottom, thickness, dash, gap, r, g, b, 255);
    }
  }

  // Additional 3x3 grid above ROI: filled opaque cells (only during scanning)
  if (isScanPhase) {
    const int spacing = 20;  // gap between ROI and the top grid
    const int gridW = roi.extent.width;
    const int gridH = roi.extent.height;
    const int cellW = gridW / 3;
    const int cellH = gridH / 3;
    const int topGridBottom = top - spacing - 1;
    const int topGridTop = topGridBottom - gridH + 1;
    const int topGridLeft = left;
    int pidx = 0;
    for (int gy = 0; gy < 3; ++gy) {
      for (int gx = 0; gx < 3; ++gx) {
        const int x0 = topGridLeft + gx * cellW;
        const int y0 = topGridTop + gy * cellH;
        const int x1 = (gx == 2) ? (topGridLeft + gridW - 1) : (x0 + cellW - 1);
        const int y1 = (gy == 2) ? (topGridTop + gridH - 1) : (y0 + cellH - 1);
        const auto& cols = patchColors9_;
        const uint8_t pr = cols[pidx][0];
        const uint8_t pg = cols[pidx][1];
        const uint8_t pb = cols[pidx][2];
        fillRect(outRgba, W, H, x0, y0, x1, y1, pr, pg, pb, 255);
        ++pidx;
      }
    }
  }

  if (isScanPhase) {
    // Arrow
    if (state_.nextArrow == ScanUiState::Arrow::Right) {
      const int acx = right + 40;            // right side of the box
      const int acy = (top + bottom) / 2;    // vertical middle
      const int asz = std::min(120, std::min(W, H) / 6);
      drawRightArrow(outRgba, W, H, acx, acy, asz, r, g, b, 255);
    } else if (state_.nextArrow == ScanUiState::Arrow::Down) {
      const int acx = (left + right) / 2;    // horizontal middle
      const int acy = bottom + 40;           // below the box
      const int asz = std::min(120, std::min(W, H) / 6);
      drawDownArrow(outRgba, W, H, acx, acy, asz, r, g, b, 255);
    }
  }

  // Stabilization progress bar (1s) on left side, vertical bottom->top
  if (isScanPhase) {
    const int spacing = 8;
    const int barW = 12;
    const int barH = roi.extent.height;
    const int barLeft = left - spacing - barW;
    const int barRight = barLeft + barW - 1;
    const int barTop = top;
    const int barBottom = bottom;
    // Background (semi-transparent)
    fillRect(outRgba, W, H, barLeft, barTop, barRight, barBottom, 30, 30, 30, 180);
    // Progress fill (bottom-up)
    float progress = 0.f;
    if (manager_) progress = manager_->GetStabilizeProgress01();
    progress = std::max(0.f, std::min(1.f, progress));
    const int fillHeight = static_cast<int>(std::round(progress * barH));
    if (fillHeight > 0) {
      const int fillTop = barBottom - fillHeight + 1;
      fillRect(outRgba, W, H, barLeft, fillTop, barRight, barBottom, r, g, b, 255);
    }
  }

  // Bottom text box
  if (!state_.bottomText.empty() || !state_.solveSteps.empty()) {
#if defined(XR_USE_PLATFORM_ANDROID)
    // Draw semi-transparent plate and text using OpenCV
    const int basePlateH = 64;
    const int margin = 20;
    const int innerPad = 24;
    const double fontScale = 1.2;
    const int thickness = 2;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    const int plateW = W - 2 * margin;
    const int contentW = plateW - 2 * innerPad;
    struct StyledToken {
      std::string text;
      cv::Scalar color;
      cv::Size size;
    };
    const cv::Scalar white(255, 255, 255, 255);
    const cv::Scalar red(255, 0, 0, 255);
    int baseLineSpace = 0;
    const cv::Size spaceSize = cv::getTextSize(" ", fontFace, fontScale, thickness, &baseLineSpace);
    (void)baseLineSpace;
    auto makeToken = [&](const std::string& t, const cv::Scalar& col) -> StyledToken {
      int bl = 0;
      cv::Size sz = cv::getTextSize(t, fontFace, fontScale, thickness, &bl);
      (void)bl;
      return StyledToken{t, col, sz};
    };

    std::vector<StyledToken> tokens;
    if (!state_.solveSteps.empty()) {
      tokens.reserve(state_.solveSteps.size());
      for (size_t i = 0; i < state_.solveSteps.size(); ++i) {
        tokens.push_back(makeToken(state_.solveSteps[i], (static_cast<int>(i) == state_.activeSolveIndex) ? red : white));
      }
    } else {
      std::istringstream iss(state_.bottomText);
      std::string word;
      while (iss >> word) {
        tokens.push_back(makeToken(word, white));
      }
      if (tokens.empty()) {
        tokens.push_back(makeToken(state_.bottomText, white));
      }
    }

    struct Line {
      std::vector<StyledToken> toks;
      int width{0};
      int height{0};
    };
    std::vector<Line> lines;
    Line current;
    auto finalizeLine = [&]() {
      if (!current.toks.empty()) {
        lines.push_back(current);
      }
      current = Line{};
    };
    for (const auto& tok : tokens) {
      const int addWidth = tok.size.width + (current.toks.empty() ? 0 : spaceSize.width);
      if (!current.toks.empty() && current.width + addWidth > contentW && lines.size() < 1) {
        finalizeLine();
      }
      if (!current.toks.empty()) current.width += spaceSize.width;
      current.width += tok.size.width;
      current.height = std::max(current.height, tok.size.height);
      current.toks.push_back(tok);
    }
    finalizeLine();
    if (lines.empty()) {
      Line l{};
      l.toks = tokens;
      for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) l.width += spaceSize.width;
        l.width += tokens[i].size.width;
        l.height = std::max(l.height, tokens[i].size.height);
      }
      lines.push_back(l);
    }
    if (lines.size() > 2) {
      // Merge extra lines into second line
      Line merged = lines[1];
      for (size_t li = 2; li < lines.size(); ++li) {
        if (!merged.toks.empty()) merged.width += spaceSize.width;
        merged.toks.insert(merged.toks.end(), lines[li].toks.begin(), lines[li].toks.end());
      }
      // Recompute width/height
      merged.width = 0;
      merged.height = 0;
      for (size_t i = 0; i < merged.toks.size(); ++i) {
        if (i > 0) merged.width += spaceSize.width;
        merged.width += merged.toks[i].size.width;
        merged.height = std::max(merged.height, merged.toks[i].size.height);
      }
      lines.resize(2);
      lines[1] = merged;
    }

    const int plateH = basePlateH * static_cast<int>(lines.size());
    const int px0 = margin;
    const int py0 = H - plateH - margin;
    const int px1 = W - margin;
    const int py1 = H - margin;
    fillRect(outRgba, W, H, px0, py0, px1, py1, 0, 0, 0, 120);
    cv::Mat rgba(H, W, CV_8UC4, outRgba.data());
    const int lineGap = 6;
    int totalH = 0;
    for (const auto& l : lines) {
      totalH += l.height;
    }
    totalH += lineGap * static_cast<int>(lines.size() - 1);

    int ty = py0 + (plateH - totalH) / 2 + lines[0].height;
    for (size_t i = 0; i < lines.size(); ++i) {
      int tx = px0 + (plateW - lines[i].width) / 2;
      bool firstTok = true;
      for (const auto& tok : lines[i].toks) {
        if (!firstTok) tx += spaceSize.width;
        cv::putText(rgba, tok.text, cv::Point(tx, ty), fontFace, fontScale, tok.color, thickness);
        tx += tok.size.width;
        firstTok = false;
      }
      ty += lines[i].height + lineGap;
    }
#else
    // Non-Android: skip text to avoid extra deps
#endif
  }

  const bool changed = (last_.size() != outRgba.size()) || (std::memcmp(last_.data(), outRgba.data(), outRgba.size()) != 0);
  if (changed) {
    last_ = outRgba;
    Log::Write(Log::Level::Debug, "ScanUi: overlay RGBA composed (static dashed box + arrow)");
  }
  return changed;
}

void ScanUi::StartThread() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) return;
  worker_ = std::thread([this]() {
    std::vector<uint8_t> buf;
    while (running_) {
      int W, H;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        W = overlayW_;
        H = overlayH_;
      }
      bool changed = false;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        changed = ComposeOverlayRgba(W, H, buf);
      }
      if (changed) {
        std::lock_guard<std::mutex> lk(mtx_);
        composed_ = buf;
        dirty_ = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
  });
}

void ScanUi::StopThread() {
  if (running_) {
    running_ = false;
    if (worker_.joinable()) worker_.join();
  }
}

}  // namespace SecureMR
