// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#pragma once

#include "pch.h"

#include <string>

namespace SecureMR {

class Solver {
 public:
  Solver() = default;
  // Returns Kociemba solution string, e.g., "R U R' U'"
  std::string Solve(const std::string& code);
};

}  // namespace SecureMR

