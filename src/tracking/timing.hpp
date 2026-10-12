#pragma once
#include <cmath>
#include <stdexcept>

namespace tracking {
inline constexpr int kPhysicsHz = 2400;
inline int capture_stride(double hz) {
  if (!std::isfinite(hz) || hz <= 0 || hz > kPhysicsHz)
    throw std::runtime_error(
        "Capture rate must be positive and divide 2400 Hz into integer steps");
  const double stride = kPhysicsHz / hz;
  const double rounded = std::round(stride);
  if (rounded > 1000000 || std::abs(stride - rounded) > 1e-9)
    throw std::runtime_error(
        "Capture rate must divide 2400 Hz into integer steps");
  return static_cast<int>(rounded);
}
} // namespace tracking
