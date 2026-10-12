#pragma once

#include "tracking/contact_wrench.hpp"
#include "tracking/frame.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace export_test {
inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, const std::string &fragment) {
  try {
    function();
  } catch (const std::exception &error) {
    check(std::string(error.what()).find(fragment) != std::string::npos,
          "Wrong rejection: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("Expected rejection: " + fragment);
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("grf-npz-test-" + std::to_string(tick));
    check(std::filesystem::create_directory(path),
          "Create temporary directory");
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

inline std::vector<char> bytes(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  check(bool(stream), "Read fixture bytes");
  return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
}

inline void set_wrench(tracking::ReplayFrame &frame, int side,
                       const tracking::ContactWrench &wrench) {
  const auto plane = tracking::plane_contact(wrench);
  (side ? frame.grf_right : frame.grf_left) = wrench.force;
  (side ? frame.moment_origin_right : frame.moment_origin_left) = wrench.moment;
  (side ? frame.cop_right : frame.cop_left) = plane.cop;
  (side ? frame.torque_z_right : frame.torque_z_left) = plane.free_moment_z;
  (side ? frame.cop_defined_right : frame.cop_defined_left) = plane.defined;
}

// Test expectations only: round-trip decimal doubles from synthetic saved
// frames. Not an exporter/compactor or an intermediate training representation.
inline void write_markers(const std::filesystem::path &path,
                          const std::vector<tracking::ReplayFrame> &frames) {
  std::ofstream stream(path);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream << std::setprecision(17);
  for (const auto &frame : frames)
    for (size_t marker = 0; marker < frame.markers_m.size(); marker += 3)
      stream << frame.markers_m[marker] << ' ' << frame.markers_m[marker + 1]
             << ' ' << frame.markers_m[marker + 2] << '\n';
}

inline void write_wrench(const std::filesystem::path &path,
                         const std::vector<tracking::ReplayFrame> &frames) {
  std::ofstream stream(path);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream << std::setprecision(17);
  for (const auto &frame : frames)
    for (int side = 0; side < 2; ++side) {
      const auto &force = side ? frame.grf_right : frame.grf_left;
      const auto &cop = side ? frame.cop_right : frame.cop_left;
      stream << force[0] << ' ' << force[1] << ' ' << force[2] << ' ' << cop[0]
             << ' ' << cop[1] << ' '
             << (side ? frame.torque_z_right : frame.torque_z_left) << '\n';
    }
}
} // namespace export_test
