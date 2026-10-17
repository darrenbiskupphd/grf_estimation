#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tracking {
inline constexpr std::array<const char *, 16> kTargets = {
    "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip"};
using ReferencePoints = std::array<double, 3 * kTargets.size()>;

// Schema v1 is a JSON object with exactly these fields:
// schema_version: 1, time_s: [T], positions_m: [T][16][3],
// target_names: kTargets (in order), metadata: object. T must be at least two.
// Input times must be finite, nonnegative and strictly increasing. Loading
// subtracts only time_s[0]; positions are never aligned, scaled or transformed.
struct ReferenceData {
  std::vector<double> time_s;
  std::vector<ReferencePoints> positions_m;
  nlohmann::json metadata = nlohmann::json::object();
  double time_origin_s = 0; // Input time_s[0], before normalization.
  std::filesystem::path source_path;
  // metadata.source_id, then metadata.source.source_id (importer provenance),
  // then the source filename stem. Metadata itself is preserved verbatim.
  std::string source_id;
  double duration_s() const { return time_s.empty() ? 0 : time_s.back(); }
};

ReferenceData parse_reference(const nlohmann::json &document);
ReferenceData load_reference(const std::filesystem::path &path);

// Z-only normalization using the pooled median sum of 3D distances along
// heel-knee-hip-shoulder-head on both sides/all frames, relative to the mean
// neutral model chain length. XY/timing/provenance unchanged; scale about Z=0.
// Returns transform provenance; rejects degenerate/nonfinite chain segments.
nlohmann::json scale_reference_to_model(
    ReferenceData &reference, const ReferencePoints &model_neutral_points);

// Check once when loading/resetting, not on every planner residual evaluation.
// The runtime uses the compiled model's key_time array as the time authority.
void validate_reference_times(const double *time_s, std::size_t count);

struct ReferenceInterpolation {
  std::size_t first, next;
  double fraction;
  double interval_s; // Zero at and after the final endpoint.

  double position(double p0, double p1) const {
    return (1 - fraction) * p0 + fraction * p1;
  }
  double velocity(double p0, double p1) const {
    return first == next ? 0 : (p1 - p0) / interval_s;
  }
};

// Requires validated times. Positions clamp at both ends. Velocities use the
// right-hand segment, including before/at time zero, and are zero at/after the
// final endpoint. This preserves the original 30 Hz endpoint convention.
ReferenceInterpolation reference_interpolation(const double *time_s,
                                                std::size_t count,
                                                double time);
struct ReferenceSample {
  ReferencePoints positions_m;
  ReferencePoints velocities_m_per_s;
};
ReferenceSample sample_reference(const ReferenceData &reference, double time);

// Conservative start of a sustained-stance history window ending at key.
// Includes the key at/before the window's left edge; nullopt means insufficient
// history. At uniform 30 Hz this is key - ceil(seconds * 30), up to roundoff.
std::optional<std::size_t> reference_history_start(const double *time_s,
                                                  std::size_t count,
                                                  std::size_t key,
                                                  double seconds);
} // namespace tracking
