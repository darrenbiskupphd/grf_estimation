#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace tracking {
namespace {
using Json = nlohmann::json;

[[noreturn]] void invalid(const std::string &message) {
  throw std::runtime_error("Invalid reference: " + message);
}

double finite_number(const Json &value, const std::string &where) {
  if (!value.is_number())
    invalid(where + " must be a finite number");
  const double number = value.get<double>();
  if (!std::isfinite(number))
    invalid(where + " must be a finite number");
  return number;
}
} // namespace

void validate_reference_times(const double *time_s, std::size_t count) {
  if (!time_s || count < 2)
    invalid("time_s must contain at least two samples");
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(time_s[i]) || time_s[i] < 0)
      invalid("time_s must be finite and nonnegative");
    if (i > 0 && !(time_s[i] > time_s[i - 1]))
      invalid("time_s must be strictly increasing");
  }
}

ReferenceData parse_reference(const Json &document) {
  if (!document.is_object())
    invalid("root must be an object");
  constexpr std::array<const char *, 5> fields = {
      "schema_version", "time_s", "positions_m", "target_names", "metadata"};
  for (const char *field : fields)
    if (!document.contains(field))
      invalid(std::string("missing ") + field);
  if (document.size() != fields.size())
    invalid("unknown top-level field (put provenance in metadata)");
  if (!document["schema_version"].is_number_integer() ||
      document["schema_version"] != 1)
    invalid("schema_version must be integer 1");
  const auto &names = document["target_names"];
  if (!names.is_array() || names.size() != kTargets.size())
    invalid("target_names must contain the 16 canonical targets in order");
  for (std::size_t i = 0; i < kTargets.size(); ++i)
    if (!names[i].is_string() || names[i] != kTargets[i])
      invalid("target_names[" + std::to_string(i) + "] must be " + kTargets[i]);
  if (!document["metadata"].is_object())
    invalid("metadata must be an object");
  const auto &times = document["time_s"];
  const auto &positions = document["positions_m"];
  if (!times.is_array() || times.size() < 2 ||
      times.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    invalid("time_s must be an array of 2..INT_MAX timestamps");
  if (!positions.is_array() || positions.size() != times.size())
    invalid("positions_m must have the same sample count as time_s");

  ReferenceData reference;
  reference.metadata = document["metadata"];
  const Json *identity = nullptr;
  if (reference.metadata.contains("source_id"))
    identity = &reference.metadata["source_id"];
  else if (reference.metadata.contains("source") &&
           reference.metadata["source"].is_object() &&
           reference.metadata["source"].contains("source_id"))
    identity = &reference.metadata["source"]["source_id"];
  if (identity) {
    if (!identity->is_string() || identity->get_ref<const std::string &>().empty())
      invalid("metadata source_id must be a nonempty string when present");
    reference.source_id = identity->get<std::string>();
    const Json::json_pointer nested("/source/source_id");
    if (reference.metadata.contains("source_id") && reference.metadata.contains(nested) &&
        reference.metadata.at(nested) != reference.metadata["source_id"])
      invalid("conflicting source recording identities in metadata");
  }
  reference.time_s.reserve(times.size());
  for (std::size_t i = 0; i < times.size(); ++i)
    reference.time_s.push_back(
        finite_number(times[i], "time_s[" + std::to_string(i) + "]"));
  validate_reference_times(reference.time_s.data(), reference.time_s.size());
  reference.time_origin_s = reference.time_s.front();
  for (double &time : reference.time_s)
    time -= reference.time_origin_s;
  validate_reference_times(reference.time_s.data(), reference.time_s.size());

  reference.positions_m.resize(times.size());
  for (std::size_t frame = 0; frame < times.size(); ++frame) {
    const auto &points = positions[frame];
    const std::string where = "positions_m[" + std::to_string(frame) + "]";
    if (!points.is_array() || points.size() != kTargets.size())
      invalid(where + " must contain exactly 16 targets");
    for (std::size_t target = 0; target < kTargets.size(); ++target) {
      if (!points[target].is_array() || points[target].size() != 3)
        invalid(where + "[" + std::to_string(target) + "] must be XYZ");
      for (std::size_t axis = 0; axis < 3; ++axis)
        reference.positions_m[frame][3 * target + axis] = finite_number(
            points[target][axis], where + "[" + std::to_string(target) +
                                      "][" + std::to_string(axis) + "]");
    }
  }
  return reference;
}

ReferenceData load_reference(const std::filesystem::path &path) {
  if (path.empty())
    throw std::runtime_error("An external reference JSON path is required");
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Could not open reference JSON: " + path.string());
  try {
    // Reject duplicate keys instead of silently accepting the last value.
    std::vector<std::set<std::string>> object_keys;
    auto callback = [&](int, Json::parse_event_t event, Json &value) {
      if (event == Json::parse_event_t::object_start)
        object_keys.emplace_back();
      else if (event == Json::parse_event_t::object_end)
        object_keys.pop_back();
      else if (event == Json::parse_event_t::key &&
               !object_keys.back().insert(value.get<std::string>()).second)
        invalid("duplicate JSON key: " + value.get<std::string>());
      return true;
    };
    auto reference = parse_reference(Json::parse(stream, callback));
    if (stream.bad())
      throw std::runtime_error("Reference JSON read failed");
    reference.source_path = std::filesystem::absolute(path).lexically_normal();
    if (reference.source_id.empty())
      reference.source_id = path.stem().string();
    return reference;
  } catch (const std::exception &error) {
    throw std::runtime_error("Reference JSON " + path.string() + ": " +
                             error.what());
  }
}

nlohmann::json scale_reference_to_model(
    ReferenceData &reference, const ReferencePoints &model_neutral_points) {
  if (reference.positions_m.empty())
    invalid("scaling requires reference frames");
  // 3D chain length is a size proxy, not standing height or anatomical length.
  auto chain_lengths = [](const ReferencePoints &points) {
    std::array<double, 2> lengths;
    auto distance = [&](int a, int b) {
      const double d = std::hypot(points[3 * a] - points[3 * b],
                                  points[3 * a + 1] - points[3 * b + 1],
                                  points[3 * a + 2] - points[3 * b + 2]);
      if (!std::isfinite(d) || d <= 1e-9)
        invalid("scaling requires finite, nondegenerate body-chain segments");
      return d;
    };
    for (int side = 0; side < 2; ++side) {
      lengths[side] = distance(4 + side, 6 + side) +
                      distance(6 + side, 14 + side) +
                      distance(14 + side, 12 + side) +
                      distance(12 + side, 1);
      if (!std::isfinite(lengths[side]))
        invalid("nonfinite body-chain length proxy");
    }
    return lengths;
  };
  const auto target = chain_lengths(model_neutral_points);
  const double model_length = .5 * target[0] + .5 * target[1];
  std::vector<double> lengths;
  lengths.reserve(2 * reference.positions_m.size());
  for (const auto &points : reference.positions_m) {
    const auto chains = chain_lengths(points);
    lengths.insert(lengths.end(), chains.begin(), chains.end());
  }
  std::sort(lengths.begin(), lengths.end());
  const std::size_t middle = lengths.size() / 2;
  const double source_length = .5 * lengths[middle - 1] + .5 * lengths[middle];
  const double scale = model_length / source_length;
  if (!std::isfinite(scale) || scale <= 0)
    invalid("invalid reference scale");
  // Scale height about Z=0 only. Preserve XY exactly, including travel path.
  // Validate before committing, including non-chain points and overflow.
  auto transformed = reference.positions_m;
  for (auto &points : transformed)
    for (std::size_t i = 0; i < points.size(); ++i) {
      if (i % 3 == 2 && scale != 1)
        points[i] *= scale;
      if (!std::isfinite(points[i]))
        invalid("nonfinite scaled reference coordinate");
    }
  reference.positions_m = std::move(transformed);
  return {{"mode", "body-height"},
          {"scale", scale},
          {"scaled_axes", "z"},
          {"ground_z_m", 0.0},
          {"source_chain_length_m", source_length},
          {"model_chain_length_m", model_length},
          {"source_chain_length_min_m", lengths.front()},
          {"source_chain_length_max_m", lengths.back()},
          {"method", "Z-only; pooled median sum of 3D heel-knee-hip-shoulder-head "
                     "distances across both sides/all frames; target uses "
                     "mean bilateral neutral tracking-site chains"},
          {"timestamps_modified", false}};
}

ReferenceInterpolation reference_interpolation(const double *time_s,
                                                std::size_t count,
                                                double time) {
  if (!time_s || count < 2 || !std::isfinite(time))
    invalid("interpolation requires validated timestamps and finite query time");
  if (time >= time_s[count - 1])
    return {count - 1, count - 1, 0, 0};
  const double clamped = std::max(time, time_s[0]);
  const std::size_t next =
      std::upper_bound(time_s, time_s + count, clamped) - time_s;
  const std::size_t first = next - 1;
  const double interval = time_s[next] - time_s[first];
  return {first, next, (clamped - time_s[first]) / interval, interval};
}

ReferenceSample sample_reference(const ReferenceData &reference, double time) {
  if (reference.positions_m.size() != reference.time_s.size())
    invalid("positions_m and time_s sample counts differ");
  const auto sample = reference_interpolation(
      reference.time_s.data(), reference.time_s.size(), time);
  ReferenceSample result;
  for (std::size_t point = 0; point < result.positions_m.size(); ++point) {
    const double p0 = reference.positions_m[sample.first][point];
    const double p1 = reference.positions_m[sample.next][point];
    result.positions_m[point] = sample.position(p0, p1);
    result.velocities_m_per_s[point] = sample.velocity(p0, p1);
  }
  return result;
}

std::optional<std::size_t> reference_history_start(const double *time_s,
                                                  std::size_t count,
                                                  std::size_t key,
                                                  double seconds) {
  if (!time_s || count < 2 || key >= count || !std::isfinite(seconds) ||
      seconds < 0)
    invalid("invalid reference history window");
  if (seconds == 0)
    return key;
  const double start = time_s[key] - seconds;
  // Only absorb arithmetic roundoff; do not quantize native reference timing.
  const double tolerance = 16 * std::numeric_limits<double>::epsilon() *
                           std::max({1.0, std::abs(time_s[key]), seconds});
  if (start < time_s[0] - tolerance)
    return std::nullopt;
  const auto end = std::upper_bound(time_s, time_s + key + 1, start + tolerance);
  return end == time_s ? 0 : static_cast<std::size_t>(end - time_s - 1);
}
} // namespace tracking
