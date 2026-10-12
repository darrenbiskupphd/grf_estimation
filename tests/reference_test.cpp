#include "tracking/reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::json;
using namespace tracking;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message,
          double tolerance = 1e-11) {
  check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
}
template <typename Fn> void rejects(Fn function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("Expected rejection: " + message);
}

Json document(const std::vector<double> &times = {0, .1, .4}) {
  Json frames = Json::array();
  for (std::size_t frame = 0; frame < times.size(); ++frame) {
    Json points = Json::array();
    for (std::size_t point = 0; point < kTargets.size(); ++point) {
      const double value = 10 * frame + point;
      points.push_back({value, -value, value / 2});
    }
    frames.push_back(points);
  }
  return {{"schema_version", 1}, {"time_s", times}, {"positions_m", frames},
          {"target_names", kTargets}, {"metadata", Json::object()}};
}

class TemporaryFile {
public:
  TemporaryFile()
      : path(std::filesystem::temp_directory_path() /
             ("grf_reference_test_" +
              std::to_string(std::chrono::steady_clock::now()
                                 .time_since_epoch().count()) + ".json")) {}
  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
  void write(const std::string &text) const {
    std::ofstream stream(path);
    stream << text;
    check(bool(stream), "write test JSON");
  }
  std::filesystem::path path;
};

void validation_and_loading() {
  const auto valid = document();
  auto parsed = parse_reference(valid);
  near(parsed.duration_s(), .4, "duration is timestamp based");
  check(parsed.positions_m.size() == 3, "reference sample count");
  auto offset = document({12.5, 12.625, 13});
  offset["metadata"] = {{"source_id", "fixture"}, {"custom_provenance", {1, 2}}};
  parsed = parse_reference(offset);
  check(parsed.time_s == std::vector<double>({0, .125, .5}),
        "normalize only time origin");
  near(parsed.time_origin_s, 12.5, "record original origin");
  check(parsed.metadata == offset["metadata"], "preserve metadata");
  check(parsed.source_id == "fixture", "metadata source identity");
  auto conflict = offset;
  conflict["metadata"]["source"] = {{"source_id", "other-recording"}};
  rejects([&] { parse_reference(conflict); }, "conflicting recording identities");
  auto imported = valid;
  imported["metadata"]["source"] = {{"source_id", "CMU:07_01"}};
  check(parse_reference(imported).source_id == "CMU:07_01", "importer source identity");
  for (std::size_t frame = 0; frame < 3; ++frame)
    for (std::size_t point = 0; point < 16; ++point)
      for (std::size_t axis = 0; axis < 3; ++axis)
        near(parsed.positions_m[frame][3 * point + axis],
             offset["positions_m"][frame][point][axis], "unchanged coordinates", 0);

  auto bad = [&](auto mutate, const std::string &message) {
    auto value = valid;
    mutate(value);
    rejects([&] { parse_reference(value); }, message);
  };
  for (const std::string field :
       {"schema_version", "time_s", "positions_m", "target_names", "metadata"})
    bad([&](Json &j) { j.erase(field); }, "missing " + field);
  bad([](Json &j) { j = Json::array(); }, "non-object root");
  bad([](Json &j) { j["schema_version"] = 2; }, "unknown schema");
  bad([](Json &j) { j["schema_version"] = 1.0; }, "non-integer schema");
  bad([](Json &j) { j["schema_version"] = true; }, "boolean schema");
  bad([](Json &j) { j["unexpected"] = 1; }, "unknown top-level field");
  bad([](Json &j) { j["metadata"] = nullptr; }, "non-object metadata");
  bad([](Json &j) { j["metadata"]["source_id"] = ""; }, "empty source id");
  bad([](Json &j) { j["metadata"]["source_id"] = 42; }, "numeric source id");
  bad([](Json &j) { j["metadata"]["source"] = {{"source_id", 42}}; },
      "numeric importer source id");
  bad([](Json &j) { std::swap(j["target_names"][2], j["target_names"][3]); },
      "swapped target names");
  bad([](Json &j) { j["target_names"][0] = "Pelvis"; }, "name case");
  bad([](Json &j) { j["target_names"].erase(0); }, "missing target name");
  bad([](Json &j) { j["target_names"][0] = 0; }, "non-string target name");
  bad([](Json &j) { j["time_s"] = Json::array(); }, "empty times");
  bad([](Json &j) { j["time_s"] = {0}; }, "one time");
  bad([](Json &j) { j["time_s"] = "0 .1 .4"; }, "non-array times");
  bad([](Json &j) { j["time_s"] = {0, .1, .1}; }, "duplicate time");
  bad([](Json &j) { j["time_s"] = {0, .4, .1}; }, "decreasing time");
  bad([](Json &j) { j["time_s"] = {-.1, 0, .1}; }, "negative time");
  bad([](Json &j) { j["time_s"][1] = true; }, "boolean time");
  bad([](Json &j) { j["positions_m"].erase(0); }, "sample count mismatch");
  bad([](Json &j) { j["positions_m"][0].erase(0); }, "target count mismatch");
  bad([](Json &j) { j["positions_m"][0][0] = {1, 2}; }, "short XYZ");
  bad([](Json &j) { j["positions_m"][0][0] = {1, 2, 3, 4}; }, "long XYZ");
  bad([](Json &j) { j["positions_m"][0][0][0] = "0"; }, "string coordinate");
  bad([](Json &j) { j["positions_m"][0][0][0] = false; }, "bool coordinate");
  bad([](Json &j) { j["positions_m"][0][0][0] = nullptr; }, "null coordinate");
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    bad([&](Json &j) { j["time_s"][1] = value; }, "non-finite time");
    bad([&](Json &j) { j["positions_m"][0][0][0] = value; },
        "non-finite coordinate");
  }

  TemporaryFile file;
  file.write(valid.dump());
  parsed = load_reference(file.path);
  check(parsed.source_id == file.path.stem(), "filename fallback identity");
  check(parsed.source_path == std::filesystem::absolute(file.path), "source path");
  file.write(offset.dump());
  check(load_reference(file.path).source_id == "fixture", "loaded source id");
  file.write(valid.dump() + " trailing garbage");
  rejects([&] { load_reference(file.path); }, "trailing garbage");
  file.write("{\"schema_version\":1,\"schema_version\":1}");
  rejects([&] { load_reference(file.path); }, "duplicate JSON keys");
  file.write("{\"time_s\":[0,1e999]}");
  rejects([&] { load_reference(file.path); }, "numeric overflow");
  file.write("not JSON");
  rejects([&] { load_reference(file.path); }, "malformed JSON");
  rejects([] { load_reference({}); }, "empty path");
  rejects([&] { load_reference(file.path / "missing.json"); }, "missing file");
}

void interpolation() {
  const auto reference = parse_reference(document());
  for (double t : {-1., 0., .05, .1, .25, .4, 1.}) {
    const auto sample = sample_reference(reference, t);
    const double position = t <= 0 ? 0 : t < .1 ? t * 100
                                        : t < .4 ? 10 + (t - .1) * 10 / .3 : 20;
    const double velocity = t < .1 ? 100 : t < .4 ? 10 / .3 : 0;
    for (std::size_t target = 0; target < 16; ++target) {
      near(sample.positions_m[3 * target], position + target, "irregular position");
      near(sample.velocities_m_per_s[3 * target], velocity, "irregular velocity");
      near(sample.velocities_m_per_s[3 * target + 1], -velocity, "negative slope");
    }
  }
  const auto end = reference_interpolation(reference.time_s.data(), 3, .4);
  check(end.first == 2 && end.next == 2 && end.interval_s == 0,
        "final endpoint has zero-velocity sentinel");
  const double before_end = std::nextafter(.4, 0.);
  check(sample_reference(reference, before_end).velocities_m_per_s[0] > 0,
        "velocity immediately before end is nonzero");
  rejects([&] { sample_reference(reference, std::numeric_limits<double>::quiet_NaN()); },
          "nonfinite query");
  const auto two = parse_reference(document({0, .2}));
  near(sample_reference(two, .1).positions_m[0], 5, "two-frame interpolation");
  near(sample_reference(two, .2).velocities_m_per_s[0], 0, "two-frame endpoint");
}

void uniform_30hz_regression() {
  constexpr int count = 91;
  std::vector<double> times(count);
  for (int key = 0; key < count; ++key)
    times[key] = key / 30.;
  auto json = document(times);
  for (int key = 0; key < count; ++key)
    for (int target = 0; target < 16; ++target)
      for (int axis = 0; axis < 3; ++axis)
        json["positions_m"][key][target][axis] =
            std::sin(key * .2 + target * .1 + axis);
  const auto reference = parse_reference(json);
  auto compare = [&](double time) {
    const double index = std::clamp(time * 30, 0., double(count - 1));
    const int first = static_cast<int>(std::floor(index));
    const int next = std::min(first + 1, count - 1);
    const double fraction = index - first;
    const auto sample = sample_reference(reference, time);
    for (std::size_t i = 0; i < 48; ++i) {
      const double p0 = reference.positions_m[first][i];
      const double p1 = reference.positions_m[next][i];
      near(sample.positions_m[i], (1 - fraction) * p0 + fraction * p1,
           "30 Hz legacy position");
      near(sample.velocities_m_per_s[i], (p1 - p0) * 30,
           "30 Hz legacy velocity");
    }
  };
  for (int segment = 0; segment < count - 1; ++segment)
    for (double fraction : {.01, .25, .5, .99})
      compare((segment + fraction) / 30.);
  for (double time : {-1., 0., 1 / 30., 2 / 30., 1., 2., 3., 3.1})
    compare(time);
  near(reference.duration_s(), (count - 1) / 30., "legacy duration");

  for (double seconds : {0., 1 / 30., .1, .12, 1., 5.}) {
    const int prior = static_cast<int>(std::ceil(seconds * 30 - 1e-9));
    for (int key = 0; key < count; ++key) {
      const auto start = reference_history_start(times.data(), times.size(), key, seconds);
      check(bool(start) == (key >= prior), "legacy stance window completeness");
      if (start)
        check(*start == std::size_t(key - prior), "legacy stance history keys");
    }
  }
}

void irregular_history() {
  const double times[] = {0, .04, .2, .21, .7};
  check(!reference_history_start(times, 5, 2, .3), "insufficient real history");
  check(reference_history_start(times, 5, 2, .16) == 1, "exact window boundary");
  check(reference_history_start(times, 5, 3, .1) == 1, "include straddling segment");
  check(reference_history_start(times, 5, 4, .1) == 3, "sparse native timing");
  check(reference_history_start(times, 5, 4, .7) == 0, "whole history");
  check(reference_history_start(times, 5, 4, 0) == 4, "zero history");
}
} // namespace

int main() try {
  validation_and_loading();
  interpolation();
  uniform_30hz_regression();
  irregular_history();
  std::cout << "Reference loading/interpolation tests passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
