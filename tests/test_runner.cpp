// Tiny synthetic episode: synchronization/capture regression, not a mocap
// pilot.
#include "tracking/contact_wrench.hpp"
#include "tracking/qmc_sampling.hpp"
#include "tracking/run_bundle.hpp"
#include "tracking/tracking.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace tracking;
void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
void near(double a, double b) {
  if (!std::isfinite(a) || std::abs(a - b) > 1e-10)
    throw std::runtime_error("Synchronization mismatch");
}
int main(int argc, char **argv) try {
  check(argc == 2, "Need model XML");
  const std::filesystem::path ref = "test_runner_reference.json",
                              output = "test_runner.grf";
  struct Cleanup {
    std::filesystem::path a, b;
    ~Cleanup() {
      std::filesystem::remove(a);
      std::filesystem::remove(b);
    }
  } cleanup{ref, output};
  const double neutral[16][3] = {
      {0, 0, 1},       {.08, 0, 1.7},    {.12, .1, .03}, {.12, -.1, .03},
      {-.05, .1, .08}, {-.05, -.1, .08}, {0, .1, .5},    {0, -.1, .5},
      {.1, .5, 1.1},   {.1, -.5, 1.1},   {0, .35, 1.3},  {0, -.35, 1.3},
      {0, .18, 1.5},   {0, -.18, 1.5},   {0, .1, .95},   {0, -.1, .95}};
  nlohmann::json points = nlohmann::json::array();
  for (const auto &p : neutral)
    points.push_back({p[0], p[1], p[2]});
  // Native duration is not a physics-step multiple; last capture off 600 Hz
  // grid.
  nlohmann::json doc = {
      {"schema_version", 1},
      {"time_s", {0, .0041}},
      {"positions_m", {points, points}},
      {"target_names", kTargets},
      {"metadata", {{"source_id", "synthetic-capture-test"}}}};
  {
    std::ofstream out(ref);
    out << doc.dump();
  }
  PrepareConfig prep;
  prep.reference_path = ref;
  auto prepared = build_reference_tracking_model(argv[1], prep);
  const auto reference = load_reference(ref);
  RunConfig run_config;
  run_config.output_path = output;
  run_config.motion = prepared.reference_id;
  run_config.planner_threads =
      std::max(1u, std::thread::hardware_concurrency());
  run(std::move(prepared), run_config);
  auto bundle = load_run_bundle(output);
  check(bundle.summary["completed_requested_duration"] == true,
        "Synthetic run did not complete");
  check(bundle.frames.size() == 4, "Expected captures at steps 0,4,8,9");
  check(bundle.frames.back().physics_step == 9,
        "Run stepped past native reference end");
  check(bundle.summary["capture_hz"] == 600., "Capture metadata incorrect");
  auto d = make_data(bundle.model.get());
  const auto sites = qmc_marker_site_ids(bundle.model.get());
  const long long expected_steps[] = {0, 4, 8, 9};
  for (size_t i = 0; i < bundle.frames.size(); ++i) {
    const auto &f = bundle.frames[i];
    check(f.physics_step == expected_steps[i], "Capture step mismatch");
    near(f.time, f.physics_step / 2400.);
    mju_copy(d->qpos, f.qpos.data(), bundle.model->nq);
    mju_copy(d->qvel, f.qvel.data(), bundle.model->nv);
    mj_kinematics(bundle.model.get(), d.get());
    for (size_t j = 0; j < sites.size(); ++j)
      for (int k = 0; k < 3; ++k)
        near(f.markers_m[3 * j + k], d->site_xpos[3 * sites[j] + k]);
    const auto targets = sample_reference(reference, f.time);
    for (size_t j = 0; j < kTargets.size(); ++j) {
      const std::string suffix = std::string("[") + kTargets[j] + "]";
      const int body =
          require_id(bundle.model.get(), mjOBJ_BODY, "mocap" + suffix);
      const int site =
          require_id(bundle.model.get(), mjOBJ_SITE, "tracking" + suffix);
      const int mocap = bundle.model->body_mocapid[body];
      for (int k = 0; k < 3; ++k) {
        near(f.mocap_pos[3 * mocap + k], targets.positions_m[3 * j + k]);
        near(f.tracking_positions_m[3 * j + k], d->site_xpos[3 * site + k]);
      }
    }
  }
  std::cout << "Synthetic runner capture/endpoint/executed-marker "
               "synchronization passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
