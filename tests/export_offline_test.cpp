// Synthetic offline-export regression. No stepping, planner, or renderer.
// --fixtures DIR retains archives/recordings for NumPy and CLI integration
// tests.
#include "export_test_helpers.hpp"
#include "tracking/export.hpp"
#include "tracking/qmc_sampling.hpp"
#include "tracking/run_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using Json = nlohmann::json;
using tracking::ReplayFrame;
using namespace export_test;

ModelPtr make_model(int per_body = 3) {
  char error[2048]{};
  SpecPtr spec(mj_parseXMLString(R"(<mujoco><worldbody>
    <body name="segment" pos="0 0 1"><freejoint/>
      <geom name="main" type="sphere" size=".1" mass="3"/>
      <body name="child" pos=".5 0 0"><joint name="hinge" axis="0 1 0"/>
        <geom name="child_geom" type="capsule" size=".05 .1" mass="2"/>
      </body>
    </body>
    <body name="reference_target" mocap="true" pos="3 0 1"><site size=".01"/></body>
    </worldbody></mujoco>)",
                                 nullptr, error, sizeof(error)),
               mj_deleteSpec);
  check(bool(spec), error);
  ModelPtr baseline(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(baseline), "Compile baseline");
  check(tracking::add_qmc_marker_sites(spec.get(), baseline.get(), per_body) ==
            2 * per_body,
        "Marker count");
  ModelPtr model(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(model), "Compile marked model");
  tracking::place_qmc_marker_sites(model.get(), 0);
  return model;
}

std::vector<ReplayFrame> make_frames(const mjModel *model) {
  auto data = make_data(model);
  const auto sites = tracking::qmc_marker_site_ids(model);
  std::vector<ReplayFrame> frames;
  for (int t = 0; t < 5; ++t) {
    data->qpos[0] = .25 * t;
    data->qpos[3] = std::cos(.1 * t);
    data->qpos[6] = std::sin(.1 * t);
    data->qpos[7] = .2 * t;
    data->mocap_pos[0] = 4 + t;
    mj_kinematics(model, data.get());
    mj_comPos(model, data.get());
    ReplayFrame frame;
    frame.time = 2.0 + .25 * t;
    frame.physics_step = 1000 + 125 * t;
    frame.wrench_available = true;
    frame.qpos.assign(data->qpos, data->qpos + model->nq);
    frame.qvel.assign(data->qvel, data->qvel + model->nv);
    frame.act.assign(data->act, data->act + model->na);
    frame.ctrl.assign(data->ctrl, data->ctrl + model->nu);
    frame.mocap_pos.assign(data->mocap_pos,
                           data->mocap_pos + 3 * model->nmocap);
    frame.mocap_quat.assign(data->mocap_quat,
                            data->mocap_quat + 4 * model->nmocap);
    frame.tracking_positions_m.resize(48, 42.0);
    for (int site : sites)
      frame.markers_m.insert(frame.markers_m.end(), data->site_xpos + 3 * site,
                             data->site_xpos + 3 * site + 3);
    // Nonphysical for this airborne fixture: recomputing contacts would erase
    // these labels, detecting that bug without simulating anything.
    tracking::ContactWrench left, right;
    if (t == 0) {
      left = {{2, 3, 10}, {-3, -2, 1.7}};
    } else if (t == 1) {
      left = {{-7, 5, 2.5}, {.75, 1.25, -.4}};
      right = {{-1e-12, 2e-12, 1e-12}, {-2.5e-13, -1.25e-13, 1e-13}};
    } else if (t == 3) {
      left = {{.12345678901234566, -2.1234567890123457, -4.5}, {1, 2, 3}};
      right = {{3, -4, 0}, {1, 2, 3}};
    } else if (t == 4) {
      left = {{1, 0, std::numeric_limits<double>::denorm_min()},
              {1e308, 1, 2}}; // Derived overflow: only CoP/Tz become undefined.
      right = {{-2, -3, 10}, {1, -2, -.7}};
    }
    set_wrench(frame, 0, left);
    set_wrench(frame, 1, right);
    if (t == 0) {
      // Valid within bundle tolerance, but distinguishable at float64
      // precision: the exporter must COPY saved CoP/Tz, not derive them again
      // from moments.
      frame.cop_left[0] = std::nextafter(frame.cop_left[0], 1.0);
      frame.torque_z_left = std::nextafter(frame.torque_z_left, 1.0);
    }
    frame.contact_proxy_left = {9, 8, 7};
    frame.contact_proxy_right = {6, 5, 4};
    frames.push_back(std::move(frame));
  }
  return frames;
}

int dynamics_callbacks = 0;
void unexpected_dynamics(const mjModel *, mjData *) { ++dynamics_callbacks; }
struct DynamicsTripwire {
  mjfGeneric control = mjcb_control, passive = mjcb_passive;
  DynamicsTripwire() {
    dynamics_callbacks = 0;
    mjcb_control = mjcb_passive = unexpected_dynamics;
  }
  ~DynamicsTripwire() {
    mjcb_control = control;
    mjcb_passive = passive;
  }
};

void legacy_fixture(const std::filesystem::path &v2,
                    const std::filesystem::path &v1, const mjModel *model) {
  // A real one-frame V1 payload: strip the V2 header count and frame extension.
  auto buffer = bytes(v2);
  std::uint64_t model_bytes = 0, summary_bytes = 0;
  std::memcpy(&model_bytes, buffer.data() + 16, 8);
  std::memcpy(&summary_bytes, buffer.data() + 24, 8);
  const std::uint32_t version = 1;
  std::memcpy(buffer.data() + 8, &version, 4);
  std::ofstream stream(v1, std::ios::binary);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.write(buffer.data(), 60);
  stream.write(buffer.data() + 64,
               static_cast<std::streamsize>(model_bytes + summary_bytes));
  const size_t frame_bytes = 8 +
                             8 * (model->nq + model->nv + model->na +
                                  model->nu + 7 * model->nmocap + 12) +
                             280;
  stream.write(buffer.data() + 64 + model_bytes + summary_bytes,
               static_cast<std::streamsize>(frame_bytes));
}

void alternate_expectations(const std::filesystem::path &path, mjModel *model,
                            const std::vector<ReplayFrame> &frames) {
  tracking::place_qmc_marker_sites(model, 11);
  const auto sites = tracking::qmc_marker_site_ids(model);
  auto data = make_data(model);
  auto expected = frames;
  for (auto &frame : expected) {
    std::copy(frame.qpos.begin(), frame.qpos.end(), data->qpos);
    std::copy(frame.mocap_pos.begin(), frame.mocap_pos.end(), data->mocap_pos);
    std::copy(frame.mocap_quat.begin(), frame.mocap_quat.end(),
              data->mocap_quat);
    mj_kinematics(model, data.get());
    frame.markers_m.clear();
    for (int site : sites) {
      // Independent local-to-world construction rather than site_xpos copying.
      const int body = model->site_bodyid[site];
      double point[3];
      mju_mulMatVec3(point, data->xmat + 9 * body, model->site_pos + 3 * site);
      mju_addTo3(point, data->xpos + 3 * body);
      frame.markers_m.insert(frame.markers_m.end(), point, point + 3);
    }
  }
  write_markers(path, expected);
  tracking::place_qmc_marker_sites(model, 0);
}

void run_tests(const std::filesystem::path &directory) {
  std::filesystem::create_directories(directory);
  const auto input = directory / "source.grf",
             output = directory / "recorded.npz";
  auto model = make_model();
  const auto frames = make_frames(model.get());
  Json summary = {
      {"mujoco_version", mj_versionString()},
      {"source_id", "synthetic-recording"},
      {"completed_requested_duration", true},
      {"arbitrary_audit", "private-to-grf"},
      {"preparation", {{"total_mass_kg", 999}, {"qmc", {{"index", 0}}}}}};
  tracking::save_run_bundle(input, model.get(), frames, summary);
  const auto input_before = bytes(input);
  {
    DynamicsTripwire tripwire;
    tracking::export_run_bundle(input, output, -1, 42);
    tracking::export_run_bundle(input, directory / "default.npz");
    tracking::export_run_bundle(input, directory / "same.npz", 0, 42);
    tracking::export_run_bundle(input, directory / "alternate.npz", 11, 42);
    tracking::export_run_bundle(input, directory / "masked.npz", -1, 42, false,
                                10);
    tracking::export_run_bundle(input, directory / "partial_mask.npz", -1, 42,
                                false, 2.5);
    check(dynamics_callbacks == 0, "Offline export invoked dynamics callbacks");
  }
  const auto recorded_bytes = bytes(output);
  check(recorded_bytes == bytes(directory / "same.npz"),
        "Online/offline same-layout agreement");
  check(recorded_bytes != bytes(directory / "alternate.npz"),
        "Independent layout did not change observations");
  write_markers(directory / "recorded_markers.txt", frames);
  write_wrench(directory / "recorded_wrench.txt", frames);
  alternate_expectations(directory / "alternate_markers.txt", model.get(),
                         frames);

  tracking::export_run_bundle(input, directory / "repeat.npz", -1, 42);
  check(recorded_bytes == bytes(directory / "repeat.npz"),
        "Reproducible export");
  tracking::export_run_bundle(input, directory / "shuffle.npz", -1, 99);
  check(recorded_bytes != bytes(directory / "shuffle.npz"),
        "Seed changes slots");
  std::filesystem::copy_file(input, directory / "relocated.grf");
  tracking::export_run_bundle(directory / "relocated.grf",
                              directory / "relocated.npz", -1, 42);
  check(recorded_bytes == bytes(directory / "relocated.npz"),
        "Path-independent permutation");

  auto modified = frames;
  modified[1].markers_m[0] += 123;
  tracking::save_run_bundle(directory / "modified.grf", model.get(), modified,
                            summary);
  tracking::export_run_bundle(directory / "modified.grf",
                              directory / "modified.npz", -1, 42);
  write_markers(directory / "modified_markers.txt", modified);

  const auto invalid = directory / "invalid.grf";
  summary["completed_requested_duration"] = false;
  tracking::save_run_bundle(directory / "incomplete.grf", model.get(), frames,
                            summary);
  rejects(
      [&] {
        tracking::export_run_bundle(directory / "incomplete.grf", output);
      },
      "Incomplete");
  rejects(
      [&] {
        tracking::export_run_bundles({input, directory / "incomplete.grf"},
                                     output);
      },
      "Incomplete");
  check(bytes(output) == recorded_bytes, "Rejection preserves existing output");
  tracking::export_run_bundle(directory / "incomplete.grf",
                              directory / "diagnostic.npz", -1, 42, true);
  summary.erase("completed_requested_duration");
  tracking::save_run_bundle(invalid, model.get(), frames, summary);
  rejects([&] { tracking::export_run_bundle(invalid, output); }, "Incomplete");
  summary["completed_requested_duration"] = true;
  summary["source_id"] = " \t\n";
  tracking::save_run_bundle(invalid, model.get(), frames, summary);
  rejects([&] { tracking::export_run_bundle(invalid, output); }, "source_id");
  summary.erase("source_id");
  tracking::save_run_bundle(directory / "no_source.grf", model.get(), frames,
                            summary);
  rejects(
      [&] { tracking::export_run_bundle(directory / "no_source.grf", output); },
      "source_id");
  summary["preparation"]["reference"] = {{"source_id", "synthetic-錄影-Δ-🦶"}};
  tracking::save_run_bundle(directory / "nested.grf", model.get(), frames,
                            summary);
  tracking::export_run_bundle(directory / "nested.grf",
                              directory / "nested.npz");
  summary["source_id"] = "conflicting-recording";
  tracking::save_run_bundle(invalid, model.get(), frames, summary);
  rejects([&] { tracking::export_run_bundle(invalid, output); }, "Conflicting");
  summary.erase("source_id");

  auto finite = frames[0];
  set_wrench(finite, 1, tracking::ContactWrench{{0, 0, 1}, {0, 0, 0}});
  const auto legacy = directory / "legacy.grf";
  tracking::save_run_bundle(directory / "finite.grf", model.get(), {finite},
                            summary);
  legacy_fixture(directory / "finite.grf", legacy, model.get());
  check(!tracking::load_run_bundle(legacy).frames[0].wrench_available,
        "Real V1 fixture");
  for (bool allow : {false, true})
    rejects([&] { tracking::export_run_bundle(legacy, output, -1, 0, allow); },
            "legacy");
  rejects([&] { tracking::export_run_bundles({}, output); }, "at least one");
  rejects([&] { tracking::export_run_bundle(input, output, -2); },
          "layout_index");
  rejects([&] { tracking::export_run_bundle(input, output, -1, -1); }, "seed");
  for (double threshold : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()})
    rejects(
        [&] {
          tracking::export_run_bundle(input, output, -1, 0, false, threshold);
        },
        "min_normal_force");
  rejects(
      [&] { tracking::export_run_bundle(input, directory / "obsolete.json"); },
      ".npz");
  rejects([&] { tracking::export_run_bundle(input, input); }, "overwrite");
  std::filesystem::create_hard_link(input, directory / "hardlink.npz");
  rejects(
      [&] { tracking::export_run_bundle(input, directory / "hardlink.npz"); },
      "overwrite");
  std::filesystem::create_symlink(input, directory / "symlink.npz");
  rejects(
      [&] { tracking::export_run_bundle(input, directory / "symlink.npz"); },
      "overwrite");

  auto nonincreasing = frames;
  nonincreasing[1].physics_step = nonincreasing[0].physics_step;
  tracking::save_run_bundle(directory / "bad_steps.grf", model.get(),
                            nonincreasing, summary);
  rejects(
      [&] { tracking::export_run_bundle(directory / "bad_steps.grf", output); },
      "physics steps");
  auto small = make_model(2);
  tracking::save_run_bundle(directory / "different_markers.grf", small.get(),
                            make_frames(small.get()), summary);
  rejects(
      [&] {
        tracking::export_run_bundles(
            {input, directory / "different_markers.grf"}, output);
      },
      "Marker counts differ");
  std::filesystem::create_directory(directory / "blocked.npz");
  rejects(
      [&] { tracking::export_run_bundle(input, directory / "blocked.npz"); },
      "rename");
  check(std::filesystem::is_directory(directory / "blocked.npz"),
        "Preserve destination on write failure");

  // Cross compression-buffer boundaries without generating motion/contacts.
  std::vector<ReplayFrame> long_frames;
  std::uint64_t noise = UINT64_C(0x123456789abcdef);
  for (int i = 0; i < 1500; ++i) {
    auto frame = frames[static_cast<size_t>(i) % frames.size()];
    frame.time = i / 600.0;
    frame.physics_step = 4 * i;
    // High-entropy saved observations exercise output-buffer boundaries too.
    for (auto &coordinate : frame.markers_m) {
      noise =
          noise * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
      coordinate = static_cast<double>(noise >> 11) * 0x1p-53;
    }
    long_frames.push_back(std::move(frame));
  }
  Json long_summary = {{"mujoco_version", mj_versionString()},
                       {"source_id", "long-synthetic"},
                       {"completed_requested_duration", true}};
  tracking::save_run_bundle(directory / "long.grf", model.get(), long_frames,
                            long_summary);
  tracking::export_run_bundle(directory / "long.grf", directory / "long.npz",
                              -1, 42);
  write_markers(directory / "long_markers.txt", long_frames);

  // A second original recording, different mass and time origin; nested source
  // identity and repeated variants remain caller-ordered in one archive.
  mj_setTotalmass(model.get(), 7.5);
  auto second = std::vector<ReplayFrame>(frames.begin(), frames.begin() + 2);
  second[0].time = 0;
  second[1].time = .1;
  summary = {{"mujoco_version", mj_versionString()},
             {"completed_requested_duration", true},
             {"reference", {{"source_id", "another-recording"}}}};
  tracking::save_run_bundle(directory / "second.grf", model.get(), second,
                            summary);
  {
    DynamicsTripwire tripwire;
    tracking::export_run_bundles(
        {input, directory / "second.grf", input, directory / "nested.grf"},
        directory / "episodes.npz", -1, 42);
    check(dynamics_callbacks == 0, "Multi-episode export invoked dynamics");
  }
  check(bytes(input) == input_before, "Never modify source bundle");
  check(bytes(output) == recorded_bytes,
        "All failures preserve existing output");
  for (const auto &item : std::filesystem::directory_iterator(directory))
    check(item.path().filename().string().find(".partial.") ==
                  std::string::npos &&
              item.path().extension() != ".json" &&
              item.path().extension() != ".npy",
          "Export left staging/metadata/intermediate files");
}
} // namespace

int main(int argc, char **argv) try {
  if (argc == 3 && std::string(argv[1]) == "--fixtures")
    run_tests(argv[2]);
  else {
    check(argc == 1, "Usage: test_export_offline [--fixtures DIR]");
    TemporaryDirectory directory;
    run_tests(directory.path);
  }
  std::cout << "Synthetic offline NPZ exporter tests passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "Exporter test failed: " << error.what() << '\n';
  return 1;
}
