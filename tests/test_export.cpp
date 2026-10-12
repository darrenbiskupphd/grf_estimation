#include "export_test_helpers.hpp"
#include "tracking/export.hpp"
#include "tracking/qmc_sampling.hpp"
#include "tracking/run_bundle.hpp"

#include <iostream>

using namespace tracking;
using namespace export_test;
using Json = nlohmann::json;

void run_tests(const char *xml, const std::filesystem::path &directory) {
  char error[1024]{};
  SpecPtr spec(mj_parseXML(xml, nullptr, error, sizeof(error)), mj_deleteSpec);
  check(bool(spec), error);
  ModelPtr baseline(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(baseline), "Baseline compile failed");
  add_qmc_marker_sites(spec.get(), baseline.get());
  ModelPtr model(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(model), "Marker model compile failed");
  place_qmc_marker_sites(model.get(), 0);
  auto d = make_data(model.get());
  const auto sites = qmc_marker_site_ids(model.get());
  check(sites.size() == 140, "Whole-foot marker budget incorrect");
  std::vector<ReplayFrame> frames;
  for (int i = 0; i < 2; ++i) {
    d->qpos[0] += .001;
    mj_kinematics(model.get(), d.get());
    mj_comPos(model.get(), d.get());
    ReplayFrame f;
    f.wrench_available = true;
    f.physics_step = 4 * i;
    f.time = i / 600.;
    f.qpos.assign(d->qpos, d->qpos + model->nq);
    f.qvel.assign(d->qvel, d->qvel + model->nv);
    f.act.assign(d->act, d->act + model->na);
    f.ctrl.assign(d->ctrl, d->ctrl + model->nu);
    f.mocap_pos.assign(d->mocap_pos, d->mocap_pos + 3 * model->nmocap);
    f.mocap_quat.assign(d->mocap_quat, d->mocap_quat + 4 * model->nmocap);
    f.tracking_positions_m.resize(48);
    for (int site : sites)
      f.markers_m.insert(f.markers_m.end(), d->site_xpos + 3 * site,
                         d->site_xpos + 3 * site + 3);
    set_wrench(f, 0, ContactWrench{{1, 2, 10}, {-3, -4, 5}});
    set_wrench(f, 1, ContactWrench{});
    frames.push_back(std::move(f));
  }
  std::filesystem::create_directories(directory);
  const auto bundle = directory / "source.grf",
             recorded = directory / "recorded.npz",
             offline = directory / "same.npz",
             changed = directory / "alternate.npz";
  const Json summary = {{"mujoco_version", mj_versionString()},
                        {"source_id", "synthetic-physical-model"},
                        {"completed_requested_duration", true},
                        {"layout_index", 0}};
  save_run_bundle(bundle, model.get(), frames, summary);
  const auto input_before = bytes(bundle);
  export_run_bundle(bundle, recorded, -1, 7);
  export_run_bundle(bundle, offline, 0, 7);
  export_run_bundle(bundle, changed, 1, 7);
  check(bytes(recorded) == bytes(offline),
        "Recorded/offline same-layout exports differ");
  check(bytes(recorded) != bytes(changed),
        "Independent layout did not change export");
  export_run_bundle(bundle, directory / "masked.npz", -1, 7, false, 10);
  export_run_bundles({bundle, bundle}, directory / "episodes.npz", -1, 7);
  check(bytes(bundle) == input_before, "Export changed input bundle");
  write_markers(directory / "recorded_markers.txt", frames);
  write_wrench(directory / "recorded_wrench.txt", frames);
}

int main(int argc, char **argv) try {
  check(argc == 2 || argc == 3, "Need physical XML path [fixture directory]");
  if (argc == 3)
    run_tests(argv[1], argv[2]);
  else {
    TemporaryDirectory directory;
    run_tests(argv[1], directory.path);
  }
  std::cout << "Physical-model offline NPZ export passed (no simulation)\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
