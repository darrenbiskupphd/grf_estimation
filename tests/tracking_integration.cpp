#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tracking/run_bundle.hpp"
#include "tracking/qmc_sampling.hpp"
#include "tracking/tracking.hpp"

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

class Cleanup {
public:
  explicit Cleanup(std::filesystem::path path) : path_(std::move(path)) {
    std::error_code error;
    std::filesystem::remove(path_, error);
    std::filesystem::remove(path_.string() + ".partial", error);
  }
  ~Cleanup() {
    std::error_code error;
    std::filesystem::remove(path_, error);
    std::filesystem::remove(path_.string() + ".partial", error);
  }

private:
  std::filesystem::path path_;
};

void exercise(const std::filesystem::path &model, const std::string &label,
              int qmc_index = 0) {
  const auto output =
      std::filesystem::current_path() / ("tracking_integration_" + label + ".grf");
  Cleanup cleanup(output);
  tracking::PrepareConfig prepare;
  prepare.motion = "walk";
  prepare.reference = tracking::ReferenceStrategy::Raw;
  prepare.qmc_index = qmc_index;
  tracking::RunConfig config;
  config.motion = prepare.motion;
  config.duration = .03;
  config.output_path = output;
  config.planner_threads = 1;
  tracking::run(tracking::prepare_custom(model.string(), prepare), config);

  require(std::filesystem::exists(output), "Run did not write its bundle");
  require(!std::filesystem::exists(output.string() + ".partial"),
          "Run left a partial bundle");
  const auto bundle = tracking::load_run_bundle(output);
  require(bundle.model != nullptr && !bundle.frames.empty(),
          "Bundle replay did not load");
  require(bundle.summary.at("completed_requested_duration").get<bool>(),
          "Short raw rollout did not complete");
  require(bundle.summary.at("state_resets_after_initialization").get<int>() == 0,
          "Reference reset the physical state");
  require(bundle.summary.at("action_evaluations").get<int>() ==
              bundle.summary.at("completed_steps").get<int>(),
          "Feedback action was not evaluated at every physics step");
  require(std::abs(bundle.summary.at("physics_timestep_s").get<double>() -
                   1.0 / 2400.0) <
              1e-12,
          "Unexpected physics timestep");
  require(bundle.summary.at("preparation")
              .at("reference_transform")
              .at("strategy") == "raw",
          "Bundle did not retain reference strategy");
  const auto marker_sites = tracking::qmc_marker_site_ids(bundle.model.get());
  require(!marker_sites.empty() &&
              bundle.summary.at("preparation")
                      .at("qmc")
                      .at("markers")
                      .at("marker_site_count") == marker_sites.size(),
          "Bundle did not preserve QMC marker sites");
  require(bundle.summary.at("preparation").at("qmc").at("index") ==
              qmc_index,
          "Bundle did not retain its QMC index");
  const auto &frame = bundle.frames.back();
  require(frame.qpos.size() == static_cast<size_t>(bundle.model->nq) &&
              frame.qvel.size() == static_cast<size_t>(bundle.model->nv) &&
              frame.act.size() == static_cast<size_t>(bundle.model->na) &&
              frame.ctrl.size() == static_cast<size_t>(bundle.model->nu) &&
              frame.mocap_pos.size() == static_cast<size_t>(3 * bundle.model->nmocap),
          "Bundle frame dimensions are wrong");
  require(std::isfinite(frame.left_contact.pitch) &&
              std::isfinite(frame.right_contact.mtp_velocity),
          "Bundle contact diagnostics are non-finite");
}
} // namespace

int main(int argc, char **argv) try {
  require(argc == 3, "Supply both baseline XMLs");
  exercise(argv[1], "male");
  exercise(argv[2], "female");
  exercise(argv[1], "male_qmc", 2);
  std::cout << "Raw tracking run-bundle integration checks passed.\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
