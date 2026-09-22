#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mujoco_raii.hpp"
#include "tracking/qmc_sampling.hpp"
#include "tracking/tracking.hpp"

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

std::array<int, 16> mocap_ids(const mjModel *model) {
  std::array<int, 16> ids{};
  for (int i = 0; i < 16; ++i) {
    const int body = tracking::require_id(
        model, mjOBJ_BODY, std::string("mocap[") + tracking::kTargets[i] + "]");
    ids[i] = model->body_mocapid[body];
    require(ids[i] >= 0, "Target body is not mocap");
  }
  return ids;
}

void verify_raw_identity(const mjModel *source, const mjModel *raw,
                         int source_first) {
  const auto source_mocap = mocap_ids(source);
  const auto raw_mocap = mocap_ids(raw);
  for (int frame = 0; frame < raw->nkey; ++frame) {
    for (int target = 0; target < 16; ++target) {
      const double *expected =
          source->key_mpos + 3 * source->nmocap * (source_first + frame) +
          3 * source_mocap[target];
      const double *actual =
          raw->key_mpos + 3 * raw->nmocap * frame + 3 * raw_mocap[target];
      require(std::equal(expected, expected + 3, actual),
              "Raw target transfer changed a source coordinate");
    }
  }
}

double sensor_weight(const mjModel *model, const char *name) {
  const int sensor = mj_name2id(model, mjOBJ_SENSOR, name);
  require(sensor >= 0, "Expected tracking cost sensor is missing");
  return model->sensor_user[sensor * model->nuser_sensor + 1];
}

void verify_controller_weights(const mjModel *model) {
  require(sensor_weight(model, "Pos[toe]") == 80.0 &&
              sensor_weight(model, "Pos[heel]") == 80.0 &&
              sensor_weight(model, "Pos[knee]") == 55.0 &&
              sensor_weight(model, "Pos[hip]") == 50.0,
          "Controller foot/leg weights are not tuned as expected");
  require(sensor_weight(model, "Pos[hand]") == 8.0 &&
              sensor_weight(model, "Pos[elbow]") == 8.0 &&
              sensor_weight(model, "Pos[shoulder]") == 8.0,
          "Controller upper-body weights are not relaxed");
}

void verify_task_contract(tracking::Prepared &prepared) {
  auto *model = prepared.model.get();
  auto data = make_data(model);
  mj_resetDataKeyframe(model, data.get(), 0);
  prepared.task->Reset(model);
  require(prepared.task->num_residual == model->nv - 6 + model->nu + 99,
          "Unexpected custom tracking residual layout");
  auto snapshot = prepared.task->Residual();
  std::vector<double> live(prepared.task->num_residual);
  std::vector<double> frozen(prepared.task->num_residual);
  const std::vector<double> qpos(data->qpos, data->qpos + model->nq);
  const std::vector<double> qvel(data->qvel, data->qvel + model->nv);
  const std::vector<double> act(data->act, data->act + model->na);
  const std::vector<double> ctrl(data->ctrl, data->ctrl + model->nu);
  for (double time : {0., .017, .5, prepared.clip.duration() + 1.}) {
    data->time = time;
    prepared.task->Transition(model, data.get());
    require(std::equal(qpos.begin(), qpos.end(), data->qpos) &&
                std::equal(qvel.begin(), qvel.end(), data->qvel) &&
                std::equal(act.begin(), act.end(), data->act) &&
                std::equal(ctrl.begin(), ctrl.end(), data->ctrl),
            "Reference transition changed physical state or controls");
    mj_forward(model, data.get());
    prepared.task->Residual(model, data.get(), live.data());
    snapshot->Residual(model, data.get(), frozen.data());
    require(live == frozen, "Planner snapshot and live objective differ");
    require(std::all_of(live.begin(), live.end(),
                        [](double value) { return std::isfinite(value); }),
            "Non-finite residual");
  }
}

std::vector<mjtNum> marker_positions(const mjModel *model,
                                     const std::vector<int> &sites) {
  std::vector<mjtNum> values;
  values.reserve(3 * sites.size());
  for (int site : sites)
    values.insert(values.end(), model->site_pos + 3 * site,
                  model->site_pos + 3 * site + 3);
  return values;
}

void verify_qmc_sampling(const tracking::Prepared &nominal,
                         tracking::Prepared &qmc_variant) {
  const auto nominal_markers =
      tracking::qmc_marker_site_ids(nominal.model.get());
  const auto variant_markers =
      tracking::qmc_marker_site_ids(qmc_variant.model.get());
  require(!nominal_markers.empty(), "Nominal model has no QMC markers");
  require(nominal_markers.size() == variant_markers.size(),
          "QMC marker count changed across otherwise compatible models");
  require(nominal.metadata.at("qmc").at("index") == 0,
          "Nominal QMC index is not recorded");
  require(nominal.metadata.at("qmc").at("markers").at("marker_site_count") ==
              nominal_markers.size(),
          "Marker metadata does not match the compiled model");
  auto marker_data = make_data(qmc_variant.model.get());
  mj_forward(qmc_variant.model.get(), marker_data.get());
  for (const int site : variant_markers) {
    const int body = qmc_variant.model->site_bodyid[site];
    require(body > 0 &&
                mju_dist3(marker_data->site_xpos + 3 * site,
                          marker_data->xpos + 3 * body) < .6,
            "QMC marker is not attached near its parent body");
  }

  const auto nominal_positions =
      marker_positions(nominal.model.get(), nominal_markers);
  const auto variant_positions =
      marker_positions(qmc_variant.model.get(), variant_markers);
  require(nominal_positions != variant_positions,
          "QMC index did not change marker placement");
  require(qmc_variant.metadata.at("qmc").at("index") == 2 &&
              qmc_variant.metadata.at("geometry_randomized").get<bool>(),
          "QMC metadata is missing");
  require(qmc_variant.metadata.at("qmc").at("morphology").at("length_scale") !=
              1.0,
          "Positive QMC index did not produce a length draw");
  require(!std::equal(nominal.model->geom_size,
                      nominal.model->geom_size + 3 * nominal.model->ngeom,
                      qmc_variant.model->geom_size),
          "QMC index did not change geometry");

  const std::vector<mjtNum> body_mass(qmc_variant.model->body_mass,
                                      qmc_variant.model->body_mass +
                                          qmc_variant.model->nbody);
  const std::vector<mjtNum> geom_size(qmc_variant.model->geom_size,
                                      qmc_variant.model->geom_size +
                                          3 * qmc_variant.model->ngeom);
  tracking::place_qmc_marker_sites(qmc_variant.model.get(), 7);
  require(std::equal(body_mass.begin(), body_mass.end(),
                     qmc_variant.model->body_mass) &&
              std::equal(geom_size.begin(), geom_size.end(),
                         qmc_variant.model->geom_size),
          "Marker placement changed physical model parameters");
  require(marker_positions(qmc_variant.model.get(), variant_markers) !=
              variant_positions,
          "QMC marker placement did not respond to its index");
}

void verify_duration_resolution() {
  const tracking::Clip clip{7, 39};
  const auto full = tracking::resolve_duration(clip, 0);
  require(std::abs(full.requested - clip.duration()) < 1e-12 &&
              std::abs(full.effective - clip.duration()) < 1e-12 &&
              !full.capped,
          "Default duration did not select the full clip");
  const auto capped = tracking::resolve_duration(clip, clip.duration() + 10);
  require(std::abs(capped.requested - (clip.duration() + 10)) < 1e-12 &&
              std::abs(capped.effective - clip.duration()) < 1e-12 &&
              capped.capped,
          "Long duration was not capped at the clip end");
}

void verify_extra_foot_only_motions(const char *model_path,
                                    const mjModel *source) {
  for (const char *motion : {"jump", "dance", "kick_spin", "spin_kick"}) {
    require(tracking::is_foot_only_motion(motion),
            "Foot-only motion is missing from the catalog");
    tracking::PrepareConfig config;
    config.motion = motion;
    auto prepared = tracking::prepare_custom(model_path, config);
    const auto source_clip = tracking::find_clip(source, motion);
    require(prepared.clip.count == source_clip.count,
            "Prepared extra motion has the wrong frame count");
    verify_raw_identity(source, prepared.model.get(), source_clip.first);
  }

  tracking::PrepareConfig unsafe;
  unsafe.motion = "cartwheel1";
  bool rejected = false;
  try {
    (void)tracking::prepare_custom(model_path, unsafe);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "Hand-supported motion was accepted for foot-only data");
}
} // namespace

int main(int argc, char **argv) try {
  require(argc == 3, "Supply both baseline XMLs");
  verify_duration_resolution();
  char error[2048] = {};
  ModelPtr source(mj_loadXML(GRF_TRACKING_MODEL, nullptr, error, sizeof(error)),
                  mj_deleteModel);
  require(static_cast<bool>(source), error);
  const auto source_clip = tracking::find_clip(source.get(), "walk");

  for (int input = 1; input < argc; ++input) {
    tracking::PrepareConfig raw_config;
    raw_config.motion = "walk";
    auto raw = tracking::prepare_custom(argv[input], raw_config);
    auto *raw_model = raw.model.get();
    require(raw_model->nq == 32 && raw_model->nv == 31 && raw_model->na == 21,
            "Unexpected custom model state dimensions");
    require(raw_model->nmocap == 16 && raw_model->nkey == source_clip.count,
            "Wrong raw reference dimensions");
    require(std::isfinite(raw.metadata.at("initialization")
                              .at("initial_fit_rmse_m")
                              .get<double>()),
            "Raw initial fit is non-finite");
    verify_raw_identity(source.get(), raw_model, source_clip.first);
    verify_controller_weights(raw_model);

    tracking::PrepareConfig qmc_config = raw_config;
    qmc_config.qmc_index = 2;
    auto qmc_variant = tracking::prepare_custom(argv[input], qmc_config);
    verify_raw_identity(source.get(), qmc_variant.model.get(),
                        source_clip.first);
    verify_qmc_sampling(raw, qmc_variant);
    verify_task_contract(raw);
  }
  verify_extra_foot_only_motions(argv[1], source.get());
  std::cout << "Raw point-reference and task-contract checks passed.\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
