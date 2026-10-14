// Kinematic/runtime regression only: no mj_step, planner rollout or episode.
#include "tracking/reference_task.hpp"
#include "tracking/qmc_sampling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tracking;
using Json = nlohmann::json;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message,
          double tolerance = 1e-10) {
  check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
}

struct Fixture {
  std::filesystem::path path;
  ReferenceData reference;
  explicit Fixture(double translation = .01) {
    path = std::filesystem::temp_directory_path() /
           ("grf_reference_runtime_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ".json");
    // Synthetic targets, deliberately not drawn from any bundled motion.
    const double neutral[16][3] = {
        {0, 0, 1}, {.08, 0, 1.7}, {.12, .1, .03}, {.12, -.1, .03},
        {-.05, .1, .08}, {-.05, -.1, .08}, {0, .1, .5}, {0, -.1, .5},
        {.1, .5, 1.1}, {.1, -.5, 1.1}, {0, .35, 1.3}, {0, -.35, 1.3},
        {0, .18, 1.5}, {0, -.18, 1.5}, {0, .1, .95}, {0, -.1, .95}};
    Json positions = Json::array();
    for (int frame = 0; frame < 3; ++frame) {
      Json points = Json::array();
      for (const auto &point : neutral)
        points.push_back({point[0] + translation * frame, point[1], point[2]});
      positions.push_back(points);
    }
    const Json document = {
        {"schema_version", 1}, {"time_s", {0, .1, .4}},
        {"positions_m", positions}, {"target_names", kTargets},
        {"metadata", {{"source_id", "synthetic-runtime-test"}}}};
    std::ofstream output(path);
    output << document.dump();
    check(bool(output), "write fixture");
    reference = parse_reference(document);
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};

void assert_raw_targets(const mjModel *model, const ReferenceData &reference) {
  for (int key = 0; key < model->nkey; ++key) {
    near(model->key_time[key], reference.time_s[key], "native compiled key time", 0);
    for (std::size_t target = 0; target < kTargets.size(); ++target) {
      const int body = require_id(model, mjOBJ_BODY,
                                  std::string("mocap[") + kTargets[target] + "]");
      const int id = model->body_mocapid[body];
      for (int axis = 0; axis < 3; ++axis)
        near(model->key_mpos[3 * model->nmocap * key + 3 * id + axis],
             reference.positions_m[key][3 * target + axis],
             "source coordinates must stay unchanged", 0);
    }
  }
}

void objective_template(const mjModel *model) {
  const std::pair<const char *, double> numerics[] = {
      {"sampling_representation", 2}, {"agent_planner", 2},
      {"agent_horizon", .5}, {"agent_timestep", .005},
      {"sampling_spline_points", 16}, {"sampling_exploration", .15},
      {"sampling_trajectories", 32}, {"gradient_spline_points", 5},
      {"ilqg_num_rollouts", 16}, {"ilqg_regularization_type", 1},
      {"ilqg_representation", 2}};
  check(model->nnumeric == 11, "frozen default planner numerics");
  for (const auto &[name, value] : numerics) {
    const int id = require_id(model, mjOBJ_NUMERIC, name);
    check(model->numeric_size[id] == 1, "scalar planner numeric");
    near(model->numeric_data[model->numeric_adr[id]], value, name, 0);
  }
  struct ExpectedCost {
    const char *name;
    int dimension;
    std::array<double, 6> userdata;
  };
  // Pinned stock norm definitions, with the project's approved default weights.
  const ExpectedCost costs[] = {
      {"Joint Vel.", model->nv - 6, {0, .001, 0, .01, 0, 0}},
      {"Control", model->nu, {3, .1, 0, 1, .3, 0}},
      {"Pos[avg]", 3, {6, 100, 0, 100, .1, 0}},
      {"Pos[pelvis]", 3, {6, 40, 0, 100, .1, 0}},
      {"Pos[head]", 3, {6, 0, 0, 100, .1, 0}},
      {"Pos[toe]", 6, {7, 200, 0, 100, .2, 4}},
      {"Pos[heel]", 6, {7, 200, 0, 100, .2, 4}},
      {"Pos[knee]", 6, {6, 44, 0, 100, .1, 0}},
      {"Pos[hand]", 6, {6, 8, 0, 100, .1, 0}},
      {"Pos[elbow]", 6, {7, 16, 0, 100, .2, 4}},
      {"Pos[shoulder]", 6, {6, 8, 0, 100, .1, 0}},
      {"Pos[hip]", 6, {6, 40, 0, 100, .1, 0}},
      {"Vel[root]", 3, {6, .12, 0, 1, .3, 0}},
      {"Vel[head]", 3, {6, 0, 0, 1, .3, 0}},
      {"Vel[toe]", 6, {6, .25, 0, 1, .3, 0}},
      {"Vel[heel]", 6, {6, .25, 0, 1, .3, 0}},
      {"Vel[knee]", 6, {6, .18, 0, 1, .3, 0}},
      {"Vel[hand]", 6, {6, .02, 0, 1, .3, 0}},
      {"Vel[elbow]", 6, {6, .02, 0, 1, .3, 0}},
      {"Vel[shoulder]", 6, {6, .02, 0, 1, .3, 0}},
      {"Vel[hip]", 6, {6, .15, 0, 1, .3, 0}},
      {"Effort rate", model->nu, {0, 100, 0, 100, 0, 0}}};
  int index = 0;
  for (const auto &cost : costs) {
    const int id = require_id(model, mjOBJ_SENSOR, cost.name);
    check(id == index++, "unchanged cost sensor order");
    check(model->sensor_type[id] == mjSENS_USER, "cost sensor type");
    check(model->sensor_dim[id] == cost.dimension, "cost sensor dimension");
    for (int slot = 0; slot < 6; ++slot)
      near(model->sensor_user[id * model->nuser_sensor + slot], cost.userdata[slot],
           std::string("Default norm/weight: ") + cost.name, 0);
  }
}

void interpolation_and_initialization(Prepared &prepared, const Fixture &fixture) {
  auto *model = prepared.model.get();
  const auto &reference = fixture.reference;
  check(prepared.reference_id == "synthetic-runtime-test", "source id propagation");
  near(prepared.clip.duration(), .4, "native clip duration", 0);
  near(resolve_duration(prepared.clip, 0).effective, .4,
       "full duration follows native timestamps", 0);
  const auto capped = resolve_duration(prepared.clip, 1.0);
  near(capped.effective, .4, "requested duration caps at native endpoint", 0);
  check(capped.capped, "native-duration cap flag");
  near(resolve_duration(prepared.clip, .2).effective, .2, "partial duration", 0);
  near(prepared.metadata["initialization"]["initial_velocity_interval_s"], .1,
       "initialization records native interval", 0);
  check(prepared.metadata["reference_metadata"] == reference.metadata,
        "reference provenance preserved");
  check(!prepared.metadata["reference_coordinates_modified"].get<bool>(),
        "explicit unchanged coordinates");
  check(!prepared.metadata.contains("reference_fps"), "no invented fixed FPS");
  assert_raw_targets(model, reference);
  objective_template(model);

  // Both IK fits are identical when only timestamps change; native velocity
  // scales by dt, and dt=1/30 recovers the original initialization convention.
  const std::vector<double> native_velocity(model->key_qvel,
                                             model->key_qvel + model->nv);
  const std::vector<double> native_pose(model->key_qpos,
                                       model->key_qpos + model->nq);
  model->key_time[1] = 1 / 30.;
  const auto legacy = initialize_reference_tracking_state(
      model, reference.positions_m[0], reference.positions_m[1]);
  near(legacy["initial_velocity_interval_s"], 1 / 30., "legacy IK interval", 0);
  bool nonzero = false;
  for (int dof = 0; dof < model->nv; ++dof) {
    near(model->key_qvel[dof], native_velocity[dof] * 3,
         "native finite-difference IK velocity");
    nonzero = nonzero || std::abs(native_velocity[dof]) > 1e-5;
  }
  check(nonzero, "nontrivial initialization velocity fixture");
  for (int q = 0; q < model->nq; ++q)
    near(model->key_qpos[q], native_pose[q], "unchanged initial pose", 0);
  model->key_time[1] = .1;
  std::copy(native_velocity.begin(), native_velocity.end(), model->key_qvel);
  assert_raw_targets(model, reference);

  prepared.task->Reset(model);
  auto data = make_data(model);
  std::vector<double> residual(prepared.task->num_residual);
  for (double time : {-.1, 0., .05, .1, .25, .4, .5}) {
    data->time = time;
    prepared.task->Transition(model, data.get());
    prepared.task->Residual(model, data.get(), residual.data());
    const auto expected = sample_reference(reference, time);
    const int velocity_offset = model->nv - 6 + model->nu + 3 + 48;
    for (std::size_t target = 0; target < 16; ++target) {
      const std::string name = kTargets[target];
      const int body = require_id(model, mjOBJ_BODY, "mocap[" + name + "]");
      const int mocap = model->body_mocapid[body];
      const int sensor = require_id(model, mjOBJ_SENSOR,
                                    "tracking_linvel[" + name + "]");
      for (int axis = 0; axis < 3; ++axis) {
        near(data->mocap_pos[3 * mocap + axis],
             expected.positions_m[3 * target + axis], "runtime transition");
        near(residual[velocity_offset + 3 * target + axis],
             expected.velocities_m_per_s[3 * target + axis] -
                 data->sensordata[model->sensor_adr[sensor] + axis],
             "runtime residual velocity, including zero endpoint");
      }
    }
  }
  model->key_time[1] = 0;
  bool rejected = false;
  try {
    prepared.task->Reset(model);
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, "runtime rejects invalid compiled key times");
  model->key_time[1] = .1;
}

void native_stance_gate(const std::string &model_path, const Fixture &fixture) {
  PrepareConfig config;
  config.reference_path = fixture.path;
  config.stance_roll_weight = 1;
  config.stance_roll_still_time_s = .15;
  auto prepared = build_reference_tracking_model(model_path, config);
  auto *model = prepared.model.get();
  prepared.task->Reset(model);
  auto data = make_data(model);
  constexpr double roll = .2;
  for (const char *name : {"foot_left", "foot_right"}) {
    const int foot = require_id(model, mjOBJ_BODY, name);
    data->xmat[9 * foot + 7] = std::sin(roll);
    data->xmat[9 * foot + 8] = std::cos(roll);
  }
  const int sensor = require_id(model, mjOBJ_SENSOR, "Stance roll");
  std::vector<double> residual(prepared.task->num_residual);
  for (const auto &[time, gate] :
       {std::pair<double, double>{.1, 0}, {.25, .5}, {.4, 1}}) {
    data->time = time;
    prepared.task->Residual(model, data.get(), residual.data());
    for (int side = 0; side < 2; ++side)
      near(residual[model->sensor_adr[sensor] + side],
           std::sqrt(gate / 2) * roll, "native speed and sustained-stance gate");
  }
}
void foot_position_override(const char *xml, const Fixture &fixture) {
  PrepareConfig config;
  config.reference_path = fixture.path;
  auto baseline = build_reference_tracking_model(xml, config);
  near(baseline.metadata["foot_position_weight"], 200, "default foot provenance");
  for (double weight : {0.0, 100.0}) {
    config.foot_position_weight = weight;
    auto candidate = build_reference_tracking_model(xml, config);
    near(candidate.metadata["foot_position_weight"], weight, "foot provenance");
    auto *a = baseline.model.get();
    auto *b = candidate.model.get();
    check(a->nsensor == b->nsensor, "unchanged sensor layout");
    for (int id = 0; id < a->nsensor; ++id) {
      const std::string name = mj_id2name(a, mjOBJ_SENSOR, id);
      for (int slot = 0; slot < a->nuser_sensor; ++slot) {
        const bool changed = slot == 1 &&
            (name == "Pos[toe]" || name == "Pos[heel]");
        near(b->sensor_user[id * b->nuser_sensor + slot],
             changed ? weight : a->sensor_user[id * a->nuser_sensor + slot],
             "only foot position weights change", 0);
      }
    }
    assert_raw_targets(b, fixture.reference);
  }
  for (double weight : {-1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    config.foot_position_weight = weight;
    bool rejected = false;
    try { build_reference_tracking_model(xml, config); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "invalid foot weight rejected");
  }
}
void default_toe_geometry(const std::string &xml, const Fixture &fixture) {
  for (int index : {0, 7}) {
    PrepareConfig config;
    config.reference_path = fixture.path;
    config.qmc_index = index;
    auto prepared = build_reference_tracking_model(xml, config);
    auto *b = prepared.model.get();
    PrepareConfig approved = config;
    approved.knee_hip_position_scale = .8;
    approved.foot_position_weight = 200;
    approved.effort_rate_weight = 100;
    approved.elbow_position_weight = 16;
    approved.hand_position_weight = 8;
    approved.stance_roll_weight = 0;
    auto explicit_config = build_reference_tracking_model(xml, approved);
    check(prepared.metadata == explicit_config.metadata, "default/approved provenance parity");
    check(b->nsensor == explicit_config.model->nsensor, "default sensor layout parity");
    for (int i = 0; i < b->nsensor * b->nuser_sensor; ++i)
      near(b->sensor_user[i], explicit_config.model->sensor_user[i], "default weight parity", 0);
    objective_template(b);
    char error[1024] = {};
    SpecPtr spec(mj_parseXML(xml.c_str(), nullptr, error, sizeof(error)), mj_deleteSpec);
    check(bool(spec), error);
    apply_morphology_qmc(spec.get(), index);
    ModelPtr physical(mj_compile(spec.get(), nullptr), mj_deleteModel);
    check(bool(physical), "physical model compiles");
    auto *a = physical.get();
    near(prepared.metadata["knee_hip_position_scale"], .8, "knee/hip provenance");
    near(prepared.metadata["foot_position_weight"], 200, "foot provenance");
    check(prepared.metadata["toe_site_height_convention"] ==
          "highest paired capsule crown in toe-local z; x/y unchanged over gap, not surface",
          "toe height provenance");
    for (const auto &side : {std::string("left"), std::string("right")}) {
      const int i = require_id(b, mjOBJ_SITE, side == "left" ? "tracking[ltoe]" : "tracking[rtoe]");
      const int toe = require_id(b, mjOBJ_BODY, "toe_" + side);
      check(b->site_bodyid[i] == toe, "toe body attachment");
      near(b->site_pos[3*i], 0, "toe x unchanged", 0);
      near(b->site_pos[3*i+1], 0, "toe y unchanged", 0);
      double crown = -1e30;
      for (const auto &prefix : {"toe1_", "toe2_"}) {
        int g = require_id(a, mjOBJ_GEOM, prefix + side);
        double rot[9]; mju_quat2Mat(rot, a->geom_quat + 4*g);
        crown = std::max(crown, a->geom_pos[3*g+2] +
            std::abs(rot[8])*a->geom_size[3*g+1] + a->geom_size[3*g]);
      }
      near(b->site_pos[3*i+2], crown, "toe crown after morphology");
      const auto &metadata = prepared.metadata["toe_sites"][side];
      check(metadata["body"] == "toe_" + side, "toe attachment provenance");
      for (int axis = 0; axis < 3; ++axis)
        near(metadata["position_m"][axis], b->site_pos[3*i+axis], "toe position provenance", 0);
      const int heel = require_id(b, mjOBJ_SITE, side == "left" ? "tracking[lheel]" : "tracking[rheel]");
      const int foot = require_id(a, mjOBJ_BODY, "foot_" + side);
      const int g = require_id(a, mjOBJ_GEOM, "foot1_" + side);
      double rot[9]; mju_quat2Mat(rot, a->geom_quat + 4*g);
      check(b->site_bodyid[heel] == foot, "heel attachment unchanged");
      near(b->site_pos[3*heel], a->geom_pos[3*g] - std::abs(rot[2])*a->geom_size[3*g+1], "heel x unchanged");
      near(b->site_pos[3*heel+1], 0, "heel y unchanged", 0);
      near(b->site_pos[3*heel+2], .04 * -a->body_pos[3*foot+2] / .39, "heel z unchanged");
    }
    check(a->ngeom == b->ngeom, "physical geom count unchanged");
    for (int g = 0; g < a->ngeom; ++g) {
      for (int k = 0; k < 3; ++k) {
        near(a->geom_pos[3*g+k], b->geom_pos[3*g+k], "geom position unchanged", 0);
        near(a->geom_size[3*g+k], b->geom_size[3*g+k], "geom size unchanged", 0);
      }
      for (int k = 0; k < 4; ++k)
        near(a->geom_quat[4*g+k], b->geom_quat[4*g+k], "geom rotation unchanged", 0);
    }
    assert_raw_targets(b, fixture.reference);
  }
}
void aligned_initialization(mjModel *model) {
  auto data = make_data(model);
  // Exactly reachable neutral landmarks, far from the model origin and facing
  // +Y. Vertical translation keeps this test independent of floor correction.
  const double rotation[4] = {std::sqrt(.5), 0, 0, std::sqrt(.5)};
  double result[4];
  mju_mulQuat(result, rotation, data->qpos + 3);
  mju_copy(data->qpos + 3, result, 4);
  data->qpos[0] = 3;
  data->qpos[1] = -2;
  data->qpos[2] += 1;
  mj_forward(model, data.get());
  ReferencePoints targets;
  for (int i = 0; i < 16; ++i) {
    const int site = require_id(model, mjOBJ_SITE,
                               std::string("tracking[") + kTargets[i] + "]");
    mju_copy(targets.data() + 3 * i, data->site_xpos + 3 * site, 3);
  }
  const auto expected = targets;
  const auto info = initialize_reference_tracking_state(model, targets, targets);
  near(info["initial_fit_rmse_m"], 0, "root alignment recovers reachable pose", 1e-8);
  near(info["second_frame_fit_rmse_m"], 0, "stationary continuation fit", 1e-8);
  for (int i = 0; i < model->nv; ++i)
    near(model->key_qvel[i], 0, "stationary reference produces zero velocity", 1e-8);
  check(targets == expected, "initialization preserves source landmarks");
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::runtime_error("Usage: reference_runtime_test <physical_model.xml>");
  // Larger translation avoids a zero-progress second IK fit on either body.
  Fixture fixture(.05);
  PrepareConfig config;
  config.reference_path = fixture.path;
  auto prepared = build_reference_tracking_model(argv[1], config);
  interpolation_and_initialization(prepared, fixture);
  Fixture stance_fixture;
  native_stance_gate(argv[1], stance_fixture);
  foot_position_override(argv[1], fixture);
  default_toe_geometry(argv[1], fixture);
  aligned_initialization(prepared.model.get());
  std::cout << "Reference runtime/IK/template tests passed (no simulation)\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
