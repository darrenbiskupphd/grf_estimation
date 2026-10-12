#include "reference_task.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tracking {
namespace {
void clear_elements(mjSpec *spec, mjtObj type) {
  while (auto *element = mjs_firstElement(spec, type))
    mjs_delete(element);
}

double controller_weight(const std::string &name, double baseline_weight,
                         double knee_hip_position_scale,
                         double elbow_velocity_weight,
                         double elbow_position_weight,
                         double hand_position_weight) {
  // This is the frozen baseline objective, not a physical contact constraint.
  // Keep these settings beside the residual layout they weight.
  if (name == "Pos[toe]" || name == "Pos[heel]")
    return 80.0;
  if (name == "Vel[toe]" || name == "Vel[heel]")
    return 0.25;
  if (name == "Pos[knee]")
    return 55.0 * knee_hip_position_scale;
  if (name == "Vel[knee]")
    return 0.18;
  if (name == "Pos[hip]")
    return 50.0 * knee_hip_position_scale;
  if (name == "Vel[hip]")
    return 0.15;
  if (name == "Pos[pelvis]")
    return 40.0;
  if (name == "Vel[root]")
    return 0.12;
  if (name == "Pos[head]" || name == "Vel[head]")
    return 0.0;
  if (name == "Pos[hand]")
    return hand_position_weight;
  if (name == "Pos[elbow]")
    return elbow_position_weight;
  if (name == "Pos[shoulder]")
    return 8.0;
  if (name == "Vel[elbow]")
    return elbow_velocity_weight;
  if (name == "Vel[hand]" || name == "Vel[shoulder]")
    return 0.02;
  return baseline_weight;
}

// Frozen from MJPC humanoid/tracking/task.xml at revision
// ff572a21e7c2bf9fda62e1862a758da7e9a8719b (Apache-2.0, DeepMind).
// Only objective/planner configuration is retained, not model physics, source
// poses, keyframe names, or motion-selection text. controller_weight applies
// the same O overrides as before; the norm definitions and order are unchanged.
struct PlannerNumeric {
  const char *name;
  double value;
};
constexpr PlannerNumeric kPlannerNumerics[] = {
    {"sampling_representation", 2}, {"agent_planner", 2},
    {"agent_horizon", 0.5}, {"agent_timestep", 0.005},
    {"sampling_spline_points", 16}, {"sampling_exploration", 0.15},
    {"sampling_trajectories", 32}, {"gradient_spline_points", 5},
    {"ilqg_num_rollouts", 16}, {"ilqg_regularization_type", 1},
    {"ilqg_representation", 2}};
struct CostSensor {
  const char *name;
  int dimension;
  std::array<double, 6> userdata;
};
constexpr CostSensor kCostSensors[] = {
    {"Joint Vel.", 0, {0, 0.001, 0, 0.01, 0, 0}},
    {"Control", 0, {3, 0.1, 0, 1, 0.3, 0}},
    {"Pos[avg]", 3, {6, 100, 0, 100, 0.1, 0}},
    {"Pos[pelvis]", 3, {6, 30, 0, 100, 0.1, 0}},
    {"Pos[head]", 3, {6, 0, 0, 100, 0.1, 0}},
    {"Pos[toe]", 6, {7, 30, 0, 100, 0.2, 4}},
    {"Pos[heel]", 6, {7, 30, 0, 100, 0.2, 4}},
    {"Pos[knee]", 6, {6, 30, 0, 100, 0.1, 0}},
    {"Pos[hand]", 6, {6, 30, 0, 100, 0.1, 0}},
    {"Pos[elbow]", 6, {7, 30, 0, 100, 0.2, 4}},
    {"Pos[shoulder]", 6, {6, 30, 0, 100, 0.1, 0}},
    {"Pos[hip]", 6, {6, 30, 0, 100, 0.1, 0}},
    {"Vel[root]", 3, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[head]", 3, {6, 0, 0, 1, 0.3, 0}},
    {"Vel[toe]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[heel]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[knee]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[hand]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[elbow]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[shoulder]", 6, {6, 0.1, 0, 1, 0.3, 0}},
    {"Vel[hip]", 6, {6, 0.1, 0, 1, 0.3, 0}}};

double low_score(double value, double full, double off) {
  const double fraction = std::clamp((value - full) / (off - full), 0.0, 1.0);
  return 1.0 - fraction * fraction * (3.0 - 2.0 * fraction);
}
} // namespace

void add_reference_tracking_objective(mjSpec *spec, const mjModel *baseline,
                                      double effort_rate_weight,
                                      double stance_roll_weight,
                                      double knee_hip_position_scale,
                                      double stance_roll_still_time_s,
                                      double elbow_velocity_weight,
                                      double elbow_position_weight,
                                      double hand_position_weight) {
  if (!std::isfinite(effort_rate_weight) || effort_rate_weight < 0)
    throw std::runtime_error("Effort-rate weight must be finite and nonnegative");
  if (!std::isfinite(stance_roll_weight) || stance_roll_weight < 0)
    throw std::runtime_error("Stance-roll weight must be finite and nonnegative");
  if (!std::isfinite(knee_hip_position_scale) || knee_hip_position_scale < 0)
    throw std::runtime_error("Knee/hip position scale must be finite and nonnegative");
  if (!std::isfinite(stance_roll_still_time_s) || stance_roll_still_time_s < 0)
    throw std::runtime_error("Stance-roll still time must be finite and nonnegative");
  if (!std::isfinite(elbow_velocity_weight) || elbow_velocity_weight < 0)
    throw std::runtime_error("Elbow-velocity weight must be finite and nonnegative");
  if (!std::isfinite(elbow_position_weight) || elbow_position_weight < 0 ||
      !std::isfinite(hand_position_weight) || hand_position_weight < 0)
    throw std::runtime_error("Elbow/hand position weights must be finite and nonnegative");
  if (stance_roll_still_time_s > 0 && stance_roll_weight == 0)
    throw std::runtime_error(
        "Positive stance-roll still time requires a positive roll weight");
  // Replace inherited task configuration; the builder adds the selected clip's
  // keyframes afterward. Physical model settings are deliberately untouched.
  clear_elements(spec, mjOBJ_SENSOR);
  clear_elements(spec, mjOBJ_NUMERIC);
  clear_elements(spec, mjOBJ_TEXT);
  clear_elements(spec, mjOBJ_KEY);
  for (const auto &setting : kPlannerNumerics) {
    auto *numeric = mjs_addNumeric(spec);
    mjs_setString(numeric->name, setting.name);
    mjs_setDouble(numeric->data, &setting.value, 1);
  }
  // Omit the extra numeric at zero so existing objective/model layout is kept.
  if (stance_roll_weight > 0 && stance_roll_still_time_s > 0) {
    auto *numeric = mjs_addNumeric(spec);
    mjs_setString(numeric->name, "stance_roll_still_time_s");
    mjs_setDouble(numeric->data, &stance_roll_still_time_s, 1);
  }
  // Adjust the stock layout for the custom model's four passive DOFs.
  for (const auto &cost : kCostSensors) {
    auto *sensor = mjs_addSensor(spec);
    const std::string name = cost.name;
    mjs_setString(sensor->name, cost.name);
    sensor->type = mjSENS_USER;
    sensor->needstage = mjSTAGE_ACC;
    sensor->datatype = mjDATATYPE_REAL;
    sensor->dim = name == "Joint Vel." ? baseline->nv - 6
                  : name == "Control"  ? baseline->nu
                                       : cost.dimension;
    auto userdata = cost.userdata;
    userdata[1] = controller_weight(name, userdata[1], knee_hip_position_scale,
                                    elbow_velocity_weight,
                                    elbow_position_weight, hand_position_weight);
    mjs_setDouble(sensor->userdata, userdata.data(), userdata.size());
  }
  // All cost sensors must precede the physical sensors. Omit the term at
  // zero weight so the original objective/model layout stays unchanged.
  if (effort_rate_weight > 0) {
    for (int i = 0; i < baseline->nu; ++i) {
      if (baseline->actuator_dyntype[i] != mjDYN_FILTER ||
          baseline->actuator_actnum[i] != 1 ||
          baseline->actuator_actadr[i] < 0 ||
          !(baseline->actuator_dynprm[mjNDYN * i] > 0))
        throw std::runtime_error(
            "Effort-rate cost requires single-state filter actuators");
    }
    auto *sensor = mjs_addSensor(spec);
    mjs_setString(sensor->name, "Effort rate");
    sensor->type = mjSENS_USER;
    sensor->needstage = mjSTAGE_ACC;
    sensor->datatype = mjDATATYPE_REAL;
    sensor->dim = baseline->nu;
    // MJPC quadratic norm is 0.5 * sum(r_i^2).
    const double userdata[] = {0, effort_rate_weight, 0,
                               std::max(1.0, effort_rate_weight)};
    mjs_setDouble(sensor->userdata, userdata, 4);
  }
  if (stance_roll_weight > 0) {
    auto *sensor = mjs_addSensor(spec);
    mjs_setString(sensor->name, "Stance roll");
    sensor->type = mjSENS_USER;
    sensor->needstage = mjSTAGE_ACC;
    sensor->datatype = mjDATATYPE_REAL;
    sensor->dim = 2;
    const double userdata[] = {0, stance_roll_weight, 0,
                               std::max(1.0, stance_roll_weight)};
    mjs_setDouble(sensor->userdata, userdata, 4);
  }
  auto add_sensor = [&](mjtSensor type, const std::string &name,
                        mjtObj object_type, const std::string &object) {
    auto *s = mjs_addSensor(spec);
    mjs_setString(s->name, name.c_str());
    s->type = type;
    s->objtype = object_type;
    mjs_setString(s->objname, object.c_str());
  };
  add_sensor(mjSENS_FRAMEPOS, "trace0", mjOBJ_BODY, "torso");
  for (const char *target : kTargets) {
    const std::string suffix = std::string("[") + target + "]";
    add_sensor(mjSENS_FRAMEPOS, "tracking_pos" + suffix, mjOBJ_SITE,
               "tracking" + suffix);
    add_sensor(mjSENS_FRAMELINVEL, "tracking_linvel" + suffix, mjOBJ_SITE,
               "tracking" + suffix);
  }
}

void ReferenceTask::ResetLocked(const mjModel *model) {
  const int effort_sensor = mj_name2id(model, mjOBJ_SENSOR, "Effort rate");
  const int roll_sensor = mj_name2id(model, mjOBJ_SENSOR, "Stance roll");
  residual_.effort_rate_offset =
      effort_sensor < 0 ? -1 : model->sensor_adr[effort_sensor];
  residual_.stance_roll_offset =
      roll_sensor < 0 ? -1 : model->sensor_adr[roll_sensor];
  const int base_dimension = model->nv - 6 + model->nu + 99;
  const int effort_dimension = effort_sensor < 0 ? 0 : model->nu;
  const int roll_dimension = roll_sensor < 0 ? 0 : 2;
  if (model->nkey < 2 || model->nmocap != 16 || model->njnt == 0 ||
      model->jnt_type[0] != mjJNT_FREE ||
      num_residual != base_dimension + effort_dimension + roll_dimension ||
      (effort_sensor >= 0 &&
       (residual_.effort_rate_offset != base_dimension ||
        model->sensor_dim[effort_sensor] != model->nu)) ||
      (roll_sensor >= 0 &&
       (residual_.stance_roll_offset != base_dimension + effort_dimension ||
        model->sensor_dim[roll_sensor] != 2)))
    throw std::runtime_error(
        "Reference tracking requires a free root and a 16-target clip");
  validate_reference_times(model->key_time, model->nkey);
  if (model->key_time[0] != 0)
    throw std::runtime_error("Reference key times must start at normalized zero");
  for (size_t i = 0; i < kTargets.size(); ++i) {
    const std::string name = kTargets[i];
    residual_.positions[i] = model->sensor_adr[require_id(
        model, mjOBJ_SENSOR, "tracking_pos[" + name + "]")];
    residual_.velocities[i] = model->sensor_adr[require_id(
        model, mjOBJ_SENSOR, "tracking_linvel[" + name + "]")];
    residual_.mocap[i] = model->body_mocapid[require_id(model, mjOBJ_BODY,
                                                        "mocap[" + name + "]")];
  }
  residual_.stance_gate.clear();
  if (roll_sensor >= 0) {
    residual_.foot_body = {
        require_id(model, mjOBJ_BODY, "foot_left"),
        require_id(model, mjOBJ_BODY, "foot_right")};
    // The gate is reference-only: the optimizer cannot suppress it by
    // unloading a foot or changing the simulated contact pattern.
    constexpr std::array<std::array<int, 2>, 2> foot_targets{{{2, 4}, {3, 5}}};
    const int width = 3 * model->nmocap;
    auto point = [&](int key, int target, int axis) {
      return model->key_mpos[width * key + 3 * residual_.mocap[target] + axis];
    };
    std::array<double, 16> minimum_height;
    minimum_height.fill(std::numeric_limits<double>::infinity());
    for (int key = 0; key < model->nkey; ++key)
      for (const auto &foot : foot_targets)
        for (int target : foot)
          minimum_height[target] =
              std::min(minimum_height[target], point(key, target, 2));
    residual_.stance_gate.resize(model->nkey);
    for (int key = 0; key < model->nkey; ++key) {
      const int previous = std::max(0, key - 1);
      const int next = std::min(model->nkey - 1, key + 1);
      for (int side = 0; side < 2; ++side) {
        double gate = 1.0;
        for (int target : foot_targets[side]) {
          const double height = point(key, target, 2) - minimum_height[target];
          const double dx = point(next, target, 0) - point(previous, target, 0);
          const double dy = point(next, target, 1) - point(previous, target, 1);
          const double speed = std::hypot(dx, dy) /
                               (model->key_time[next] - model->key_time[previous]);
          gate = std::min(gate,
                          low_score(height, kStanceHeightFull, kStanceHeightOff) *
                              low_score(speed, kStanceSpeedFull, kStanceSpeedOff));
        }
        residual_.stance_gate[key][side] = gate;
      }
    }
    const int still_numeric =
        mj_name2id(model, mjOBJ_NUMERIC, "stance_roll_still_time_s");
    if (still_numeric >= 0) {
      if (model->numeric_size[still_numeric] != 1)
        throw std::runtime_error("Invalid stance-roll still-time numeric");
      const double seconds =
          model->numeric_data[model->numeric_adr[still_numeric]];
      if (!std::isfinite(seconds) || seconds < 0)
        throw std::runtime_error("Invalid stance-roll still time");
      const auto instantaneous_gate = residual_.stance_gate;
      for (int key = 0; key < model->nkey; ++key) {
        const auto first = reference_history_start(model->key_time, model->nkey,
                                                   key, seconds);
        for (int side = 0; side < 2; ++side) {
          if (!first) {
            residual_.stance_gate[key][side] = 0;
            continue;
          }
          double sustained = 1;
          for (std::size_t earlier = *first; earlier <= std::size_t(key); ++earlier)
            sustained = std::min(sustained, instantaneous_gate[earlier][side]);
          residual_.stance_gate[key][side] = sustained;
        }
      }
    }
  }
}

void ReferenceTask::TransitionLocked(mjModel *model, mjData *data) {
  const auto sample =
      reference_interpolation(model->key_time, model->nkey, data->time);
  const int width = 3 * model->nmocap;
  for (int i = 0; i < width; ++i)
    data->mocap_pos[i] = sample.position(
        model->key_mpos[width * sample.first + i],
        model->key_mpos[width * sample.next + i]);
}

void ReferenceTask::ResidualFn::Residual(const mjModel *model,
                                         const mjData *data,
                                         double *residual) const {
  const auto sample =
      reference_interpolation(model->key_time, model->nkey, data->time);
  int offset = model->nv - 6;
  mju_copy(residual, data->qvel + 6, offset);
  mju_copy(residual + offset, data->ctrl, model->nu);
  offset += model->nu;
  double errors[48], average[3] = {};
  const int width = 3 * model->nmocap;
  for (size_t i = 0; i < kTargets.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const int point = 3 * mocap[i] + axis;
      const double p0 = model->key_mpos[width * sample.first + point];
      const double p1 = model->key_mpos[width * sample.next + point];
      errors[3 * i + axis] = sample.position(p0, p1) -
                             data->sensordata[positions[i] + axis];
      average[axis] += errors[3 * i + axis] / kTargets.size();
      residual[offset + 3 + 48 + 3 * i + axis] =
          sample.velocity(p0, p1) - data->sensordata[velocities[i] + axis];
    }
  }
  mju_copy3(residual + offset, average);
  for (int i = 0; i < 48; ++i)
    residual[offset + 3 + i] = errors[i] - average[i % 3];
  if (effort_rate_offset >= 0) {
    const double scale = kEffortRateScale * std::sqrt(double(model->nu));
    for (int i = 0; i < model->nu; ++i) {
      double command = data->ctrl[i];
      if (model->actuator_ctrllimited[i])
        command = std::clamp(command, model->actuator_ctrlrange[2 * i],
                             model->actuator_ctrlrange[2 * i + 1]);
      const double activation = data->act[model->actuator_actadr[i]];
      const double time_constant = model->actuator_dynprm[mjNDYN * i];
      residual[effort_rate_offset + i] =
          (command - activation) / (time_constant * scale);
    }
  }
  if (stance_roll_offset >= 0) {
    for (int side = 0; side < 2; ++side) {
      const double gate =
          (1 - sample.fraction) * stance_gate[sample.first][side] +
          sample.fraction * stance_gate[sample.next][side];
      const double *rotation = data->xmat + 9 * foot_body[side];
      // World vertical components of foot-local lateral and normal axes:
      // yaw-invariant lateral roll, with no direct heel/toe pitch cost.
      const double roll = std::atan2(rotation[7], rotation[8]);
      residual[stance_roll_offset + side] = std::sqrt(gate / 2.0) * roll;
    }
  }
}
} // namespace tracking
