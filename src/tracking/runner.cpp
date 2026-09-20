#include "tracking.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "mjpc/agent.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mujoco_raii.hpp"
#include "residual_callback.hpp"
#include "run_bundle.hpp"
#include "viewer.hpp"

namespace tracking {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr double kContactThreshold = 1.0;
constexpr double kJointLimitTolerance = 1e-3;
constexpr double kForefootBoundaryM = .02;
constexpr double kLateralEdgeBoundaryM = .03;

Clip find_clip(const mjModel *model, const std::string &motion) {
  Clip clip;
  const std::string prefix = motion + "_";
  for (int key = 0; key < model->nkey; ++key) {
    const char *name = mj_id2name(model, mjOBJ_KEY, key);
    if (name && std::string(name).rfind(prefix, 0) == 0) {
      if (clip.first < 0)
        clip.first = key;
      if (key != clip.first + clip.count)
        throw std::runtime_error("Non-contiguous motion keyframes");
      ++clip.count;
    }
  }
  if (clip.count < 2)
    throw std::runtime_error("Reference clip not found");
  return clip;
}

int require_id(const mjModel *model, mjtObj type, const std::string &name) {
  const int id = mj_name2id(model, type, name.c_str());
  if (id < 0)
    throw std::runtime_error("Tracking model is missing " + name);
  return id;
}

namespace {
struct FootIds {
  int body = -1;
  std::array<int, 4> geom{};
  int ankle_y = -1;
  int ankle_x = -1;
  int mtp = -1;
};

struct Sample {
  double position_mse = 0;
  std::array<double, 16> target_squared_error{};
  double slip_speed = 0;
  std::array<double, 2> foot_load{};
  bool loaded = false;
  bool fall = false;
  ReplayFrame frame;
};

// Evaluate observations on a data copy: mj_step advances qpos/time after
// computing contacts. A fresh forward pass on the copy aligns all diagnostics
// to the recorded timestamp without altering the simulation's solver state.
class Measurements {
public:
  explicit Measurements(const mjModel *model)
      : scratch_(make_data(model)), jacobian_(3 * model->nv),
        floor_(require_id(model, mjOBJ_GEOM, "floor")) {
    for (const char *name : kTargets) {
      sites_.push_back(
          require_id(model, mjOBJ_SITE, std::string("tracking[") + name + "]"));
      const int body =
          require_id(model, mjOBJ_BODY, std::string("mocap[") + name + "]");
      mocap_.push_back(model->body_mocapid[body]);
      if (mocap_.back() < 0)
        throw std::runtime_error("Tracking target is not a mocap body");
    }
    for (int side = 0; side < 2; ++side) {
      const std::string name = side == 0 ? "left" : "right";
      auto &foot = feet_[side];
      foot.body = require_id(model, mjOBJ_BODY, "foot_" + name);
      for (int slot = 0; slot < 4; ++slot) {
        const std::string prefix = slot < 2 ? "foot" : "toe";
        foot.geom[slot] = require_id(
            model, mjOBJ_GEOM,
            prefix + std::to_string(slot % 2 + 1) + "_" + name);
      }
      foot.ankle_y = require_id(model, mjOBJ_JOINT, "ankle_y_" + name);
      foot.ankle_x = require_id(model, mjOBJ_JOINT, "ankle_x_" + name);
      foot.mtp = require_id(model, mjOBJ_JOINT, "mtp_" + name);
    }
  }

  Sample take(const mjModel *model, const mjData *data, bool capture) {
    mjData *d = scratch_.get();
    mj_copyData(d, model, data);
    mj_forward(model, d);
    Sample sample;
    auto &frame = sample.frame;
    frame.time = d->time;
    if (capture) {
      frame.qpos.assign(d->qpos, d->qpos + model->nq);
      frame.qvel.assign(d->qvel, d->qvel + model->nv);
      frame.mocap_pos.assign(d->mocap_pos, d->mocap_pos + 3 * model->nmocap);
      frame.mocap_quat.assign(d->mocap_quat, d->mocap_quat + 4 * model->nmocap);
      frame.act.assign(d->act, d->act + model->na);
      frame.ctrl.assign(d->ctrl, d->ctrl + model->nu);
    }
    for (size_t i = 0; i < sites_.size(); ++i) {
      const double *position = d->site_xpos + 3 * sites_[i];
      const double *target = d->mocap_pos + 3 * mocap_[i];
      for (int axis = 0; axis < 3; ++axis) {
        sample.target_squared_error[i] +=
            std::pow(position[axis] - target[axis], 2);
      }
      sample.position_mse += sample.target_squared_error[i] / sites_.size();
    }

    double slip_load = 0;
    std::array<std::array<double, 3>, 2> local_contact_sum{};
    for (int i = 0; i < d->ncon; ++i) {
      const auto &contact = d->contact[i];
      if (contact.geom[0] != floor_ && contact.geom[1] != floor_)
        continue;
      const int other =
          contact.geom[0] == floor_ ? contact.geom[1] : contact.geom[0];
      if (other < 0)
        continue;
      const int body = model->geom_bodyid[other];
      double wrench[6], force[3];
      mj_contactForce(model, d, i, wrench);
      mju_mulMatTVec3(force, contact.frame, wrench);
      if (contact.geom[1] == floor_)
        mju_scl3(force, force, -1);
      const int side = foot_side(model, body);
      if (side < 0) {
        sample.fall = sample.fall || force[2] > kContactThreshold;
        continue;
      }

      auto &diagnostic = side == 0 ? frame.left_contact : frame.right_contact;
      auto &grf = side == 0 ? frame.grf_left : frame.grf_right;
      auto &cop = side == 0 ? frame.cop_left : frame.cop_right;
      mju_addTo3(grf.data(), force);
      ++diagnostic.contact_count;
      if (force[2] <= 0)
        continue;

      sample.foot_load[side] += force[2];
      mju_addToScl3(cop.data(), contact.pos, force[2]);
      const auto local = contact_local_position(d, feet_[side].body, contact.pos);
      for (int axis = 0; axis < 3; ++axis)
        local_contact_sum[side][axis] += force[2] * local[axis];
      const int slot = geom_slot(side, other);
      if (slot >= 0)
        diagnostic.geom_normal_load[slot] += force[2];
      if (slot >= 2)
        diagnostic.toe_normal_load += force[2];
      else if (local[0] < kForefootBoundaryM)
        diagnostic.heel_normal_load += force[2];
      else
        diagnostic.forefoot_normal_load += force[2];
      if (std::abs(local[1]) > kLateralEdgeBoundaryM)
        diagnostic.lateral_edge_normal_load += force[2];

      if (force[2] > kContactThreshold) {
        double velocity[3];
        mj_jac(model, d, jacobian_.data(), nullptr, contact.pos, body);
        mju_mulMatVec(velocity, jacobian_.data(), d->qvel, 3, model->nv);
        sample.slip_speed += force[2] * std::hypot(velocity[0], velocity[1]);
        slip_load += force[2];
      }
    }
    for (int side = 0; side < 2; ++side) {
      auto &diagnostic = side == 0 ? frame.left_contact : frame.right_contact;
      auto &cop = side == 0 ? frame.cop_left : frame.cop_right;
      if (sample.foot_load[side] > 0) {
        mju_scl3(cop.data(), cop.data(), 1 / sample.foot_load[side]);
        for (int axis = 0; axis < 3; ++axis)
          diagnostic.contact_point_local[axis] =
              local_contact_sum[side][axis] / sample.foot_load[side];
      }
      populate_foot_kinematics(model, d, side, diagnostic);
    }
    sample.loaded =
        sample.foot_load[0] + sample.foot_load[1] > kContactThreshold;
    if (slip_load > 0)
      sample.slip_speed /= slip_load;
    return sample;
  }

private:
  int foot_side(const mjModel *model, int body) const {
    while (body > 0) {
      if (body == feet_[0].body)
        return 0;
      if (body == feet_[1].body)
        return 1;
      body = model->body_parentid[body];
    }
    return -1;
  }

  int geom_slot(int side, int geom) const {
    const auto &geoms = feet_[side].geom;
    const auto found = std::find(geoms.begin(), geoms.end(), geom);
    return found == geoms.end() ? -1
                                : static_cast<int>(found - geoms.begin());
  }

  static std::array<double, 3>
  contact_local_position(const mjData *data, int body, const double *world) {
    double offset[3] = {world[0] - data->xpos[3 * body],
                        world[1] - data->xpos[3 * body],
                        world[2] - data->xpos[3 * body + 2]};
    std::array<double, 3> local{};
    mju_mulMatTVec3(local.data(), data->xmat + 9 * body, offset);
    return local;
  }

  void populate_foot_kinematics(const mjModel *model, const mjData *data,
                                int side, FootContactFrame &diagnostic) const {
    const auto &foot = feet_[side];
    const double *rotation = data->xmat + 9 * foot.body;
    diagnostic.roll = std::atan2(rotation[7], rotation[8]);
    diagnostic.pitch =
        std::atan2(-rotation[6], std::hypot(rotation[7], rotation[8]));
    diagnostic.ankle_y = data->qpos[model->jnt_qposadr[foot.ankle_y]];
    diagnostic.ankle_x = data->qpos[model->jnt_qposadr[foot.ankle_x]];
    diagnostic.mtp = data->qpos[model->jnt_qposadr[foot.mtp]];
    diagnostic.mtp_velocity = data->qvel[model->jnt_dofadr[foot.mtp]];
  }

  DataPtr scratch_;
  std::vector<double> jacobian_;
  int floor_;
  std::array<FootIds, 2> feet_{};
  std::vector<int> sites_, mocap_;
};

bool numerical_failure(const mjModel *model, const mjData *data) {
  for (int warning :
       {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC, mjWARN_BADCTRL}) {
    if (data->warning[warning].number)
      return true;
  }
  return !std::all_of(data->qpos, data->qpos + model->nq,
                      [](double x) { return std::isfinite(x); }) ||
         !std::all_of(data->qvel, data->qvel + model->nv,
                      [](double x) { return std::isfinite(x); });
}

Json model_numerics(const mjModel *model) {
  Json result = Json::object();
  for (int i = 0; i < model->nnumeric; ++i) {
    const char *name = mj_id2name(model, mjOBJ_NUMERIC, i);
    if (name)
      result[name] = std::vector<double>(
          model->numeric_data + model->numeric_adr[i],
          model->numeric_data + model->numeric_adr[i] +
              model->numeric_size[i]);
  }
  return result;
}

Json warning_counts(const mjData *data) {
  Json result = Json::array();
  for (int i = 0; i < mjNWARNING; ++i)
    result.push_back(data->warning[i].number);
  return result;
}

struct JointLimitAccumulator {
  explicit JointLimitAccumulator(const mjModel *model) {
    for (int joint = 1; joint < model->njnt; ++joint) {
      if (!model->jnt_limited[joint])
        continue;
      const char *name = mj_id2name(model, mjOBJ_JOINT, joint);
      if (!name)
        continue;
      joints.push_back(joint);
      names.emplace_back(name);
      lower_time.push_back(0);
      upper_time.push_back(0);
    }
  }

  void sample(const mjModel *model, const mjData *data, double interval) {
    for (size_t index = 0; index < joints.size(); ++index) {
      const int joint = joints[index];
      const double position = data->qpos[model->jnt_qposadr[joint]];
      const double lower = model->jnt_range[2 * joint];
      const double upper = model->jnt_range[2 * joint + 1];
      if (position <= lower + kJointLimitTolerance)
        lower_time[index] += interval;
      if (position >= upper - kJointLimitTolerance)
        upper_time[index] += interval;
    }
  }

  Json report(double measured_time) const {
    Json result = Json::object();
    for (size_t index = 0; index < joints.size(); ++index) {
      result[names[index]] = {
          {"lower_fraction", measured_time > 0 ? lower_time[index] / measured_time
                                                : 0.0},
          {"upper_fraction", measured_time > 0 ? upper_time[index] / measured_time
                                                : 0.0}};
    }
    return result;
  }

  std::vector<int> joints;
  std::vector<std::string> names;
  std::vector<double> lower_time, upper_time;
};

Json foot_diagnostic_report(
    const std::array<double, 2> &loaded_time,
    const std::array<int, 2> &transitions,
    const std::array<std::array<double, 4>, 2> &geom_impulse,
    const std::array<std::array<double, 4>, 2> &region_impulse,
    double measured_time) {
  Json result = Json::object();
  for (int side = 0; side < 2; ++side) {
    const auto name = side == 0 ? "left" : "right";
    result[name] = {
        {"loaded_fraction", measured_time > 0 ? loaded_time[side] / measured_time
                                              : 0.0},
        {"contact_transitions", transitions[side]},
        {"normal_impulse_ns",
         {{"foot1", geom_impulse[side][0]},
          {"foot2", geom_impulse[side][1]},
          {"toe1", geom_impulse[side][2]},
          {"toe2", geom_impulse[side][3]}}},
        {"region_normal_impulse_ns",
         {{"heel", region_impulse[side][0]},
          {"forefoot", region_impulse[side][1]},
          {"toe", region_impulse[side][2]},
          {"lateral_edge", region_impulse[side][3]}}}};
  }
  return result;
}

} // namespace

void run(Prepared prepared, const RunConfig &config) {
  std::cout << prepared.label << "\n" << std::flush;
  if (config.output_path.empty())
    throw std::runtime_error("Run requires an explicit --output path");
  if (config.plan_every_n_steps < 1 || config.planner_threads < 1 ||
      config.warmup_iterations < 1 || config.replay_every_n_steps < 1)
    throw std::runtime_error("Invalid tracking configuration");
  if (!std::isfinite(config.duration) || config.duration < 0 ||
      !std::isfinite(config.physics_timestep) || config.physics_timestep < 0)
    throw std::runtime_error("Duration and timestep must be finite and nonnegative");
  if (std::filesystem::exists(config.output_path))
    throw std::runtime_error("Output file already exists: " +
                             config.output_path.string());

  auto *model = prepared.model.get();
  const Clip clip = prepared.clip;
  const double duration =
      config.duration == 0 ? clip.duration() : config.duration;
  if (duration <= 0 || duration > clip.duration() + 1e-9)
    throw std::runtime_error("Duration must fit inside the non-looping reference");
  if (config.physics_timestep > 0)
    model->opt.timestep = config.physics_timestep;
  const double dt = model->opt.timestep;
  if (dt > duration || duration / dt > 1'000'000)
    throw std::runtime_error(
        "Choose a timestep spanning 1 to 1,000,000 steps per episode");
  const int steps = static_cast<int>(std::ceil(duration / dt - 1e-9));
  if (steps < 1)
    throw std::runtime_error("Duration must span at least one physics step");

  auto data = make_data(model);
  auto *d = data.get();
  auto task = prepared.task;
  mjpc::Agent agent(model, task);
  agent.estimator_enabled = false;
  agent.plan_enabled = true;
  ResidualSensorScope residual_callback(agent, true);
  mjpc::ThreadPool pool(config.planner_threads);
  mj_resetDataKeyframe(model, d, clip.first);
  task->Transition(
      model, d); // Initialization only; subsequent transitions cannot reset state.
  mj_forward(model, d);
  Measurements measurements(model);
  std::vector<ReplayFrame> frames;
  frames.reserve(steps / config.replay_every_n_steps + 2);

  int planning_iterations = 0, action_evaluations = 0, resets = 0;
  double planning_seconds = 0;
  const auto start = Clock::now();
  auto plan = [&] {
    agent.state.Set(model, d);
    const auto plan_start = Clock::now();
    agent.PlanIteration(&pool);
    planning_seconds +=
        std::chrono::duration<double>(Clock::now() - plan_start).count();
    ++planning_iterations;
  };
  for (int i = 0; i < config.warmup_iterations; ++i)
    plan();

  std::string termination = "duration_reached";
  double error_integral = 0, slip_integral = 0, loaded_time = 0,
         flight_time = 0;
  std::array<double, 16> target_error_integral{};
  std::array<double, 2> foot_loaded_time{};
  std::array<int, 2> contact_transitions{};
  std::array<bool, 2> previous_foot_loaded{};
  std::array<std::array<double, 4>, 2> geom_impulse{};
  std::array<std::array<double, 4>, 2> region_impulse{};
  JointLimitAccumulator joint_limits(model);
  double max_error = 0, max_control = 0, max_activation = 0, elapsed = 0,
         first_fall = -1;
  int completed_steps = 0;
  std::vector<double> previous_qpos(model->nq), previous_qvel(model->nv);

  for (int step = 0; step <= steps; ++step) {
    if (step > 0) {
      std::copy(d->qpos, d->qpos + model->nq, previous_qpos.begin());
      std::copy(d->qvel, d->qvel + model->nv, previous_qvel.begin());
      task->Transition(model, d); // Refresh mocap targets at the current time.
      if (!std::equal(previous_qpos.begin(), previous_qpos.end(), d->qpos) ||
          !std::equal(previous_qvel.begin(), previous_qvel.end(), d->qvel))
        ++resets;
    }
    if (resets > 0) {
      termination = "unexpected_state_reset";
      break;
    }
    const bool replan =
        step > 0 && step < steps && step % config.plan_every_n_steps == 0;
    if (replan)
      plan();
    if (step < steps) {
      agent.state.Set(model, d);
      agent.ActivePlanner().ActionFromPolicy(
          d->ctrl, agent.state.state().data(), d->time, false);
      ++action_evaluations;
    }

    const bool capture =
        step % config.replay_every_n_steps == 0 || step == steps;
    auto sample = measurements.take(model, d, capture);
    const double rmse = std::sqrt(sample.position_mse);
    max_error = std::max(max_error, rmse);
    if (capture)
      frames.push_back(sample.frame);
    else if (sample.fall)
      frames.push_back(measurements.take(model, d, true).frame);
    elapsed = d->time;
    if (sample.fall) {
      first_fall = d->time;
      termination = "nonfoot_floor_contact";
      break;
    }
    if (step == steps)
      break;

    const double before = d->time;
    const double interval = dt;
    error_integral += sample.position_mse * interval;
    for (int target = 0; target < 16; ++target)
      target_error_integral[target] +=
          sample.target_squared_error[target] * interval;
    const std::array<bool, 2> foot_loaded{
        sample.foot_load[0] > kContactThreshold,
        sample.foot_load[1] > kContactThreshold};
    for (int side = 0; side < 2; ++side) {
      if (step > 0 && foot_loaded[side] != previous_foot_loaded[side])
        ++contact_transitions[side];
      previous_foot_loaded[side] = foot_loaded[side];
      if (foot_loaded[side])
        foot_loaded_time[side] += interval;
      const auto &diagnostic =
          side == 0 ? sample.frame.left_contact : sample.frame.right_contact;
      for (int geom = 0; geom < 4; ++geom)
        geom_impulse[side][geom] += diagnostic.geom_normal_load[geom] * interval;
      region_impulse[side][0] += diagnostic.heel_normal_load * interval;
      region_impulse[side][1] += diagnostic.forefoot_normal_load * interval;
      region_impulse[side][2] += diagnostic.toe_normal_load * interval;
      region_impulse[side][3] += diagnostic.lateral_edge_normal_load * interval;
    }
    joint_limits.sample(model, d, interval);
    for (int actuator = 0; actuator < model->nu; ++actuator)
      max_control = std::max(max_control, std::abs(d->ctrl[actuator]));
    for (int activation = 0; activation < model->na; ++activation)
      max_activation = std::max(max_activation, std::abs(d->act[activation]));
    if (sample.loaded) {
      slip_integral += sample.slip_speed * interval;
      loaded_time += interval;
    } else {
      flight_time += interval;
    }
    mj_step(model, d);
    if (numerical_failure(model, d) || d->time <= before) {
      termination = "numerical_failure";
      break;
    }
    ++completed_steps;
    if (static_cast<int>(d->time) > static_cast<int>(before)) {
      std::cout << config.motion << ": " << d->time
                << " s, tracking RMSE " << rmse << " m\n"
                << std::flush;
    }
  }

  const double wall_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  const double measured_time = loaded_time + flight_time;
  const bool completed =
      termination == "duration_reached" && completed_steps == steps;
  Json per_target_rmse = Json::object();
  for (int target = 0; target < 16; ++target)
    per_target_rmse[kTargets[target]] =
        measured_time > 0 ? std::sqrt(target_error_integral[target] / measured_time)
                          : 0.0;
  Json report = {
      {"schema_version", 1},
      {"motion", config.motion},
      {"mujoco_version", mj_versionString()},
      {"mjpc_revision", GRF_MJPC_REVISION},
      {"model_source", prepared.source},
      {"reference_first_key", clip.first},
      {"reference_frames", clip.count},
      {"reference_fps", kReferenceFps},
      {"reference_duration_s", clip.duration()},
      {"requested_duration_s", duration},
      {"simulated_duration_s", elapsed},
      {"physics_timestep_s", dt},
      {"planning_interval_s", dt * config.plan_every_n_steps},
      {"plan_every_n_steps", config.plan_every_n_steps},
      {"planner_threads", config.planner_threads},
      {"warmup_iterations", config.warmup_iterations},
      {"planning_iterations", planning_iterations},
      {"action_evaluations", action_evaluations},
      {"completed_steps", completed_steps},
      {"model_numerics", model_numerics(model)},
      {"initial_state_resets", 1},
      {"state_resets_after_initialization", resets},
      {"termination", termination},
      {"completed_requested_duration", completed},
      {"completed_reference_clip",
       completed && duration >= clip.duration() - 1e-9},
      {"first_fall_time_s", first_fall < 0 ? Json(nullptr) : Json(first_fall)},
      {"tracking_rmse_m", measured_time > 0
                              ? Json(std::sqrt(error_integral / measured_time))
                              : Json(nullptr)},
      {"per_target_tracking_rmse_m", per_target_rmse},
      {"max_frame_tracking_rmse_m", max_error},
      {"contact_slip_mean_m_per_s",
       loaded_time > 0 ? Json(slip_integral / loaded_time) : Json(nullptr)},
      {"flight_time_s", flight_time},
      {"loaded_time_s", loaded_time},
      {"joint_limit_occupancy", joint_limits.report(measured_time)},
      {"peak_abs_control", max_control},
      {"peak_abs_activation", max_activation},
      {"foot_contact_diagnostics",
       foot_diagnostic_report(foot_loaded_time, contact_transitions,
                              geom_impulse, region_impulse, measured_time)},
      {"wall_time_s", wall_seconds},
      {"planning_wall_time_s", planning_seconds},
      {"simulated_seconds_per_wall_second",
       wall_seconds > 0 ? elapsed / wall_seconds : 0.0},
      {"mujoco_warning_counts", warning_counts(d)},
      {"metric_definitions",
       {{"tracking", "Time-weighted RMS Euclidean error across 16 tracking "
                     "sites in world coordinates"},
        {"per_target_tracking",
         "Time-weighted RMS Euclidean error for each named tracking site"},
        {"slip", "Mean over loaded time of vertical-load-weighted horizontal "
                 "contact-point speed; individual contact Fz > 1 N"},
        {"fall",
         "Non-foot floor contact with upward world force greater than 1 N"},
        {"foot_contact",
         "Per-frame GRF, world contact-position proxy, per-geometry loads, "
         "foot-frame contact point, foot attitude, ankle and MTP state"},
        {"replay_arrows", "Visualization-only force arrows begin at the "
                           "vertical-force-weighted contact-position proxy; "
                           "they are not validated CoP labels"}}}};
  report["label"] = prepared.label;
  report["replay_every_n_steps"] = config.replay_every_n_steps;
  report["preparation"] = prepared.metadata;
  save_run_bundle(config.output_path, model, frames, report);
  std::cout << config.motion << ": " << termination << " at " << elapsed
            << " s; RMSE " << report.at("tracking_rmse_m") << " m; wall "
            << wall_seconds << " s. Bundle: " << config.output_path << '\n';
  if (config.render)
    render_replay(model, frames, true, prepared.label.c_str());
}

} // namespace tracking
