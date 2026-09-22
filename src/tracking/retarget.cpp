#include "reference_task.hpp"
#include "qmc_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tracking {
namespace {
using Points = std::array<double, 48>;

ModelPtr compile(mjSpec *spec) {
  ModelPtr model(mj_compile(spec, nullptr), mj_deleteModel);
  if (!model)
    throw std::runtime_error(std::string("Custom tracking compile: ") +
                             mjs_getError(spec));
  return model;
}

void clear_elements(mjSpec *spec, mjtObj type) {
  while (auto *element = mjs_firstElement(spec, type))
    mjs_delete(element);
}

void site(mjSpec *spec, const std::string &body_name, const std::string &name,
          const std::array<double, 3> &position, bool target = false,
          mjsBody *body = nullptr) {
  if (!body)
    body = mjs_findBody(spec, body_name.c_str());
  if (!body)
    throw std::runtime_error("Missing anatomical body: " + body_name);
  auto *s = mjs_addSite(body, nullptr);
  mjs_setString(s->name, name.c_str());
  mju_copy3(s->pos, position.data());
  s->type = mjGEOM_SPHERE;
  s->size[0] = target ? 0.018 : 0.009;
  s->group = 2;
  s->rgba[0] = target ? 0.15f : 1.0f;
  s->rgba[1] = target ? 0.45f : 0.25f;
  s->rgba[2] = target ? 1.0f : 0.1f;
  s->rgba[3] = 1;
}

void add_tracking_sites(mjSpec *spec, const mjModel *baseline) {
  site(spec, "pelvis", "tracking[pelvis]", {0, 0, .075});
  const int head = require_id(baseline, mjOBJ_GEOM, "head");
  // Virtual anterior-head marker, at 90% of the model's head radius.
  site(spec, "head", "tracking[head]",
       {baseline->geom_size[3 * head] * .9, 0, 0});
  for (const auto &side : {std::string("left"), std::string("right")}) {
    const std::string prefix = side == "left" ? "l" : "r";
    const int shin = require_id(baseline, mjOBJ_BODY, "shin_" + side);
    const int foot = require_id(baseline, mjOBJ_BODY, "foot_" + side);
    const double thigh_scale = -baseline->body_pos[3 * shin + 2] / .4;
    const double shin_scale = -baseline->body_pos[3 * foot + 2] / .39;
    site(spec, "thigh_" + side, "tracking[" + prefix + "hip]",
         {0, side == "left" ? -.025 : .025, .025 * thigh_scale});
    site(spec, "shin_" + side, "tracking[" + prefix + "knee]",
         {0, 0, .05 * thigh_scale});
    // Heel sits above the rear sole; toe target follows the passive MTP body.
    const int geom = require_id(baseline, mjOBJ_GEOM, "foot1_" + side);
    const int toe = require_id(baseline, mjOBJ_BODY, "toe_" + side);
    const double toe_x = baseline->body_pos[3 * toe];
    // The capsule's local z axis spans the two endpoints in the compiled model.
    double rotation[9];
    mju_quat2Mat(rotation, baseline->geom_quat + 4 * geom);
    const double heel_x =
        baseline->geom_pos[3 * geom] -
        std::abs(rotation[2]) * baseline->geom_size[3 * geom + 1];
    site(spec, "foot_" + side, "tracking[" + prefix + "heel]",
         {heel_x, 0, .04 * shin_scale});
    site(spec, "toe_" + side, "tracking[" + prefix + "toe]",
         {0, 0, -.01 * toe_x / .07});
    site(spec, "upper_arm_" + side, "tracking[" + prefix + "shoulder]",
         {0, 0, 0});
    site(spec, "lower_arm_" + side, "tracking[" + prefix + "elbow]",
         {0, 0, 0});
    site(spec, "hand_" + side, "tracking[" + prefix + "hand]", {0, 0, 0});
  }
  auto *world = mjs_findBody(spec, "world");
  for (const char *target : kTargets) {
    const std::string name = std::string("mocap[") + target + "]";
    auto *body = mjs_addBody(world, nullptr);
    mjs_setString(body->name, name.c_str());
    body->mocap = true;
    site(spec, name, name, {0, 0, 0}, true, body);
  }
}

double controller_weight(const std::string &name, double baseline_weight) {
  // Keep the centroid/root term and effort regularization unchanged, but make
  // the heel/toe and supporting-leg points more important than upper-body
  // points. This is controller tuning, not a physical contact constraint:
  // foot geometry and contact settings still need to be evaluated separately.
  if (name == "Pos[toe]" || name == "Pos[heel]")
    return 80.0;
  if (name == "Vel[toe]" || name == "Vel[heel]")
    return 0.25;
  if (name == "Pos[knee]")
    return 55.0;
  if (name == "Vel[knee]")
    return 0.18;
  if (name == "Pos[hip]")
    return 50.0;
  if (name == "Vel[hip]")
    return 0.15;
  if (name == "Pos[pelvis]")
    return 40.0;
  if (name == "Vel[root]")
    return 0.12;

  if (name == "Pos[head]" || name == "Vel[head]")
    return 0.0;
  if (name == "Pos[hand]" || name == "Pos[elbow]" ||
      name == "Pos[shoulder]")
    return 8.0;
  if (name == "Vel[hand]" || name == "Vel[elbow]" ||
      name == "Vel[shoulder]")
    return 0.02;
  return baseline_weight;
}

void add_objective(mjSpec *spec, const mjModel *source,
                   const mjModel *baseline) {
  // Tracking owns its planner settings and residual sensors. Clearing the
  // baseline Walk include here ensures it cannot affect the new objective.
  clear_elements(spec, mjOBJ_SENSOR);
  clear_elements(spec, mjOBJ_NUMERIC);
  clear_elements(spec, mjOBJ_TEXT);
  clear_elements(spec, mjOBJ_KEY);
  for (int i = 0; i < source->nnumeric; ++i) {
    auto *numeric = mjs_addNumeric(spec);
    mjs_setString(numeric->name, mj_id2name(source, mjOBJ_NUMERIC, i));
    mjs_setDouble(numeric->data, source->numeric_data + source->numeric_adr[i],
                  source->numeric_size[i]);
  }
  // Preserve the stock tracking objective, adjusting dimensions for the four
  // passive custom-model DOFs.
  for (int i = 0; i < source->nsensor; ++i) {
    if (source->sensor_type[i] != mjSENS_USER)
      continue;
    auto *sensor = mjs_addSensor(spec);
    const std::string name = mj_id2name(source, mjOBJ_SENSOR, i);
    mjs_setString(sensor->name, name.c_str());
    sensor->type = mjSENS_USER;
    sensor->needstage = mjSTAGE_ACC;
    sensor->datatype = mjDATATYPE_REAL;
    sensor->dim = name == "Joint Vel." ? baseline->nv - 6
                  : name == "Control"  ? baseline->nu
                                       : source->sensor_dim[i];
    std::vector<double> userdata(
        source->sensor_user + i * source->nuser_sensor,
        source->sensor_user + (i + 1) * source->nuser_sensor);
    if (userdata.size() > 1)
      userdata[1] = controller_weight(name, userdata[1]);
    mjs_setDouble(sensor->userdata, userdata.data(), userdata.size());
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

std::array<int, 16> tracking_sites(const mjModel *model) {
  std::array<int, 16> ids;
  for (size_t i = 0; i < ids.size(); ++i)
    ids[i] = require_id(model, mjOBJ_SITE,
                        std::string("tracking[") + kTargets[i] + "]");
  return ids;
}

std::array<int, 16> target_mocap_ids(const mjModel *model) {
  std::array<int, 16> ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    const int body = require_id(model, mjOBJ_BODY,
                                std::string("mocap[") + kTargets[i] + "]");
    ids[i] = model->body_mocapid[body];
    if (ids[i] < 0)
      throw std::runtime_error("Tracking target is not a mocap body");
  }
  return ids;
}

Points source_targets(const mjModel *source, int key,
                      const std::array<int, 16> &mocap) {
  Points points{};
  const double *values = source->key_mpos + 3 * source->nmocap * key;
  for (int i = 0; i < 16; ++i)
    mju_copy3(points.data() + 3 * i, values + 3 * mocap[i]);
  return points;
}

void write_targets(mjModel *model, int key, const Points &points,
                   const std::array<int, 16> &mocap) {
  double *values = model->key_mpos + 3 * model->nmocap * key;
  for (int i = 0; i < 16; ++i)
    mju_copy3(values + 3 * mocap[i], points.data() + 3 * i);
}

double lowest_foot(const mjModel *model, const mjData *data) {
  double height = std::numeric_limits<double>::infinity();
  for (int geom = 0; geom < model->ngeom; ++geom) {
    const char *name = mj_id2name(model, mjOBJ_GEOM, geom);
    if (!name || (std::string(name).rfind("foot", 0) != 0 &&
                  std::string(name).rfind("toe", 0) != 0))
      continue;
    if (model->geom_type[geom] != mjGEOM_CAPSULE)
      throw std::runtime_error("Tracking currently expects capsule feet");
    const double extent = model->geom_size[3 * geom] +
                          model->geom_size[3 * geom + 1] *
                              std::abs(data->geom_xmat[9 * geom + 8]);
    height = std::min(height, data->geom_xpos[3 * geom + 2] - extent);
  }
  if (!std::isfinite(height))
    throw std::runtime_error("No foot capsules found");
  return height;
}

// Damped least squares in MuJoCo's tangent coordinates, with a bounded step
// and line search. The caller supplies a fresh custom-model neutral state;
// this function never consults source joint trajectories.
double fit_pose(const mjModel *model, mjData *data, const Points &target,
                const std::array<int, 16> &sites) {
  std::vector<int> dofs(model->nv);
  std::iota(dofs.begin(), dofs.end(), 0);
  const int n = dofs.size();
  std::vector<double> jacobian(3 * model->nv), matrix(n * n), rhs(n), delta(n),
      velocity(model->nv), previous(model->nq);
  auto error = [&] {
    mj_forward(model, data);
    double sum = 0;
    for (size_t i = 0; i < sites.size(); ++i)
      for (int axis = 0; axis < 3; ++axis)
        sum += std::pow(
            target[3 * i + axis] - data->site_xpos[3 * sites[i] + axis], 2);
    return sum;
  };
  double current = error();
  for (int iteration = 0; iteration < 150; ++iteration) {
    std::fill(matrix.begin(), matrix.end(), 0);
    std::fill(rhs.begin(), rhs.end(), 0);
    for (size_t site = 0; site < sites.size(); ++site) {
      mj_jacSite(model, data, jacobian.data(), nullptr, sites[site]);
      for (int axis = 0; axis < 3; ++axis) {
        const double residual =
            target[3 * site + axis] - data->site_xpos[3 * sites[site] + axis];
        for (int row = 0; row < n; ++row) {
          const double jr = jacobian[axis * model->nv + dofs[row]];
          rhs[row] += jr * residual;
          for (int col = 0; col < n; ++col)
            matrix[row * n + col] +=
                jr * jacobian[axis * model->nv + dofs[col]];
        }
      }
    }
    for (int i = 0; i < n; ++i)
      matrix[i * n + i] += 1e-4;
    mju_cholFactor(matrix.data(), n, 1e-12);
    mju_cholSolve(delta.data(), matrix.data(), rhs.data(), n);
    mju_copy(previous.data(), data->qpos, model->nq);
    double largest = 0;
    for (double value : delta)
      largest = std::max(largest, std::abs(value));
    const double scale = largest > .2 ? .2 / largest : 1;
    bool accepted = false;
    for (double step = scale; step >= scale / 64; step *= .5) {
      std::fill(velocity.begin(), velocity.end(), 0);
      for (int i = 0; i < n; ++i)
        velocity[dofs[i]] = delta[i] * step;
      mju_copy(data->qpos, previous.data(), model->nq);
      mj_integratePos(model, data->qpos, velocity.data(), 1);
      for (int joint = 1; joint < model->njnt; ++joint)
        if (model->jnt_limited[joint]) {
          auto &q = data->qpos[model->jnt_qposadr[joint]];
          q = std::clamp(q, model->jnt_range[2 * joint],
                         model->jnt_range[2 * joint + 1]);
        }
      const double candidate = error();
      if (candidate < current - 1e-12) {
        current = candidate;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      mju_copy(data->qpos, previous.data(), model->nq);
      break;
    }
  }
  return std::sqrt(error() / sites.size());
}

double site_rmse(const mjModel *model, mjData *data, const Points &target,
                 const std::array<int, 16> &sites) {
  mj_forward(model, data);
  double squared_error = 0;
  for (int i = 0; i < 16; ++i)
    squared_error += std::pow(mju_dist3(data->site_xpos + 3 * sites[i],
                                        target.data() + 3 * i),
                              2);
  return std::sqrt(squared_error / 16);
}

double lift_to_clearance(const mjModel *model, mjData *data) {
  mj_forward(model, data);
  const double lift = std::max(0.0, .001 - lowest_foot(model, data));
  if (lift > 0) {
    data->qpos[model->jnt_qposadr[0] + 2] += lift;
    mj_forward(model, data);
  }
  return lift;
}

void verify_dynamics(const mjModel *baseline, const mjModel *model) {
  if (baseline->nq != model->nq || baseline->nv != model->nv ||
      baseline->nu != model->nu || baseline->na != model->na ||
      baseline->ngeom != model->ngeom)
    throw std::runtime_error(
        "Tracking preparation changed the physical model dimensions");
  auto equal = [](const auto *a, const auto *b, int count) {
    if (!std::equal(a, a + count, b))
      throw std::runtime_error(
          "Tracking preparation changed a physical model parameter");
  };
  equal(baseline->body_mass, model->body_mass, baseline->nbody);
  equal(baseline->body_inertia, model->body_inertia, 3 * baseline->nbody);
  equal(baseline->body_pos, model->body_pos, 3 * baseline->nbody);
  equal(baseline->jnt_axis, model->jnt_axis, 3 * baseline->njnt);
  equal(baseline->jnt_range, model->jnt_range, 2 * baseline->njnt);
  equal(baseline->dof_damping, model->dof_damping, baseline->nv);
  equal(baseline->dof_armature, model->dof_armature, baseline->nv);
  equal(baseline->jnt_stiffness, model->jnt_stiffness, baseline->njnt);
  equal(baseline->geom_size, model->geom_size, 3 * baseline->ngeom);
  equal(baseline->geom_pos, model->geom_pos, 3 * baseline->ngeom);
  equal(baseline->geom_quat, model->geom_quat, 4 * baseline->ngeom);
  equal(baseline->geom_friction, model->geom_friction, 3 * baseline->ngeom);
  equal(baseline->geom_solref, model->geom_solref, mjNREF * baseline->ngeom);
  equal(baseline->geom_solimp, model->geom_solimp, mjNIMP * baseline->ngeom);
  equal(baseline->geom_contype, model->geom_contype, baseline->ngeom);
  equal(baseline->geom_conaffinity, model->geom_conaffinity, baseline->ngeom);
  equal(baseline->geom_condim, model->geom_condim, baseline->ngeom);
  equal(baseline->actuator_gear, model->actuator_gear, 6 * baseline->nu);
  equal(baseline->actuator_dynprm, model->actuator_dynprm,
        mjNDYN * baseline->nu);
  equal(baseline->actuator_dyntype, model->actuator_dyntype, baseline->nu);
}

} // namespace

Prepared prepare_custom(const std::string &model_path,
                        const PrepareConfig &config) {
  if (config.qmc_index < 0)
    throw std::runtime_error("QMC index must be nonnegative");

  char error[2048] = {};
  ModelPtr source(mj_loadXML(GRF_TRACKING_MODEL, nullptr, error, sizeof(error)),
                  mj_deleteModel);
  if (!source)
    throw std::runtime_error(std::string("Could not load reference clips: ") +
                             error);
  const Clip source_clip = find_clip(source.get(), config.motion);
  if (!is_foot_only_motion(config.motion))
    throw std::runtime_error(
        "Motion is not approved for foot-only episodes: " + config.motion +
        " (choose " + foot_only_motion_names() + ")");
  const int source_first = source_clip.first;
  const int frame_count = source_clip.count;

  SpecPtr spec(mj_parseXML(model_path.c_str(), nullptr, error, sizeof(error)),
               mj_deleteSpec);
  if (!spec)
    throw std::runtime_error(std::string("Could not load custom XML: ") +
                             error);
  const MorphologyQmcSample morphology =
      apply_morphology_qmc(spec.get(), config.qmc_index);
  auto baseline = compile(spec.get());
  add_tracking_sites(spec.get(), baseline.get());
  add_objective(spec.get(), source.get(), baseline.get());
  const int marker_site_count = add_qmc_marker_sites(spec.get(), baseline.get());
  for (int frame = 0; frame < frame_count; ++frame) {
    auto *key = mjs_addKey(spec.get());
    mjs_setString(key->name,
                  (config.motion + "_" + std::to_string(frame)).c_str());
    key->time = frame / kReferenceFps;
  }

  Prepared prepared;
  prepared.model = compile(spec.get());
  auto *model = prepared.model.get();
  verify_dynamics(baseline.get(), model);
  place_qmc_marker_sites(model, config.qmc_index);
  const auto sites = tracking_sites(model);
  const auto destination_mocap = target_mocap_ids(model);
  const auto source_mocap = target_mocap_ids(source.get());

  // The runtime intentionally transfers the source point coordinates exactly:
  // no rigid alignment, scaling, or source joint trajectory is applied.
  std::vector<Points> targets(frame_count);
  for (int frame = 0; frame < frame_count; ++frame) {
    targets[frame] = source_targets(source.get(), source_first + frame,
                                    source_mocap);
    write_targets(model, frame, targets[frame], destination_mocap);
  }

  // Each fit starts from the model's own neutral qpos. Source key_qpos is
  // deliberately never read: the target-point contract is the only motion
  // information transferred to the custom model.
  auto initial = make_data(model);
  const double initial_fit_before_lift =
      fit_pose(model, initial.get(), targets.front(), sites);
  const double initial_floor_lift = lift_to_clearance(model, initial.get());
  const double initial_fit =
      site_rmse(model, initial.get(), targets.front(), sites);
  auto next = make_data(model);
  const double second_fit_before_lift =
      fit_pose(model, next.get(), targets[1], sites);
  const double second_floor_lift = lift_to_clearance(model, next.get());
  const double second_fit = site_rmse(model, next.get(), targets[1], sites);
  for (int frame = 0; frame < frame_count; ++frame)
    mju_copy(model->key_qpos + model->nq * frame, initial->qpos, model->nq);
  mj_differentiatePos(model, model->key_qvel, 1 / kReferenceFps,
                      initial->qpos, next->qpos);

  std::array<double, 16> initial_errors{};
  for (int i = 0; i < 16; ++i)
    initial_errors[i] =
        mju_dist3(initial->site_xpos + 3 * sites[i], targets[0].data() + 3 * i);
  nlohmann::json passive_initial = nlohmann::json::object();
  for (const char *name :
       {"mtp_left", "mtp_right", "shoulder3_left", "shoulder3_right"})
    passive_initial[name] =
        initial->qpos[model->jnt_qposadr[require_id(model, mjOBJ_JOINT, name)]];

  prepared.source = std::filesystem::absolute(model_path).string();
  prepared.label = "Custom tracking model: " +
                   std::filesystem::path(model_path).filename().string() +
                   " / qmc " + std::to_string(config.qmc_index);
  prepared.clip = {0, frame_count};
  prepared.task = std::make_shared<ReferenceTask>();
  prepared.metadata = {
      {"model_kind", "custom_tracking"},
      {"geometry_randomized", config.qmc_index != 0},
      {"tracking_preparation_preserves_morphed_physics", true},
      {"qmc",
       {{"index", config.qmc_index},
        {"morphology",
         {{"length_scale", morphology.length_scale},
          {"radius_scale", morphology.radius_scale},
          {"bases", {2, 3}},
          {"sampler",
           "global length/radius Halton sampler; experimental and not population-calibrated"}}},
        {"markers",
         {{"sequence_bases", {5, 7, 11}},
        {"marker_site_count", marker_site_count},
        {"markers_per_eligible_body", kMarkersPerEligibleBody},
        {"site_group", 3},
          {"affects_physics", false}}}}},
      {"total_mass_kg", mj_getTotalmass(model)},
      {"nq", model->nq},
      {"nv", model->nv},
      {"nu", model->nu},
      {"na", model->na},
      {"reference_source", GRF_TRACKING_MODEL},
      {"source_first_key", source_first},
      {"reference_frames", frame_count},
      {"reference_fps", kReferenceFps},
      {"target_order", kTargets},
      {"initialization",
       {{"seed", "custom model neutral qpos only"},
        {"initial_fit_rmse_before_floor_lift_m", initial_fit_before_lift},
        {"initial_fit_rmse_m", initial_fit},
        {"initial_site_errors_m", initial_errors},
        {"initial_floor_lift_m", initial_floor_lift},
        {"initial_foot_clearance_m", lowest_foot(model, initial.get())},
        {"second_frame_fit_rmse_before_floor_lift_m", second_fit_before_lift},
        {"second_frame_fit_rmse_m", second_fit},
        {"second_frame_floor_lift_m", second_floor_lift},
        {"initial_velocity",
         "finite difference between two neutral-seeded point-only IK fits at 30 Hz"},
        {"initial_passive_joint_angles_rad", passive_initial}}},
      {"training_labels_validated", false}};
  return prepared;
}

} // namespace tracking
