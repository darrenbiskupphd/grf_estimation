#include "qmc_sampling.hpp"
#include "reference_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tracking {
namespace {
using Points = ReferencePoints;

ModelPtr compile(mjSpec *spec) {
  ModelPtr model(mj_compile(spec, nullptr), mj_deleteModel);
  if (!model)
    throw std::runtime_error(std::string("Custom tracking compile: ") +
                             mjs_getError(spec));
  return model;
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
    // The capsule's local z axis spans the two endpoints in the compiled model.
    double rotation[9];
    mju_quat2Mat(rotation, baseline->geom_quat + 4 * geom);
    const double heel_x =
        baseline->geom_pos[3 * geom] -
        std::abs(rotation[2]) * baseline->geom_size[3 * geom + 1];
    site(spec, "foot_" + side, "tracking[" + prefix + "heel]",
         {heel_x, 0, .04 * shin_scale});
    // Highest paired capsule crown in toe-body coordinates. This is a
    // HEIGHT convention: x=y=0 remains over the gap, not on the surface.
    double toe_z = -1e30;
    for (const auto &part : {"toe1_", "toe2_"}) {
      const int g = require_id(baseline, mjOBJ_GEOM, part + side);
      if (baseline->geom_bodyid[g] != toe ||
          baseline->geom_type[g] != mjGEOM_CAPSULE)
        throw std::runtime_error("Expected toe-local paired capsules");
      mju_quat2Mat(rotation, baseline->geom_quat + 4 * g);
      toe_z = std::max(toe_z, baseline->geom_pos[3 * g + 2] +
          std::abs(rotation[8]) * baseline->geom_size[3 * g + 1] +
          baseline->geom_size[3 * g]);
    }
    site(spec, "toe_" + side, "tracking[" + prefix + "toe]",
         {0, 0, toe_z});
    site(spec, "upper_arm_" + side, "tracking[" + prefix + "shoulder]",
         {0, 0, 0});
    site(spec, "lower_arm_" + side, "tracking[" + prefix + "elbow]", {0, 0, 0});
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

void write_targets(mjModel *model, int key, const Points &points,
                   const std::array<int, 16> &mocap) {
  double *values = model->key_mpos + 3 * std::size_t(model->nmocap) * key;
  for (int i = 0; i < 16; ++i)
    mju_copy3(values + 3 * mocap[i], points.data() + 3 * i);
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

Prepared build_reference_tracking_model(const std::string &model_path,
                                        const PrepareConfig &config) {
  if (config.qmc_index < 0 || config.layout_index < 0)
    throw std::runtime_error(
        "QMC morphology and marker-layout indices must be nonnegative");
  // Validate the external data before parsing or modifying the physical model.
  // The MJPC/O objective is installed separately; no bundled motion is loaded.
  const ReferenceData reference = load_reference(config.reference_path);
  const int frame_count = static_cast<int>(reference.time_s.size());
  const std::string reference_id = reference.source_id;

  char error[2048] = {};
  SpecPtr spec(mj_parseXML(model_path.c_str(), nullptr, error, sizeof(error)),
               mj_deleteSpec);
  if (!spec)
    throw std::runtime_error(std::string("Could not load custom XML: ") +
                             error);
  const MorphologyQmcSample morphology =
      apply_morphology_qmc(spec.get(), config.qmc_index);
  auto baseline = compile(spec.get());
  add_tracking_sites(spec.get(), baseline.get());
  add_reference_tracking_objective(spec.get(), baseline.get(),
                                    config.effort_rate_weight,
                                    config.stance_roll_weight,
                                    config.knee_hip_position_scale,
                                    config.stance_roll_still_time_s,
                                    config.elbow_velocity_weight,
                                    config.elbow_position_weight,
                                    config.hand_position_weight,
                                    config.foot_position_weight);
  const int marker_site_count =
      add_qmc_marker_sites(spec.get(), baseline.get());
  for (int frame = 0; frame < frame_count; ++frame) {
    auto *key = mjs_addKey(spec.get());
    mjs_setString(key->name,
                  ("reference_" + std::to_string(frame)).c_str());
    key->time = reference.time_s[frame];
  }

  Prepared prepared;
  prepared.model = compile(spec.get());
  auto *model = prepared.model.get();
  verify_dynamics(baseline.get(), model);
  place_qmc_marker_sites(model, config.layout_index);
  const auto destination_mocap = target_mocap_ids(model);

  // The runtime intentionally transfers the source point coordinates exactly:
  // no rigid alignment, scaling, or source joint trajectory is applied.
  for (int frame = 0; frame < frame_count; ++frame)
    write_targets(model, frame, reference.positions_m[frame], destination_mocap);

  const auto initialization = initialize_reference_tracking_state(
      model, reference.positions_m[0], reference.positions_m[1]);

  prepared.source = std::filesystem::absolute(model_path).string();
  prepared.label = "Custom tracking model: " +
                   std::filesystem::path(model_path).filename().string() +
                   " / qmc " + std::to_string(config.qmc_index) +
                   " / layout " + std::to_string(config.layout_index) +
                   " / " + (config.motion.empty() ? reference_id : config.motion);
  prepared.reference_id = reference_id;
  prepared.clip = {0, frame_count, reference.duration_s()};
  prepared.task = std::make_shared<ReferenceTask>();
  prepared.metadata = {
      {"model_kind", "custom_tracking"},
      {"geometry_randomized", config.qmc_index != 0},
      {"morphology_index", config.qmc_index},
      {"layout_index", config.layout_index},
      {"tracking_preparation_preserves_morphed_physics", true},
      {"qmc",
       {{"index", config.qmc_index},
        {"morphology",
         {{"length_scale", morphology.length_scale},
          {"radius_scale", morphology.radius_scale},
          {"bases", {2, 3}},
          {"sampler", "global length/radius Halton sampler; experimental and "
                      "not population-calibrated"}}},
        {"markers",
         {{"layout_index", config.layout_index},
          {"algorithm", kQmcMarkerLayoutAlgorithm},
          {"sequence_bases", {5, 7, 11}},
          {"marker_site_count", marker_site_count},
          {"markers_per_eligible_group", kMarkersPerEligibleGroup},
          {"foot_group", "main foot and articulated toes share one budget"},
          {"coverage", "complete primitive surface area; reject buried points at qpos0; hands excluded"},
          {"site_group", 3},
          {"affects_physics", false}}}}},
      {"total_mass_kg", mj_getTotalmass(model)},
      {"nq", model->nq},
      {"nv", model->nv},
      {"nu", model->nu},
      {"na", model->na},
      {"reference_source", reference.source_path.string()},
      {"reference_id", reference_id},
      {"reference_schema_version", 1},
      {"reference_frames", frame_count},
      {"reference_duration_s", reference.duration_s()},
      {"reference_time_s", reference.time_s},
      {"reference_time_origin_s", reference.time_origin_s},
      {"reference_metadata", reference.metadata},
      {"reference_coordinates_modified", false},
      {"reference_interpolation", "native timestamp piecewise linear; right-"
                                  "segment velocity; zero velocity at/after end"},
      {"controller_template", "frozen MJPC humanoid tracking objective with O "
                              "overrides; no bundled reference data"},
      {"target_order", kTargets},
      {"initialization", initialization},
      {"training_labels_validated", false}};
  prepared.metadata["toe_site_height_convention"] =
      "highest paired capsule crown in toe-local z; x/y unchanged over gap, not surface";
  for (const auto &side : {std::string("left"), std::string("right")}) {
    const int id = require_id(model, mjOBJ_SITE,
        side == "left" ? "tracking[ltoe]" : "tracking[rtoe]");
    prepared.metadata["toe_sites"][side] = {
        {"body", "toe_" + side},
        {"position_m", {model->site_pos[3 * id], model->site_pos[3 * id + 1], model->site_pos[3 * id + 2]}}};
  }
  prepared.metadata["effort_rate_cost"] = {
      {"weight", config.effort_rate_weight},
      {"activation_rate_scale_per_s", kEffortRateScale},
      {"definition", "0.5 * weight * mean((activation_rate / scale)^2); "
                     "activation_rate = (clamped_ctrl - act) / filter_time"},
      {"enabled", config.effort_rate_weight > 0}};
  prepared.metadata["stance_roll_cost"] = {
      {"enabled", config.stance_roll_weight > 0},
      {"weight", config.stance_roll_weight},
      {"definition", "0.5 * weight * mean(reference_stance_gate * "
                     "foot_lateral_roll_rad^2); no pitch penalty"},
      {"gate", "minimum of each foot's toe/heel smooth height and horizontal-"
               "speed scores from raw reference keys, interpolated in time"},
      {"height_full_m", kStanceHeightFull},
      {"height_off_m", kStanceHeightOff},
      {"horizontal_speed_full_m_per_s", kStanceSpeedFull},
      {"horizontal_speed_off_m_per_s", kStanceSpeedOff},
      {"minimum_continuous_reference_stance_s",
       config.stance_roll_still_time_s}};
  prepared.metadata["foot_position_weight"] = config.foot_position_weight;
  prepared.metadata["knee_hip_position_scale"] =
      config.knee_hip_position_scale;
  prepared.metadata["elbow_velocity_tracking"] = {
      {"weight", config.elbow_velocity_weight},
      {"baseline_weight", 0.02},
      {"definition", "existing Vel[elbow] reference minus site velocity; "
                     "smooth-absolute norm, scale 0.3; NOT absolute speed"}};
  prepared.metadata["upper_position_tracking"] = {
      {"elbow_weight", config.elbow_position_weight},
      {"hand_weight", config.hand_position_weight},
      {"shoulder_weight", 8.0},
      {"definition", "existing bilateral demeaned Pos[elbow] (smooth-absolute-2) "
                     "and Pos[hand] (smooth-absolute) weights; no physical change"}};
  return prepared;
}

} // namespace tracking
