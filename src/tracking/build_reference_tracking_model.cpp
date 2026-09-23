#include "qmc_sampling.hpp"
#include "reference_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

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
  if (config.qmc_index < 0)
    throw std::runtime_error("QMC index must be nonnegative");
  if (config.motion == "run")
    throw std::runtime_error(
        "The bundled run clip was retired (39 frames, 1.267 s); sustained "
        "running needs a replacement reference. Choose " +
        foot_only_motion_names());
  if (!is_foot_only_motion(config.motion))
    throw std::runtime_error(
        "Motion is not approved for foot-only episodes: " + config.motion +
        " (choose " + foot_only_motion_names() + ")");

  char error[2048] = {};
  ModelPtr source(mj_loadXML(GRF_TRACKING_MODEL, nullptr, error, sizeof(error)),
                  mj_deleteModel);
  if (!source)
    throw std::runtime_error(std::string("Could not load reference clips: ") +
                             error);
  const Clip source_clip = find_clip(source.get(), config.motion);
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
  add_reference_tracking_objective(spec.get(), source.get(), baseline.get());
  const int marker_site_count =
      add_qmc_marker_sites(spec.get(), baseline.get());
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
  const auto destination_mocap = target_mocap_ids(model);
  const auto source_mocap = target_mocap_ids(source.get());

  // The runtime intentionally transfers the source point coordinates exactly:
  // no rigid alignment, scaling, or source joint trajectory is applied.
  for (int frame = 0; frame < frame_count; ++frame)
    write_targets(model, frame,
                  source_targets(source.get(), source_first + frame, source_mocap),
                  destination_mocap);

  const auto initialization = initialize_reference_tracking_state(
      model, source_targets(source.get(), source_first, source_mocap),
      source_targets(source.get(), source_first + 1, source_mocap));

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
          {"sampler", "global length/radius Halton sampler; experimental and "
                      "not population-calibrated"}}},
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
      {"initialization", initialization},
      {"training_labels_validated", false}};
  return prepared;
}

} // namespace tracking
