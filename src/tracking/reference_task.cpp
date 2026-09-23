#include "reference_task.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tracking {
namespace {
void clear_elements(mjSpec *spec, mjtObj type) {
  while (auto *element = mjs_firstElement(spec, type))
    mjs_delete(element);
}

double controller_weight(const std::string &name, double baseline_weight) {
  // This is the frozen baseline objective, not a physical contact constraint.
  // Keep these settings beside the residual layout they weight.
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
  if (name == "Pos[hand]" || name == "Pos[elbow]" || name == "Pos[shoulder]")
    return 8.0;
  if (name == "Vel[hand]" || name == "Vel[elbow]" || name == "Vel[shoulder]")
    return 0.02;
  return baseline_weight;
}

struct Interpolation {
  int first, next;
  double fraction;
};
Interpolation interpolate(const mjModel *model, double time) {
  const double index =
      std::clamp(time * kReferenceFps, 0.0, double(model->nkey - 1));
  const int first = static_cast<int>(std::floor(index));
  return {first, std::min(first + 1, model->nkey - 1), index - first};
}
} // namespace

void add_reference_tracking_objective(mjSpec *spec, const mjModel *source,
                                      const mjModel *baseline) {
  // Replace inherited task configuration; the builder adds the selected clip's
  // keyframes afterward. Physical model settings are deliberately untouched.
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
  // Adjust the stock layout for the custom model's four passive DOFs.
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
    std::vector<double> userdata(source->sensor_user + i * source->nuser_sensor,
                                 source->sensor_user +
                                     (i + 1) * source->nuser_sensor);
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

void ReferenceTask::ResetLocked(const mjModel *model) {
  if (model->nkey < 2 || model->nmocap != 16 || model->njnt == 0 ||
      model->jnt_type[0] != mjJNT_FREE ||
      num_residual != model->nv - 6 + model->nu + 99)
    throw std::runtime_error(
        "Reference tracking requires a free root and a 16-target clip");
  for (size_t i = 0; i < kTargets.size(); ++i) {
    const std::string name = kTargets[i];
    residual_.positions[i] = model->sensor_adr[require_id(
        model, mjOBJ_SENSOR, "tracking_pos[" + name + "]")];
    residual_.velocities[i] = model->sensor_adr[require_id(
        model, mjOBJ_SENSOR, "tracking_linvel[" + name + "]")];
    residual_.mocap[i] = model->body_mocapid[require_id(model, mjOBJ_BODY,
                                                        "mocap[" + name + "]")];
  }
}

void ReferenceTask::TransitionLocked(mjModel *model, mjData *data) {
  const auto sample = interpolate(model, data->time);
  const int width = 3 * model->nmocap;
  for (int i = 0; i < width; ++i)
    data->mocap_pos[i] =
        (1 - sample.fraction) * model->key_mpos[width * sample.first + i] +
        sample.fraction * model->key_mpos[width * sample.next + i];
}

void ReferenceTask::ResidualFn::Residual(const mjModel *model,
                                         const mjData *data,
                                         double *residual) const {
  const auto sample = interpolate(model, data->time);
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
      errors[3 * i + axis] = (1 - sample.fraction) * p0 + sample.fraction * p1 -
                             data->sensordata[positions[i] + axis];
      average[axis] += errors[3 * i + axis] / kTargets.size();
      residual[offset + 3 + 48 + 3 * i + axis] =
          (p1 - p0) * kReferenceFps - data->sensordata[velocities[i] + axis];
    }
  }
  mju_copy3(residual + offset, average);
  for (int i = 0; i < 48; ++i)
    residual[offset + 3 + i] = errors[i] - average[i % 3];
}
} // namespace tracking
