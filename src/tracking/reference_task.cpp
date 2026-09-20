#include "reference_task.hpp"

#include <algorithm>
#include <cmath>

namespace tracking {
namespace {
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
