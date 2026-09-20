#include "residual_callback.hpp"

#include "mjpc/agent.h"

namespace {
mjpc::Agent *active_agent = nullptr;
bool use_snapshot = false;

void residual_sensor(const mjModel *model, mjData *data, int stage) {
  if (stage != mjSTAGE_ACC || !active_agent)
    return;
  const auto *snapshot = active_agent->PlanningResidual();
  if (use_snapshot && active_agent->IsPlanningModel(model) && snapshot) {
    snapshot->Residual(model, data, data->sensordata);
  } else {
    active_agent->ActiveTask()->Residual(model, data, data->sensordata);
  }
}
} // namespace

ResidualSensorScope::ResidualSensorScope(mjpc::Agent &agent,
                                         bool use_planning_snapshot)
    : previous_agent_(active_agent), previous_callback_(mjcb_sensor),
      previous_snapshot_mode_(use_snapshot) {
  active_agent = &agent;
  use_snapshot = use_planning_snapshot;
  mjcb_sensor = residual_sensor;
}

ResidualSensorScope::~ResidualSensorScope() {
  mjcb_sensor = previous_callback_;
  active_agent = previous_agent_;
  use_snapshot = previous_snapshot_mode_;
}
