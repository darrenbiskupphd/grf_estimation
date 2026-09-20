#pragma once

#include <mujoco/mujoco.h>

namespace mjpc {
class Agent;
}

// MuJoCo's callback is process-global. Run episodes sequentially within a
// process; the scope must outlive all planner workers and precede Agent
// teardown.
class ResidualSensorScope {
public:
  explicit ResidualSensorScope(mjpc::Agent &agent,
                               bool use_planning_snapshot = false);
  ~ResidualSensorScope();
  ResidualSensorScope(const ResidualSensorScope &) = delete;
  ResidualSensorScope &operator=(const ResidualSensorScope &) = delete;

private:
  mjpc::Agent *previous_agent_;
  mjfSensor previous_callback_;
  bool previous_snapshot_mode_;
};
