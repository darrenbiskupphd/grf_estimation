#pragma once

#include "tracking.hpp"

#include <vector>

namespace tracking {
// Normalized activation rate: one full-scale activation change in 50 ms.
inline constexpr double kEffortRateScale = 20.0; // per second
inline constexpr double kStanceHeightFull = 0.06; // m above marker clip minimum
inline constexpr double kStanceHeightOff = 0.12;  // m
inline constexpr double kStanceSpeedFull = 0.05;  // horizontal m/s
inline constexpr double kStanceSpeedOff = 0.20;   // horizontal m/s

// Install planner settings and cost/sensor layout without changing physics.
void add_reference_tracking_objective(mjSpec *spec, const mjModel *source,
                                      const mjModel *baseline,
                                      double effort_rate_weight = 100,
                                      double stance_roll_weight = 0,
                                      double knee_hip_position_scale = 1,
                                      double stance_roll_still_time_s = 0,
                                      double elbow_velocity_weight = 0.02,
                                      double elbow_position_weight = 16,
                                      double hand_position_weight = 8);

// Point tracking with optional activation-rate and reference-gated roll costs;
// one raw clip and no physical state changes inside Transition.
class ReferenceTask final : public mjpc::Task {
public:
  ReferenceTask() : residual_(this) {}
  std::string Name() const override {
    return "Custom humanoid reference tracking";
  }
  std::string XmlPath() const override { return {}; }

protected:
  class ResidualFn final : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const ReferenceTask *task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
    std::array<int, 16> positions{}, velocities{}, mocap{};
    int effort_rate_offset = -1;
    int stance_roll_offset = -1;
    std::array<int, 2> foot_body{};
    std::vector<std::array<double, 2>> stance_gate;
  };
  ResidualFn *InternalResidual() override { return &residual_; }
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  void ResetLocked(const mjModel *model) override;
  void TransitionLocked(mjModel *model, mjData *data) override;

private:
  ResidualFn residual_;
};
} // namespace tracking
