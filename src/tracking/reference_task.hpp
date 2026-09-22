#pragma once

#include "tracking.hpp"

namespace tracking {
// Same objective and cost layout as the pinned MJPC humanoid Tracking task,
// with one raw point clip and no state changes inside Transition.
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
