#ifndef GRF_STAND_TASK_HPP_
#define GRF_STAND_TASK_HPP_

#include <mujoco/mujoco.h>
#include <mjpc/task.h>
#include <memory>
#include <string>

class StandTask : public mjpc::Task {
public:
    class ResidualFn : public mjpc::BaseResidualFn {
    public:
        explicit ResidualFn(const StandTask* task) : mjpc::BaseResidualFn(task), task_(task) {}
        void Residual(const mjModel* model, const mjData* data, double* residual) const override;
    private:
        const StandTask* task_;
    };

    StandTask() : residual_(this) {}
    
    std::string Name() const override { return "Custom Stand"; }
    std::string XmlPath() const override { return ""; }

    void ResetLocked(const mjModel* model) override;

    // Target height calculated from the specific morphology
    double target_height_ = 1.4; 

    // Cached sensor addresses for O(1) lookups in the hot loop
    int adr_head_ = -1;
    int adr_comvel_ = -1;
    
    // Cached qpos addresses for ankle posture
    int adr_ank_x_r_ = -1;
    int adr_ank_y_r_ = -1;
    int adr_ank_x_l_ = -1;
    int adr_ank_y_l_ = -1;

protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
        return std::make_unique<ResidualFn>(this);
    }
    mjpc::BaseResidualFn* InternalResidual() override { return &residual_; }

private:
    ResidualFn residual_;
};

#endif
