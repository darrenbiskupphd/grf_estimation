#include "stand_task.hpp"
#include <mjpc/utilities.h>
#include <iostream>

void StandTask::ResetLocked(const mjModel* model) {
    // Spin up a temporary mjData from the static model
    // copy the drop impact keyframe and compute forward kinematics.
    mjData* data = mj_makeData(model);
    int key_id = mj_name2id(model, mjOBJ_KEY, "drop_impact");
    if (key_id >= 0) {
        mju_copy(data->qpos, model->key_qpos + key_id * model->nq, model->nq);
    }
    mj_forward(model, data);
    
    // Get nominal head height and relax it by ~0.3m (approx 1-2 head lengths)
    double* head_pos = mjpc::SensorByName(model, data, "head_position");
    target_height_ = head_pos[2] * .95;
    mj_deleteData(data); // free the temporary mjData

    // Bypass XML <user> sensors and configure costs manually
    // 0: Height (Asymmetric, Quadratic)
    // 1: CoM Velocity (Quadratic)
    // 2: Joint Velocity (Quadratic)
    // 3: Control Effort (Quadratic)
    // 4: Foot Flatness / Pitch (Quadratic)
    
    num_term = 5;
    num_residual = 1 + 2 + (model->nv - 6) + model->nu + 4;
    
    // The 5th term is now Ankle Posture (4 dimensions: x and y for both feet).
    dim_norm_residual.assign({1, 2, model->nv - 6, model->nu, 4});
    norm.assign(5, mjpc::NormType::kQuadratic);
    weight.assign({1000.0, 50.0, 0.5, 0.001, 15.0});
    num_norm_parameter.assign(5, 0);
    norm_parameter.clear();

    // Cache sensor addresses to avoid O(N) string lookups in the hot loop
    int id_head = mj_name2id(model, mjOBJ_SENSOR, "head_position");
    adr_head_ = id_head >= 0 ? model->sensor_adr[id_head] : -1;
    
    int id_comvel = mj_name2id(model, mjOBJ_SENSOR, "torso_subtreelinvel");
    adr_comvel_ = id_comvel >= 0 ? model->sensor_adr[id_comvel] : -1;
    
    // Cache ankle joint addresses for posture cost
    int j_x_r = mj_name2id(model, mjOBJ_JOINT, "ankle_x_right");
    adr_ank_x_r_ = j_x_r >= 0 ? model->jnt_qposadr[j_x_r] : -1;
    
    int j_y_r = mj_name2id(model, mjOBJ_JOINT, "ankle_y_right");
    adr_ank_y_r_ = j_y_r >= 0 ? model->jnt_qposadr[j_y_r] : -1;
    
    int j_x_l = mj_name2id(model, mjOBJ_JOINT, "ankle_x_left");
    adr_ank_x_l_ = j_x_l >= 0 ? model->jnt_qposadr[j_x_l] : -1;
    
    int j_y_l = mj_name2id(model, mjOBJ_JOINT, "ankle_y_left");
    adr_ank_y_l_ = j_y_l >= 0 ? model->jnt_qposadr[j_y_l] : -1;
}

void StandTask::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
    int counter = 0;

    // Use pointer arithmetic on cached landmark addresses instead of string lookups
    double* head_pos = data->sensordata + task_->adr_head_;
    
    double current_height = head_pos[2];
    
    // ----- 0: Height Penalty ----- Only penalize if dipping below target_height_
    double height_err = task_->target_height_ - current_height;
    residual[counter++] = height_err > 0.0 ? height_err : 0.0;

    // ----- 1: CoM XY Velocity -----
    double* com_vel = data->sensordata + task_->adr_comvel_;
    residual[counter++] = com_vel[0];
    residual[counter++] = com_vel[1];

    // ----- 2: Joint Velocity -----
    mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
    counter += model->nv - 6;

    // ----- 3: Control Effort -----
    mju_copy(residual + counter, data->ctrl, model->nu);
    counter += model->nu;

    // ----- 4: Ankle Posture -----
    // Penalize deviation from neutral joint angles (0.0) to prevent foot rolling
    residual[counter++] = task_->adr_ank_x_r_ >= 0 ? data->qpos[task_->adr_ank_x_r_] : 0.0; // Right roll
    residual[counter++] = task_->adr_ank_y_r_ >= 0 ? data->qpos[task_->adr_ank_y_r_] : 0.0; // Right pitch
    residual[counter++] = task_->adr_ank_x_l_ >= 0 ? data->qpos[task_->adr_ank_x_l_] : 0.0; // Left roll
    residual[counter++] = task_->adr_ank_y_l_ >= 0 ? data->qpos[task_->adr_ank_y_l_] : 0.0; // Left pitch
}
