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
    
    // Get nominal head height and relax it by 5%
    double* head_pos = mjpc::SensorByName(model, data, "head_position");
    target_height_ = head_pos[2] * .95;
    mj_deleteData(data); // free the temporary mjData

    // Bypass XML <user> sensors and configure costs manually
    // 0: Height (Asymmetric)
    // 1: CoM Velocity
    // 2: Upright Posture (Torso Z-axis)
    // 3: Control Effort
    // 4: Foot Flatness / Pitch

    num_term = 5;
    num_residual = 1 + 2 + 1 + model->nu + 2;
    
    dim_norm_residual.assign({1, 2, 1, model->nu, 2});
    norm.assign(5, mjpc::NormType::kSmoothAbsLoss); // Pseudo-Huber loss
    weight.assign({2500.0, 1000.0, 500.0, 0.0005, 100.0});
    num_norm_parameter.assign(5, 1);                // This norm requires 1 parameter
    norm_parameter.assign(5, 0.1);                  // The 'p' parameter (quadratic bowl width)

    int id_head = mj_name2id(model, mjOBJ_SENSOR, "head_position");
    adr_head_ = id_head >= 0 ? model->sensor_adr[id_head] : -1;
    
    int id_comvel = mj_name2id(model, mjOBJ_SENSOR, "torso_subtreelinvel");
    adr_comvel_ = id_comvel >= 0 ? model->sensor_adr[id_comvel] : -1;
    
    int id_zaxis = mj_name2id(model, mjOBJ_SENSOR, "torso_zaxis");
    adr_torso_zaxis_ = id_zaxis >= 0 ? model->sensor_adr[id_zaxis] : -1;
    
    int id_foot_r = mj_name2id(model, mjOBJ_SENSOR, "foot_right_zaxis");
    adr_foot_right_zaxis_ = id_foot_r >= 0 ? model->sensor_adr[id_foot_r] : -1;
    
    int id_foot_l = mj_name2id(model, mjOBJ_SENSOR, "foot_left_zaxis");
    adr_foot_left_zaxis_ = id_foot_l >= 0 ? model->sensor_adr[id_foot_l] : -1;
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

    // ----- 2: Upright Posture -----
    double* zaxis = data->sensordata + task_->adr_torso_zaxis_;
    residual[counter++] = 1.0 - zaxis[2];

    // ----- 3: Control Effort -----
    mju_copy(residual + counter, data->ctrl, model->nu);
    counter += model->nu;

    // ----- 4: Foot Flatness (Global Pitch/Roll) -----
    // Penalize deviation of foot Z-axis from world Z-axis (0, 0, 1)
    double* foot_r_z = data->sensordata + task_->adr_foot_right_zaxis_;
    residual[counter++] = 1.0 - foot_r_z[2];
    
    double* foot_l_z = data->sensordata + task_->adr_foot_left_zaxis_;
    residual[counter++] = 1.0 - foot_l_z[2];
}
