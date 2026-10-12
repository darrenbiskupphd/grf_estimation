#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "mjpc/task.h"
#include "mujoco_raii.hpp"
#include "reference.hpp"

namespace tracking {
struct Clip {
  int first = -1;
  int count = 0;
  double duration_s = 0; // Last minus first native reference timestamp.
  double duration() const { return duration_s; }
};
int require_id(const mjModel *model, mjtObj type, const std::string &name);

struct ResolvedDuration {
  double requested = 0;
  double effective = 0;
  bool capped = false;
};
ResolvedDuration resolve_duration(const Clip &clip, double requested_duration);

struct PrepareConfig {
  std::filesystem::path reference_path; // Required schema-v1 external JSON.
  std::string motion; // Optional display/keyframe id; defaults to reference id.
  int qmc_index = 0;
  int layout_index = 0; // Independent observation-marker layout.
  double effort_rate_weight = 100; // O default; zero disables the rate cost.
  double stance_roll_weight = 0; // Experimental; zero preserves the objective.
  double stance_roll_still_time_s = 0; // Require sustained reference stance.
  double knee_hip_position_scale = 1; // Experimental; 1 preserves weights.
  double elbow_velocity_weight = 0.02; // Existing Vel[elbow] reference-tracking weight.
  double elbow_position_weight = 16; // O default, bilateral Pos[elbow].
  double hand_position_weight = 8; // Existing Pos[hand] weight, both sides.
};

struct RunConfig {
  std::string motion;
  double duration = 0;         // Zero selects the full, non-looping clip.
  double physics_timestep = 1.0 / 2400.0;
  int plan_every_n_steps = 36;
  int planner_threads = 1;
  int warmup_iterations = 2;
  int replay_every_n_steps = 4; // Capture 600 Hz; viewer uses frame timestamps.
  std::filesystem::path output_path;
  bool render = false;
};

struct Prepared {
  ModelPtr model{nullptr, mj_deleteModel};
  std::shared_ptr<mjpc::Task> task;
  Clip clip;
  std::string source;
  std::string label;
  std::string reference_id; // Set RunConfig.motion to this for run reporting.
  nlohmann::json metadata = nlohmann::json::object();
};

void run(Prepared prepared, const RunConfig &config);
Prepared build_reference_tracking_model(const std::string &model_path,
                                        const PrepareConfig &config);

// Set initial physical key states from the first two raw reference samples,
// using model->key_time[1] - model->key_time[0] for the velocity difference.
// This never modifies the reference coordinates or model physics.
nlohmann::json initialize_reference_tracking_state(
    mjModel *model, const std::array<double, 48> &first_targets,
    const std::array<double, 48> &second_targets);
} // namespace tracking
