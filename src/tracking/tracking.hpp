#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "mjpc/task.h"
#include "mujoco_raii.hpp"

namespace tracking {
inline constexpr double kReferenceFps = 30;
inline constexpr std::array<const char *, 16> kTargets = {
    "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip"};

// Approved bundled clips with foot-only support. The 39-frame run clip is
// retired; hand- and knee-supported acrobatics are also excluded.
inline constexpr std::array<const char *, 5> kFootOnlyMotions = {
    "walk", "jump", "dance", "kick_spin", "spin_kick"};

struct Clip {
  int first = -1;
  int count = 0;
  double duration() const { return (count - 1) / kReferenceFps; }
};
Clip find_clip(const mjModel *model, const std::string &motion);
bool is_foot_only_motion(const std::string &motion);
std::string foot_only_motion_names();
int require_id(const mjModel *model, mjtObj type, const std::string &name);

struct ResolvedDuration {
  double requested = 0;
  double effective = 0;
  bool capped = false;
};
ResolvedDuration resolve_duration(const Clip &clip, double requested_duration);

struct PrepareConfig {
  std::string motion = "walk";
  int qmc_index = 0;
};

struct RunConfig {
  std::string motion = "walk";
  double duration = 0;         // Zero selects the full, non-looping clip.
  double physics_timestep = 1.0 / 2400.0;
  int plan_every_n_steps = 36;
  int planner_threads = 1;
  int warmup_iterations = 2;
  int replay_every_n_steps = 12;
  std::filesystem::path output_path;
  bool render = false;
};

struct Prepared {
  ModelPtr model{nullptr, mj_deleteModel};
  std::shared_ptr<mjpc::Task> task;
  Clip clip;
  std::string source;
  std::string label;
  nlohmann::json metadata = nlohmann::json::object();
};

void run(Prepared prepared, const RunConfig &config);
Prepared build_reference_tracking_model(const std::string &model_path,
                                        const PrepareConfig &config);

// Set initial physical key states from two raw reference samples and return
// diagnostics. This never modifies the reference coordinates or model physics.
nlohmann::json initialize_reference_tracking_state(
    mjModel *model, const std::array<double, 48> &first_targets,
    const std::array<double, 48> &second_targets);
} // namespace tracking
