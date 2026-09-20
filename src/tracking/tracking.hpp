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

struct Clip {
  int first = -1;
  int count = 0;
  double duration() const { return (count - 1) / kReferenceFps; }
};
Clip find_clip(const mjModel *model, const std::string &motion);
int require_id(const mjModel *model, mjtObj type, const std::string &name);

enum class ReferenceStrategy { Raw, Rigid, Retargeted };

const char *reference_strategy_name(ReferenceStrategy strategy);
ReferenceStrategy parse_reference_strategy(const std::string &text);

struct PrepareConfig {
  std::string motion = "walk";
  int start_frame = 0;
  ReferenceStrategy reference = ReferenceStrategy::Raw;
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
Prepared prepare_custom(const std::string &model_path,
                        const PrepareConfig &config);
} // namespace tracking
