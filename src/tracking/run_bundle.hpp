#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>

#include "frame.hpp"
#include "mujoco_raii.hpp"

namespace tracking {

struct RunBundle {
  ModelPtr model{nullptr, mj_deleteModel};
  std::vector<ReplayFrame> frames;
  nlohmann::json summary;
};

// Stores the compiled model, sampled frames, and a small JSON summary in one
// self-contained file.  The write is atomic at the output-path level: a
// sibling .partial file is removed on failure and renamed only on success.
void save_run_bundle(const std::filesystem::path &path, const mjModel *model,
                     const std::vector<ReplayFrame> &frames,
                     const nlohmann::json &summary);

// Loads the embedded model through MuJoCo's in-memory VFS, so replay creates
// no side files and does not need the original XML or source reference clip.
RunBundle load_run_bundle(const std::filesystem::path &path);

} // namespace tracking
