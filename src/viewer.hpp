#pragma once

#include <mujoco/mujoco.h>
#include <vector>

#include "tracking/frame.hpp"

void render_replay(mjModel *model,
                   const std::vector<tracking::ReplayFrame> &frames,
                   bool follow_root = false,
                   const char *title = "MuJoCo GRF Viewer",
                   double playback_speed = 1.0);
