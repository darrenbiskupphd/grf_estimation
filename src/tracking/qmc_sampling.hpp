#pragma once

#include <mujoco/mujoco.h>

#include <vector>

namespace tracking {

// The original exploratory sampler used seven surface markers on each eligible
// articulated body. One QMC episode index drives separate Halton dimensions for
// morphology and observation placement.
inline constexpr int kMarkersPerEligibleBody = 7;

struct MorphologyQmcSample {
  int index = 0;
  double length_scale = 1.0;
  double radius_scale = 1.0;
};

// Index zero is the untouched nominal morphology. Positive indices retain the
// project's existing Halton global length/radius sampler.
MorphologyQmcSample apply_morphology_qmc(mjSpec *spec, int index);

// Adds massless visual sites to eligible physical bodies. The compiled baseline
// identifies those bodies so tracking target and mocap bodies are excluded.
int add_qmc_marker_sites(mjSpec *spec, const mjModel *baseline,
                         int markers_per_body = kMarkersPerEligibleBody);

// Sets deterministic site positions on the compiled geometry. This changes no
// physical MuJoCo parameter; the sites are observation-only.
void place_qmc_marker_sites(mjModel *model, int qmc_index);

std::vector<int> qmc_marker_site_ids(const mjModel *model);

} // namespace tracking
