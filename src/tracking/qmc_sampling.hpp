#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <string>
#include <vector>

namespace tracking {

// Kept under its original name for source compatibility. The budget is per
// eligible GROUP: each foot_left/right + its articulated toe_left/right gets
// ten sites TOTAL, across all four capsules. Other eligible bodies get ten
// each; hand_left/right remain excluded. Both bundled models have 14 groups.
inline constexpr int kMarkersPerEligibleBody = 10;
inline constexpr int kMarkersPerEligibleGroup = kMarkersPerEligibleBody;
inline constexpr const char *kQmcMarkerLayoutAlgorithm =
    "halton_full_surface_area_rejection_v1";

struct MorphologyQmcSample {
  int index = 0;
  double length_scale = 1.0;
  double radius_scale = 1.0;
};

// Index zero is the untouched nominal morphology. Positive indices retain the
// project's existing global length/radius Halton draw (bases 2, 3), including
// its density/inertia policy. Body-local joint anchors scale with length too.
MorphologyQmcSample apply_morphology_qmc(mjSpec *spec, int index);

struct QmcMarkerGroup {
  std::string name; // stable group identity, also encoded in marker site names
  std::vector<int> body_ids;
  std::vector<int> geom_ids;
  std::vector<int> site_ids;
  // Full primitive areas, aligned with geom_ids (before overlap rejection).
  std::vector<double> geom_surface_areas;
};

struct QmcMarkerPlacement {
  int site_id = -1;
  int group_index = -1;
  int geom_id = -1;
  int body_id = -1; // selected geom's actual articulated body, NOT group root
  std::array<mjtNum, 3> local_pos{}; // in body_id coordinates
  // "sphere", "side", "negative_cap", or "positive_cap" in geom coordinates.
  std::string surface_region;
};

struct QmcMarkerLayout {
  int index = 0;
  std::vector<QmcMarkerGroup> groups;
  // In qmc_marker_site_ids order; suitable for private audit provenance, not
  // anatomical labels in an unlabelled estimator input.
  std::vector<QmcMarkerPlacement> placements;
  // Indexed by model geom id. Counts document rejection, not an independent
  // exposed-area estimate. Per-geom/cap output counts follow from placements.
  std::vector<int> geom_candidate_counts;
  std::vector<int> geom_buried_counts;
};

// Discover groups from the compiled model, including a saved MJB, without any
// mjSpec or process-local registry. Names preserve membership after
// reparenting. Legacy models retain their existing site counts; rebuild them to
// get the new ten-per-group budget. An unmarked baseline returns groups with no
// site_ids.
std::vector<QmcMarkerGroup> qmc_marker_groups(const mjModel *model);

// Adds massless observation-only sites, initially on each group's root body.
// The baseline excludes tracking/mocap additions. markers_per_body is retained
// as a compatibility argument, but now means markers per eligible GROUP.
int add_qmc_marker_sites(mjSpec *spec, const mjModel *baseline,
                         int markers_per_body = kMarkersPerEligibleBody);

// Pure, reproducible layout + geometry/coverage audit. Only layout_index seeds
// the marker sequence; morphology has no shared random state or episode index.
// Propose geoms proportional to COMPLETE surface area, and sample that surface
// uniformly (Halton bases 5, 7, 11). Reject points inside another supported
// non-world, non-mocap primitive at qpos0, including adjacent groups and hands.
// Thus caps are included and overlapping/buried surfaces are not overweighted.
// No equal-per-capsule quota: a finite layout may omit a small
// primitive/region. Unsupported primitives/meshes and occlusion during motion
// are not assessed. Bounded rejection throws on inaccessible groups rather than
// placing inside.
QmcMarkerLayout qmc_marker_layout(const mjModel *model, int layout_index);

// Apply the above layout, changing ONLY marker site_bodyid, site_pos and
// site_sameframe. MuJoCo kinematics/jacobians use site_bodyid directly; no
// recompilation or mj_setConst is needed. Call mj_forward (or mj_kinematics for
// positions alone) on any existing mjData afterwards. Do not mutate a model
// concurrently with stepping/rendering; observation sites must not drive
// physics.
void place_qmc_marker_sites(mjModel *model, int layout_index);

// Only owned marker[body/group][slot] names in visualization group 3, not all
// sites that happen to share that visualization group.
std::vector<int> qmc_marker_site_ids(const mjModel *model);

} // namespace tracking
