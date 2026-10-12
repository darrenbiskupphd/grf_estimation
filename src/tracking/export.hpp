#pragma once

#include <filesystem>
#include <vector>

namespace tracking {

// Direct, compressed NPZ training export; V2 recorded labels only. Exactly six
// arrays, in caller episode order (all episodes must have the same M):
//   markers_m[T,M,3], mass_kg[E], time_s[T], contact_wrench[T,2,6],
//   episode_offsets[E+1], source_id[E]. Numeric data stays float64, offsets are
// int64, and source_id is a Unicode array (numpy.load(...,
// allow_pickle=False)). Feet are left/right; wrench is
// [Fx,Fy,Fz,CoPx,CoPy,Tz_at_CoP]. CoP/Tz are NaN when undefined or Fz <=
// min_normal_force (default 0 N). Forces, including flight samples, are never
// thresholded. Raw moments/audit stay in the .grf.
//
// layout_index == -1 uses recorded executed markers; nonnegative indices sample
// independent attachments on the embedded model and use only mj_kinematics /
// mj_comPos on saved poses. No physics, planning, or force recomputation. One
// seeded, run-salted anonymous permutation is fixed throughout each episode.
//
// The summary must identify the original recording with a nonempty source_id
// (top-level, in reference, or in preparation.reference). Variants retain this
// grouping ID. Incomplete recordings are rejected unless explicitly allowed;
// legacy V1 labels are always rejected. Writes atomically, with no sidecar.
void export_run_bundles(const std::vector<std::filesystem::path> &input_paths,
                        const std::filesystem::path &output_npz_path,
                        int layout_index = -1, int permutation_seed = 0,
                        bool allow_incomplete = false,
                        double min_normal_force = 0.0);

void export_run_bundle(const std::filesystem::path &input_path,
                       const std::filesystem::path &output_npz_path,
                       int layout_index = -1, int permutation_seed = 0,
                       bool allow_incomplete = false,
                       double min_normal_force = 0.0);

} // namespace tracking
