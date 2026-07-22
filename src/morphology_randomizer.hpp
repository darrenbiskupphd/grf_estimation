#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <fstream>
#include <string>

// Add randomized massless markers to the spec before compilation
void add_qmc_markers_to_spec(mjSpec* spec, int markers_per_body = 7);

// Randomize the positions of the markers on the surfaces of the compiled geometry
void randomize_marker_positions(mjModel* m, int markers_per_body = 7);

// Domain Randomization: Mutate the mjSpec geometry using QMC (Halton) sequence
void randomize_mjspec_geometry(mjSpec* spec, int qmc_index);

struct ReplayFrame {
    double time;
    std::vector<double> qpos;
    std::vector<double> qvel;
    
    // Physics Data for CSV
    double grf_left[3];
    double grf_right[3];
    double cop_left[3];
    double cop_right[3];
    std::vector<double> markers; // flattened 3D coordinates
};

class StateRecorder {
public:
    StateRecorder(const mjModel* m);
    ~StateRecorder() = default;

    // Extracts physics data to the frame. Returns false if non-foot contacts floor.
    bool extract_physics(const mjModel* m, const mjData* d, ReplayFrame& frame);
    
    // Writes the entire replay buffer to CSV at the end of the simulation
    void write_csv(const std::string& filename, const std::vector<ReplayFrame>& buffer);

private:
    int floor_geom_id_ = -1;
    
    // Use raw integer caching instead of unordered_set/vector for maximum O(1) performance
    int fl_body_id_ = -1, tl_body_id_ = -1;
    int fr_body_id_ = -1, tr_body_id_ = -1;
    
    std::vector<int> procedural_marker_site_ids_;
};
