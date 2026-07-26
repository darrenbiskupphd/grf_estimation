#include "morphology_randomizer.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Halton sequence generator for QMC sampling
double halton(int index, int base) {
    double f = 1.0;
    double result = 0.0;
    while (index > 0) {
        f = f / base;
        result = result + f * (index % base);
        index = index / base;
    }
    return result;
}

// Map a [0,1] value to a uniform range [min, max]
double map_uniform(double value, double min, double max) {
    return min + value * (max - min);
}

void randomize_mjspec_geometry(mjSpec* spec, int qmc_index) {
    // Generate Uniform QMC scale factors for global height and mass variance.
    // Since classic scaling models are linear (L = c*H, m = c*M), the segment variance 
    // is mathematically identical to the variance of the global Height and Mass!
    // We use a tight 3-sigma bound of ±10% to ensure resulting segment mass remains realistic.
    double L_scale = map_uniform(halton(qmc_index, 2), 0.90, 1.10);
    double r_scale = map_uniform(halton(qmc_index, 3), 0.90, 1.10);

    for (mjsElement* el = mjs_firstElement(spec, mjOBJ_BODY); el != nullptr; el = mjs_nextElement(spec, el)) {
        mjsBody* b = mjs_asBody(el);
        if (b->name && b->name->compare("world") == 0) continue;

        // Scale body position (this shifts child joints down correctly)
        b->pos[0] *= L_scale;
        b->pos[1] *= L_scale;
        b->pos[2] *= L_scale;

        // Iterate through all geoms attached to this body
        for (mjsElement* gel = mjs_firstChild(b, mjOBJ_GEOM, 0); gel != nullptr; gel = mjs_nextChild(b, gel, 0)) {
            mjsGeom* g = mjs_asGeom(gel);
            
            // Only scale primitive collision geometries
            if (g->type == mjGEOM_CAPSULE || g->type == mjGEOM_CYLINDER || g->type == mjGEOM_SPHERE) {
                // Radius is always size[0]
                g->size[0] *= r_scale;
                
                // If using fromto, scale the endpoints
                bool has_fromto = false;
                for (int i=0; i<6; ++i) {
                    if (g->fromto[i] != 0.0) has_fromto = true;
                }
                
                if (has_fromto) {
                    for(int i=0; i<6; ++i) g->fromto[i] *= L_scale;
                } else {
                    // Length is size[1]
                    g->size[1] *= L_scale;
                    g->size[2] *= L_scale; // Scale other dims just in case
                }
                
                // Also scale geom pos if it's offset from the body
                g->pos[0] *= L_scale;
                g->pos[1] *= L_scale;
                g->pos[2] *= L_scale;
            }
        }
    }
}

void add_qmc_markers_to_spec(mjSpec* spec, int markers_per_body) {
    // Iterate over all bodies in the mjSpec builder
    for (mjsElement* el = mjs_firstElement(spec, mjOBJ_BODY); el != nullptr; el = mjs_nextElement(spec, el)) {
        mjsBody* b = mjs_asBody(el);
        
        // Skip worldbody (we don't attach markers to the environment)
        if (b->name && (b->name->compare("world") == 0 || b->name->compare("hand_left") == 0 || b->name->compare("hand_right") == 0)) continue;
        
        // Add exact number of massless sites to the body
        for (int i = 0; i < markers_per_body; ++i) {
            mjsSite* s = mjs_addSite(b, nullptr);
            s->size[0] = 0.01;
            s->size[1] = 0.01;
            s->size[2] = 0.01;
            s->rgba[0] = 1.0f; // Red
            s->rgba[1] = 0.0f;
            s->rgba[2] = 0.0f;
            s->rgba[3] = 1.0f;
            s->type = mjGEOM_SPHERE;
            s->group = 3; // Use group 3 to flag these as procedurally generated markers
        }
    }
}

void randomize_marker_positions(mjModel* m, int markers_per_body, int global_qmc_index) {
    // Offset the index into a non-overlapping range so each episode gets unique markers.
    int qmc_index = global_qmc_index * 1000;
    
    // Iterate through compiled sites and randomize position for procedural ones
    for (int s = 0; s < m->nsite; ++s) {
        if (m->site_group[s] != 3) continue; // Only randomize our procedural markers
        
        int body_id = m->site_bodyid[s];
        int geom_adr = m->body_geomadr[body_id];
        int geom_num = m->body_geomnum[body_id];
        
        // Target primitive geometries of the body (avoiding visual meshes if possible)
        int valid_geoms[32];
        int num_valid_geoms = 0;
        for (int j = 0; j < geom_num && num_valid_geoms < 32; ++j) {
            int g_type = m->geom_type[geom_adr + j];
            if (g_type == mjGEOM_CAPSULE || g_type == mjGEOM_CYLINDER || g_type == mjGEOM_SPHERE) {
                valid_geoms[num_valid_geoms++] = geom_adr + j;
            }
        }
        
        int geom_id = geom_adr; 
        if (num_valid_geoms > 0) {
            geom_id = valid_geoms[qmc_index % num_valid_geoms];
        }
        
        int geom_type = m->geom_type[geom_id];
        double* geom_size = m->geom_size + geom_id * 3;
        double* geom_pos = m->geom_pos + geom_id * 3; // offset relative to body origin
        
        // Draw low-discrepancy 2D point (u, v)
        double u = halton(qmc_index, 2);
        double v = halton(qmc_index, 3);
        qmc_index++;
        double x = 0, y = 0, z = 0;
        
        // Procedural mapping to primitive surface
        if (geom_type == mjGEOM_CAPSULE || geom_type == mjGEOM_CYLINDER) {
            double radius = geom_size[0];
            double half_length = geom_size[1];
            
            double theta = u * 2.0 * M_PI;
            double h = (v - 0.5) * 2.0 * half_length;
            
            x = radius * cos(theta);
            y = radius * sin(theta);
            z = h;
        } else if (geom_type == mjGEOM_SPHERE) {
            double radius = geom_size[0];
            double phi = acos(1.0 - 2.0 * u);
            double theta = v * 2.0 * M_PI;
            
            x = radius * sin(phi) * cos(theta);
            y = radius * sin(phi) * sin(theta);
            z = radius * cos(phi);
        }
        
        // Local offset in the geometry's local frame
        mjtNum local_vec[3] = {x, y, z};
        mjtNum rot_vec[3];
        
        // Rotate vector by geom_quat to map to the oriented surface
        mjtNum* geom_quat = m->geom_quat + geom_id * 4;
        mju_rotVecQuat(rot_vec, local_vec, geom_quat);
        
        // Final position relative to the parent body frame
        m->site_pos[s * 3 + 0] = geom_pos[0] + rot_vec[0];
        m->site_pos[s * 3 + 1] = geom_pos[1] + rot_vec[1];
        m->site_pos[s * 3 + 2] = geom_pos[2] + rot_vec[2];
        
        // Disable sameframe optimization so mj_kinematics respects the new position
        m->site_sameframe[s] = 0;
    }
}

StateRecorder::StateRecorder(const mjModel* m) {
    // Cache floor geom
    floor_geom_id_ = mj_name2id(m, mjOBJ_GEOM, "floor");

    // Cache foot bodies directly for instant O(1) comparison (no unordered_set/hashing overhead needed for 2 items)
    fr_body_id_ = mj_name2id(m, mjOBJ_BODY, "foot_right");
    tr_body_id_ = mj_name2id(m, mjOBJ_BODY, "toe_right");
    fl_body_id_ = mj_name2id(m, mjOBJ_BODY, "foot_left");
    tl_body_id_ = mj_name2id(m, mjOBJ_BODY, "toe_left");

    // Cache procedural markers
    for (int s = 0; s < m->nsite; ++s) {
        if (m->site_group[s] == 3) {
            procedural_marker_site_ids_.push_back(s);
        }
    }
}

bool StateRecorder::extract_physics(const mjModel* m, const mjData* d, ReplayFrame& frame) {
    // Zero out frame arrays
    for(int i=0; i<3; ++i) {
        frame.grf_left[i] = frame.grf_right[i] = 0;
        frame.cop_left[i] = frame.cop_right[i] = 0;
    }
    double sum_Fz_left = 0, sum_Fz_right = 0;

    for (int i = 0; i < d->ncon; ++i) {
        const mjContact* c = d->contact + i;
        
        // We only care about contacts involving the floor
        if (c->geom[0] != floor_geom_id_ && c->geom[1] != floor_geom_id_) continue;

        int other_geom = (c->geom[0] == floor_geom_id_) ? c->geom[1] : c->geom[0];
        int other_body = m->geom_bodyid[other_geom];

        // Check if contact is on the foot using flat integer comparison
        bool is_left = (other_body == fl_body_id_ || other_body == tl_body_id_);
        bool is_right = (other_body == fr_body_id_ || other_body == tr_body_id_);

        if (!is_left && !is_right) {
            // Early Termination! A non-foot body touched the floor.
            return false; 
        }

        double force6[6];
        mj_contactForce(m, d, i, force6);

        // Map local contact force to global frame
        double global_force[3] = {0, 0, 0};
        mju_mulMatTVec3(global_force, c->frame, force6);

        // Ensure force direction is pointing ON the foot BY the floor
        if (c->geom[1] == floor_geom_id_) {
            global_force[0] = -global_force[0];
            global_force[1] = -global_force[1];
            global_force[2] = -global_force[2];
        }

        if (is_left) {
            frame.grf_left[0] += global_force[0];
            frame.grf_left[1] += global_force[1];
            frame.grf_left[2] += global_force[2];
            
            // CoP is weighted by vertical force
            if (global_force[2] > 0) {
                sum_Fz_left += global_force[2];
                frame.cop_left[0] += global_force[2] * c->pos[0];
                frame.cop_left[1] += global_force[2] * c->pos[1];
                frame.cop_left[2] += global_force[2] * c->pos[2];
            }
        } else if (is_right) {
            frame.grf_right[0] += global_force[0];
            frame.grf_right[1] += global_force[1];
            frame.grf_right[2] += global_force[2];

            if (global_force[2] > 0) {
                sum_Fz_right += global_force[2];
                frame.cop_right[0] += global_force[2] * c->pos[0];
                frame.cop_right[1] += global_force[2] * c->pos[1];
                frame.cop_right[2] += global_force[2] * c->pos[2];
            }
        }
    }

    // Finalize CoP by dividing by total vertical force
    if (sum_Fz_left > 1e-5) {
        frame.cop_left[0] /= sum_Fz_left;
        frame.cop_left[1] /= sum_Fz_left;
        frame.cop_left[2] /= sum_Fz_left;
    }
    if (sum_Fz_right > 1e-5) {
        frame.cop_right[0] /= sum_Fz_right;
        frame.cop_right[1] /= sum_Fz_right;
        frame.cop_right[2] /= sum_Fz_right;
    }

    // Capture procedural marker clouds
    frame.markers.clear();
    frame.markers.reserve(procedural_marker_site_ids_.size() * 3);
    for (int s : procedural_marker_site_ids_) {
        frame.markers.push_back(d->site_xpos[s*3 + 0]);
        frame.markers.push_back(d->site_xpos[s*3 + 1]);
        frame.markers.push_back(d->site_xpos[s*3 + 2]);
    }

    return true; // Valid step
}

void StateRecorder::write_csv(const std::string& filename, const std::vector<ReplayFrame>& buffer) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for recording." << std::endl;
        return;
    }

    // Write CSV Header
    file << "time,grf_left_x,grf_left_y,grf_left_z,grf_right_x,grf_right_y,grf_right_z,"
         << "cop_left_x,cop_left_y,cop_left_z,cop_right_x,cop_right_y,cop_right_z";
    for (size_t i = 0; i < procedural_marker_site_ids_.size(); ++i) {
        file << ",marker_" << procedural_marker_site_ids_[i] << "_x"
             << ",marker_" << procedural_marker_site_ids_[i] << "_y"
             << ",marker_" << procedural_marker_site_ids_[i] << "_z";
    }
    file << "\n";

    // Dump all frames to file
    for (const auto& frame : buffer) {
        file << frame.time << ","
             << frame.grf_left[0] << "," << frame.grf_left[1] << "," << frame.grf_left[2] << ","
             << frame.grf_right[0] << "," << frame.grf_right[1] << "," << frame.grf_right[2] << ","
             << frame.cop_left[0] << "," << frame.cop_left[1] << "," << frame.cop_left[2] << ","
             << frame.cop_right[0] << "," << frame.cop_right[1] << "," << frame.cop_right[2];

        for (double val : frame.markers) {
            file << "," << val;
        }
        file << "\n";
    }
    file.close();
}
