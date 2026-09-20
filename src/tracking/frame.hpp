#pragma once

#include <array>
#include <vector>

namespace tracking {

// Values sampled from one replay instant.  Dynamic state vectors are kept in
// the bundle because their dimensions belong to the embedded compiled model;
// the contact fields have a fixed, documented shape.
struct FootContactFrame {
  // Upward world-force contribution in newtons from
  // {foot1, foot2, toe1, toe2}.  The first pair is the model's two main-foot
  // capsules; the second pair is the passive-toe capsule pair.
  std::array<double, 4> geom_normal_load{};

  // Main-foot contacts are also partitioned by their x coordinate in the
  // foot frame, while toe contacts retain their own category.  This records
  // heel-to-forefoot-toe progression even though the baseline main-foot
  // capsules span most of the sole.
  double heel_normal_load = 0;
  double forefoot_normal_load = 0;
  double toe_normal_load = 0;
  double lateral_edge_normal_load = 0;

  // Vertical-force-weighted floor-contact position expressed in the foot
  // body's local frame.  Zero means that the foot was unloaded.
  std::array<double, 3> contact_point_local{};

  // Foot orientation in world axes and the relevant joint state.  Roll and
  // pitch use the foot body's xmat convention; angles are radians.
  double roll = 0;
  double pitch = 0;
  double ankle_y = 0;
  double ankle_x = 0;
  double mtp = 0;
  double mtp_velocity = 0;
  int contact_count = 0;
};

struct ReplayFrame {
  double time = 0;
  std::vector<double> qpos;
  std::vector<double> qvel;
  std::vector<double> act;
  std::vector<double> ctrl;
  std::vector<double> mocap_pos;
  std::vector<double> mocap_quat;

  // Force exerted by the floor on each foot in world axes, and its
  // vertical-force-weighted contact-position proxy in world coordinates.
  std::array<double, 3> grf_left{};
  std::array<double, 3> grf_right{};
  std::array<double, 3> cop_left{};
  std::array<double, 3> cop_right{};
  FootContactFrame left_contact;
  FootContactFrame right_contact;
};

} // namespace tracking
