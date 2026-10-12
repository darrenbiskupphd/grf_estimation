#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <mujoco/mujoco.h>

namespace tracking {

inline int foot_side(const mjModel *model, int body, int left, int right) {
  while (body > 0) {
    if (body == left)
      return 0;
    if (body == right)
      return 1;
    body = model->body_parentid[body];
  }
  return -1;
}

// Floor-on-foot wrench in world axes, moment about world origin (on Z=0).
struct ContactWrench {
  std::array<double, 3> force{};
  std::array<double, 3> moment{};

  void add(const mjContact &contact, const double local[6],
           bool floor_is_geom2) {
    double f[3], torque[3], lever[3];
    mju_mulMatTVec3(f, contact.frame, local);
    mju_mulMatTVec3(torque, contact.frame, local + 3);
    // MuJoCo contact frame normal points geom[0] -> geom[1]; wrench is on
    // geom[1].
    if (floor_is_geom2) {
      mju_scl3(f, f, -1);
      mju_scl3(torque, torque, -1);
    }
    mju_cross(lever, contact.pos, f);
    mju_addTo3(force.data(), f);
    mju_addTo3(moment.data(), torque);
    mju_addTo3(moment.data(), lever);
  }
};

struct PlaneContact {
  std::array<double, 3> cop;
  double free_moment_z;
  bool defined;
};

// No training/contact-quality threshold here. Raw wrench is always retained.
// Positive normal load and finite arithmetic are only mathematical guards.
inline PlaneContact plane_contact(const ContactWrench &wrench) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PlaneContact result{{nan, nan, nan}, nan, false};
  if (!(wrench.force[2] > 0))
    return result;
  const double x = -wrench.moment[1] / wrench.force[2];
  const double y = wrench.moment[0] / wrench.force[2];
  const double tz =
      wrench.moment[2] - (x * wrench.force[1] - y * wrench.force[0]);
  if (std::isfinite(x) && std::isfinite(y) && std::isfinite(tz))
    result = {{x, y, 0}, tz, true};
  return result;
}

} // namespace tracking
