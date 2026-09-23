#include "tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tracking {
namespace {
using Points = std::array<double, 48>;

std::array<int, 16> tracking_sites(const mjModel *model) {
  std::array<int, 16> ids;
  for (size_t i = 0; i < ids.size(); ++i)
    ids[i] = require_id(model, mjOBJ_SITE,
                        std::string("tracking[") + kTargets[i] + "]");
  return ids;
}

double lowest_foot(const mjModel *model, const mjData *data) {
  double height = std::numeric_limits<double>::infinity();
  for (int geom = 0; geom < model->ngeom; ++geom) {
    const char *name = mj_id2name(model, mjOBJ_GEOM, geom);
    if (!name || (std::string(name).rfind("foot", 0) != 0 &&
                  std::string(name).rfind("toe", 0) != 0))
      continue;
    if (model->geom_type[geom] != mjGEOM_CAPSULE)
      throw std::runtime_error("Tracking currently expects capsule feet");
    const double extent = model->geom_size[3 * geom] +
                          model->geom_size[3 * geom + 1] *
                              std::abs(data->geom_xmat[9 * geom + 8]);
    height = std::min(height, data->geom_xpos[3 * geom + 2] - extent);
  }
  if (!std::isfinite(height))
    throw std::runtime_error("No foot capsules found");
  return height;
}

// Damped least squares in MuJoCo's tangent coordinates, with a bounded step
// and line search. The caller supplies a fresh custom-model neutral state;
// this function never consults source joint trajectories.
double fit_pose(const mjModel *model, mjData *data, const Points &target,
                const std::array<int, 16> &sites) {
  const int n = model->nv;
  std::vector<double> jacobian(3 * model->nv), matrix(n * n), rhs(n), delta(n),
      velocity(model->nv), previous(model->nq);
  auto error = [&] {
    mj_forward(model, data);
    double sum = 0;
    for (size_t i = 0; i < sites.size(); ++i)
      for (int axis = 0; axis < 3; ++axis)
        sum += std::pow(
            target[3 * i + axis] - data->site_xpos[3 * sites[i] + axis], 2);
    return sum;
  };
  double current = error();
  for (int iteration = 0; iteration < 150; ++iteration) {
    std::fill(matrix.begin(), matrix.end(), 0);
    std::fill(rhs.begin(), rhs.end(), 0);
    for (size_t site = 0; site < sites.size(); ++site) {
      mj_jacSite(model, data, jacobian.data(), nullptr, sites[site]);
      for (int axis = 0; axis < 3; ++axis) {
        const double residual =
            target[3 * site + axis] - data->site_xpos[3 * sites[site] + axis];
        for (int row = 0; row < n; ++row) {
          const double jr = jacobian[axis * model->nv + row];
          rhs[row] += jr * residual;
          for (int col = 0; col < n; ++col)
            matrix[row * n + col] +=
                jr * jacobian[axis * model->nv + col];
        }
      }
    }
    for (int i = 0; i < n; ++i)
      matrix[i * n + i] += 1e-4;
    mju_cholFactor(matrix.data(), n, 1e-12);
    mju_cholSolve(delta.data(), matrix.data(), rhs.data(), n);
    mju_copy(previous.data(), data->qpos, model->nq);
    double largest = 0;
    for (double value : delta)
      largest = std::max(largest, std::abs(value));
    const double scale = largest > .2 ? .2 / largest : 1;
    bool accepted = false;
    for (double step = scale; step >= scale / 64; step *= .5) {
      for (int i = 0; i < n; ++i)
        velocity[i] = delta[i] * step;
      mju_copy(data->qpos, previous.data(), model->nq);
      mj_integratePos(model, data->qpos, velocity.data(), 1);
      for (int joint = 1; joint < model->njnt; ++joint)
        if (model->jnt_limited[joint]) {
          auto &q = data->qpos[model->jnt_qposadr[joint]];
          q = std::clamp(q, model->jnt_range[2 * joint],
                         model->jnt_range[2 * joint + 1]);
        }
      const double candidate = error();
      if (candidate < current - 1e-12) {
        current = candidate;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      mju_copy(data->qpos, previous.data(), model->nq);
      break;
    }
  }
  return std::sqrt(error() / sites.size());
}

double site_rmse(const mjModel *model, mjData *data, const Points &target,
                 const std::array<int, 16> &sites) {
  mj_forward(model, data);
  double squared_error = 0;
  for (int i = 0; i < 16; ++i)
    squared_error += std::pow(
        mju_dist3(data->site_xpos + 3 * sites[i], target.data() + 3 * i), 2);
  return std::sqrt(squared_error / 16);
}

double lift_to_clearance(const mjModel *model, mjData *data) {
  mj_forward(model, data);
  const double lift = std::max(0.0, .001 - lowest_foot(model, data));
  if (lift > 0) {
    data->qpos[model->jnt_qposadr[0] + 2] += lift;
    mj_forward(model, data);
  }
  return lift;
}
} // namespace

nlohmann::json
initialize_reference_tracking_state(mjModel *model, const Points &first_targets,
                                    const Points &second_targets) {
  if (!model || model->nkey < 2)
    throw std::runtime_error("Tracking initialization requires two keyframes");
  const auto sites = tracking_sites(model);

  // Preserve the baseline initialization: both fits start independently from
  // neutral. Continuity seeding and support constraints are later experiments,
  // not part of this behavior-preserving split. Target coordinates never
  // change.
  auto initial = make_data(model);
  const double initial_fit_before_lift =
      fit_pose(model, initial.get(), first_targets, sites);
  const double initial_floor_lift = lift_to_clearance(model, initial.get());
  const double initial_fit =
      site_rmse(model, initial.get(), first_targets, sites);
  auto next = make_data(model);
  const double second_fit_before_lift =
      fit_pose(model, next.get(), second_targets, sites);
  const double second_floor_lift = lift_to_clearance(model, next.get());
  const double second_fit = site_rmse(model, next.get(), second_targets, sites);
  for (int frame = 0; frame < model->nkey; ++frame)
    mju_copy(model->key_qpos + model->nq * frame, initial->qpos, model->nq);
  mj_differentiatePos(model, model->key_qvel, 1 / kReferenceFps, initial->qpos,
                      next->qpos);

  std::array<double, 16> initial_errors{};
  for (int i = 0; i < 16; ++i)
    initial_errors[i] = mju_dist3(initial->site_xpos + 3 * sites[i],
                                  first_targets.data() + 3 * i);
  nlohmann::json passive_initial = nlohmann::json::object();
  for (const char *name :
       {"mtp_left", "mtp_right", "shoulder3_left", "shoulder3_right"})
    passive_initial[name] =
        initial->qpos[model->jnt_qposadr[require_id(model, mjOBJ_JOINT, name)]];

  return {{"seed", "custom model neutral qpos only"},
          {"initial_fit_rmse_before_floor_lift_m", initial_fit_before_lift},
          {"initial_fit_rmse_m", initial_fit},
          {"initial_site_errors_m", initial_errors},
          {"initial_floor_lift_m", initial_floor_lift},
          {"initial_foot_clearance_m", lowest_foot(model, initial.get())},
          {"second_frame_fit_rmse_before_floor_lift_m", second_fit_before_lift},
          {"second_frame_fit_rmse_m", second_fit},
          {"second_frame_floor_lift_m", second_floor_lift},
          {"initial_velocity", "finite difference between two neutral-seeded "
                               "point-only IK fits at 30 Hz"},
          {"initial_passive_joint_angles_rad", passive_initial}};
}
} // namespace tracking
