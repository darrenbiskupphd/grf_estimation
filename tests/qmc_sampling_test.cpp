// Standalone MuJoCo-only regression test. No MJPC, renderer, or test framework.
// Link this file and src/tracking/qmc_sampling.cpp against mujoco::mujoco;
// run with the repository root as argv[1] (or from the repository root).
#include "tracking/qmc_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Model = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
using Spec = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;
using Data = std::unique_ptr<mjData, decltype(&mj_deleteData)>;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance,
          const std::string &message) {
  check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": " + std::to_string(actual) +
            " != " + std::to_string(expected));
}
template <class F> void throws(F &&function, const std::string &message) {
  bool caught = false;
  try {
    function();
  } catch (const std::runtime_error &) {
    caught = true;
  }
  check(caught, message);
}
Spec parse_file(const std::string &path) {
  char error[2048]{};
  Spec spec(mj_parseXML(path.c_str(), nullptr, error, sizeof(error)),
            mj_deleteSpec);
  check(bool(spec), std::string("parse: ") + error);
  return spec;
}
Spec parse(const char *xml) {
  char error[2048]{};
  Spec spec(mj_parseXMLString(xml, nullptr, error, sizeof(error)),
            mj_deleteSpec);
  check(bool(spec), std::string("parse string: ") + error);
  return spec;
}
Model compile(mjSpec *spec) {
  Model model(mj_compile(spec, nullptr), mj_deleteModel);
  check(bool(model), std::string("compile: ") + mjs_getError(spec));
  return model;
}
Data data(const mjModel *model) {
  Data result(mj_makeData(model), mj_deleteData);
  check(bool(result), "allocate data");
  return result;
}
std::vector<char> bytes(const mjModel *model) {
  std::vector<char> result(mj_sizeModel(model));
  mj_saveModel(model, nullptr, result.data(), static_cast<int>(result.size()));
  return result;
}
Model embedded_roundtrip(const mjModel *model) {
  const auto buffer = bytes(model);
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  check(mj_addBufferVFS(&vfs, "embedded.mjb", buffer.data(),
                        static_cast<int>(buffer.size())) == 0,
        "VFS buffer");
  Model result(mj_loadModel("embedded.mjb", &vfs), mj_deleteModel);
  mj_deleteVFS(&vfs);
  check(bool(result), "load embedded MJB");
  return result;
}
int id(const mjModel *model, mjtObj type, const std::string &name) {
  const int result = mj_name2id(model, type, name.c_str());
  check(result >= 0, "missing object " + name);
  return result;
}
template <class T>
void equal_array(const T *a, const T *b, int count, const char *name) {
  check(std::equal(a, a + count, b), std::string("changed ") + name);
}

void same_physics(const mjModel *a, const mjModel *b) {
  check(a->nq == b->nq && a->nv == b->nv && a->nu == b->nu && a->na == b->na &&
            a->nbody == b->nbody && a->njnt == b->njnt &&
            a->ngeom == b->ngeom && a->ntendon == b->ntendon &&
            a->neq == b->neq,
        "physical model dimensions");
#define SAME(field, count) equal_array(a->field, b->field, count, #field)
  SAME(qpos0, a->nq);
  SAME(qpos_spring, a->nq);
  SAME(body_parentid, a->nbody);
  SAME(body_mass, a->nbody);
  SAME(body_inertia, 3 * a->nbody);
  SAME(body_ipos, 3 * a->nbody);
  SAME(body_iquat, 4 * a->nbody);
  SAME(body_pos, 3 * a->nbody);
  SAME(body_quat, 4 * a->nbody);
  SAME(jnt_pos, 3 * a->njnt);
  SAME(jnt_axis, 3 * a->njnt);
  SAME(jnt_range, 2 * a->njnt);
  SAME(jnt_stiffness, a->njnt);
  SAME(dof_damping, a->nv);
  SAME(dof_armature, a->nv);
  SAME(dof_frictionloss, a->nv);
  SAME(geom_bodyid, a->ngeom);
  SAME(geom_size, 3 * a->ngeom);
  SAME(geom_pos, 3 * a->ngeom);
  SAME(geom_quat, 4 * a->ngeom);
  SAME(geom_friction, 3 * a->ngeom);
  SAME(geom_solref, mjNREF * a->ngeom);
  SAME(geom_solimp, mjNIMP * a->ngeom);
  SAME(geom_contype, a->ngeom);
  SAME(geom_conaffinity, a->ngeom);
  SAME(geom_condim, a->ngeom);
  SAME(tendon_stiffness, a->ntendon);
  SAME(tendon_damping, a->ntendon);
  SAME(actuator_gear, 6 * a->nu);
  SAME(actuator_gainprm, mjNGAIN * a->nu);
  SAME(actuator_biasprm, mjNBIAS * a->nu);
  SAME(actuator_dynprm, mjNDYN * a->nu);
#undef SAME
}

// Independent distance/transform checks (not production's surface sampler).
std::array<mjtNum, 3> world_point(const mjData *d, int body,
                                  const mjtNum *local) {
  std::array<mjtNum, 3> result;
  mju_mulMatVec3(result.data(), d->xmat + 9 * body, local);
  mju_addTo3(result.data(), d->xpos + 3 * body);
  return result;
}
std::array<mjtNum, 3> geom_point(const mjData *d, int geom,
                                 const mjtNum *world) {
  mjtNum delta[3];
  mju_sub3(delta, world, d->geom_xpos + 3 * geom);
  std::array<mjtNum, 3> result;
  mju_mulMatTVec3(result.data(), d->geom_xmat + 9 * geom, delta);
  return result;
}
double surface_distance(const mjModel *m, int geom,
                        const std::array<mjtNum, 3> &local) {
  const double r = m->geom_size[3 * geom];
  const double h = m->geom_size[3 * geom + 1];
  if (m->geom_type[geom] == mjGEOM_SPHERE)
    return mju_norm3(local.data()) - r;
  if (m->geom_type[geom] == mjGEOM_CAPSULE) {
    const double dz = local[2] - std::clamp(local[2], -h, h);
    return std::sqrt(local[0] * local[0] + local[1] * local[1] + dz * dz) - r;
  }
  if (m->geom_type[geom] == mjGEOM_CYLINDER)
    return std::max(std::hypot(local[0], local[1]) - r, std::abs(local[2]) - h);
  return std::numeric_limits<double>::infinity();
}

void validate_layout(const mjModel *m,
                     const tracking::QmcMarkerLayout &layout) {
  auto reference = data(m);
  mj_kinematics(m, reference.get());
  const auto sites = tracking::qmc_marker_site_ids(m);
  check(layout.placements.size() == sites.size(), "layout site count");
  std::vector<int> accepted(m->ngeom);
  for (size_t ordinal = 0; ordinal < sites.size(); ++ordinal) {
    const auto &p = layout.placements[ordinal];
    check(p.site_id == sites[ordinal], "stable site ordering");
    check(m->geom_bodyid[p.geom_id] == p.body_id, "attachment follows geom");
    const auto &group = layout.groups.at(p.group_index);
    check(std::find(group.geom_ids.begin(), group.geom_ids.end(), p.geom_id) !=
              group.geom_ids.end(),
          "geom in correct group");
    const auto world =
        world_point(reference.get(), p.body_id, p.local_pos.data());
    const auto local = geom_point(reference.get(), p.geom_id, world.data());
    near(surface_distance(m, p.geom_id, local), 0, 2e-12, "on full surface");
    const double h = m->geom_size[3 * p.geom_id + 1];
    if (p.surface_region == "positive_cap")
      check(local[2] >= h - 1e-12, "positive cap mapping");
    else if (p.surface_region == "negative_cap")
      check(local[2] <= -h + 1e-12, "negative cap mapping");
    else if (p.surface_region == "side")
      check(std::abs(local[2]) <= h + 1e-12, "side mapping");
    else
      check(p.surface_region == "sphere" &&
                m->geom_type[p.geom_id] == mjGEOM_SPHERE,
            "sphere mapping");
    for (int geom = 0; geom < m->ngeom; ++geom) {
      const int body = m->geom_bodyid[geom];
      if (geom == p.geom_id || !body || m->body_mocapid[body] >= 0)
        continue;
      check(surface_distance(m, geom,
                             geom_point(reference.get(), geom, world.data())) >=
                -2e-10,
            "marker not buried at qpos0");
    }
    ++accepted[p.geom_id];
  }
  for (int geom = 0; geom < m->ngeom; ++geom)
    check(accepted[geom] + layout.geom_buried_counts[geom] ==
              layout.geom_candidate_counts[geom],
          "coverage audit accounting");
}

void compare_layout(const tracking::QmcMarkerLayout &a,
                    const tracking::QmcMarkerLayout &b) {
  check(a.index == b.index && a.placements.size() == b.placements.size(),
        "deterministic layout dimensions");
  for (size_t i = 0; i < a.placements.size(); ++i) {
    const auto &x = a.placements[i];
    const auto &y = b.placements[i];
    check(x.site_id == y.site_id && x.group_index == y.group_index &&
              x.geom_id == y.geom_id && x.body_id == y.body_id &&
              x.local_pos == y.local_pos &&
              x.surface_region == y.surface_region,
          "deterministic independent layout");
  }
  check(a.geom_candidate_counts == b.geom_candidate_counts &&
            a.geom_buried_counts == b.geom_buried_counts,
        "deterministic audit");
}

void morphology_test(const std::string &path) {
  auto spec = parse_file(path);
  auto baseline = compile(spec.get());
  const auto before = bytes(baseline.get());
  const auto nominal = tracking::apply_morphology_qmc(spec.get(), 0);
  auto unchanged = compile(spec.get());
  check(nominal.index == 0 && nominal.length_scale == 1 &&
            nominal.radius_scale == 1,
        "nominal sample");
  check(before == bytes(unchanged.get()),
        "index zero leaves model byte-identical");

  for (int index : {1, 2, 3, 17}) {
    auto morphed_spec = parse_file(path);
    const auto sample =
        tracking::apply_morphology_qmc(morphed_spec.get(), index);
    auto morphed = compile(morphed_spec.get());
    const auto *a = baseline.get();
    const auto *b = morphed.get();
    auto radical_inverse = [](int n, int base) {
      double result = 0, denominator = base;
      while (n) {
        result += (n % base) / denominator;
        denominator *= base;
        n /= base;
      }
      return result;
    };
    near(sample.length_scale, .9 + .2 * radical_inverse(index, 2), 2e-16,
         "unchanged length distribution");
    near(sample.radius_scale, .9 + .2 * radical_inverse(index, 3), 2e-16,
         "unchanged radius distribution");
    for (int body = 1; body < a->nbody; ++body)
      for (int axis = 0; axis < 3; ++axis)
        near(b->body_pos[3 * body + axis],
             a->body_pos[3 * body + axis] * sample.length_scale, 1e-14,
             "body offset scaling");
    for (int joint = 0; joint < a->njnt; ++joint) {
      if (a->jnt_type[joint] == mjJNT_FREE)
        continue;
      for (int axis = 0; axis < 3; ++axis)
        near(b->jnt_pos[3 * joint + axis],
             a->jnt_pos[3 * joint + axis] * sample.length_scale, 1e-14,
             "joint anchor scaling");
    }
    for (const char *name : {"abdomen_x", "abdomen_y", "abdomen_z"}) {
      const int joint = id(a, mjOBJ_JOINT, name);
      check(a->jnt_pos[3 * joint + 2] != 0,
            "regression requires explicit anchor");
    }
    equal_array(a->jnt_axis, b->jnt_axis, 3 * a->njnt, "joint axes");
    equal_array(a->jnt_range, b->jnt_range, 2 * a->njnt, "angular ranges");
    equal_array(a->jnt_stiffness, b->jnt_stiffness, a->njnt, "stiffness");
    equal_array(a->dof_damping, b->dof_damping, a->nv, "damping");
    equal_array(a->dof_armature, b->dof_armature, a->nv, "armature");
    equal_array(a->actuator_gear, b->actuator_gear, 6 * a->nu, "actuators");
    std::vector<double> expected_mass(a->nbody);
    for (int geom = 0; geom < a->ngeom; ++geom) {
      const int body = a->geom_bodyid[geom];
      if (!body)
        continue;
      near(b->geom_size[3 * geom], a->geom_size[3 * geom] * sample.radius_scale,
           1e-14, "geom radius scaling");
      if (a->geom_type[geom] == mjGEOM_CAPSULE)
        near(b->geom_size[3 * geom + 1],
             a->geom_size[3 * geom + 1] * sample.length_scale, 1e-14,
             "capsule length scaling");
      for (int axis = 0; axis < 3; ++axis)
        near(b->geom_pos[3 * geom + axis],
             a->geom_pos[3 * geom + axis] * sample.length_scale, 1e-14,
             "geom offset scaling");
      const char *name = mj_id2name(b, mjOBJ_GEOM, geom);
      const auto *g =
          mjs_asGeom(mjs_findElement(morphed_spec.get(), mjOBJ_GEOM, name));
      near(g->density, 1000, 0, "uniform density preserved");
      const double r = b->geom_size[3 * geom];
      const double h = b->geom_size[3 * geom + 1];
      double volume = 4.0 / 3.0 * mjPI * r * r * r;
      if (b->geom_type[geom] == mjGEOM_CAPSULE)
        volume += 2 * mjPI * r * r * h;
      expected_mass[body] += g->density * volume;
    }
    for (int body = 1; body < a->nbody; ++body)
      near(b->body_mass[body], expected_mass[body], 1e-12,
           "mass from uniform density");
  }
}

void size_defined_morphology_test() {
  auto spec = parse(R"(<mujoco><compiler inertiafromgeom="true"/><default>
      <joint pos=".1 .2 .3"/></default><worldbody><body name="body" pos="0 0 1">
      <joint name="inherited"/><geom name="capsule" type="capsule" size=".1 .4"/>
      <geom name="cylinder" type="cylinder" size=".2 .3" pos="1 0 0"/>
      </body></worldbody></mujoco>)");
  auto base = compile(spec.get());
  const auto sample = tracking::apply_morphology_qmc(spec.get(), 2);
  auto model = compile(spec.get());
  for (int geom = 0; geom < model->ngeom; ++geom) {
    near(model->geom_size[3 * geom],
         base->geom_size[3 * geom] * sample.radius_scale, 1e-14,
         "size-defined radius");
    near(model->geom_size[3 * geom + 1],
         base->geom_size[3 * geom + 1] * sample.length_scale, 1e-14,
         "NaN fromto sentinel must not skip size-defined length");
  }
  for (int axis = 0; axis < 3; ++axis)
    near(model->jnt_pos[axis], base->jnt_pos[axis] * sample.length_scale, 1e-14,
         "inherited anchor scaling");
}

void articulation_test(mjModel *model,
                       const tracking::QmcMarkerLayout &layout) {
  auto d = data(model);
  mj_forward(model, d.get());
  const int toe = id(model, mjOBJ_BODY, "toe_left");
  const int foot = id(model, mjOBJ_BODY, "foot_left");
  int toe_site = -1, foot_site = -1;
  for (const auto &p : layout.placements) {
    check(model->site_bodyid[p.site_id] == p.body_id, "reparented site body");
    if (p.body_id == toe)
      toe_site = p.site_id;
    if (p.body_id == foot)
      foot_site = p.site_id;
  }
  check(toe_site >= 0 && foot_site >= 0, "fixture samples main foot and toe");
  const auto initial_toe = std::array<mjtNum, 3>{
      d->site_xpos[3 * toe_site], d->site_xpos[3 * toe_site + 1],
      d->site_xpos[3 * toe_site + 2]};
  const auto initial_foot = std::array<mjtNum, 3>{
      d->site_xpos[3 * foot_site], d->site_xpos[3 * foot_site + 1],
      d->site_xpos[3 * foot_site + 2]};
  const int joint = id(model, mjOBJ_JOINT, "mtp_left");
  const int qadr = model->jnt_qposadr[joint];
  const int dof = model->jnt_dofadr[joint];
  d->qpos[qadr] += .4;
  mj_forward(model, d.get());
  double motion = 0;
  for (int axis = 0; axis < 3; ++axis) {
    near(d->site_xpos[3 * foot_site + axis], initial_foot[axis], 1e-14,
         "main foot site must not follow MTP");
    motion += std::abs(d->site_xpos[3 * toe_site + axis] - initial_toe[axis]);
  }
  check(motion > 1e-4, "toe site follows articulated toe");
  for (const auto &p : layout.placements) {
    const auto world = world_point(d.get(), p.body_id, p.local_pos.data());
    for (int axis = 0; axis < 3; ++axis)
      near(d->site_xpos[3 * p.site_id + axis], world[axis], 2e-14,
           "MuJoCo reparented site world transform");
  }
  std::vector<mjtNum> jac(3 * model->nv);
  mj_jacSite(model, d.get(), jac.data(), nullptr, toe_site);
  const auto current = std::array<mjtNum, 3>{d->site_xpos[3 * toe_site],
                                             d->site_xpos[3 * toe_site + 1],
                                             d->site_xpos[3 * toe_site + 2]};
  d->qpos[qadr] += 1e-7;
  mj_forward(model, d.get());
  for (int axis = 0; axis < 3; ++axis)
    near((d->site_xpos[3 * toe_site + axis] - current[axis]) / 1e-7,
         jac[axis * model->nv + dof], 1e-8, "reparented site Jacobian");
}

void grouped_layout_test(const std::string &path) {
  auto spec = parse_file(path);
  // Layout is independent even on a positively morphed specimen.
  tracking::apply_morphology_qmc(spec.get(), 2);
  auto base = compile(spec.get());
  check(tracking::qmc_marker_groups(base.get()).size() == 14,
        "14 eligible groups");
  check(tracking::add_qmc_marker_sites(spec.get(), base.get()) == 140,
        "ten per group, not per foot/toe body");
  auto model = compile(spec.get());
  same_physics(base.get(), model.get());
  throws([&] { tracking::add_qmc_marker_sites(spec.get(), base.get()); },
         "duplicate sites rejected");
  const auto groups = tracking::qmc_marker_groups(model.get());
  for (const auto &group : groups) {
    check(group.site_ids.size() == 10, "ten sites per eligible group");
    check(group.name != "toe_left" && group.name != "toe_right" &&
              group.name != "hand_left" && group.name != "hand_right",
          "group exclusions");
    if (group.name == "foot_left" || group.name == "foot_right")
      check(group.body_ids.size() == 2 && group.geom_ids.size() == 4,
            "whole-foot four-capsule group");
  }
  const auto initial = tracking::qmc_marker_layout(model.get(), 0);
  validate_layout(model.get(), initial);
  const auto before = bytes(model.get());
  compare_layout(initial, tracking::qmc_marker_layout(model.get(), 0));
  check(before == bytes(model.get()), "layout/audit generation is pure");
  tracking::place_qmc_marker_sites(model.get(), 0);
  articulation_test(model.get(), initial);
  auto embedded = embedded_roundtrip(model.get());
  compare_layout(initial, tracking::qmc_marker_layout(embedded.get(), 0));
  articulation_test(embedded.get(), initial);
  bool moved_body = false;
  int negative_caps = 0, positive_caps = 0, rejections = 0;
  std::vector<int> coverage(model->ngeom);
  for (int layout_index = 1; layout_index <= 32; ++layout_index) {
    const auto layout = tracking::qmc_marker_layout(model.get(), layout_index);
    validate_layout(model.get(), layout);
    auto pristine_copy =
        Model(mj_copyModel(nullptr, model.get()), mj_deleteModel);
    tracking::place_qmc_marker_sites(model.get(), layout_index);
    // Stronger than a physics array subset: ONLY these three arrays may change.
    mju_copy(pristine_copy->site_pos, model->site_pos, 3 * model->nsite);
    std::copy_n(model->site_bodyid, model->nsite, pristine_copy->site_bodyid);
    std::copy_n(model->site_sameframe, model->nsite,
                pristine_copy->site_sameframe);
    check(bytes(pristine_copy.get()) == bytes(model.get()),
          "only site binding/position changes");
    compare_layout(layout,
                   tracking::qmc_marker_layout(embedded.get(), layout_index));
    tracking::place_qmc_marker_sites(embedded.get(), layout_index);
    equal_array(model->site_pos, embedded->site_pos, 3 * model->nsite,
                "offline/online site position agreement");
    equal_array(model->site_bodyid, embedded->site_bodyid, model->nsite,
                "offline/online body agreement");
    for (size_t i = 0; i < layout.placements.size(); ++i) {
      const auto &p = layout.placements[i];
      moved_body |= p.body_id != initial.placements[i].body_id;
      negative_caps += p.surface_region == "negative_cap";
      positive_caps += p.surface_region == "positive_cap";
      ++coverage[p.geom_id];
    }
    rejections += std::accumulate(layout.geom_buried_counts.begin(),
                                  layout.geom_buried_counts.end(), 0);
  }
  check(moved_body, "layouts can change a foot slot's articulated attachment");
  check(negative_caps > 0 && positive_caps > 0, "both capsule caps covered");
  check(rejections > 0, "real model buried overlaps rejected");
  for (const auto &group : groups)
    for (int geom : group.geom_ids)
      check(coverage[geom] > 0, "aggregate real-model primitive coverage");
  tracking::place_qmc_marker_sites(model.get(), 0);
  compare_layout(initial, tracking::qmc_marker_layout(model.get(), 0));
  same_physics(base.get(), model.get());
  compare_layout(
      tracking::qmc_marker_layout(model.get(), std::numeric_limits<int>::max()),
      tracking::qmc_marker_layout(embedded.get(),
                                  std::numeric_limits<int>::max()));

  // Identical dynamics with no sites, with one layout, and while changing the
  // layout using existing mjData objects. Includes contacts and activation
  // state.
  auto a = data(base.get());
  auto b = data(model.get());
  auto c = data(embedded.get());
  for (int step = 0; step < 20; ++step) {
    if (step % 4 == 0)
      tracking::place_qmc_marker_sites(embedded.get(), step);
    for (int actuator = 0; actuator < base->nu; ++actuator) {
      const double ctrl = .01 * std::sin(step + actuator);
      a->ctrl[actuator] = b->ctrl[actuator] = c->ctrl[actuator] = ctrl;
    }
    mj_step(base.get(), a.get());
    mj_step(model.get(), b.get());
    mj_step(embedded.get(), c.get());
    for (const auto *other : {b.get(), c.get()}) {
      equal_array(a->qpos, other->qpos, base->nq, "trajectory qpos");
      equal_array(a->qvel, other->qvel, base->nv, "trajectory qvel");
      equal_array(a->qacc, other->qacc, base->nv, "trajectory qacc");
      equal_array(a->act, other->act, base->na, "trajectory activation");
      equal_array(a->qfrc_constraint, other->qfrc_constraint, base->nv,
                  "contact forces");
      check(a->ncon == other->ncon, "contact count");
    }
  }
  std::cout << path << ": 14 groups/140 sites; caps (-/+): " << negative_caps
            << '/' << positive_caps
            << "; buried proposals rejected: " << rejections << '\n';
}

void area_and_overlap_test() {
  auto spec = parse(R"(<mujoco><worldbody>
    <body name="spheres" pos="-3 0 1"><geom type="sphere" size=".2" pos="-.5 0 0"/>
      <geom type="sphere" size=".1" pos=".5 0 0"/></body>
    <body name="capsule" pos="0 0 1"><geom type="capsule" size=".1 .2"/></body>
    <body name="cylinder" pos="3 0 1"><geom type="cylinder" size=".1 .2"/></body>
    <body name="overlap" pos="6 0 1">
      <geom name="outer" type="sphere" size=".3"/>
      <geom name="buried" type="sphere" size=".1"/>
      <geom name="duplicate" type="sphere" size=".3"/>
      <geom name="partial" type="sphere" size=".25" pos=".35 0 0"/></body>
    <body name="hand_left" pos="9 0 1"><geom type="sphere" size=".1"/></body>
    <body name="target" mocap="true" pos="12 0 1"><geom type="sphere" size=".1"/></body>
    <body name="unsupported" pos="15 0 1"><geom type="box" size=".1 .1 .1"/></body>
    <site name="unrelated_group3" group="3" pos="1 2 3"/>
    </worldbody></mujoco>)");
  auto base = compile(spec.get());
  check(tracking::add_qmc_marker_sites(spec.get(), base.get()) == 40,
        "hands/mocap/unsupported excluded");
  auto model = compile(spec.get());
  const int inner = id(model.get(), mjOBJ_GEOM, "buried");
  const int duplicate = id(model.get(), mjOBJ_GEOM, "duplicate");
  int big = 0, small = 0, capsule_caps = 0, cylinder_caps = 0,
      inner_rejected = 0;
  int duplicate_rejected = 0;
  constexpr int layouts = 400;
  for (int index = 0; index < layouts; ++index) {
    const auto layout = tracking::qmc_marker_layout(model.get(), index);
    validate_layout(model.get(), layout);
    for (const auto &p : layout.placements) {
      check(p.geom_id != inner && p.geom_id != duplicate,
            "fully buried/duplicate primitives must never own sites");
      const auto &group = layout.groups[p.group_index];
      if (group.name == "spheres") {
        if (model->geom_size[3 * p.geom_id] > .15)
          ++big;
        else
          ++small;
      }
      if (p.surface_region == "negative_cap" ||
          p.surface_region == "positive_cap") {
        capsule_caps += group.name == "capsule";
        cylinder_caps += group.name == "cylinder";
      }
    }
    inner_rejected += layout.geom_buried_counts[inner];
    duplicate_rejected += layout.geom_buried_counts[duplicate];
  }
  // These broad deterministic statistical checks reject equal-per-geom
  // selection and side-only placement, without claiming an IID probabilistic
  // guarantee.
  near(double(big) / (big + small), .8, .025,
       "surface-area weighted geom choice");
  near(double(capsule_caps) / (layouts * 10), 1.0 / 3.0, .025,
       "capsule cap area fraction");
  near(double(cylinder_caps) / (layouts * 10), .2, .025,
       "cylinder disk area fraction");
  check(inner_rejected > 0 && duplicate_rejected > 0,
        "overlap audit records buried draws");
  const int unrelated = id(model.get(), mjOBJ_SITE, "unrelated_group3");
  tracking::place_qmc_marker_sites(model.get(), 5);
  check(model->site_bodyid[unrelated] == 0 &&
            model->site_pos[3 * unrelated] == 1 &&
            model->site_pos[3 * unrelated + 1] == 2 &&
            model->site_pos[3 * unrelated + 2] == 3,
        "unrelated group-3 sites untouched");
  std::cout << "Area audit: large sphere fraction="
            << double(big) / (big + small)
            << ", capsule caps=" << double(capsule_caps) / (layouts * 10)
            << ", cylinder caps=" << double(cylinder_caps) / (layouts * 10)
            << '\n';
}

void failure_and_custom_count_test() {
  auto spec = parse(R"(<mujoco><worldbody><body name="outer">
    <geom type="sphere" size=".3"/><body name="hidden"><geom type="sphere" size=".1"/>
    </body></body></worldbody></mujoco>)");
  auto base = compile(spec.get());
  check(tracking::add_qmc_marker_sites(spec.get(), base.get(), 3) == 6,
        "custom budget compatibility");
  auto model = compile(spec.get());
  const auto before = bytes(model.get());
  throws([&] { tracking::place_qmc_marker_sites(model.get(), 0); },
         "bounded rejection");
  check(before == bytes(model.get()), "failed placement must be atomic");
  throws([&] { tracking::place_qmc_marker_sites(model.get(), -1); },
         "negative layout rejected");
  throws([&] { tracking::apply_morphology_qmc(spec.get(), -1); },
         "negative morphology rejected");
  throws([&] { tracking::place_qmc_marker_sites(nullptr, 0); },
         "null model rejected");
  throws([&] { tracking::apply_morphology_qmc(nullptr, 0); },
         "null spec rejected");
  check(tracking::qmc_marker_site_ids(nullptr).empty(),
        "legacy null site enumeration");
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string root = argc > 1 ? argv[1] : ".";
    for (const char *file :
         {"winter_baseline_male.xml", "plagenhoef_baseline_female.xml"}) {
      const auto path = root + "/assets/" + file;
      morphology_test(path);
      grouped_layout_test(path);
    }
    size_defined_morphology_test();
    area_and_overlap_test();
    failure_and_custom_count_test();
    std::cout << "QMC morphology/grouped-layout tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "QMC test failed: " << error.what() << '\n';
    return 1;
  }
}
