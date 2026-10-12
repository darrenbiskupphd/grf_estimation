#include "mujoco_raii.hpp"
#include "tracking/contact_wrench.hpp"
#include "tracking/timing.hpp"
#include <iostream>
#include <stdexcept>

void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void near(double x, double y, double tolerance = 1e-9) {
  if (std::abs(x - y) > tolerance || !std::isfinite(x))
    throw std::runtime_error("Numeric mismatch: " + std::to_string(x) +
                             " != " + std::to_string(y));
}

int main() try {
  using namespace tracking;
  check(capture_stride(600) == 4, "600 Hz stride");
  check(capture_stride(200) == 12, "200 Hz stride");
  check(capture_stride(2400) == 1, "2400 Hz stride");
  for (double rate : {0., -1., 700., 2401.}) {
    bool rejected = false;
    try {
      capture_stride(rate);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    check(rejected, "Invalid capture rate accepted");
  }
  mjContact c{};
  // Local normal +X maps to world +Z, tangent Y -> X, tangent Z -> Y.
  const double rotation[] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
  mju_copy(c.frame, rotation, 9);
  c.pos[0] = 2;
  c.pos[1] = 3;
  c.pos[2] = .1;
  const double local[] = {100, 4, 5, 6, 7, 8};
  ContactWrench wrench;
  wrench.add(c, local, false);
  near(wrench.force[0], 4);
  near(wrench.force[1], 5);
  near(wrench.force[2], 100);
  near(wrench.moment[0], 306.5);
  near(wrench.moment[1], -191.6);
  near(wrench.moment[2], 4);
  const auto plane = plane_contact(wrench);
  check(plane.defined, "Loaded CoP undefined");
  near(plane.cop[0], 1.916);
  near(plane.cop[1], 3.065);
  near(plane.cop[2], 0);
  double cross[3];
  mju_cross(cross, plane.cop.data(), wrench.force.data());
  near(cross[0], wrench.moment[0]);
  near(cross[1], wrench.moment[1]);
  near(cross[2] + plane.free_moment_z, wrench.moment[2]);
  ContactWrench reversed;
  reversed.add(c, local, true);
  for (int k = 0; k < 3; ++k) {
    near(reversed.force[k], -wrench.force[k]);
    near(reversed.moment[k], -wrench.moment[k]);
  }
  check(!plane_contact(ContactWrench{}).defined, "Flight CoP defined");
  check(std::isnan(plane_contact(ContactWrench{}).free_moment_z),
        "Flight torque must be NaN");
  ContactWrench low;
  low.force[2] = 1e-10;
  check(plane_contact(low).defined,
        "Low load was subjected to a training threshold");

  const char *xml =
      R"(<mujoco><option timestep="0.0004166666666666667" solver="Newton" tolerance="1e-12"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1" condim="6"/>
    <body name="left" pos="0 0 .099"><freejoint/>
      <geom name="main" type="sphere" size=".03" mass="1" pos="0 0 .5"/>
      <body name="toe"><geom name="toegeom" type="sphere" size=".1" mass="1" condim="6"/></body>
    </body><body name="right" pos="1 0 1"><geom type="sphere" size=".1" mass="0"/></body></worldbody></mujoco>)";
  char error[1024]{};
  SpecPtr spec(mj_parseXMLString(xml, nullptr, error, sizeof(error)),
               mj_deleteSpec);
  check(bool(spec), error);
  ModelPtr model(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(model), "Compile contact fixture");
  near(mj_getTotalmass(model.get()), 2);
  auto data = make_data(model.get());
  mj_forward(model.get(), data.get());
  const int left = mj_name2id(model.get(), mjOBJ_BODY, "left");
  const int right = mj_name2id(model.get(), mjOBJ_BODY, "right");
  const int floor = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  ContactWrench actual;
  int count = 0;
  for (int i = 0; i < data->ncon; ++i) {
    const auto &contact = data->contact[i];
    int other = contact.geom[0] == floor ? contact.geom[1] : contact.geom[0];
    if (foot_side(model.get(), model->geom_bodyid[other], left, right) != 0)
      continue;
    double f[6];
    mj_contactForce(model.get(), data.get(), i, f);
    actual.add(contact, f, contact.geom[1] == floor);
    ++count;
  }
  check(count > 0 && actual.force[2] > 0,
        "Toe-only floor-on-foot force missing/wrong sign");
  // Whole free subtree: total external force = mass * COM acceleration.
  double jac[18]{}, acceleration[3]{};
  mj_jacSubtreeCom(model.get(), data.get(), jac, left);
  mju_mulMatVec(acceleration, jac, data->qacc, 3, model->nv);
  for (int k = 0; k < 3; ++k)
    near(actual.force[k] + 2 * model->opt.gravity[k], 2 * acceleration[k],
         1e-7);
  // At rest, full angular momentum balance about subtree CoM via inverse
  // dynamics.
  mj_rnePostConstraint(model.get(), data.get());
  double moment_com[3], lever[3];
  mju_cross(lever, data->subtree_com + 3 * left, actual.force.data());
  mju_sub3(moment_com, actual.moment.data(), lever);
  double external_moment[3]{};
  for (int body = 1; body < model->nbody; ++body)
    if (foot_side(model.get(), body, left, right) == 0)
      mju_addTo3(external_moment, data->cfrc_ext + 6 * body);
  for (int k = 0; k < 3; ++k)
    near(moment_com[k], external_moment[k], 1e-7);
  // Independent finite-difference momentum check at a moving/rotating state.
  data->qvel[0] = .1;
  data->qvel[1] = -.2;
  data->qvel[2] = .03;
  data->qvel[3] = .2;
  data->qvel[4] = -.1;
  data->qvel[5] = .3;
  mj_forward(model.get(), data.get());
  ContactWrench moving;
  for (int i = 0; i < data->ncon; ++i) {
    const auto &contact = data->contact[i];
    if (contact.geom[0] != floor && contact.geom[1] != floor)
      continue;
    const int other =
        contact.geom[0] == floor ? contact.geom[1] : contact.geom[0];
    if (foot_side(model.get(), model->geom_bodyid[other], left, right) != 0)
      continue;
    double f[6];
    mj_contactForce(model.get(), data.get(), i, f);
    moving.add(contact, f, contact.geom[1] == floor);
  }
  auto momentum = [&](double offset) {
    auto scratch = make_data(model.get());
    mj_copyData(scratch.get(), model.get(), data.get());
    mj_integratePos(model.get(), scratch->qpos, data->qvel, offset);
    for (int j = 0; j < model->nv; ++j)
      scratch->qvel[j] += offset * data->qacc[j];
    mj_kinematics(model.get(), scratch.get());
    mj_comPos(model.get(), scratch.get());
    mj_comVel(model.get(), scratch.get());
    mj_subtreeVel(model.get(), scratch.get());
    std::array<double, 6> result{};
    mju_scl3(result.data(), scratch->subtree_linvel + 3 * left, 2);
    double orbital[3];
    mju_cross(orbital, scratch->subtree_com + 3 * left, result.data());
    mju_add3(result.data() + 3, scratch->subtree_angmom + 3 * left, orbital);
    return result;
  };
  const double epsilon = 1e-6;
  const auto before = momentum(-epsilon), after = momentum(epsilon);
  double gravity[3], gravity_moment[3];
  mju_scl3(gravity, model->opt.gravity, 2);
  mju_cross(gravity_moment, data->subtree_com + 3 * left, gravity);
  for (int k = 0; k < 3; ++k) {
    near((after[k] - before[k]) / (2 * epsilon), moving.force[k] + gravity[k],
         1e-5);
    near((after[k + 3] - before[k + 3]) / (2 * epsilon),
         moving.moment[k] + gravity_moment[k], 1e-5);
  }
  // Settled static support and a true no-contact flight state.
  mj_resetData(model.get(), data.get());
  for (int step = 0; step < 2400; ++step)
    mj_step(model.get(), data.get());
  mj_forward(model.get(), data.get());
  ContactWrench settled;
  for (int i = 0; i < data->ncon; ++i) {
    const auto &contact = data->contact[i];
    if (contact.geom[0] != floor && contact.geom[1] != floor)
      continue;
    double f[6];
    mj_contactForce(model.get(), data.get(), i, f);
    settled.add(contact, f, contact.geom[1] == floor);
  }
  near(settled.force[2], -2 * model->opt.gravity[2], 1e-7);
  check(plane_contact(settled).defined, "Static-support CoP undefined");
  mj_resetData(model.get(), data.get());
  data->qpos[2] = 2;
  mj_forward(model.get(), data.get());
  check(data->ncon == 0, "Airborne fixture unexpectedly contacting floor");
  near(data->qacc[2], model->opt.gravity[2]);
  std::cout << "Contact wrench, capture, toe aggregation, static/flight and "
               "linear/angular "
               "balance tests passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
