#include "viewer.hpp"

#include <GLFW/glfw3.h>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "mujoco_raii.hpp"
#include "tracking/qmc_sampling.hpp"

namespace {
struct GlfwSession {
  GlfwSession() {
    if (!glfwInit())
      throw std::runtime_error(
          "GLFW initialization failed; replay needs a display");
  }
  ~GlfwSession() { glfwTerminate(); }
};

struct ViewerState {
  mjModel *model = nullptr;
  mjvCamera cam{};
  mjvOption opt{};
  mjvScene scn{};
  mjrContext con{};
  bool follow_root = false;
  bool button_left = false, button_middle = false, button_right = false;
  double last_mouse_x = 0, last_mouse_y = 0;
  ~ViewerState() {
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
  }
};

void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
  if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  } else if (act == GLFW_PRESS && key == GLFW_KEY_F) {
    auto &viewer =
        *static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
    viewer.follow_root = !viewer.follow_root;
  }
}

void mouse_button(GLFWwindow *window, int button, int act, int mods) {
  auto &viewer = *static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  viewer.button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  viewer.button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  viewer.button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
  glfwGetCursorPos(window, &viewer.last_mouse_x, &viewer.last_mouse_y);
}

void mouse_move(GLFWwindow *window, double xpos, double ypos) {
  auto &viewer = *static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  if (!viewer.button_left && !viewer.button_middle && !viewer.button_right)
    return;

  // Compute mouse displacement
  double dx = xpos - viewer.last_mouse_x;
  double dy = ypos - viewer.last_mouse_y;
  viewer.last_mouse_x = xpos;
  viewer.last_mouse_y = ypos;

  int width, height;
  glfwGetWindowSize(window, &width, &height);

  if (height <= 0)
    return;

  // Determine viewer action based on modifier keys and active button
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if (viewer.button_right)
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (viewer.button_left)
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  else
    action = mjMOUSE_ZOOM;

  mjv_moveCamera(viewer.model, action, dx / height, dy / height, &viewer.scn,
                 &viewer.cam);
}

void scroll(GLFWwindow *window, double xoffset, double yoffset) {
  auto &viewer = *static_cast<ViewerState *>(glfwGetWindowUserPointer(window));
  mjv_moveCamera(viewer.model, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &viewer.scn,
                 &viewer.cam);
}

} // namespace

void render_replay(mjModel *m,
                   const std::vector<tracking::ReplayFrame> &replay_buffer,
                   bool follow_root, const char *title,
                   double playback_speed) {
  if (replay_buffer.empty())
    return;
  if (!std::isfinite(playback_speed) || playback_speed <= 0)
    throw std::runtime_error("Replay speed must be finite and positive");
  GlfwSession session;

  std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_owner(
      glfwCreateWindow(1200, 900, title, nullptr, nullptr), glfwDestroyWindow);
  GLFWwindow *window = window_owner.get();
  if (!window)
    throw std::runtime_error("Could not create viewer window");

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  auto data = make_data(m);
  mjData *d = data.get();
  ViewerState viewer;
  viewer.model = m;
  viewer.follow_root = follow_root;
  auto &cam = viewer.cam;
  auto &opt = viewer.opt;
  auto &scn = viewer.scn;
  auto &con = viewer.con;

  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // Initial Camera Position
  cam.lookat[0] = 0.0;
  cam.lookat[1] = 0.0;
  cam.lookat[2] = 1.0;
  cam.distance = 3.5;
  cam.azimuth = 90;
  cam.elevation = -15;

  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // Group 2 contains the blue reference targets and orange tracking sites.
  opt.sitegroup[2] = 1;
  // QMC sites are drawn below as decorative geometry. Showing their native
  // dynamic sites would make every marker cast a physical-body shadow.
  opt.sitegroup[3] = 0;
  const auto marker_sites = tracking::qmc_marker_site_ids(m);

  glfwSetWindowUserPointer(window, &viewer);
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  size_t frame_idx = 0;
  double start_real_time = glfwGetTime();
  double start_sim_time = replay_buffer[0].time;

  while (!glfwWindowShouldClose(window)) {
    double elapsed_real =
        (glfwGetTime() - start_real_time) * playback_speed;

    // kill viewer if the trajectory completes
    if (elapsed_real > (replay_buffer.back().time - start_sim_time)) {
      break;
    }

    // skip the buffer frame index to the next one matching real elapsed time
    while (frame_idx < replay_buffer.size() - 1 &&
           (replay_buffer[frame_idx + 1].time - start_sim_time) <=
               elapsed_real) {
      frame_idx++;
    }

    // Update engine state
    mju_copy(d->qpos, replay_buffer[frame_idx].qpos.data(), m->nq);
    mju_copy(d->qvel, replay_buffer[frame_idx].qvel.data(), m->nv);
    if (!replay_buffer[frame_idx].act.empty())
      mju_copy(d->act, replay_buffer[frame_idx].act.data(), m->na);
    if (!replay_buffer[frame_idx].ctrl.empty())
      mju_copy(d->ctrl, replay_buffer[frame_idx].ctrl.data(), m->nu);
    d->time = replay_buffer[frame_idx].time;

    if (!replay_buffer[frame_idx].mocap_pos.empty()) {
      mju_copy(d->mocap_pos, replay_buffer[frame_idx].mocap_pos.data(),
               3 * m->nmocap);
      mju_copy(d->mocap_quat, replay_buffer[frame_idx].mocap_quat.data(),
               4 * m->nmocap);
    }
    if (viewer.follow_root) {
      cam.lookat[0] = d->qpos[0];
      cam.lookat[1] = d->qpos[1];
    }

    // Forward kinematics pass to recalculate contacts and geometry
    mj_forward(m, d);

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

    // Render the massless marker cloud after scene construction so it remains
    // visible but cannot distort the ground with a large collective shadow.
    for (const int site : marker_sites) {
      if (scn.ngeom >= scn.maxgeom)
        break;
      auto &geom = scn.geoms[scn.ngeom++];
      mjv_initGeom(&geom, mjGEOM_SPHERE, m->site_size + 3 * site,
                   d->site_xpos + 3 * site, nullptr,
                   m->site_rgba + 4 * site);
      geom.category = mjCAT_DECOR;
      geom.objtype = mjOBJ_SITE;
      geom.objid = site;
    }

    // Ground-reaction arrows begin at the vertical-force-weighted contact
    // position. They are a visual diagnostic, not validated CoP labels.
    auto draw_grf = [&](const auto &cop, const auto &grf, float *color) {
      if (grf[2] > 1.0 &&
          scn.ngeom < scn.maxgeom) { // Draw if vertical force is meaningful
        double scale = 0.005;
        double p2[3] = {cop[0] + grf[0] * scale, cop[1] + grf[1] * scale,
                        cop[2] + grf[2] * scale};
        mjv_initGeom(&scn.geoms[scn.ngeom], mjGEOM_ARROW, nullptr, nullptr,
                     nullptr, color);
        mjv_connector(&scn.geoms[scn.ngeom++], mjGEOM_ARROW, 0.015,
                      cop.data(), p2);
      }
    };
    const auto &current_frame = replay_buffer[frame_idx];
    float cyan[4] = {0.0f, 1.0f, 1.0f, 1.0f}; // Opaque Cyan (R, G, B, A)
    draw_grf(current_frame.cop_left, current_frame.grf_left, cyan);
    draw_grf(current_frame.cop_right, current_frame.grf_right, cyan);

    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}
