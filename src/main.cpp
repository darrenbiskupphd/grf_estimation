#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "morphology_randomizer.hpp"
#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/tasks/humanoid/stand/stand.h"

struct SimConfig {
    double duration = 1.0;
    bool render = false;
    std::string output_path = "";
};

// Viewer Globals
mjModel* m_viewer = nullptr;
mjData* d_viewer = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

// Mouse state for viewer interaction
bool viewer_button_left = false;
bool viewer_button_middle = false;
bool viewer_button_right = false;
double viewer_last_mouse_x = 0;
double viewer_last_mouse_y = 0;

// Viewer Callbacks
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    viewer_button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    viewer_button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    viewer_button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &viewer_last_mouse_x, &viewer_last_mouse_y);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!viewer_button_left && !viewer_button_middle && !viewer_button_right) return;
    
    // Compute mouse displacement
    double dx = xpos - viewer_last_mouse_x;
    double dy = ypos - viewer_last_mouse_y;
    viewer_last_mouse_x = xpos;
    viewer_last_mouse_y = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Determine viewer action based on modifier keys and active button
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (viewer_button_right) action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (viewer_button_left) action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else action = mjMOUSE_ZOOM;

    mjv_moveCamera(m_viewer, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m_viewer, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// Global task pointer for the mjcb_sensor residual callback
static mjpc::Task* g_task = nullptr;
static void residual_sensor_cb(const mjModel* m, mjData* d, int stage) {
    if (stage == mjSTAGE_ACC && g_task) g_task->Residual(m, d, d->sensordata);
}

// Simulation Core
void run_simulation(mjModel* m, std::vector<ReplayFrame>& replay_buffer, const SimConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Physics at 2400Hz, recording at 600Hz, planning every `k` steps
    double capture_rate_hz = 600.0;
    double physics_timestep = 1.0 / (capture_rate_hz * 4.0);
    constexpr int kPlanEveryNSteps = 12;

    // Set the high-fidelity physics timestep
    m->opt.timestep = physics_timestep;
    mjData* d = mj_makeData(m);

    // --- Stand Task Agent Setup ---
    auto stand_task = std::make_shared<mjpc::humanoid::Stand>();
    mjpc::Agent agent(m, stand_task);
    agent.Initialize(m);
    agent.Allocate();
    agent.Reset();
    agent.plan_enabled = true;
    agent.estimator_enabled = false;

    // Install residual callback so mj_step triggers cost evaluation
    g_task = agent.ActiveTask();
    mjcb_sensor = &residual_sensor_cb;

    // Single planning thread: caller (bash) handles parallelism via multiple processes
    mjpc::ThreadPool pool(1);

    int key_id = mj_name2id(m, mjOBJ_KEY, "drop_impact");
    if (key_id >= 0) {
        mju_copy(d->qpos, m->key_qpos + key_id * m->nq, m->nq);
        mju_copy(d->qvel, m->key_qvel + key_id * m->nv, m->nv);
        mju_copy(d->act,  m->key_act  + key_id * m->na, m->na);
    } else {
        std::cerr << "Warning: Keyframe 'drop_impact' not found." << std::endl;
    }

    // Seed the planner with the initial state
    mj_forward(m, d);
    agent.ActiveTask()->Transition(m, d);
    agent.state.Set(m, d);
    agent.PlanIteration(&pool);

    replay_buffer.reserve(static_cast<size_t>(config.duration * capture_rate_hz) + 2);
    double next_record_time = 0.0;
    int step_count = 0;
    StateRecorder recorder(m);

    while (d->time < config.duration) {
        // Run one planning iteration every N physics steps (synchronous, blocks physics)
        if (step_count % kPlanEveryNSteps == 0) {
            agent.ActiveTask()->Transition(m, d);
            agent.state.Set(m, d);
            agent.PlanIteration(&pool);
        }

        // Apply the planned action to ctrl
        agent.ActivePlanner().ActionFromPolicy(
            d->ctrl, agent.state.state().data(), agent.state.time(), false);

        mj_step(m, d);
        step_count++;

        if (d->time >= next_record_time) {
            ReplayFrame frame;
            frame.time = d->time;
            frame.qpos.assign(d->qpos, d->qpos + m->nq);
            frame.qvel.assign(d->qvel, d->qvel + m->nv);

            if (!recorder.extract_physics(m, d, frame)) {
                std::cout << "Early Termination: Non-foot body contact with floor at t=" << d->time << std::endl;
                break;
            }

            replay_buffer.push_back(frame);
            next_record_time += 1.0 / capture_rate_hz;
        }
    }

    if (!config.output_path.empty()) {
        recorder.write_csv(config.output_path, replay_buffer);
        std::cout << "Data saved to " << config.output_path << std::endl;
    }

    // Cleanup
    mjcb_sensor = nullptr;
    g_task = nullptr;
    mj_deleteData(d);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_time = end_time - start_time;
    std::cout << "Simulation Wall Time: " << wall_time.count() << " seconds" << std::endl;
}

// Visualization
void render_replay(mjModel* m, const std::vector<ReplayFrame>& replay_buffer) {
    if (!glfwInit()) return;

    GLFWwindow* window = glfwCreateWindow(1200, 900, "MuJoCo GRF Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjData* d = mj_makeData(m);

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

    // Enable physics visualizations
    opt.flags[mjVIS_CONTACTFORCE] = 1;
    // opt.flags[mjVIS_CONTACTPOINT] = 1;

    opt.sitegroup[3] = 1; // Enable rendering for site group 3 (QMC markers)

    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    m_viewer = m;
    d_viewer = d;

    size_t frame_idx = 0;
    double start_real_time = glfwGetTime();
    double start_sim_time = replay_buffer[0].time;

    while (!glfwWindowShouldClose(window)) {
        double elapsed_real = glfwGetTime() - start_real_time;

        // kill viewer if the trajectory completes
        if (elapsed_real > (replay_buffer.back().time - start_sim_time)) {
            break;
        }

        // skip the buffer frame index to the next one matching real elapsed time
        while (frame_idx < replay_buffer.size() - 1 && 
               (replay_buffer[frame_idx + 1].time - start_sim_time) <= elapsed_real) {
            frame_idx++;
        }

        // Update engine state
        mju_copy(d->qpos, replay_buffer[frame_idx].qpos.data(), m->nq);
        mju_copy(d->qvel, replay_buffer[frame_idx].qvel.data(), m->nv);
        d->time = replay_buffer[frame_idx].time;

        // Forward kinematics pass to recalculate contacts and geometry
        mj_forward(m, d);

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    glfwTerminate();
}

int main(int argc, char** argv) {
    SimConfig config;

    std::string model_path = "assets/winter_baseline_male.xml";
    int qmc_index = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" && i + 1 < argc) {
            config.duration = std::stod(argv[++i]);
        } else if (arg == "--render") {
            config.render = true;
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--qmc-index" && i + 1 < argc) {
            qmc_index = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or malformed argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--duration <sec>] [--render] [--output <path>] [--model <path>] [--qmc-index <int>]" << std::endl;
            return 1;
        }
    }

    char error[1000] = "";
    
    std::cout << "Starting episode with " << model_path << " (QMC Index: " << qmc_index << ")" << std::endl;

    mjSpec* spec = mj_parseXML(model_path.c_str(), nullptr, error, 1000);
    if (!spec) {
        std::cerr << "MuJoCo Load Error: " << error << std::endl;
        return 1;
    }

    // Apply geometric Domain Randomization using QMC
    randomize_mjspec_geometry(spec, qmc_index);

    // Add QMC procedural markers before compilation
    add_qmc_markers_to_spec(spec, 7);

    // Compile into final rigorous mjModel
    mjModel* m = mj_compile(spec, nullptr);
    mj_deleteSpec(spec);

    if (!m) {
        std::cerr << "MuJoCo Compile Error." << std::endl;
        return 1;
    }

    // Snap markers to the randomized capsule surfaces
    randomize_marker_positions(m, 7, qmc_index);

    // =====================================================================
    // [DELETE ME] MASS VERIFICATION
    // =====================================================================
    double total_mass = 0.0;
    for (int i = 1; i < m->nbody; ++i) { // Skip worldbody
        total_mass += m->body_mass[i];
    }
    std::cout << "--- MASS VERIFICATION (DELETE ME) ---" << std::endl;
    std::cout << "Total Body Mass: " << total_mass << " kg" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    // =====================================================================

    // Simulate and optionally render
    std::vector<ReplayFrame> replay_buffer;
    run_simulation(m, replay_buffer, config);

    if (config.render && !replay_buffer.empty()) {
        render_replay(m, replay_buffer);
    }

    mj_deleteModel(m);

    return 0;
}