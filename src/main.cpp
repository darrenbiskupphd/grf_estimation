#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tracking/run_bundle.hpp"
#include "tracking/export.hpp"
#include "tracking/timing.hpp"
#include "tracking/tracking.hpp"
#include "viewer.hpp"

namespace {
double number(const std::string &text) {
  size_t used = 0;
  const double result = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(result))
    throw std::runtime_error("Expected a finite number: " + text);
  return result;
}

int integer(const std::string &text) {
  size_t used = 0;
  const int result = std::stoi(text, &used);
  if (used != text.size())
    throw std::runtime_error("Expected an integer: " + text);
  return result;
}

void usage(std::ostream &stream) {
  stream
      << "Usage:\n"
      << "  data_factory run --reference <reference.json> --output <run.grf> [options]\n"
      << "  data_factory replay <run.grf> [--speed <multiplier>]\n"
      << "  data_factory export <run.grf>... --output <dataset.npz> [options]\n\n"
      << "Run options:\n"
      << "  --model <XML>                 assets/winter_baseline_male.xml\n"
      << "  --reference <JSON>            timestamped 16-target reference (required)\n"
      << "  --duration <seconds>          full clip (caps at clip end)\n"
      << "  --qmc-index <index>           0 (nominal morphology)\n"
      << "  --layout-index <index>        0 (independent marker layout)\n"
      << "  --capture-hz <rate>           600 (must divide 2400 into integer steps)\n"
      << "  --threads <count>             1\n"
      << "  --effort-rate-weight <value>  100 (activation-rate cost; 0 disables)\n"
      << "  --stance-roll-weight <value> 0 (reference-gated lateral foot roll)\n"
      << "  --stance-roll-still-time <s> 0 (prior low/slow reference duration)\n"
      << "  --knee-hip-pos-scale <value> 1 (experimental knee/hip position weights)\n"
      << "  --elbow-velocity-weight <v>  0.02 (reference elbow-velocity tracking)\n"
      << "  --elbow-position-weight <v>  16 (both elbow position targets)\n"
      << "  --hand-position-weight <v>   8 (both hand position targets)\n"
      << "  --render                      render the saved in-memory replay\n\n"
      << "Replay options:\n"
      << "  --speed <multiplier>          1 (for example .25 for quarter speed)\n\n"
      << "Export options (offline; no simulation or force recomputation):\n"
      << "  --layout-index <index>        recorded markers by default; saved-pose alternative\n"
      << "  --permutation-seed <index>    0 (run-salted, fixed within each episode)\n"
      << "  --min-normal-force <N>        0 (Fz > N for CoP/Tz only; never masks GRF)\n"
      << "  --allow-incomplete            include incomplete V2 runs for diagnostics\n";
}

int run(int argc, char **argv) {
  tracking::PrepareConfig prepare;
  tracking::RunConfig config;
  config.replay_every_n_steps = tracking::capture_stride(600);
  std::string model_path = "assets/winter_baseline_male.xml";
  bool output_set = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--reference" && i + 1 < argc) {
      prepare.reference_path = argv[++i];
    } else if (arg == "--layout-index" && i + 1 < argc) {
      prepare.layout_index = integer(argv[++i]);
      if (prepare.layout_index < 0)
        throw std::runtime_error("--layout-index must be nonnegative");
    } else if (arg == "--capture-hz" && i + 1 < argc) {
      config.replay_every_n_steps = tracking::capture_stride(number(argv[++i]));
    } else if (arg == "--duration" && i + 1 < argc) {
      config.duration = number(argv[++i]);
      if (config.duration <= 0)
        throw std::runtime_error("--duration must be positive");
    } else if (arg == "--qmc-index" && i + 1 < argc) {
      prepare.qmc_index = integer(argv[++i]);
      if (prepare.qmc_index < 0)
        throw std::runtime_error("--qmc-index must be nonnegative");
    } else if (arg == "--output" && i + 1 < argc) {
      config.output_path = argv[++i];
      output_set = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      config.planner_threads = integer(argv[++i]);
      if (config.planner_threads < 1)
        throw std::runtime_error("--threads must be positive");
    } else if (arg == "--effort-rate-weight" && i + 1 < argc) {
      prepare.effort_rate_weight = number(argv[++i]);
      if (prepare.effort_rate_weight < 0)
        throw std::runtime_error("--effort-rate-weight must be nonnegative");
    } else if (arg == "--stance-roll-weight" && i + 1 < argc) {
      prepare.stance_roll_weight = number(argv[++i]);
      if (prepare.stance_roll_weight < 0)
        throw std::runtime_error("--stance-roll-weight must be nonnegative");
    } else if (arg == "--stance-roll-still-time" && i + 1 < argc) {
      prepare.stance_roll_still_time_s = number(argv[++i]);
      if (prepare.stance_roll_still_time_s < 0)
        throw std::runtime_error("--stance-roll-still-time must be nonnegative");
    } else if (arg == "--knee-hip-pos-scale" && i + 1 < argc) {
      prepare.knee_hip_position_scale = number(argv[++i]);
      if (prepare.knee_hip_position_scale < 0)
        throw std::runtime_error("--knee-hip-pos-scale must be nonnegative");
    } else if (arg == "--elbow-velocity-weight" && i + 1 < argc) {
      prepare.elbow_velocity_weight = number(argv[++i]);
      if (prepare.elbow_velocity_weight < 0)
        throw std::runtime_error("--elbow-velocity-weight must be nonnegative");
    } else if (arg == "--elbow-position-weight" && i + 1 < argc) {
      prepare.elbow_position_weight = number(argv[++i]);
      if (prepare.elbow_position_weight < 0)
        throw std::runtime_error("--elbow-position-weight must be nonnegative");
    } else if (arg == "--hand-position-weight" && i + 1 < argc) {
      prepare.hand_position_weight = number(argv[++i]);
      if (prepare.hand_position_weight < 0)
        throw std::runtime_error("--hand-position-weight must be nonnegative");
    } else if (arg == "--render") {
      config.render = true;
    } else if (arg == "--help") {
      usage(std::cout);
      return 0;
    } else {
      throw std::runtime_error("Unknown or malformed run option: " + arg);
    }
  }
  if (prepare.reference_path.empty())
    throw std::runtime_error("run requires --reference <reference.json>; bundled motions are no longer inputs");
  if (!output_set)
    throw std::runtime_error("run requires --output <run.grf>");
  if (config.output_path.extension() != ".grf")
    throw std::runtime_error("Run output must use the .grf extension");
  auto prepared = tracking::build_reference_tracking_model(model_path, prepare);
  config.motion = prepared.reference_id;
  tracking::run(std::move(prepared), config);
  return 0;
}

int export_data(int argc, char **argv) {
  std::vector<std::filesystem::path> inputs;
  std::filesystem::path output;
  int layout = -1, seed = 0;
  double min_normal_force = 0;
  bool allow_incomplete = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc)
      output = argv[++i];
    else if (arg == "--layout-index" && i + 1 < argc) {
      layout = integer(argv[++i]);
      if (layout < 0) throw std::runtime_error("Layout index must be nonnegative");
    } else if (arg == "--permutation-seed" && i + 1 < argc) {
      seed = integer(argv[++i]);
      if (seed < 0) throw std::runtime_error("Permutation seed must be nonnegative");
    } else if (arg == "--min-normal-force" && i + 1 < argc) {
      min_normal_force = number(argv[++i]);
      if (min_normal_force < 0)
        throw std::runtime_error("--min-normal-force must be nonnegative");
    } else if (arg == "--allow-incomplete")
      allow_incomplete = true;
    else if (arg == "--help") {
      usage(std::cout);
      return 0;
    } else if (!arg.empty() && arg[0] != '-')
      inputs.emplace_back(arg);
    else
      throw std::runtime_error("Unknown or malformed export option: " + arg);
  }
  if (inputs.empty())
    throw std::runtime_error("export requires <run.grf>... --output <dataset.npz>");
  if (output.empty() || output.extension() != ".npz")
    throw std::runtime_error("export requires --output <dataset.npz>");
  for (const auto &input : inputs)
    if (input.extension() != ".grf")
      throw std::runtime_error("Export inputs must use the .grf extension");
  tracking::export_run_bundles(inputs, output, layout, seed, allow_incomplete,
                              min_normal_force);
  return 0;
}

int replay(int argc, char **argv) {
  if (argc < 3)
    throw std::runtime_error("Usage: data_factory replay <run.grf>");
  double speed = 1.0;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--speed" && i + 1 < argc) {
      speed = number(argv[++i]);
      if (speed <= 0)
        throw std::runtime_error("Replay --speed must be positive");
    } else {
      throw std::runtime_error("Unknown or malformed replay option: " + arg);
    }
  }
  const auto bundle = tracking::load_run_bundle(argv[2]);
  std::cout << "Loaded " << bundle.frames.size() << " frames, "
            << bundle.frames.back().time << " s"
            << " (" << bundle.summary.value("termination", "unknown") << ")\n";
  // Report saved whole-run diagnostics, not metrics recomputed during replay.
  const auto print_metric = [&](const char *label, const char *key,
                                const char *unit) {
    std::cout << label << ": ";
    const auto metric = bundle.summary.find(key);
    if (metric != bundle.summary.end() && metric->is_number())
      std::cout << metric->get<double>() << ' ' << unit;
    else
      std::cout << "unavailable";
    std::cout << '\n';
  };
  print_metric("Saved tracking RMSE", "tracking_rmse_m", "m");
  print_metric("Saved mean contact slip (loaded time)",
               "contact_slip_mean_m_per_s", "m/s");
  std::cout << std::flush;
  render_replay(bundle.model.get(), bundle.frames, true, "Saved tracking run",
                speed);
  return 0;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc < 2 || std::string(argv[1]) == "--help" ||
      std::string(argv[1]) == "help") {
    usage(argc < 2 ? std::cerr : std::cout);
    return argc < 2 ? 1 : 0;
  }
  const std::string command = argv[1];
  if (command == "run")
    return run(argc, argv);
  if (command == "replay")
    return replay(argc, argv);
  if (command == "export")
    return export_data(argc, argv);
  throw std::runtime_error("Unknown command: " + command);
} catch (const std::exception &error) {
  std::cerr << "Error: " << error.what() << '\n';
  return 1;
}
