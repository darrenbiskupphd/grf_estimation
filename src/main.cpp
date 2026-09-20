#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tracking/run_bundle.hpp"
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
      << "  data_factory run --output <run.grf> [options]\n"
      << "  data_factory replay <run.grf> [--speed <multiplier>]\n\n"
      << "Run options:\n"
      << "  --model <XML>                 assets/winter_baseline_male.xml\n"
      << "  --motion walk|run             walk\n"
      << "  --start <seconds>             0 (must align to 30 Hz source frames)\n"
      << "  --duration <seconds>          full remaining reference\n"
      << "  --reference raw|rigid|retargeted  raw\n"
      << "  --qmc-index <index>          0 (nominal body, canonical markers)\n"
      << "  --threads <count>             1\n"
      << "  --render                      render the saved in-memory replay\n\n"
      << "Replay options:\n"
      << "  --speed <multiplier>          1 (for example .25 for quarter speed)\n";
}

int frame_at(double seconds) {
  if (seconds < 0)
    throw std::runtime_error("--start must be nonnegative");
  const double frames = seconds * tracking::kReferenceFps;
  const int rounded = static_cast<int>(std::llround(frames));
  if (std::abs(frames - rounded) > 1e-8)
    throw std::runtime_error("--start must align to the 30 Hz source frames");
  return rounded;
}

int run(int argc, char **argv) {
  tracking::PrepareConfig prepare;
  tracking::RunConfig config;
  std::string model_path = "assets/winter_baseline_male.xml";
  bool output_set = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--motion" && i + 1 < argc) {
      prepare.motion = argv[++i];
      config.motion = prepare.motion;
    } else if (arg == "--start" && i + 1 < argc) {
      prepare.start_frame = frame_at(number(argv[++i]));
    } else if (arg == "--duration" && i + 1 < argc) {
      config.duration = number(argv[++i]);
      if (config.duration <= 0)
        throw std::runtime_error("--duration must be positive");
    } else if (arg == "--reference" && i + 1 < argc) {
      prepare.reference = tracking::parse_reference_strategy(argv[++i]);
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
    } else if (arg == "--render") {
      config.render = true;
    } else if (arg == "--help") {
      usage(std::cout);
      return 0;
    } else {
      throw std::runtime_error("Unknown or malformed run option: " + arg);
    }
  }
  if (!output_set)
    throw std::runtime_error("run requires --output <run.grf>");
  if (config.output_path.extension() != ".grf")
    throw std::runtime_error("Run output must use the .grf extension");
  tracking::run(tracking::prepare_custom(model_path, prepare), config);
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
  throw std::runtime_error("Unknown command: " + command);
} catch (const std::exception &error) {
  std::cerr << "Error: " << error.what() << '\n';
  return 1;
}
