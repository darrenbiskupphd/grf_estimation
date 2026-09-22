#include "run_bundle.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

namespace tracking {
namespace {
constexpr std::array<char, 8> kMagic = {'G', 'R', 'F', 'R', 'U', 'N', '1', 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kMaxModelBytes = 1ULL << 30;
constexpr std::uint64_t kMaxSummaryBytes = 64ULL << 20;
constexpr std::uint64_t kMaxFrames = 10'000'000;

template <typename T> void write_value(std::ostream &stream, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T> T read_value(std::istream &stream) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  stream.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!stream)
    throw std::runtime_error("Unexpected end of GRF run bundle");
  return value;
}

void write_doubles(std::ostream &stream, const double *values, size_t count) {
  stream.write(reinterpret_cast<const char *>(values),
               static_cast<std::streamsize>(count * sizeof(double)));
}

void read_doubles(std::istream &stream, double *values, size_t count) {
  stream.read(reinterpret_cast<char *>(values),
              static_cast<std::streamsize>(count * sizeof(double)));
  if (!stream)
    throw std::runtime_error("Unexpected end of GRF run bundle frame");
}

void require_finite(const std::vector<double> &values, size_t expected,
                    const char *name) {
  if (values.size() != expected ||
      !std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error(std::string("Invalid ") + name +
                             " in GRF run bundle");
  }
}

void require_finite(const double *values, size_t count, const char *name) {
  if (!std::all_of(values, values + count,
                   [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error(std::string("Invalid ") + name +
                             " in GRF run bundle");
  }
}

void write_foot(std::ostream &stream, const FootContactFrame &foot) {
  write_doubles(stream, foot.geom_normal_load.data(),
                foot.geom_normal_load.size());
  for (const double value : {foot.heel_normal_load, foot.forefoot_normal_load,
                             foot.toe_normal_load,
                             foot.lateral_edge_normal_load})
    write_value(stream, value);
  write_doubles(stream, foot.contact_point_local.data(),
                foot.contact_point_local.size());
  for (const double value : {foot.roll, foot.pitch, foot.ankle_y,
                             foot.ankle_x, foot.mtp, foot.mtp_velocity})
    write_value(stream, value);
  write_value(stream, static_cast<std::int32_t>(foot.contact_count));
}

FootContactFrame read_foot(std::istream &stream) {
  FootContactFrame foot;
  read_doubles(stream, foot.geom_normal_load.data(),
               foot.geom_normal_load.size());
  for (double *value : {&foot.heel_normal_load, &foot.forefoot_normal_load,
                        &foot.toe_normal_load, &foot.lateral_edge_normal_load})
    *value = read_value<double>(stream);
  read_doubles(stream, foot.contact_point_local.data(),
               foot.contact_point_local.size());
  for (double *value : {&foot.roll, &foot.pitch, &foot.ankle_y,
                        &foot.ankle_x, &foot.mtp, &foot.mtp_velocity})
    *value = read_value<double>(stream);
  const auto contacts = read_value<std::int32_t>(stream);
  if (contacts < 0)
    throw std::runtime_error("Negative contact count in GRF run bundle");
  foot.contact_count = contacts;
  require_finite(foot.geom_normal_load.data(), foot.geom_normal_load.size(),
                 "per-geometry contact load");
  require_finite(&foot.heel_normal_load, 4, "contact region load");
  require_finite(foot.contact_point_local.data(),
                 foot.contact_point_local.size(), "local contact position");
  require_finite(&foot.roll, 6, "foot diagnostic");
  return foot;
}

void validate_frame(const ReplayFrame &frame, const mjModel *model,
                    double previous_time) {
  if (!std::isfinite(frame.time) || frame.time <= previous_time)
    throw std::runtime_error(
        "Run-bundle timestamps must be finite and strictly increasing");
  require_finite(frame.qpos, model->nq, "qpos");
  require_finite(frame.qvel, model->nv, "qvel");
  require_finite(frame.act, model->na, "act");
  require_finite(frame.ctrl, model->nu, "ctrl");
  require_finite(frame.mocap_pos, 3 * model->nmocap, "mocap position");
  require_finite(frame.mocap_quat, 4 * model->nmocap, "mocap quaternion");
  require_finite(frame.grf_left.data(), frame.grf_left.size(), "left GRF");
  require_finite(frame.grf_right.data(), frame.grf_right.size(), "right GRF");
  require_finite(frame.cop_left.data(), frame.cop_left.size(), "left CoP");
  require_finite(frame.cop_right.data(), frame.cop_right.size(), "right CoP");
  require_finite(frame.left_contact.geom_normal_load.data(),
                 frame.left_contact.geom_normal_load.size(),
                 "left per-geometry contact load");
  require_finite(&frame.left_contact.heel_normal_load, 4,
                 "left contact region load");
  require_finite(frame.left_contact.contact_point_local.data(),
                 frame.left_contact.contact_point_local.size(),
                 "left local contact position");
  require_finite(&frame.left_contact.roll, 6, "left foot diagnostic");
  require_finite(frame.right_contact.geom_normal_load.data(),
                 frame.right_contact.geom_normal_load.size(),
                 "right per-geometry contact load");
  require_finite(&frame.right_contact.heel_normal_load, 4,
                 "right contact region load");
  require_finite(frame.right_contact.contact_point_local.data(),
                 frame.right_contact.contact_point_local.size(),
                 "right local contact position");
  require_finite(&frame.right_contact.roll, 6, "right foot diagnostic");
  if (frame.left_contact.contact_count < 0 ||
      frame.right_contact.contact_count < 0)
    throw std::runtime_error("Negative contact count in GRF run bundle");
}

std::filesystem::path
unique_partial_path(const std::filesystem::path &destination) {
  static std::atomic_uint64_t sequence{0};
  const auto parent = destination.parent_path();
  const auto prefix = destination.filename().string() + ".partial.";
  for (int attempt = 0; attempt < 100; ++attempt) {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto candidate =
        parent / (prefix + std::to_string(tick) + "." +
                  std::to_string(sequence.fetch_add(1)));
    std::error_code error;
    const bool exists = std::filesystem::exists(candidate, error);
    if (error)
      throw std::runtime_error("Could not create temporary run bundle: " +
                               error.message());
    if (!exists)
      return candidate;
  }
  throw std::runtime_error("Could not allocate a temporary run bundle path");
}

class PartialFile {
public:
  explicit PartialFile(const std::filesystem::path &destination)
      : destination_(destination) {
    if (const auto parent = destination_.parent_path(); !parent.empty())
      std::filesystem::create_directories(parent);
    partial_ = unique_partial_path(destination_);
  }
  ~PartialFile() {
    if (!committed_) {
      std::error_code error;
      std::filesystem::remove(partial_, error);
    }
  }
  const std::filesystem::path &partial() const { return partial_; }
  void commit() {
    std::error_code error;
    std::filesystem::rename(partial_, destination_, error);
    if (error)
      throw std::runtime_error("Could not finalize run bundle: " +
                               error.message());
    committed_ = true;
  }

private:
  std::filesystem::path destination_;
  std::filesystem::path partial_;
  bool committed_ = false;
};

struct VfsScope {
  VfsScope() { mj_defaultVFS(&vfs); }
  ~VfsScope() { mj_deleteVFS(&vfs); }
  mjVFS vfs{};
};

} // namespace

void save_run_bundle(const std::filesystem::path &path, const mjModel *model,
                     const std::vector<ReplayFrame> &frames,
                     const nlohmann::json &summary) {
  if (!model || frames.empty())
    throw std::runtime_error("A run bundle needs a model and at least one frame");
  if (frames.size() > kMaxFrames)
    throw std::runtime_error("Too many frames for a GRF run bundle");
  const int model_size = mj_sizeModel(model);
  if (model_size <= 0 || static_cast<std::uint64_t>(model_size) > kMaxModelBytes)
    throw std::runtime_error("Invalid compiled model size");
  std::vector<char> model_bytes(model_size);
  mj_saveModel(model, nullptr, model_bytes.data(), model_size);
  const std::string summary_text = summary.dump();
  if (summary_text.size() > kMaxSummaryBytes)
    throw std::runtime_error("Run summary is too large");

  double previous_time = -std::numeric_limits<double>::infinity();
  for (const auto &frame : frames) {
    validate_frame(frame, model, previous_time);
    previous_time = frame.time;
  }

  PartialFile output(path);
  std::ofstream stream(output.partial(), std::ios::binary | std::ios::trunc);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_value(stream, kVersion);
  write_value(stream, static_cast<std::uint32_t>(0));
  write_value(stream, static_cast<std::uint64_t>(model_bytes.size()));
  write_value(stream, static_cast<std::uint64_t>(summary_text.size()));
  write_value(stream, static_cast<std::uint64_t>(frames.size()));
  for (const int dimension :
       {model->nq, model->nv, model->na, model->nu, model->nmocap})
    write_value(stream, static_cast<std::uint32_t>(dimension));
  stream.write(model_bytes.data(), static_cast<std::streamsize>(model_bytes.size()));
  stream.write(summary_text.data(), static_cast<std::streamsize>(summary_text.size()));
  for (const auto &frame : frames) {
    write_value(stream, frame.time);
    write_doubles(stream, frame.qpos.data(), frame.qpos.size());
    write_doubles(stream, frame.qvel.data(), frame.qvel.size());
    write_doubles(stream, frame.act.data(), frame.act.size());
    write_doubles(stream, frame.ctrl.data(), frame.ctrl.size());
    write_doubles(stream, frame.mocap_pos.data(), frame.mocap_pos.size());
    write_doubles(stream, frame.mocap_quat.data(), frame.mocap_quat.size());
    write_doubles(stream, frame.grf_left.data(), frame.grf_left.size());
    write_doubles(stream, frame.grf_right.data(), frame.grf_right.size());
    write_doubles(stream, frame.cop_left.data(), frame.cop_left.size());
    write_doubles(stream, frame.cop_right.data(), frame.cop_right.size());
    write_foot(stream, frame.left_contact);
    write_foot(stream, frame.right_contact);
  }
  stream.close();
  output.commit();
}

RunBundle load_run_bundle(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Could not open run bundle: " + path.string());
  std::array<char, kMagic.size()> magic{};
  stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!stream || magic != kMagic)
    throw std::runtime_error("Not a GRF run bundle");
  if (read_value<std::uint32_t>(stream) != kVersion)
    throw std::runtime_error("Unsupported GRF run-bundle format version");
  (void)read_value<std::uint32_t>(stream); // Reserved for format flags.
  const auto model_size = read_value<std::uint64_t>(stream);
  const auto summary_size = read_value<std::uint64_t>(stream);
  const auto frame_count = read_value<std::uint64_t>(stream);
  std::array<std::uint32_t, 5> dimensions{};
  for (auto &dimension : dimensions)
    dimension = read_value<std::uint32_t>(stream);
  if (model_size == 0 || model_size > kMaxModelBytes ||
      summary_size > kMaxSummaryBytes || frame_count == 0 ||
      frame_count > kMaxFrames || model_size > static_cast<std::uint64_t>(INT_MAX)) {
    throw std::runtime_error("Invalid GRF run-bundle header");
  }
  std::vector<char> model_bytes(static_cast<size_t>(model_size));
  stream.read(model_bytes.data(), static_cast<std::streamsize>(model_bytes.size()));
  if (!stream)
    throw std::runtime_error("Truncated compiled model in GRF run bundle");
  std::string summary_text(static_cast<size_t>(summary_size), '\0');
  stream.read(summary_text.data(), static_cast<std::streamsize>(summary_text.size()));
  if (!stream)
    throw std::runtime_error("Truncated summary in GRF run bundle");

  RunBundle result;
  try {
    result.summary = nlohmann::json::parse(summary_text);
  } catch (const nlohmann::json::exception &error) {
    throw std::runtime_error(std::string("Invalid run-bundle summary: ") +
                             error.what());
  }
  if (!result.summary.is_object() ||
      result.summary.value("mujoco_version", std::string()) != mj_versionString()) {
    throw std::runtime_error("Run bundle requires a different MuJoCo version");
  }

  VfsScope vfs;
  if (mj_addBufferVFS(&vfs.vfs, "run.mjb", model_bytes.data(),
                      static_cast<int>(model_bytes.size())) != 0)
    throw std::runtime_error("Could not stage compiled model for replay");
  result.model.reset(mj_loadModel("run.mjb", &vfs.vfs));
  if (!result.model)
    throw std::runtime_error("Could not load compiled model from run bundle");
  const auto *model = result.model.get();
  if (dimensions != std::array<std::uint32_t, 5>{
                        static_cast<std::uint32_t>(model->nq),
                        static_cast<std::uint32_t>(model->nv),
                        static_cast<std::uint32_t>(model->na),
                        static_cast<std::uint32_t>(model->nu),
                        static_cast<std::uint32_t>(model->nmocap)}) {
    throw std::runtime_error("Run-bundle model dimensions do not match header");
  }

  result.frames.reserve(static_cast<size_t>(frame_count));
  double previous_time = -std::numeric_limits<double>::infinity();
  for (std::uint64_t index = 0; index < frame_count; ++index) {
    ReplayFrame frame;
    frame.time = read_value<double>(stream);
    frame.qpos.resize(model->nq);
    frame.qvel.resize(model->nv);
    frame.act.resize(model->na);
    frame.ctrl.resize(model->nu);
    frame.mocap_pos.resize(3 * model->nmocap);
    frame.mocap_quat.resize(4 * model->nmocap);
    read_doubles(stream, frame.qpos.data(), frame.qpos.size());
    read_doubles(stream, frame.qvel.data(), frame.qvel.size());
    read_doubles(stream, frame.act.data(), frame.act.size());
    read_doubles(stream, frame.ctrl.data(), frame.ctrl.size());
    read_doubles(stream, frame.mocap_pos.data(), frame.mocap_pos.size());
    read_doubles(stream, frame.mocap_quat.data(), frame.mocap_quat.size());
    read_doubles(stream, frame.grf_left.data(), frame.grf_left.size());
    read_doubles(stream, frame.grf_right.data(), frame.grf_right.size());
    read_doubles(stream, frame.cop_left.data(), frame.cop_left.size());
    read_doubles(stream, frame.cop_right.data(), frame.cop_right.size());
    frame.left_contact = read_foot(stream);
    frame.right_contact = read_foot(stream);
    validate_frame(frame, model, previous_time);
    previous_time = frame.time;
    result.frames.push_back(std::move(frame));
  }
  return result;
}

} // namespace tracking
