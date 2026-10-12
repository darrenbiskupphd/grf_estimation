#include "tracking/contact_wrench.hpp"
#include "tracking/run_bundle.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

void check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
int main() try {
  using namespace tracking;
  char error[512]{};
  SpecPtr spec(mj_parseXMLString("<mujoco><worldbody><body><freejoint/><geom "
                                 "size='.1'/></body></worldbody></mujoco>",
                                 nullptr, error, sizeof(error)),
               mj_deleteSpec);
  check(bool(spec), error);
  ModelPtr model(mj_compile(spec.get(), nullptr), mj_deleteModel);
  check(bool(model), "Model compile failed");
  auto d = make_data(model.get());
  ReplayFrame f;
  f.wrench_available = true;
  f.time = 0;
  f.qpos.assign(d->qpos, d->qpos + model->nq);
  f.qvel.assign(d->qvel, d->qvel + model->nv);
  f.tracking_positions_m.resize(48);
  const auto flight = plane_contact(ContactWrench{});
  f.cop_left = f.cop_right = flight.cop;
  f.torque_z_left = f.torque_z_right = flight.free_moment_z;
  nlohmann::json summary = {{"mujoco_version", mj_versionString()}};
  const std::filesystem::path path = "test_bundle_v2.grf",
                              legacy = "test_bundle_v1.grf";
  struct Cleanup {
    std::filesystem::path a, b;
    ~Cleanup() {
      std::filesystem::remove(a);
      std::filesystem::remove(b);
    }
  } cleanup{path, legacy};
  save_run_bundle(path, model.get(), {f}, summary);
  auto loaded = load_run_bundle(path);
  check(loaded.frames.size() == 1, "Round-trip frame count");
  check(loaded.frames[0].wrench_available && !loaded.frames[0].cop_defined_left,
        "Round-trip mask");
  check(std::isnan(loaded.frames[0].cop_left[0]), "Round-trip NaN");

  // Construct a real V1 payload from V2's unchanged prefix (legacy proxy is
  // finite).
  f.cop_left = {0, 0, 0};
  f.cop_right = {0, 0, 0};
  f.grf_left = f.grf_right = {0, 0, 1};
  f.cop_defined_left = f.cop_defined_right = true;
  f.torque_z_left = f.torque_z_right = 0;
  save_run_bundle(path, model.get(), {f}, summary);
  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
  std::uint64_t model_bytes = 0, summary_bytes = 0;
  std::memcpy(&model_bytes, bytes.data() + 16, 8);
  std::memcpy(&summary_bytes, bytes.data() + 24, 8);
  std::uint32_t version = 1;
  std::memcpy(bytes.data() + 8, &version, 4);
  std::ofstream old(legacy, std::ios::binary);
  old.write(bytes.data(), 60); // omit V2 marker count
  old.write(bytes.data() + 64, model_bytes + summary_bytes);
  const size_t old_frame_bytes = 8 +
                                 8 * (model->nq + model->nv + model->na +
                                      model->nu + 7 * model->nmocap + 12) +
                                 280;
  old.write(bytes.data() + 64 + model_bytes + summary_bytes, old_frame_bytes);
  old.close();
  auto v1 = load_run_bundle(legacy);
  check(!v1.frames[0].wrench_available && !v1.frames[0].cop_defined_left,
        "Legacy proxies incorrectly promoted to labels");
  check(v1.frames[0].contact_proxy_left == f.cop_left, "Legacy proxy lost");
  bool rejected = false;
  try {
    save_run_bundle(path, model.get(), v1.frames, summary);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "Legacy frames written as V2 labels");
  f.cop_left[0] = 1;
  rejected = false;
  try {
    save_run_bundle(path, model.get(), {f}, summary);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CoP inconsistent with raw wrench accepted");
  std::cout << "V2 round-trip and legacy V1 tests passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
