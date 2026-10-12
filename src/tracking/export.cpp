#include "export.hpp"

#include "qmc_sampling.hpp"
#include "run_bundle.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace tracking {
namespace {
using Json = nlohmann::json;

void require_finite(const double *values, size_t count, const char *name) {
  if (!std::all_of(values, values + count,
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error(std::string("Nonfinite ") + name + " for export");
}

void require_finite(const std::vector<double> &values, size_t expected,
                    const char *name) {
  if (values.size() != expected)
    throw std::runtime_error(std::string("Invalid ") + name + " for export");
  if (!values.empty())
    require_finite(values.data(), values.size(), name);
}

std::string source_id(const Json &summary) {
  // Never substitute a bundle path or a run/model/layout variant identity.
  std::string result;
  for (const char *path : {"/source_id", "/reference/source_id",
                           "/preparation/reference/source_id"}) {
    const Json::json_pointer pointer(path);
    if (!summary.contains(pointer))
      continue;
    const auto &value = summary.at(pointer);
    if (!value.is_string() || value.get<std::string>().find_first_not_of(
                                  " \t\r\n") == std::string::npos)
      throw std::runtime_error(std::string("Invalid recording source_id at ") +
                               path);
    const auto candidate = value.get<std::string>();
    if (!result.empty() && result != candidate)
      throw std::runtime_error(
          "Conflicting original recording source_id values");
    result = candidate;
  }
  if (result.empty())
    throw std::runtime_error(
        "Export needs the original recording source_id in the run summary "
        "(source_id, reference.source_id, or preparation.reference.source_id)");
  return result;
}

std::u32string unicode_source(const std::string &utf8) {
  std::u32string result;
  for (size_t i = 0; i < utf8.size();) {
    const auto lead = static_cast<unsigned char>(utf8[i]);
    const size_t count = lead < 0x80                    ? 1
                         : lead >= 0xc2 && lead <= 0xdf ? 2
                         : lead >= 0xe0 && lead <= 0xef ? 3
                         : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                        : 0;
    if (!count || count > utf8.size() - i)
      throw std::runtime_error("Invalid UTF-8 recording source_id");
    std::uint32_t code = lead & (count == 1 ? 0x7f : (0x7f >> count));
    for (size_t j = 1; j < count; ++j) {
      const auto byte = static_cast<unsigned char>(utf8[i + j]);
      if ((byte & 0xc0) != 0x80)
        throw std::runtime_error("Invalid UTF-8 recording source_id");
      code = (code << 6) | (byte & 0x3f);
    }
    if ((count == 2 && code < 0x80) || (count == 3 && code < 0x800) ||
        (count == 4 && code < 0x10000) || code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff) || code == 0)
      throw std::runtime_error("Invalid Unicode recording source_id");
    result.push_back(static_cast<char32_t>(code));
    i += count;
  }
  return result;
}

// A content fingerprint, not a security checksum. Stable across file moves and
// offline layout/permutation variants; source grouping uses source_id instead.
std::string bundle_id(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Could not fingerprint input run bundle");
  std::uint64_t hash = UINT64_C(14695981039346656037);
  std::array<char, 65536> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    for (std::streamsize i = 0; i < stream.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
      hash *= UINT64_C(1099511628211);
    }
  }
  if (!stream.eof())
    throw std::runtime_error("Could not read input run bundle fingerprint");
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
  return text.str();
}

std::vector<size_t> marker_permutation(size_t count, int seed,
                                       const std::string &run_id) {
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), size_t{0});
  std::uint64_t salt = UINT64_C(14695981039346656037);
  for (unsigned char byte : run_id) {
    salt ^= byte;
    salt *= UINT64_C(1099511628211);
  }
  std::seed_seq seeds{static_cast<std::uint32_t>(seed),
                      static_cast<std::uint32_t>(salt),
                      static_cast<std::uint32_t>(salt >> 32)};
  std::mt19937 random(seeds);
  // Specify the bounded draw rather than std::shuffle's implementation-defined
  // distribution so the seed is reproducible across standard libraries.
  constexpr std::uint64_t range = UINT64_C(1) << 32;
  for (size_t end = count; end > 1; --end) {
    const auto limit = range - range % end;
    std::uint64_t draw;
    do {
      draw = random();
    } while (draw >= limit);
    std::swap(order[end - 1], order[static_cast<size_t>(draw % end)]);
  }
  return order;
}

// Minimal ZIP64/NPY writer. Stream raw DEFLATE and CRC32 with bounded buffers;
// never create JSON, uncompressed NPY intermediates, or metadata sidecars.
// ZIP64 avoids a 4 GiB limit on a concatenated training array/archive.
void little_endian(std::ostream &stream, std::uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) {
    stream.put(static_cast<char>(value & 0xff));
    value >>= 8;
  }
}

std::uint64_t position(std::ostream &stream) {
  const auto value = stream.tellp();
  if (value < 0)
    throw std::runtime_error("Could not determine NPZ output position");
  return static_cast<std::uint64_t>(value);
}

class DeflateSink {
public:
  explicit DeflateSink(std::ostream &stream) : stream_(stream) {
    if (deflateInit2(&zip_, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
      throw std::runtime_error("Could not initialize NPZ compression");
  }
  ~DeflateSink() { deflateEnd(&zip_); }
  DeflateSink(const DeflateSink &) = delete;
  DeflateSink &operator=(const DeflateSink &) = delete;

  void write(const void *data, size_t bytes) {
    const auto *input = static_cast<const unsigned char *>(data);
    while (bytes) {
      const auto count = static_cast<uInt>(std::min(bytes, buffer_.size()));
      crc_ = crc32(crc_, input, count);
      if (size_ > std::numeric_limits<std::uint64_t>::max() - count)
        throw std::runtime_error("NPZ array size overflow");
      size_ += count;
      zip_.next_in = const_cast<Bytef *>(input);
      zip_.avail_in = count;
      pump(Z_NO_FLUSH);
      input += count;
      bytes -= count;
    }
  }
  void finish() {
    zip_.next_in = nullptr;
    zip_.avail_in = 0;
    pump(Z_FINISH);
  }
  std::uint64_t size() const { return size_; }
  std::uint32_t crc() const { return static_cast<std::uint32_t>(crc_); }

private:
  void pump(int flush) {
    int result;
    do {
      zip_.next_out = buffer_.data();
      zip_.avail_out = static_cast<uInt>(buffer_.size());
      result = deflate(&zip_, flush);
      if (result != Z_OK && result != Z_STREAM_END &&
          !(result == Z_BUF_ERROR && flush == Z_NO_FLUSH && !zip_.avail_in))
        throw std::runtime_error("NPZ compression failed");
      stream_.write(
          reinterpret_cast<const char *>(buffer_.data()),
          static_cast<std::streamsize>(buffer_.size() - zip_.avail_out));
    } while (flush == Z_FINISH ? result != Z_STREAM_END
                               : (zip_.avail_in || zip_.avail_out == 0));
  }
  std::ostream &stream_;
  z_stream zip_{};
  std::array<unsigned char, 65536> buffer_{};
  uLong crc_ = crc32(0L, Z_NULL, 0);
  std::uint64_t size_ = 0;
};

std::string npy_header(const std::string &dtype,
                       const std::vector<size_t> &shape) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t dimension : shape)
    text << dimension << ", ";
  text << "), }";
  std::string header = text.str();
  // NPY v1: ten-byte preamble; pad the header to a 64-byte boundary.
  header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
  header += '\n';
  if (header.size() > std::numeric_limits<std::uint16_t>::max())
    throw std::runtime_error("NPY header is too large");
  std::ostringstream result;
  result.write("\x93NUMPY\x01\x00", 8);
  little_endian(result, header.size(), 2);
  result << header;
  return result.str();
}

class NpzWriter {
public:
  explicit NpzWriter(std::ostream &stream) : stream_(stream) {}

  template <typename T>
  void array(const std::string &name, const std::string &dtype,
             const std::vector<size_t> &shape, const std::vector<T> &data) {
    if (data.size() > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::runtime_error("NPZ array byte count overflow");
    Entry entry;
    entry.name = name + ".npy";
    entry.offset = position(stream_);
    little_endian(stream_, 0x04034b50, 4); // Local ZIP header.
    little_endian(stream_, 45, 2);         // ZIP64, no data descriptor.
    little_endian(stream_, 0, 2);
    little_endian(stream_, 8, 2);    // DEFLATE.
    little_endian(stream_, 0, 2);    // Fixed timestamp for reproducibility.
    little_endian(stream_, 0x21, 2); // 1980-01-01.
    little_endian(stream_, 0, 4);    // CRC patched after compression.
    little_endian(stream_, 0xffffffff, 4);
    little_endian(stream_, 0xffffffff, 4);
    little_endian(stream_, entry.name.size(), 2);
    little_endian(stream_, 20, 2);
    stream_ << entry.name;
    little_endian(stream_, 1, 2); // ZIP64 extra field.
    little_endian(stream_, 16, 2);
    const auto sizes_offset = position(stream_);
    little_endian(stream_, 0, 8);
    little_endian(stream_, 0, 8);
    const auto data_offset = position(stream_);
    {
      DeflateSink sink(stream_);
      const auto header = npy_header(dtype, shape);
      sink.write(header.data(), header.size());
      sink.write(data.data(), data.size() * sizeof(T));
      sink.finish();
      entry.size = sink.size();
      entry.crc = sink.crc();
    }
    const auto end = position(stream_);
    entry.compressed_size = end - data_offset;
    stream_.seekp(static_cast<std::streamoff>(entry.offset + 14));
    little_endian(stream_, entry.crc, 4);
    stream_.seekp(static_cast<std::streamoff>(sizes_offset));
    little_endian(stream_, entry.size, 8);
    little_endian(stream_, entry.compressed_size, 8);
    stream_.seekp(static_cast<std::streamoff>(end));
    entries_.push_back(entry);
  }

  void finish() {
    const auto directory_offset = position(stream_);
    for (const auto &entry : entries_) {
      little_endian(stream_, 0x02014b50, 4); // Central directory entry.
      little_endian(stream_, 45, 2);
      little_endian(stream_, 45, 2);
      little_endian(stream_, 0, 2);
      little_endian(stream_, 8, 2);
      little_endian(stream_, 0, 2);
      little_endian(stream_, 0x21, 2);
      little_endian(stream_, entry.crc, 4);
      little_endian(stream_, 0xffffffff, 4);
      little_endian(stream_, 0xffffffff, 4);
      little_endian(stream_, entry.name.size(), 2);
      little_endian(stream_, 28, 2); // ZIP64 sizes and local-header offset.
      little_endian(stream_, 0, 2);  // Comment length.
      little_endian(stream_, 0, 2);  // Disk.
      little_endian(stream_, 0, 2);  // Internal attributes.
      little_endian(stream_, 0, 4);  // External attributes.
      little_endian(stream_, 0xffffffff, 4);
      stream_ << entry.name;
      little_endian(stream_, 1, 2);
      little_endian(stream_, 24, 2);
      little_endian(stream_, entry.size, 8);
      little_endian(stream_, entry.compressed_size, 8);
      little_endian(stream_, entry.offset, 8);
    }
    const auto zip64_offset = position(stream_);
    const auto directory_size = zip64_offset - directory_offset;
    little_endian(stream_, 0x06064b50, 4); // ZIP64 end of central directory.
    little_endian(stream_, 44, 8);
    little_endian(stream_, 45, 2);
    little_endian(stream_, 45, 2);
    little_endian(stream_, 0, 4);
    little_endian(stream_, 0, 4);
    little_endian(stream_, entries_.size(), 8);
    little_endian(stream_, entries_.size(), 8);
    little_endian(stream_, directory_size, 8);
    little_endian(stream_, directory_offset, 8);
    little_endian(stream_, 0x07064b50, 4); // ZIP64 locator.
    little_endian(stream_, 0, 4);
    little_endian(stream_, zip64_offset, 8);
    little_endian(stream_, 1, 4);
    little_endian(stream_, 0x06054b50,
                  4); // Classic end record, ZIP64 sentinels.
    little_endian(stream_, 0, 2);
    little_endian(stream_, 0, 2);
    little_endian(stream_, 0xffff, 2);
    little_endian(stream_, 0xffff, 2);
    little_endian(stream_, 0xffffffff, 4);
    little_endian(stream_, 0xffffffff, 4);
    little_endian(stream_, 0, 2);
  }

private:
  struct Entry {
    std::string name;
    std::uint64_t offset = 0, size = 0, compressed_size = 0;
    std::uint32_t crc = 0;
  };
  std::ostream &stream_;
  std::vector<Entry> entries_;
};

struct Dataset {
  size_t marker_count = 0;
  std::vector<double> markers, mass, time, wrench;
  std::vector<std::int64_t> offsets{0};
  std::vector<std::u32string> sources;
};

void write_npz(const std::filesystem::path &destination, const Dataset &data) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "NPZ export requires IEEE-754 float64");
  static_assert(sizeof(char32_t) == 4,
                "NPY Unicode requires 32-bit codepoints");
  const std::uint16_t endian_probe = 1;
  const std::string endian =
      *reinterpret_cast<const unsigned char *>(&endian_probe) ? "<" : ">";
  size_t source_width = 1;
  for (const auto &id : data.sources)
    source_width = std::max(source_width, id.size());
  if (source_width > std::numeric_limits<size_t>::max() / data.sources.size())
    throw std::runtime_error("NPZ source_id size overflow");
  std::vector<char32_t> sources(source_width * data.sources.size(), U'\0');
  for (size_t e = 0; e < data.sources.size(); ++e)
    std::copy(data.sources[e].begin(), data.sources[e].end(),
              sources.begin() + e * source_width);

  if (!destination.parent_path().empty())
    std::filesystem::create_directories(destination.parent_path());
  static std::atomic_uint64_t sequence{0};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  auto partial = destination;
  partial += ".partial." + std::to_string(tick) + "." +
             std::to_string(sequence.fetch_add(1));
  try {
    std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    NpzWriter zip(stream);
    const size_t t = data.time.size(), e = data.mass.size();
    zip.array("markers_m", endian + "f8", {t, data.marker_count, 3},
              data.markers);
    zip.array("mass_kg", endian + "f8", {e}, data.mass);
    zip.array("time_s", endian + "f8", {t}, data.time);
    zip.array("contact_wrench", endian + "f8", {t, 2, 6}, data.wrench);
    zip.array("episode_offsets", endian + "i8", {e + 1}, data.offsets);
    zip.array("source_id", endian + "U" + std::to_string(source_width), {e},
              sources);
    zip.finish();
    stream.close();
    std::filesystem::rename(partial, destination);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(partial, ignored);
    throw;
  }
}
} // namespace

void export_run_bundles(const std::vector<std::filesystem::path> &input_paths,
                        const std::filesystem::path &output_npz_path,
                        int layout_index, int permutation_seed,
                        bool allow_incomplete, double min_normal_force) {
  if (input_paths.empty())
    throw std::runtime_error("Export needs at least one input run bundle");
  if (layout_index < -1 || permutation_seed < 0)
    throw std::runtime_error("Export layout_index must be -1 or nonnegative; "
                             "permutation seed must be nonnegative");
  if (!std::isfinite(min_normal_force) || min_normal_force < 0)
    throw std::runtime_error("Export min_normal_force must be finite and "
                             "nonnegative (newtons)");
  for (const auto &path : input_paths)
    if (std::filesystem::weakly_canonical(path) ==
            std::filesystem::weakly_canonical(output_npz_path) ||
        (std::filesystem::exists(output_npz_path) &&
         std::filesystem::equivalent(path, output_npz_path)))
      throw std::runtime_error(
          "Export must not overwrite its input run bundle");
  if (output_npz_path.extension() != ".npz")
    throw std::runtime_error("Export output must use the .npz extension");

  Dataset result;
  for (const auto &input_path : input_paths) {
    // Retain only training arrays across episodes, not all compiled models and
    // replay state. Each input is loaded once; labels are copied, never
    // derived.
    auto bundle = load_run_bundle(input_path);
    if (!bundle.model || bundle.frames.empty())
      throw std::runtime_error("Cannot export an empty run bundle");
    for (const auto &frame : bundle.frames)
      if (!frame.wrench_available)
        throw std::runtime_error(
            "Export requires V2 recorded wrenches; legacy contact-position "
            "proxies are not CoP labels (even with allow_incomplete)");

    const auto &summary = bundle.summary;
    const bool completed = summary.contains("completed_requested_duration") &&
                           summary.at("completed_requested_duration") == true;
    if (!completed && !allow_incomplete)
      throw std::runtime_error(
          "Incomplete episode rejected; allow_incomplete is diagnostic-only");
    result.sources.push_back(unicode_source(source_id(summary)));
    const auto run_id = bundle_id(input_path);
    auto *model = bundle.model.get();
    const double mass = mj_getTotalmass(model);
    if (!std::isfinite(mass) || mass <= 0)
      throw std::runtime_error(
          "Export requires finite positive total model mass");
    const auto sites = qmc_marker_site_ids(model);
    if (sites.empty())
      throw std::runtime_error(
          "Export requires embedded observation marker sites");
    if (result.marker_count && result.marker_count != sites.size())
      throw std::runtime_error("Marker counts differ across episodes; export "
                               "them separately");
    result.marker_count = sites.size();
    if (layout_index >= 0)
      place_qmc_marker_sites(model, layout_index);
    const auto permutation =
        marker_permutation(sites.size(), permutation_seed, run_id);

    DataPtr data(nullptr, mj_deleteData);
    if (layout_index >= 0)
      data = make_data(model);
    double previous_time = -std::numeric_limits<double>::infinity();
    long long previous_step = -1;
    for (const auto &frame : bundle.frames) {
      if (!std::isfinite(frame.time) || frame.time <= previous_time ||
          frame.physics_step < 0 || frame.physics_step <= previous_step)
        throw std::runtime_error(
            "Export needs strictly increasing times/physics steps");
      previous_time = frame.time;
      previous_step = frame.physics_step;
      require_finite(frame.markers_m, 3 * sites.size(), "recorded markers");
      if (data) {
        require_finite(frame.qpos, model->nq, "saved qpos");
        require_finite(frame.mocap_pos, 3 * model->nmocap,
                       "saved mocap positions");
        require_finite(frame.mocap_quat, 4 * model->nmocap,
                       "saved mocap quaternions");
        std::copy(frame.qpos.begin(), frame.qpos.end(), data->qpos);
        std::copy(frame.mocap_pos.begin(), frame.mocap_pos.end(),
                  data->mocap_pos);
        std::copy(frame.mocap_quat.begin(), frame.mocap_quat.end(),
                  data->mocap_quat);
        data->time = frame.time;
        // No mj_forward/mj_step: observation sites need only saved poses, and
        // recorded contact forces must never be recomputed.
        mj_kinematics(model, data.get());
        mj_comPos(model, data.get());
      }
      for (const size_t ordinal : permutation) {
        const double *point = data ? data->site_xpos + 3 * sites[ordinal]
                                   : frame.markers_m.data() + 3 * ordinal;
        require_finite(point, 3, "executed marker");
        result.markers.insert(result.markers.end(), point, point + 3);
      }
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (int side = 0; side < 2; ++side) {
        const auto &f = side == 0 ? frame.grf_left : frame.grf_right;
        const auto &c = side == 0 ? frame.cop_left : frame.cop_right;
        const double tz =
            side == 0 ? frame.torque_z_left : frame.torque_z_right;
        const bool valid =
            side == 0 ? frame.cop_defined_left : frame.cop_defined_right;
        require_finite(f.data(), 3, "recorded GRF");
        result.wrench.insert(result.wrench.end(), f.begin(), f.end());
        if (valid && f[2] > min_normal_force) {
          require_finite(c.data(), 3, "recorded CoP");
          if (!std::isfinite(tz) || c[2] != 0)
            throw std::runtime_error(
                "Invalid defined CoP/torque in recorded wrench");
          result.wrench.insert(result.wrench.end(), {c[0], c[1], tz});
        } else {
          result.wrench.insert(result.wrench.end(), {nan, nan, nan});
        }
      }
      result.time.push_back(frame.time);
    }
    if (result.time.size() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::runtime_error("Too many samples for int64 episode_offsets");
    result.offsets.push_back(static_cast<std::int64_t>(result.time.size()));
    result.mass.push_back(mass);
  }
  write_npz(output_npz_path, result);
}

void export_run_bundle(const std::filesystem::path &input_path,
                       const std::filesystem::path &output_npz_path,
                       int layout_index, int permutation_seed,
                       bool allow_incomplete, double min_normal_force) {
  export_run_bundles({input_path}, output_npz_path, layout_index,
                     permutation_seed, allow_incomplete, min_normal_force);
}

} // namespace tracking
