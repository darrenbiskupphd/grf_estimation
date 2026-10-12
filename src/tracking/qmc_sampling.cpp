#include "qmc_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tracking {
namespace {

double halton(std::uint64_t index, int base) {
  double factor = 1.0;
  double value = 0.0;
  while (index > 0) {
    factor /= base;
    value += factor * (index % base);
    index /= base;
  }
  return value;
}

bool supported_primitive(int type) {
  return type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER ||
         type == mjGEOM_SPHERE;
}

std::vector<int> supported_geometries(const mjModel *model, int body) {
  std::vector<int> geoms;
  for (int offset = 0; offset < model->body_geomnum[body]; ++offset) {
    const int geom = model->body_geomadr[body] + offset;
    if (supported_primitive(model->geom_type[geom]))
      geoms.push_back(geom);
  }
  return geoms;
}

std::string body_name(const mjsBody *body) {
  return body && body->name ? body->name->c_str() : std::string();
}

std::string marker_group_name(const mjModel *model, int site) {
  const char *raw = mj_id2name(model, mjOBJ_SITE, site);
  if (model->site_group[site] != 3 || !raw)
    return {};
  const std::string name(raw);
  const auto separator = name.rfind("][");
  if (name.compare(0, 7, "marker[") != 0 || separator == std::string::npos ||
      separator <= 7 || name.back() != ']' || separator + 3 >= name.size())
    return {};
  if (!std::all_of(name.begin() + separator + 2, name.end() - 1,
                   [](char c) { return c >= '0' && c <= '9'; }))
    return {};
  return name.substr(7, separator - 7);
}

double surface_area(const mjModel *model, int geom) {
  const double r = model->geom_size[3 * geom];
  const double h = model->geom_size[3 * geom + 1];
  double area = 4 * mjPI * r * r;
  if (model->geom_type[geom] == mjGEOM_CAPSULE)
    area += 4 * mjPI * r * h;
  else if (model->geom_type[geom] == mjGEOM_CYLINDER)
    area = 4 * mjPI * r * h + 2 * mjPI * r * r;
  if (!(area > 0) || !std::isfinite(area))
    throw std::runtime_error("Invalid primitive area for QMC markers");
  return area;
}

// Area-uniform point on the COMPLETE primitive surface, in geom coordinates.
// A capsule has constant area per axial distance, 2*pi*r, even on its caps.
std::array<mjtNum, 3> surface_point(const mjModel *model, int geom, double u,
                                    double v, std::string &region) {
  const double r = model->geom_size[3 * geom];
  const double h = model->geom_size[3 * geom + 1];
  double radial = r;
  double z;
  if (model->geom_type[geom] == mjGEOM_SPHERE) {
    z = r * (2 * u - 1);
    radial = std::sqrt(std::max(0.0, r * r - z * z));
    region = "sphere";
  } else if (model->geom_type[geom] == mjGEOM_CAPSULE) {
    z = (2 * u - 1) * (h + r);
    const double cap_height = std::max(0.0, std::abs(z) - h);
    radial = std::sqrt(std::max(0.0, r * r - cap_height * cap_height));
    region = z < -h ? "negative_cap" : z > h ? "positive_cap" : "side";
  } else {
    // Areas divided by 2*pi*r: side=2*h, each disk=r/2. Reuse the
    // within-region area coordinate as squared disk radius on cylinder caps.
    const double area_coordinate = u * (2 * h + r);
    if (area_coordinate < 2 * h) {
      z = area_coordinate - h;
      region = "side";
    } else {
      const double cap_coordinate = (area_coordinate - 2 * h) * 2 / r;
      const bool positive = cap_coordinate >= 1;
      z = positive ? h : -h;
      radial = r * std::sqrt(std::clamp(cap_coordinate - (positive ? 1 : 0),
                                        0.0, 1.0));
      region = positive ? "positive_cap" : "negative_cap";
    }
  }
  const double theta = 2 * mjPI * v;
  return {radial * std::cos(theta), radial * std::sin(theta), z};
}

bool buried(const mjModel *model, const mjData *reference, int selected_geom,
            const std::vector<int> &occluders,
            const std::array<mjtNum, 3> &geom_local) {
  mjtNum world[3];
  mju_mulMatVec3(world, reference->geom_xmat + 9 * selected_geom,
                 geom_local.data());
  mju_addTo3(world, reference->geom_xpos + 3 * selected_geom);
  for (int other : occluders) {
    if (other == selected_geom)
      continue;
    mjtNum delta[3], local[3];
    mju_sub3(delta, world, reference->geom_xpos + 3 * other);
    mju_mulMatTVec3(local, reference->geom_xmat + 9 * other, delta);
    const double r = model->geom_size[3 * other];
    const double h = model->geom_size[3 * other + 1];
    const double radial = std::hypot(local[0], local[1]);
    double distance;
    if (model->geom_type[other] == mjGEOM_SPHERE)
      distance = std::hypot(radial, local[2]) - r;
    else if (model->geom_type[other] == mjGEOM_CAPSULE)
      distance = std::hypot(radial, std::max(0.0, std::abs(local[2]) - h)) - r;
    else
      distance = std::max(radial - r, std::abs(local[2]) - h);
    const double tolerance = 1e-10 * std::max(1.0, r + h);
    // Lower geom id owns coincident boundary patches, avoiding duplicate
    // weighting for identical primitives. Ordinary intersection seams have
    // zero surface measure. Cylinder distance need only have the correct sign.
    if (distance < -tolerance ||
        (std::abs(distance) <= tolerance && other < selected_geom))
      return true;
  }
  return false;
}

} // namespace

MorphologyQmcSample apply_morphology_qmc(mjSpec *spec, int index) {
  if (!spec || index < 0)
    throw std::runtime_error("Morphology QMC index must be nonnegative");

  MorphologyQmcSample sample;
  sample.index = index;
  if (index == 0)
    return sample;

  // Preserve the original global draw and uniform-density inertia policy.
  sample.length_scale = 0.90 + 0.20 * halton(index, 2);
  sample.radius_scale = 0.90 + 0.20 * halton(index, 3);

  for (mjsElement *element = mjs_firstElement(spec, mjOBJ_BODY); element;
       element = mjs_nextElement(spec, element)) {
    auto *body = mjs_asBody(element);
    if (body_name(body) == "world")
      continue;

    for (double &coordinate : body->pos)
      coordinate *= sample.length_scale;

    // Anchors are body-local lengths, just like geometry offsets. This also
    // scales inherited nonzero anchors; zero/default anchors remain zero.
    // Do not touch axes, angular ranges, stiffness, damping, or joint refs.
    for (mjsElement *child = mjs_firstChild(body, mjOBJ_JOINT, 0); child;
         child = mjs_nextChild(body, child, 0)) {
      auto *joint = mjs_asJoint(child);
      for (double &coordinate : joint->pos)
        coordinate *= sample.length_scale;
    }

    for (mjsElement *child = mjs_firstChild(body, mjOBJ_GEOM, 0); child;
         child = mjs_nextChild(body, child, 0)) {
      auto *geom = mjs_asGeom(child);
      if (!supported_primitive(geom->type))
        continue;
      geom->size[0] *= sample.radius_scale;
      // MuJoCo marks an unspecified fromto with NaN, not all zeroes. The old
      // any(nonzero) test treated NaN as true, missing size-defined lengths.
      if (!std::isnan(geom->fromto[0])) {
        for (double &coordinate : geom->fromto)
          coordinate *= sample.length_scale;
      } else {
        geom->size[1] *= sample.length_scale;
        geom->size[2] *= sample.length_scale;
      }
      for (double &coordinate : geom->pos)
        coordinate *= sample.length_scale;
    }
  }
  return sample;
}

std::vector<int> qmc_marker_site_ids(const mjModel *model) {
  std::vector<int> sites;
  if (model) {
    for (int site = 0; site < model->nsite; ++site)
      if (!marker_group_name(model, site).empty())
        sites.push_back(site);
  }
  return sites;
}

std::vector<QmcMarkerGroup> qmc_marker_groups(const mjModel *model) {
  if (!model)
    throw std::runtime_error("Null model for QMC marker groups");
  std::vector<int> root(model->nbody, -1);
  for (int body = 1; body < model->nbody; ++body) {
    const char *name = mj_id2name(model, mjOBJ_BODY, body);
    if (!name || !*name || std::string(name) == "hand_left" ||
        std::string(name) == "hand_right" || model->body_mocapid[body] >= 0 ||
        supported_geometries(model, body).empty())
      continue;
    root[body] = body;
  }
  for (const char *side : {"left", "right"}) {
    const int foot =
        mj_name2id(model, mjOBJ_BODY, (std::string("foot_") + side).c_str());
    const int toe =
        mj_name2id(model, mjOBJ_BODY, (std::string("toe_") + side).c_str());
    if (foot < 0 || toe < 0 || root[foot] < 0 || root[toe] < 0)
      continue;
    int ancestor = model->body_parentid[toe];
    while (ancestor > 0 && ancestor != foot)
      ancestor = model->body_parentid[ancestor];
    if (ancestor != foot)
      throw std::runtime_error("QMC foot/toe bodies are not in one subtree");
    root[toe] = foot;
  }

  std::vector<QmcMarkerGroup> groups;
  std::vector<int> group_for_root(model->nbody, -1);
  for (int body = 1; body < model->nbody; ++body) {
    if (root[body] < 0)
      continue;
    int &group_index = group_for_root[root[body]];
    if (group_index < 0) {
      group_index = static_cast<int>(groups.size());
      QmcMarkerGroup group;
      group.name = mj_id2name(model, mjOBJ_BODY, root[body]);
      groups.push_back(std::move(group));
    }
    auto &group = groups[group_index];
    group.body_ids.push_back(body);
    for (int geom : supported_geometries(model, body)) {
      group.geom_ids.push_back(geom);
      group.geom_surface_areas.push_back(surface_area(model, geom));
    }
  }
  for (int site : qmc_marker_site_ids(model)) {
    const auto name = marker_group_name(model, site);
    const int body = mj_name2id(model, mjOBJ_BODY, name.c_str());
    if (body < 0 || root[body] < 0)
      throw std::runtime_error("QMC marker names an ineligible group: " + name);
    groups[group_for_root[root[body]]].site_ids.push_back(site);
  }
  return groups;
}

int add_qmc_marker_sites(mjSpec *spec, const mjModel *baseline,
                         int markers_per_body) {
  if (!spec || !baseline || markers_per_body < 1)
    throw std::runtime_error("Invalid QMC marker configuration");
  const auto groups = qmc_marker_groups(baseline);
  if (groups.empty())
    throw std::runtime_error("No eligible bodies for QMC marker sites");

  // Check before modifying the spec, including accidental repeated addition.
  for (const auto &group : groups) {
    if (!mjs_findBody(spec, group.name.c_str()))
      throw std::runtime_error("QMC baseline body missing from spec: " +
                               group.name);
    for (int marker = 0; marker < markers_per_body; ++marker) {
      const std::string name =
          "marker[" + group.name + "][" + std::to_string(marker) + "]";
      if (mjs_findElement(spec, mjOBJ_SITE, name.c_str()))
        throw std::runtime_error("QMC marker site already exists: " + name);
    }
  }

  int added = 0;
  for (const auto &group : groups) {
    auto *body = mjs_findBody(spec, group.name.c_str());
    for (int marker = 0; marker < markers_per_body; ++marker) {
      auto *site = mjs_addSite(body, nullptr);
      const std::string name =
          "marker[" + group.name + "][" + std::to_string(marker) + "]";
      mjs_setString(site->name, name.c_str());
      site->type = mjGEOM_SPHERE;
      site->size[0] = .01;
      site->group = 3;
      site->rgba[0] = 1.0f;
      site->rgba[1] = 0.0f;
      site->rgba[2] = 0.0f;
      site->rgba[3] = 1.0f;
      ++added;
    }
  }
  return added;
}

QmcMarkerLayout qmc_marker_layout(const mjModel *model, int layout_index) {
  if (!model || layout_index < 0)
    throw std::runtime_error("Marker layout index must be nonnegative");
  QmcMarkerLayout layout;
  layout.index = layout_index;
  layout.groups = qmc_marker_groups(model);
  layout.geom_candidate_counts.resize(model->ngeom);
  layout.geom_buried_counts.resize(model->ngeom);
  const auto sites = qmc_marker_site_ids(model);
  if (sites.empty())
    return layout;

  // Disjoint sequence blocks for each layout and marker. 4096 and 2^32 are
  // coprime to all marker bases (5,7,11). Rejection for one marker never shifts
  // another marker or aliases the next layout. uint64 avoids int overflow for
  // the full nonnegative int layout-index range. Morphology is never consulted.
  constexpr std::uint64_t kLayoutStride = std::uint64_t{1} << 32;
  constexpr int kMaxCandidates = 4096;
  if (sites.size() >= kLayoutStride / kMaxCandidates)
    throw std::runtime_error("Too many QMC marker sites for sequence blocks");
  std::vector<int> site_group(model->nsite, -1);
  std::vector<std::vector<double>> cumulative_areas;
  for (size_t group_index = 0; group_index < layout.groups.size();
       ++group_index) {
    const auto &group = layout.groups[group_index];
    std::vector<double> areas;
    double cumulative = 0;
    for (double area : group.geom_surface_areas) {
      cumulative += area;
      areas.push_back(cumulative);
    }
    cumulative_areas.push_back(std::move(areas));
    for (int site : group.site_ids)
      site_group[site] = static_cast<int>(group_index);
  }

  std::unique_ptr<mjData, decltype(&mj_deleteData)> reference(
      mj_makeData(model), mj_deleteData);
  if (!reference)
    throw std::runtime_error("Could not allocate QMC reference-pose data");
  // mj_makeData initializes qpos to qpos0. Kinematics alone avoids callbacks,
  // dynamics, collision detection, and dependence on any live simulation pose.
  mj_kinematics(model, reference.get());
  std::vector<int> occluders;
  for (int geom = 0; geom < model->ngeom; ++geom) {
    const int body = model->geom_bodyid[geom];
    if (body > 0 && model->body_mocapid[body] < 0 &&
        supported_primitive(model->geom_type[geom]))
      occluders.push_back(geom);
  }

  for (size_t ordinal = 0; ordinal < sites.size(); ++ordinal) {
    QmcMarkerPlacement placement;
    placement.site_id = sites[ordinal];
    placement.group_index = site_group[placement.site_id];
    const auto &group = layout.groups[placement.group_index];
    const auto &areas = cumulative_areas[placement.group_index];
    bool accepted = false;
    for (int attempt = 0; attempt < kMaxCandidates; ++attempt) {
      const std::uint64_t sequence =
          static_cast<std::uint64_t>(layout_index) * kLayoutStride +
          ordinal * kMaxCandidates + attempt + 1;
      const double area_draw = halton(sequence, 5) * areas.back();
      const size_t selected =
          std::min(static_cast<size_t>(
                       std::upper_bound(areas.begin(), areas.end(), area_draw) -
                       areas.begin()),
                   areas.size() - 1);
      placement.geom_id = group.geom_ids[selected];
      ++layout.geom_candidate_counts[placement.geom_id];
      const auto local =
          surface_point(model, placement.geom_id, halton(sequence, 7),
                        halton(sequence, 11), placement.surface_region);
      if (buried(model, reference.get(), placement.geom_id, occluders, local)) {
        ++layout.geom_buried_counts[placement.geom_id];
        continue;
      }
      placement.body_id = model->geom_bodyid[placement.geom_id];
      mju_rotVecQuat(placement.local_pos.data(), local.data(),
                     model->geom_quat + 4 * placement.geom_id);
      mju_addTo3(placement.local_pos.data(),
                 model->geom_pos + 3 * placement.geom_id);
      accepted = true;
      break;
    }
    if (!accepted)
      throw std::runtime_error(
          "No exposed QMC marker surface found for group " + group.name +
          " after 4096 proposals");
    layout.placements.push_back(std::move(placement));
  }
  return layout;
}

void place_qmc_marker_sites(mjModel *model, int layout_index) {
  // Generate fully before applying, so a failed layout leaves the model intact.
  const auto layout = qmc_marker_layout(model, layout_index);
  for (const auto &placement : layout.placements) {
    model->site_bodyid[placement.site_id] = placement.body_id;
    mju_copy3(model->site_pos + 3 * placement.site_id,
              placement.local_pos.data());
    // MuJoCo iterates sites directly, not via per-body contiguous site ranges.
    // Clearing this optimization is essential after changing local position.
    model->site_sameframe[placement.site_id] = mjSAMEFRAME_NONE;
  }
}

} // namespace tracking
