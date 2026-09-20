#include "qmc_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tracking {
namespace {

double halton(int index, int base) {
  if (index < 0 || base < 2)
    throw std::runtime_error("Invalid Halton sequence request");
  double factor = 1.0;
  double value = 0.0;
  while (index > 0) {
    factor /= base;
    value += factor * (index % base);
    index /= base;
  }
  return value;
}

bool supported_primitive(mjtGeom type) {
  return type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER ||
         type == mjGEOM_SPHERE;
}

bool has_supported_primitive(const mjModel *model, int body) {
  for (int offset = 0; offset < model->body_geomnum[body]; ++offset) {
    const int geom = model->body_geomadr[body] + offset;
    if (supported_primitive(static_cast<mjtGeom>(model->geom_type[geom])))
      return true;
  }
  return false;
}

std::vector<int> supported_geometries(const mjModel *model, int body) {
  std::vector<int> geoms;
  for (int offset = 0; offset < model->body_geomnum[body]; ++offset) {
    const int geom = model->body_geomadr[body] + offset;
    if (supported_primitive(static_cast<mjtGeom>(model->geom_type[geom])))
      geoms.push_back(geom);
  }
  return geoms;
}

std::string body_name(const mjsBody *body) {
  return body && body->name ? body->name->c_str() : std::string();
}

} // namespace

MorphologyQmcSample apply_morphology_qmc(mjSpec *spec, int index) {
  if (!spec || index < 0)
    throw std::runtime_error("Morphology QMC index must be nonnegative");

  MorphologyQmcSample sample;
  sample.index = index;
  if (index == 0)
    return sample;

  // Preserve the project's original exploratory sampler for positive indices.
  // It is deliberately recorded as a global length/radius draw rather than
  // presented as a calibrated population model.
  sample.length_scale = 0.90 + 0.20 * halton(index, 2);
  sample.radius_scale = 0.90 + 0.20 * halton(index, 3);

  for (mjsElement *element = mjs_firstElement(spec, mjOBJ_BODY); element;
       element = mjs_nextElement(spec, element)) {
    auto *body = mjs_asBody(element);
    if (body_name(body) == "world")
      continue;

    for (double &coordinate : body->pos)
      coordinate *= sample.length_scale;

    for (mjsElement *child = mjs_firstChild(body, mjOBJ_GEOM, 0); child;
         child = mjs_nextChild(body, child, 0)) {
      auto *geom = mjs_asGeom(child);
      if (!supported_primitive(geom->type))
        continue;
      geom->size[0] *= sample.radius_scale;
      const bool uses_fromto = std::any_of(
          std::begin(geom->fromto), std::end(geom->fromto),
          [](double value) { return value != 0.0; });
      if (uses_fromto) {
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

int add_qmc_marker_sites(mjSpec *spec, const mjModel *baseline,
                         int markers_per_body) {
  if (!spec || !baseline || markers_per_body < 1)
    throw std::runtime_error("Invalid QMC marker configuration");

  int added = 0;
  for (mjsElement *element = mjs_firstElement(spec, mjOBJ_BODY); element;
       element = mjs_nextElement(spec, element)) {
    auto *body = mjs_asBody(element);
    const std::string name = body_name(body);
    if (name.empty() || name == "world" || name == "hand_left" ||
        name == "hand_right")
      continue;
    const int baseline_body = mj_name2id(baseline, mjOBJ_BODY, name.c_str());
    if (baseline_body < 0 || !has_supported_primitive(baseline, baseline_body))
      continue;

    for (int marker = 0; marker < markers_per_body; ++marker) {
      auto *site = mjs_addSite(body, nullptr);
      const std::string site_name =
          "marker[" + name + "][" + std::to_string(marker) + "]";
      mjs_setString(site->name, site_name.c_str());
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
  if (added == 0)
    throw std::runtime_error("No eligible bodies for QMC marker sites");
  return added;
}

std::vector<int> qmc_marker_site_ids(const mjModel *model) {
  std::vector<int> sites;
  if (!model)
    return sites;
  for (int site = 0; site < model->nsite; ++site) {
    if (model->site_group[site] == 3)
      sites.push_back(site);
  }
  return sites;
}

void place_qmc_marker_sites(mjModel *model, int qmc_index) {
  if (!model || qmc_index < 0)
    throw std::runtime_error("Marker QMC index must be nonnegative");

  const auto sites = qmc_marker_site_ids(model);
  // Each marker gets a stable offset in the same episode-indexed sequence.
  // Thirteen is coprime to the marker bases (5, 7, and 11), so advancing an
  // episode advances every placement coordinate instead of fixing a radix
  // digit as the retired index*1000 scheme did for base 5.
  constexpr int kMarkerOrdinalStride = 13;
  for (size_t ordinal = 0; ordinal < sites.size(); ++ordinal) {
    const long long sequence_long =
        static_cast<long long>(qmc_index) +
        static_cast<long long>(ordinal + 1) * kMarkerOrdinalStride;
    if (sequence_long > std::numeric_limits<int>::max())
      throw std::runtime_error("Marker QMC index is too large");
    const int sequence = static_cast<int>(sequence_long);
    const int site = sites[ordinal];
    const auto geoms = supported_geometries(model, model->site_bodyid[site]);
    if (geoms.empty())
      throw std::runtime_error("QMC marker body has no supported geometry");
    // Bases 2 and 3 parameterize morphology. Use distinct bases for marker
    // geometry selection and surface coordinates so one episode index forms a
    // low-discrepancy joint sample without duplicating those morphology draws.
    const int geom = geoms[std::min(
        static_cast<int>(halton(sequence, 5) * geoms.size()),
        static_cast<int>(geoms.size()) - 1)];
    const double u = halton(sequence, 7);
    const double v = halton(sequence, 11);
    const double radius = model->geom_size[3 * geom];
    double local[3]{};
    if (model->geom_type[geom] == mjGEOM_SPHERE) {
      const double phi = std::acos(1.0 - 2.0 * u);
      const double theta = 2.0 * mjPI * v;
      local[0] = radius * std::sin(phi) * std::cos(theta);
      local[1] = radius * std::sin(phi) * std::sin(theta);
      local[2] = radius * std::cos(phi);
    } else {
      const double theta = 2.0 * mjPI * u;
      const double half_length = model->geom_size[3 * geom + 1];
      local[0] = radius * std::cos(theta);
      local[1] = radius * std::sin(theta);
      local[2] = (v - .5) * 2.0 * half_length;
    }
    double rotated[3];
    mju_rotVecQuat(rotated, local, model->geom_quat + 4 * geom);
    for (int axis = 0; axis < 3; ++axis)
      model->site_pos[3 * site + axis] =
          model->geom_pos[3 * geom + axis] + rotated[axis];
    model->site_sameframe[site] = 0;
  }
}

} // namespace tracking
