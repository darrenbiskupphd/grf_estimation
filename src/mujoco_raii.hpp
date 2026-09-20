#pragma once

#include <memory>
#include <mujoco/mujoco.h>
#include <stdexcept>

using ModelPtr = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
using DataPtr = std::unique_ptr<mjData, decltype(&mj_deleteData)>;
using SpecPtr = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;

inline DataPtr make_data(const mjModel *model) {
  DataPtr data(mj_makeData(model), mj_deleteData);
  if (!data)
    throw std::runtime_error("Could not allocate MuJoCo data");
  return data;
}
