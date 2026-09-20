cmake_minimum_required(VERSION 3.20)
project(grf_estimation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required for CMake 4.x: suppresses hard errors in vendored deps with old cmake_minimum_required
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# GCC 15 promotes -Wdiscarded-qualifiers to a hard error; MuJoCo 3.x has a strchr const-assignment
# bug in engine_print.c that trips it. Scoped to this project only.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=discarded-qualifiers")

# MJPC links MuJoCo's simulator UI adapter; retain its build support.
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_TESTS    OFF)

include(FetchContent)
# Fetch and build GLFW hermetically (Must be BEFORE MuJoCo!)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    mujoco
    GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
    GIT_TAG        3.2.6 # 3.1.6 has a const-qualifier bug that errors under GCC 15
)

FetchContent_MakeAvailable(mujoco)

# Fetch MuJoCo MPC (MJPC) for the walking policy
set(GRF_MJPC_REVISION ff572a21e7c2bf9fda62e1862a758da7e9a8719b)
FetchContent_Declare(
    mujoco_mpc
    GIT_REPOSITORY https://github.com/google-deepmind/mujoco_mpc.git
    GIT_TAG        ${GRF_MJPC_REVISION}
)
# Disable dependency tests and the optional gRPC service.
set(MJPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MJPC_BUILD_GRPC_SERVICE OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mujoco_mpc)

add_library(grf_runtime STATIC
    src/residual_callback.cpp
    src/viewer.cpp
    src/tracking/runner.cpp
    src/tracking/run_bundle.cpp
    src/tracking/reference_task.cpp
    src/tracking/retarget.cpp
    src/tracking/qmc_sampling.cpp
)
target_include_directories(grf_runtime PUBLIC src)
target_link_libraries(grf_runtime PUBLIC mujoco::mujoco glfw libmjpc)
target_compile_definitions(grf_runtime PUBLIC
    GRF_TRACKING_MODEL="${mujoco_mpc_BINARY_DIR}/mjpc/tasks/humanoid/tracking/task.xml"
    GRF_MJPC_REVISION="${GRF_MJPC_REVISION}"
)

add_executable(data_factory src/main.cpp)
target_link_libraries(data_factory PRIVATE grf_runtime)

option(GRF_BUILD_TESTS "Enable project integration checks" ON)
if(GRF_BUILD_TESTS)
    enable_testing()
    add_executable(tracking_contract_test tests/tracking_contract.cpp)
    target_link_libraries(tracking_contract_test PRIVATE grf_runtime)
    add_test(NAME tracking_reference_contract
        COMMAND tracking_contract_test
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/winter_baseline_male.xml
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/plagenhoef_baseline_female.xml)
    add_executable(tracking_integration_test tests/tracking_integration.cpp)
    target_link_libraries(tracking_integration_test PRIVATE grf_runtime)
    add_test(NAME tracking_run_bundle_integration
        COMMAND tracking_integration_test
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/winter_baseline_male.xml
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/plagenhoef_baseline_female.xml)
endif()
