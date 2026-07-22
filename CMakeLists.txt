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

# Skip MuJoCo sub-targets we don't need (also avoids X11 dependency from the simulator)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_TESTS    OFF)

include(FetchContent)
# Fetch and build GLFW hermetically for Wayland natively (Must be BEFORE MuJoCo!)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    mujoco
    GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
    GIT_TAG        3.2.6 # 3.1.6 has a const-qualifier bug that errors under GCC 15
)

FetchContent_MakeAvailable(mujoco)

# Fetch MuJoCo MPC (MJPC) for the walking policy
FetchContent_Declare(
    mujoco_mpc
    GIT_REPOSITORY https://github.com/google-deepmind/mujoco_mpc.git
    GIT_TAG        main
)
# Disable MJPC python/test builds to save time
set(MJPC_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(MJPC_BUILD_GRPC_SERVICE OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mujoco_mpc)

add_executable(data_factory src/main.cpp src/morphology_randomizer.cpp)

target_link_libraries(data_factory PRIVATE mujoco::mujoco glfw libmjpc)

# toggle debug
# target_compile_options(data_factory PRIVATE -g -O0)
