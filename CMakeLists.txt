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

# Direct compressed NPZ export uses raw DEFLATE/CRC32 from zlib.
find_package(ZLIB REQUIRED)

add_library(grf_runtime STATIC
    src/residual_callback.cpp
    src/viewer.cpp
    src/tracking/runner.cpp
    src/tracking/run_bundle.cpp
    src/tracking/export.cpp
    src/tracking/reference.cpp
    src/tracking/reference_task.cpp
    src/tracking/build_reference_tracking_model.cpp
    src/tracking/initialize_reference_tracking_state.cpp
    src/tracking/qmc_sampling.cpp
)
target_include_directories(grf_runtime PUBLIC src)
target_link_libraries(grf_runtime PUBLIC mujoco::mujoco glfw libmjpc
                                  PRIVATE ZLIB::ZLIB)
target_compile_definitions(grf_runtime PUBLIC
    GRF_MJPC_REVISION="${GRF_MJPC_REVISION}"
)

add_executable(data_factory src/main.cpp)
target_link_libraries(data_factory PRIVATE grf_runtime)

option(GRF_BUILD_TESTS "Build project regression tests" ON)
if(GRF_BUILD_TESTS)
  enable_testing()
  foreach(name contacts bundle)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE grf_runtime)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
  foreach(name reference qmc_sampling)
    add_executable(test_${name} tests/${name}_test.cpp)
    target_link_libraries(test_${name} PRIVATE grf_runtime)
    add_test(NAME ${name} COMMAND test_${name} ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
  add_executable(test_reference_runtime tests/reference_runtime_test.cpp)
  target_link_libraries(test_reference_runtime PRIVATE grf_runtime)
  add_test(NAME reference_runtime COMMAND test_reference_runtime
           ${CMAKE_CURRENT_SOURCE_DIR}/assets/winter_baseline_male.xml)
  add_test(NAME reference_runtime_female COMMAND test_reference_runtime
           ${CMAKE_CURRENT_SOURCE_DIR}/assets/plagenhoef_baseline_female.xml)
  add_executable(test_export tests/test_export.cpp)
  target_link_libraries(test_export PRIVATE grf_runtime)
  add_test(NAME export COMMAND test_export
           ${CMAKE_CURRENT_SOURCE_DIR}/assets/winter_baseline_male.xml)
  add_executable(test_export_offline tests/export_offline_test.cpp)
  target_link_libraries(test_export_offline PRIVATE grf_runtime)
  add_test(NAME export_offline COMMAND test_export_offline)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    # NumPy is only a test dependency, never needed by the C++ exporter.
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                    RESULT_VARIABLE GRF_TEST_NUMPY_STATUS
                    OUTPUT_QUIET ERROR_QUIET)
    if(GRF_TEST_NUMPY_STATUS EQUAL 0)
      add_test(NAME export_dataset COMMAND ${Python3_EXECUTABLE}
               ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_export_dataset.py)
      set_tests_properties(export_dataset PROPERTIES ENVIRONMENT
          "GRF_EXPORT_TEST_BUILD_DIR=$<TARGET_FILE_DIR:data_factory>;PYTHONDONTWRITEBYTECODE=1")
    else()
      message(STATUS "NumPy NPZ tests require a NumPy-enabled Python3_EXECUTABLE")
    endif()
  endif()
  add_executable(test_runner tests/test_runner.cpp)
  target_link_libraries(test_runner PRIVATE grf_runtime)
  add_test(NAME runner COMMAND test_runner
           ${CMAKE_CURRENT_SOURCE_DIR}/assets/winter_baseline_male.xml)
endif()
