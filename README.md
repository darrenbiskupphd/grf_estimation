# Ground Reaction Force and Center of Pressure Estimation

An early research project exploring estimation of ground reaction force (GRF) and center of pressure (CoP) from motion-capture marker trajectories. The current repository implements a synthetic-data prototype using MuJoCo and MuJoCo MPC (MJPC).

The C++ `data_factory` executable randomizes a humanoid model, attaches procedural markers, runs an MJPC walking task, and exports per-foot forces, contact-position estimates, and marker coordinates. It can also replay an episode with force arrows and planner traces. A trained estimator, training pipeline, and validated real-world results are future work; reliable walking across morphologies is still under development.

## Build

The instructions below target Linux. You need a C/C++ toolchain with C++17 support, CMake 3.20 or newer, Git, and the graphics development dependencies used by GLFW and MuJoCo. The first configuration downloads dependencies and requires network access.

For Debian/Ubuntu, typical prerequisites are:

```bash
sudo apt install build-essential cmake git pkg-config libgl1-mesa-dev zlib1g-dev xorg-dev libwayland-dev libxkbcommon-dev
```

GLFW builds both X11 and Wayland backends by default on Linux; see its [platform dependency instructions](https://www.glfw.org/docs/3.4/compile.html#compile_deps_wayland). Graphics libraries are build dependencies even when running an episode without a viewer.

```bash
git clone https://github.com/darrenbiskupphd/grf_estimation.git
cd grf_estimation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target data_factory --parallel 4
```

[CMakeLists.txt](CMakeLists.txt) requests MuJoCo **3.2.6**, GLFW **3.4**, and MJPC **`main`**. MJPC is not pinned, so a fresh checkout can resolve a different dependency revision. The project currently has no automated clean-build verification across platforms.

## Generate an episode

Run from the repository root so relative asset paths resolve. Create the output directory first; the program does not create it.

```bash
mkdir -p data
./build/data_factory --duration 1 --speed 0.8 --qmc-index 1 --output data/episode_001.csv
```

Use the other baseline and optionally replay the completed simulation:

```bash
./build/data_factory --model assets/plagenhoef_baseline_female.xml --duration 1 --speed 0.8 --qmc-index 2 --output data/episode_002.csv --render
```

`--render` requires a working graphical display. Replay runs after generation and closes at the end of the episode; Escape closes it early. Drag with the left mouse button to rotate, the right button to pan, and use the wheel to zoom.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model <path>` | `assets/winter_baseline_male.xml` | MJCF model compatible with the humanoid walking task and recorder. |
| `--duration <seconds>` | `1.0` | Requested simulation time; an episode can terminate early. |
| `--speed <m/s>` | Random | Nonnegative target speed. A negative value selects a random target in `[0.5, 2.5)` m/s. |
| `--qmc-index <integer>` | `1` | Halton-sequence index controlling geometry and marker placement. Use a positive integer. |
| `--output <path>` | No file | CSV destination; an existing file at this path is overwritten. |
| `--render` | Off | Replay the generated episode. |

Geometry randomization is always enabled; index `1` is not the unmodified baseline. An explicit speed and QMC index control these inputs but do not provide a complete reproducibility guarantee. There is no seed option or batch-generation command. Numeric arguments have only limited validation.

The simulator targets 600 Hz recording with 2400 Hz physics. Planning is synchronous, so generating an episode can take substantially longer than its simulated duration. Each process currently creates 22 planner worker threads; account for this before launching several processes.

## Output and current limitations

Each CSV contains:

| Columns | Contents / units |
| --- | --- |
| `time` | Simulation timestamp in seconds. |
| `grf_left_x/y/z`, `grf_right_x/y/z` | Summed force exerted by the floor on each foot, in world axes, in newtons. |
| `cop_left_x/y/z`, `cop_right_x/y/z` | Vertical-force-weighted contact positions, in world axes, in metres. |
| `marker_<site_id>_x/y/z` | Procedural marker positions, in world axes, in metres. |

The supplied models produce 112 markers and 349 columns. Use the recorded `time` column when reading an episode. Model state, contact moments, marker-to-body mappings, and episode configuration metadata are not exported.

The current `cop_*` values are a contact-position approximation: they do not incorporate contact torques or project onto a specified force-plate plane. An unloaded foot has zero CoP values and no separate validity flag. Episodes stop when the recorder detects floor contact by a body other than a foot or toe, retaining the preceding frames. This check does not establish that a gait is biologically realistic or that all unwanted contacts are detected.

## Inspect a baseline model

The optional Python helper prints body masses, an inertia comparison, and height/centre-of-mass diagnostics, then opens MuJoCo's viewer:

```bash
python3 -m venv /tmp/grf-inspection-venv
/tmp/grf-inspection-venv/bin/python -m pip install 'mujoco==3.2.6' numpy
/tmp/grf-inspection-venv/bin/python python/xml_visualizer.py assets/winter_baseline_male.xml
```

This displays the unrandomized XML. The inertia comparison uses female Plagenhoef coefficients for either input model and prints differences without pass/fail criteria; treat it as an exploratory diagnostic.

## Repository layout

- [`assets/`](assets/): male and female baseline MJCF models and shared walking-task settings.
- [`src/main.cpp`](src/main.cpp): command-line entry point, planning, simulation, and replay.
- [`src/morphology_randomizer.cpp`](src/morphology_randomizer.cpp): geometry/marker sampling, force extraction, and CSV recording.
- [`python/xml_visualizer.py`](python/xml_visualizer.py): baseline inspection helper.
- `build/` and `data/`: local build products and generated data, ignored by Git.
