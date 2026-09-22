# Ground Reaction Force and Center of Pressure Estimation

This early research prototype uses MuJoCo and MuJoCo MPC (MJPC) to test whether the project’s male and female humanoid models can dynamically follow motion-capture-derived anatomical point trajectories. Its current job is a lean motion tracer bullet: produce one replayable tracking run and expose the contact behavior needed to judge it.

The source motion is represented by 16 three-dimensional target points: pelvis, head, bilateral toe, heel, knee, hand, elbow, shoulder, and hip. The controller always transfers those blue target trajectories in **raw** source coordinates, without dimensional scaling or source joint poses. It does not yet produce validated estimator-training labels or a learned estimator.

## Build

The instructions target Linux. You need a C/C++17 toolchain, CMake 3.20 or newer, Git, and the graphics development packages used by GLFW and MuJoCo. The first configuration fetches dependencies.

For Debian/Ubuntu:

~~~bash
sudo apt install build-essential cmake git patch pkg-config libgl1-mesa-dev zlib1g-dev xorg-dev libwayland-dev libxkbcommon-dev
~~~

~~~bash
git clone https://github.com/darrenbiskupphd/grf_estimation.git
cd grf_estimation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target data_factory --parallel 4
~~~

The build pins MuJoCo 3.2.6, GLFW 3.4, and MJPC revision ff572a21e7c2bf9fda62e1862a758da7e9a8719b.

## Run a tracking episode

Run from the repository root. The output path is explicit; **data/runs/** is ignored by Git and is created when needed.

~~~bash
./build/data_factory run \
  --model assets/winter_baseline_male.xml \
  --motion walk --duration 8 \
  --qmc-index 0 \
  --output data/runs/male_walk_raw.grf
~~~

Use --render to show the recorded replay immediately after the run:

~~~bash
./build/data_factory run \
  --model assets/plagenhoef_baseline_female.xml \
  --motion walk --duration 8 \
  --output data/runs/female_walk_raw.grf \
  --render
~~~

**walk** includes the source clip’s standing lead-in. The foot-only source clips are `walk`, `run`, `jump`, `dance`, `kick_spin`, and `spin_kick`; hand- or knee-supported acrobatics are intentionally excluded. Every run starts at source frame zero. Omit `--duration` for the full non-looping clip; a longer value stops at the clip end and the bundle records requested/effective/capped duration. An existing `--output` bundle is replaced only after the new bundle is written successfully. The runtime uses 2400 Hz physics, replans every 15 ms, evaluates the current policy at every physics step, and uses one planner worker unless `--threads` is supplied.

The tracker copies the ordered 16 source points exactly, including for nonzero QMC morphology draws: there is no scale, alignment, or retargeting option. A one-time neutral-seeded pose fit and floor clearance configure the custom model’s initial physical state only; they never alter stored target coordinates. This is the intended input contract for future sources after their trajectories are distilled into the same ordered 16 points; external-trajectory import is not implemented yet.

| QMC option | Default | Meaning |
| --- | ---: | --- |
| `--qmc-index <index>` | `0` | `0` keeps the nominal physical model and uses a canonical marker layout. A positive value is one joint Halton sample: bases 2 and 3 set the experimental global length/radius draw, while bases 5, 7, and 11 place seven massless surface markers on each eligible body. It is recorded in the bundle. |

One episode index avoids two coupled command-line knobs while keeping morphology and marker placement in distinct QMC dimensions. Marker sites change no contact, inertia, or controller parameter. For example, use `--qmc-index 2` for a reproducible non-nominal visual test.

## Replay a saved run

~~~bash
./build/data_factory replay data/runs/male_walk_raw.grf --speed 0.25
~~~

`--speed` is a positive playback multiplier: `.25` is quarter speed and `1` is real time. Scroll to zoom, left-drag to rotate, right-drag to pan vertically, Shift+right-drag to pan horizontally, and press `F` to toggle root-follow while inspecting a foot.

Each .grf file is a self-contained binary bundle containing the compiled model, 200 Hz sampled states and targets, force/contact diagnostics, and a JSON summary. New bundles include QMC marker sites; older saved bundles do not gain them retroactively. Replay loads its model from memory and does not replan or write side files. A new QMC bundle shows blue targets, orange tracking sites, red massless markers, and cyan GRF arrows. The red markers are rendered as decorative overlays, so they remain attached to the model without casting a collective ground shadow.

The force arrows begin at a vertical-force-weighted contact-position proxy. They are useful diagnostics, not validated CoP labels. A run stops before recording a loaded non-foot floor-contact frame or on numerical failure; use only bundles whose `completed_requested_duration` is true for downstream episodes.

## Current scope

- The physical XML models, contact settings, passive joints, and actuator parameters are preserved during tracking preparation.
- Replay frames record per-foot GRF, the contact-position proxy, per-geometry normal loads, foot-frame contact position, foot pitch/roll, ankle state, and MTP state.
- Positive morphology QMC indices retain the earlier global length/radius sampler. It is an experimental coverage tool, not a population-calibrated anthropometric distribution; do not begin a broad sweep before the foot-contact review.
- One recorded QMC index supplies distinct Halton dimensions for morphology and marker placement. A later controlled cross of fixed morphology against several layouts belongs in the batch design, not in the runtime CLI.
- The baseline feet use rounded capsules and a passive MTP joint. Heel/edge/toe behavior remains an active diagnostic question; no contact-physics tuning should be inferred from the current defaults.

## Tests

~~~bash
cmake --build build --target tracking_contract_test tracking_integration_test --parallel 4
ctest --test-dir build --output-on-failure
~~~

The contract test verifies raw target identity for nominal and QMC models, extra foot-only source clips, duration capping, controller cost weights, QMC marker behavior, interpolation, and target-only transitions. The integration test performs short nominal/nonzero-QMC rollouts, verifies replacement of an existing bundle, reloads each result, and removes its build-local artifacts.

## Repository layout

- [assets/](assets/): nominal male and female MuJoCo models.
- [src/tracking/](src/tracking/): target preparation, tracking objective, runner, compact run bundle, and contact diagnostics.
- [src/viewer.cpp](src/viewer.cpp): replay viewer.
- [tests/](tests/): source-based contract and integration checks.
- [python/xml_visualizer.py](python/xml_visualizer.py): exploratory baseline-model inspection helper.
- data/runs/: ignored, explicitly requested local run bundles.
