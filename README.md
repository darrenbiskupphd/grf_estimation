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

## Nominal models and controller

The selected **O configuration is now the default**, not a separate preset. `assets/` contains only `winter_baseline_male.xml` and `plagenhoef_baseline_female.xml`. Both have bilateral `ankle_x` −12°/+30° and passive `shoulder3` limits of right −60°/+80°, left −80°/+60°; axial stiffness/damping remain 1/0.2 and ankle_x stiffness remains 3. The controller defaults to effort-rate weight 100, elbow position weight 16, hand position 8, elbow reference-velocity weight 0.02, roll 0, QMC 0, and one planner worker.

These are the adopted engineering settings, not validated anatomical ROM or GRF labels. Heel behavior and shoulder soft-limit overruns remain unresolved. Earlier nominal XMLs remain available in Git history; trial XML copies have been removed.

## Run a tracking episode

Run from the repository root. The output path is explicit; **data/runs/** is ignored by Git and is created when needed. No wrapper or tuning flags are required for the default controller.

~~~bash
./build/data_factory run \
  --model assets/winter_baseline_male.xml \
  --motion walk \
  --qmc-index 0 \
  --output data/runs/male_walk_raw.grf
~~~

Use --render to show the recorded replay immediately after the run:

~~~bash
./build/data_factory run \
  --model assets/plagenhoef_baseline_female.xml \
  --motion walk \
  --output data/runs/female_walk_raw.grf \
  --render
~~~

**walk** includes the source clip’s standing lead-in. The foot-only source clips are `walk`, `jump`, `dance`, `kick_spin`, and `spin_kick`; hand- or knee-supported acrobatics are intentionally excluded. The bundled `run` clip is retired (39 frames / 1.267 s); sustained running needs a replacement reference. Every run starts at source frame zero. Omit `--duration` for the full non-looping clip; a longer value stops at the clip end and the bundle records requested/effective/capped duration. An existing `--output` bundle is replaced only after the new bundle is written successfully. The runtime uses 2400 Hz physics, replans every 15 ms, evaluates the current policy at every physics step, and uses one planner worker unless `--threads` is supplied.

The tracker copies the ordered 16 source points exactly, including for nonzero QMC morphology draws: there is no scale, alignment, or retargeting option. A one-time neutral-seeded pose fit and floor clearance configure the custom model’s initial physical state only; they never alter stored target coordinates. This is the intended input contract for future sources after their trajectories are distilled into the same ordered 16 points; external-trajectory import is not implemented yet.

| QMC option | Default | Meaning |
| --- | ---: | --- |
| `--qmc-index <index>` | `0` | `0` keeps the nominal physical model and uses a canonical marker layout. A positive value is one joint Halton sample: bases 2 and 3 set the experimental global length/radius draw, while bases 5, 7, and 11 place seven massless surface markers on each eligible body. It is recorded in the bundle. |

One episode index avoids two coupled command-line knobs while keeping morphology and marker placement in distinct QMC dimensions. Marker sites change no contact, inertia, or controller parameter. For example, use `--qmc-index 2` for a reproducible non-nominal visual test.

`--effort-rate-weight <value>` controls the activation-rate penalty and defaults to **`100`**. Set it to `0` to disable the term. The cost is `0.5 * weight * mean((activation_rate / 20)^2)`, with activation rate in inverse seconds, computed from the existing filtered actuators. This regularizes changes in actuator effort; it does not directly penalize impacts or smooth recorded forces. The weight and definition are saved in the bundle's preparation metadata.

Experimental `--stance-roll-weight <value>` (default `0`) adds a reference-gated lateral foot-roll penalty, with no direct pitch penalty; the gate uses low/slow reference toe and heel markers, not simulated contact. Optional `--stance-roll-still-time <seconds>` (default `0`) requires that low/slow reference condition to persist before the roll cost engages. `--knee-hip-pos-scale <value>` (default `1`) multiplies only knee/hip position-tracking weights. `--elbow-velocity-weight <value>` defaults to `0.02` and changes **only** the existing left/right `Vel[elbow]` reference-tracking weight; it is not a penalty on absolute elbow speed or GRF changes. `--elbow-position-weight` defaults to **`16`** and `--hand-position-weight` to **`8`**; they change only the respective bilateral point-position terms. Shoulder and lower-body weights remain unchanged. These settings are recorded in preparation metadata and leave model physics and raw target trajectories unchanged. Overrides support controlled experiments; no setting implies validated GRF labels.

## Replay a saved run

~~~bash
./build/data_factory replay data/runs/male_walk_raw.grf --speed 1
~~~

`--speed` is a positive playback multiplier: `.25` is quarter speed and `1` is real time. For research replay reviews, use **1× Walk**, **0.5× Jump and Dance**, and **0.25× Kick Spin and Spin Kick**. Report the exact replay command for **every full clip and both sexes**, grouped by controller candidate (all A, then all B, then all C), with a short candidate legend; never make the reviewer infer settings from filenames. Scroll to zoom, left-drag to rotate, right-drag to pan vertically, Shift+right-drag to pan horizontally, and press `F` to toggle root-follow while inspecting a foot.

The replay viewer draws cyan GRF arrows at `0.003 m/N` and leaves only the overhead model light casting a shadow; these display settings do not change saved forces or simulation physics.

Each .grf file is a self-contained binary bundle containing the compiled model, 200 Hz sampled states and targets, force/contact diagnostics, and a JSON summary. New bundles include QMC marker sites; older saved bundles do not gain them retroactively. Replay loads its model from memory and does not replan or write side files. A new QMC bundle shows blue targets, orange tracking sites, red massless markers, and cyan GRF arrows. The red markers are rendered as decorative overlays, so they remain attached to the model without casting a collective ground shadow.

The force arrows begin at a vertical-force-weighted contact-position proxy. They are useful diagnostics, not validated CoP labels. A run stops before recording a loaded non-foot floor-contact frame or on numerical failure; use only bundles whose `completed_requested_duration` is true for downstream episodes.

## Current scope

- The physical XML models, contact settings, passive joints, and actuator parameters are preserved during tracking preparation.
- Replay frames record per-foot GRF, the contact-position proxy, per-geometry normal loads, foot-frame contact position, foot pitch/roll, ankle state, and MTP state.
- Positive morphology QMC indices retain the earlier global length/radius sampler. It is an experimental coverage tool, not a population-calibrated anthropometric distribution; do not begin a broad sweep before the foot-contact review.
- One recorded QMC index supplies distinct Halton dimensions for morphology and marker placement. A later controlled cross of fixed morphology against several layouts belongs in the batch design, not in the runtime CLI.
- The baseline feet use rounded capsules and a passive MTP joint. Heel/edge/toe behavior remains an active diagnostic question; no contact-physics tuning should be inferred from the current defaults.

## Repository layout

- [assets/](assets/): nominal male and female MuJoCo models.
- [src/tracking/](src/tracking/): model assembly (`build_reference_tracking_model.cpp`), initialization (`initialize_reference_tracking_state.cpp`), objective (`reference_task.cpp`), runner, run bundles, and contact diagnostics.
- [src/viewer.cpp](src/viewer.cpp): replay viewer.
- [python/xml_visualizer.py](python/xml_visualizer.py): exploratory baseline-model inspection helper.
- data/runs/: ignored, explicitly requested local run bundles.
