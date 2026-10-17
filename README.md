# GRF data factory

Generates simulated motion and bilateral ground-contact labels from mocap trajectories using MuJoCo Predictive Control. Training inputs are executed, randomly ordered marker positions and total subject mass. Each foot's target is a six-parameter contact wrench: world-frame force `[Fx, Fy, Fz]`, ground-plane center of pressure `[CoPx, CoPy]`, and torsional friction moment `Tz` about the CoP.

Coordinates use metres, a right-handed frame, Z up, and ground at Z=0. Forces are in N and moments in Nm.

## Build

Requires Linux, C/C++17, CMake ≥3.20, Git and GLFW development dependencies. Debian/Ubuntu packages:

```bash
sudo apt install build-essential cmake git patch pkg-config libgl1-mesa-dev \
  zlib1g-dev xorg-dev libwayland-dev libxkbcommon-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"

# Python tools for mocap conversion and dataset export
python3 -m venv .venv-mocap
.venv-mocap/bin/pip install -r python/requirements-mocap.txt
```

CMake fetches MuJoCo 3.2.6, GLFW 3.4 and MJPC at `ff572a21e7c2bf9fda62e1862a758da7e9a8719b`. Upstream tracking-code license: `LICENSES/MJPC-Apache-2.0.txt`.

## Storage layout

`mocap/` contains input recordings, converted tracking references and previews. Only reviewed mapping files are version-controlled; recordings and `.venv-mocap/` stay ignored. Generated runs and datasets live outside the repository, with explicit CLI paths and no symlinks.

Local SSD setup (adjust for another machine; `GRF_DATA` is a shell variable, not application configuration):

```bash
mountpoint -q /run/media/darren/darren_ssd || { echo 'Mount the SSD first'; exit 1; }
export GRF_DATA=/run/media/darren/darren_ssd/data/grf_estimation
mkdir -p "$GRF_DATA/runs" "$GRF_DATA/datasets"
```

Existing campaigns are under `$GRF_DATA/runs/cmu_screening_30/` and `$GRF_DATA/runs/cmu_screening_centerline/`. Their historical commands/provenance retain the original paths; use their new location for replay. Keep an independent backup of the SSD. A training machine can use a different data root and download only the needed NPZ exports.

## Convert mocap

Source recordings are C3D plus an explicit mapping JSON. The mapping selects source labels or weighted marker averages and defines expected source units and a rigid transform into the simulator's coordinate frame. Mappings live beside their dataset recordings; reviewed mappings are version-controlled while recordings remain ignored. See `mocap/cmu/mapping.json` for an example.

The importer generates a reference JSON (`schema_version`, `time_s[T]`, `positions_m[T,16,3]`, `target_names`, `metadata`) for `run --reference`. This generated file contains the 16 tracking-landmark trajectories; it is not an alternative source format. The simulator does not read C3D directly.

```bash
.venv-mocap/bin/python python/mocap_import.py inspect \
  mocap/cmu/82/82_08.c3d --format cmu-c3d

.venv-mocap/bin/python python/mocap_import.py convert \
  mocap/cmu/82/82_08.c3d --format cmu-c3d \
  --mapping mocap/cmu/mapping.json --label-prefix Male1: \
  --output mocap/cmu/82/82_08.reference.json \
  --preview mocap/cmu/82/82_08.html
xdg-open mocap/cmu/82/82_08.html
```

The offline HTML shows landmark names, native-time playback, coordinates and foot-height traces. Conversion preserves native sample intervals, normalizes the time origin and records the original offset.

The reference file is JSON with `schema_version: 1`, `time_s[T]`, `positions_m[T,16,3]`, `target_names[16]` and `metadata`. Landmark order:

```text
pelvis, head, ltoe, rtoe, lheel, rheel, lknee, rknee,
lhand, rhand, lelbow, relbow, lshoulder, rshoulder, lhip, rhip
```

`metadata.source_id` or `metadata.source.source_id` identifies the original recording across model and marker-layout variants. Inspect the mapping and preview before simulation; no approval-status field is required.

### Import arguments

| Argument | Required | Default | Purpose |
|---|---|---|---|
| `input` | Yes | — | Source file; positional argument |
| `--format` | Yes | — | `c3d` or `cmu-c3d` (same reader, CMU provenance); applies to inspect/convert |
| `--mapping` | Convert | — | Source labels, derivations, units and coordinate transform |
| `--output` | Convert | — | Reference JSON destination; optional inspection-report destination for `inspect` |
| `--preview` | No | None | Also write offline HTML |
| `--label-prefix` | No | Mapping value | Override source-label namespace |
| `--source-id` | No | Adapter-derived | Original recording identifier |
| `--source-url` | No | Adapter-derived, if available | Source provenance |
| `--force` | No | Off | Replace existing outputs |
| `-h`, `--help` | No | Off | Show command help |

To regenerate HTML from an existing reference: `python/mocap_preview.py REFERENCE --c3d SOURCE.c3d --output PREVIEW.html [--force]`. Conversion previews include all raw markers; standalone previews without `--c3d` show only the 16 tracking landmarks. Toggle the layers independently and select a landmark to highlight its source markers and weights. Invalid raw samples are hidden, never filled. Raw trajectories are embedded only in HTML, not runtime reference JSON.

## Simulate and replay

```bash
./build/data_factory run --reference mocap/cmu/82/82_08.reference.json \
  --model assets/winter_baseline_male.xml --output "${GRF_DATA:?Set GRF_DATA first}/runs/walk.grf"
./build/data_factory replay "${GRF_DATA:?Set GRF_DATA first}/runs/walk.grf"
```

Available body models are `assets/winter_baseline_male.xml` and `assets/plagenhoef_baseline_female.xml`. Geometry determines mass and inertia through uniform density. Morphology and marker-layout indices are independent. Each of 14 eligible groups has 10 markers: **140 total**, with hands excluded. Each whole foot, including articulated toes, shares 10 markers. Sites are sampled over primitive surfaces, rejecting points buried inside other geometry at the neutral pose.

Physics runs at **2400 Hz**. The planner replans every **15 ms (36 physics steps)**. Capture defaults to **600 Hz** and must divide 2400 into integer steps. Reference timing, capture rate and replay speed are independent. Simulation stops at the last physics step within the requested duration and includes that final frame even if it falls off the capture grid.

Non-foot ground support above 1 N, numerical failure or an unexpected reset stops the run. The saved file retains the valid prefix and termination reason; incomplete runs are excluded from dataset export by default.

### Run flags

| Flag | Required | Default | Purpose |
|---|---|---|---|
| `--reference` | Yes | — | Timestamped reference JSON |
| `--reference-scaling` | No | `body-height` | Whole-body-chain size estimate, Z-only scaling; `none` preserves source coordinates |
| `--output` | Yes | — | Destination `.grf` file |
| `--model` | No | `assets/winter_baseline_male.xml` | Body model XML |
| `--duration` | No | Full reference | Duration in seconds, capped at reference end |
| `--qmc-index` | No | `0` | Morphology index; 0 uses nominal dimensions |
| `--layout-index` | No | `0` | Independent marker-layout index |
| `--capture-hz` | No | `600` | Recorded samples per second |
| `--threads` | No | `1` | Planner worker count |
| `--render` | No | Off | Replay the run after simulation |
| `--help` | No | Off | Show command help |

### Reference size normalization

New runs default to `--reference-scaling body-height`. For every frame and both sides, sum **3D Euclidean distances** along heel → knee → hip → shoulder → head. The pooled median is the source body-size proxy. The target proxy is the mean of the same two chains at the actual morphed model's neutral tracking sites. One constant scale `s = target / source` adjusts only height:

```text
x' = x
y' = y
z' = s * z
```

This does not require a neutral source start or use standing height. XY, horizontal travel speed, stride width/length and timestamps stay unchanged; vertical excursions/velocities scale by `s`. Z=0 and stationary reference feet are preserved. This is an approximate landmark-based adjustment: chain lengths can vary with pose/marker placement, and Z-only scaling changes segment angles and proportions. It does not correct marker offsets, smooth noise, enforce joint limits or guarantee contact feasibility.

Scaling happens in C++ preparation before both initialization and tracking. Imported JSON, physical model parameters and controller weights are untouched. Run metadata records the mode, scale, scaled axis, ground plane, source/model chain proxies and source proxy range. Degenerate/nonfinite chain segments are rejected rather than skipped. Use `--reference-scaling none` for unscaled coordinates. The former leg-only XYZ method has been removed, not retained as an option. Existing `.grf` recordings replay unchanged.

Toe tracking sites use the articulated toe-body origin, local `[0,0,0]`. Heel sites retain their rear capsule endpoint X and use foot-local Y=Z=0, between the rear capsule endpoint centres. Neither site has an upward offset. These are tracking conventions, not calibrated surface-marker or sole-contact locations. Physical geometry is unchanged; both site positions and conventions are saved in preparation provenance. Heel placement participates in the body-height scale estimate; toe placement does not. Heel/toe position and velocity weights remain equal.

Controller overrides are optional; defaults are defined in `src/tracking/tracking.hpp` and the tracking model builder.

| Flag | Default | Purpose |
|---|---|---|
| `--effort-rate-weight` | `100` | Activation-change penalty |
| `--stance-roll-weight` | `0` | Reference-gated lateral foot-roll penalty |
| `--stance-roll-still-time` | `0` | Required prior low/slow foot interval, seconds |
| `--knee-hip-pos-scale` | `0.8` | Knee/hip position-weight multiplier |
| `--elbow-velocity-weight` | `0.02` | Elbow reference-velocity weight |
| `--foot-position-weight` | `200` | Shared bilateral toe/heel position weight (finite, nonnegative; velocities unchanged) |
| `--elbow-position-weight` | `16` | Elbow position weight |
| `--hand-position-weight` | `8` | Hand position weight |

`replay` takes a required positional `.grf` path and optional `--speed` (default `1`). Replay speed changes playback only, not the recorded dynamics.

## What is a `.grf` file?

A `.grf` is a custom **self-contained binary simulation recording**, not just a force array and not the training dataset. It contains the compiled `mjModel`, captured states and contact labels needed to replay a run or generate different marker observations without simulating it again.

The runner collects frames in memory. At the end of a run, it serializes the model with MuJoCo's `mj_saveModel`, validates the frames, writes a temporary file, then atomically replaces the destination. A version-2 file consists of:

| Block | Contents |
|---|---|
| Header | Format identifier/version, model and summary byte counts, frame count, state dimensions and marker count |
| Compiled model | MuJoCo MJB bytes: geometry, mass/inertia, joints, actuators, sites and reference keyframes |
| Run summary | Embedded JSON containing source identity, configuration, timing, marker-layout audit, diagnostics and termination status |
| Frame records | Timestamp and physics-step index; joint positions/velocities; actuator activations and controls; reference mocap positions/orientations; executed observation markers and tracking landmarks; bilateral forces, moments, CoP, torsional friction moment about CoP and validity masks; contact diagnostics |

Numeric frame arrays are stored as doubles. The format uses native binary representations and embeds a MuJoCo-version-dependent model; it is not a portable interchange standard. The loader requires the same MuJoCo version recorded in the summary. No original XML or mocap file is needed for replay. Replay restores saved states rather than rerunning the planner. The recording is not a restart checkpoint: planner policies and full solver history are not saved.

Forces are reconstructed at the captured state using a forward pass on a copy of simulation data. Labels are instantaneous, not capture-interval averages. Each foot's wrench aggregates all floor contacts on its descendants, including toes and contact torques. Force is floor-on-foot in world coordinates; raw moment is about world origin. Each foot's own aggregated contact wrench is expressed at its ground-plane CoP, not at the whole-body CoP.

The six-parameter target is `[Fx, Fy, Fz, CoPx, CoPy, Tz_at_CoP]`. `Tz_at_CoP` is the resultant torsional friction moment at the foot–ground interface, including distributed contact forces and contact torques—not the moment about world origin. CoP Z is fixed at zero; the resultant horizontal moment components about CoP are zero by construction.

CoP and its torsional moment are undefined for nonpositive vertical force or nonfinite results. Raw forces and world-origin moments remain in the recording for later threshold selection. Flight samples are retained. Legacy version-1 recordings are replay-only because they lack the full-wrench labels.

## Training dataset

Export produces one compressed NumPy archive directly from `.grf` recordings, without intermediate JSON or sidecar files:

```bash
./build/data_factory export "${GRF_DATA:?Set GRF_DATA first}/runs/walk.grf" \
  --output "$GRF_DATA/datasets/walk.npz"
```

### Export arguments

| Argument | Required | Default | Purpose |
|---|---|---|---|
| `RUN.grf...` | Yes | — | One or more recordings |
| `--output` | Yes | — | Destination `.npz` archive |
| `--layout-index` | No | Recorded layout | Sample alternative marker attachments using saved poses |
| `--permutation-seed` | No | `0` | Fixed-within-episode anonymous marker permutation |
| `--min-normal-force` | No | `0` N | Require Fz above this value for CoP/torsional-moment labels |
| `--allow-incomplete` | No | Off | Include incomplete recordings for diagnostics |
| `--help` | No | Off | Show command help |

`T` is the total sample count, `E` the episode count, and `M` the marker count. Foot order is left, right; wrench components are `[Fx, Fy, Fz, CoPx, CoPy, Tz_at_CoP]` in N, m and Nm respectively.

| Array | Shape | Contents |
|---|---|---|
| `markers_m` | `[T,M,3]` | Executed marker positions; fixed slot correspondence within each episode |
| `mass_kg` | `[E]` | Subject mass per episode |
| `time_s` | `[T]` | Recorded sample times |
| `contact_wrench` | `[T,2,6]` | Bilateral six-parameter contact-wrench targets |
| `episode_offsets` | `[E+1]` | Episode boundaries |
| `source_id` | `[E]` | Original recording IDs for partition grouping |

During flight, GRF components are zero and CoP/torsional-moment components are NaN. Derive the target loss mask with `numpy.isfinite` and exclude or replace undefined components before loss arithmetic. Raw moments, model state and marker/body identities stay in `.grf`, not the training archive. Alternate observation layouts change only marker positions, not the recorded motion or contact labels.

Load with `numpy.load(path, allow_pickle=False)`. Slice episode `e` using `episode_offsets[e]:episode_offsets[e+1]` and pair it with `mass_kg[e]`. Do not form temporal windows across episode boundaries. Keep all variants of an original recording in the same dataset partition.

## Development tests

These are regression tests for maintaining the importer, simulator and exporter—not a procedure for accepting a new mocap recording. Run them after code, model-building or file-format changes to catch broken contact conventions, timing, marker attachments or serialization.

- `tests/` C++ tests cover force signs, toe aggregation, static/flight contact, momentum balance, capture timing, marker placement, reference interpolation, bundle compatibility and offline export.
- `tests/test_mocap_import.py` covers source parsing, mapping, units, timestamps and preview generation.
- `tests/test_export_dataset.py` validates direct NPZ export, array shapes, episode boundaries and flight labels using NumPy.

```bash
ctest --test-dir build --output-on-failure
PYTHONDONTWRITEBYTECODE=1 .venv-mocap/bin/python -m unittest discover -s tests
```

For a fresh recording, use `inspect`, convert with its mapping, inspect the reference preview, then run and replay the simulator. Passing regression tests does not establish that a particular recording is mapped correctly or produces realistic motion.
