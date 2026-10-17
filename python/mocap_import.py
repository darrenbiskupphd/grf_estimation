#!/usr/bin/env python3
"""Strict, reference-only C3D importer (Python >= 3.10).

Install: python -m pip install -r python/requirements-mocap.txt
Inspect: python python/mocap_import.py inspect mocap/cmu/82/82_08.c3d --format cmu-c3d
Import:  python python/mocap_import.py convert mocap/cmu/82/82_08.c3d \
  --format cmu-c3d --mapping mocap/cmu/mapping.json --label-prefix Male1: \
  --output mocap/cmu/82/82_08.reference.json --preview mocap/cmu/82/82_08.html

Source input is C3D plus an explicit mapping JSON. Use --format c3d for general
recordings; cmu-c3d uses the same reader with CMU provenance conventions.
Reference JSON is generated output for the simulator, not an alternative source.
An invalid *selected* point is an error, never filled, trimmed or interpolated.
Unselected invalid points are counted but do not prevent conversion. Source
labels are exact and unique (C3D fixed-width padding is removed, not namespaces).
JSON duplicate keys, unknown configuration keys, NaN/Infinity and ambiguous
labels fail closed. C3D requires POINT:UNITS and native POINT:RATE; ASF/AMC is
intentionally unsupported when measured C3D trajectories are available. C3D
0.6.0 extended starting-frame inconsistencies (>65535) are rejected explicitly,
not silently shifted.

Mapping contract (see mocap/cmu/mapping.json):
- Exactly the workspace's 16 named targets, in TARGET_NAMES order on output.
- Each target is either kind='marker' with 'label', or kind='weighted_mean'
  with explicit source 'labels' and nonnegative 'weights' summing to one.
  No inferred aliases, joint-center reconstruction, chained expressions, or
  anatomical offsets. Every target needs anatomical_meaning/evidence/limitations.
- label_prefix is prepended literally; --label-prefix is an explicit override
  recorded with the original mapping checksum and effective configuration.
- expected_units must match input units. The sole scale is conversion to metres.
- world_from_source_m is a proper rigid transform: p_out = R @ (scale*p_in) + t.
  Rotation must be orthonormal with det +1. Ground assumptions and axis evidence
  are mandatory; there is no automatic foot-floor fit or morphology scaling.

Canonical output has schema_version=1, time_s[T] starting at zero,
positions_m[T][16][3], target_names[16], metadata. rhip is the 16th workspace
target. metadata.time_offset_s is the
subtracted source time; C3D uses (frame_number-1)/POINT:RATE, not wall-clock time.
The source frame range, checksums, complete effective mapping, axis/ground
assumptions, and quality diagnostics accompany the reference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
import re
import sys
import warnings

import numpy as np

TARGET_NAMES = (
    "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
    "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip", "rhip",
)
UNIT_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
CMU_HOME = "http://mocap.cs.cmu.edu/"


class ImportValidationError(ValueError):
    """Input would require an undocumented assumption or change to observations."""


def _require(condition, message):
    if not condition:
        raise ImportValidationError(message)


def _keys(obj, required, optional=(), *, context):
    _require(isinstance(obj, dict), f"{context} must be an object")
    missing = set(required) - obj.keys()
    extra = obj.keys() - set(required) - set(optional)
    _require(not missing and not extra,
             f"{context}: missing keys {sorted(missing)}; unknown keys {sorted(extra)}")


def _text(value, context):
    _require(isinstance(value, str) and bool(value.strip()), f"{context} must be nonempty text")


def _strings(value, context, *, nonempty=False):
    _require(isinstance(value, list) and (bool(value) or not nonempty),
             f"{context} must be a list" + (" with at least one entry" if nonempty else ""))
    for entry in value:
        _text(entry, context)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ImportValidationError(f"Nonstandard JSON number: {value}")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_no_duplicate_keys, parse_constant=_bad_constant)


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _numeric(value, context):
    try:
        raw = np.asarray(value) if isinstance(value, np.ndarray) else np.asarray(value, dtype=object)
        if raw.dtype.kind not in "iuf":
            _require(raw.dtype.kind == "O" and all(
                item is None or (isinstance(item, Real) and not isinstance(item, (bool, np.bool_)))
                for item in raw.flat), f"{context} must contain numbers, not strings/booleans")
        return np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ImportValidationError(f"{context} must be numeric: {error}") from error


def _times(value):
    time = _numeric(value, "time_s")
    _require(time.ndim == 1 and len(time) >= 2, "At least two timestamps are required")
    _require(np.isfinite(time).all() and np.all(np.diff(time) > 0),
             "time_s must be finite and strictly increasing")
    return time


@dataclass
class LabeledPoints:
    """Adapter boundary. Positions are NOT scaled; validity is per observation."""

    labels: list[str]
    time_s: np.ndarray
    positions: np.ndarray
    units: str
    valid: np.ndarray | None = None
    metadata: dict | None = None

    def __post_init__(self):
        _strings(self.labels, "labels", nonempty=True)
        _require(len(self.labels) == len(set(self.labels)), "Duplicate source labels")
        self.time_s = _times(self.time_s)
        self.positions = _numeric(self.positions, "positions")
        shape = (len(self.time_s), len(self.labels))
        _require(self.positions.shape == (*shape, 3), f"positions must have shape {(*shape, 3)}")
        _require(isinstance(self.units, str) and self.units in UNIT_TO_M,
                 f"Unsupported units {self.units!r}; choose {list(UNIT_TO_M)} explicitly")
        finite = np.isfinite(self.positions).all(axis=2)
        if self.valid is None:
            self.valid = finite
        else:
            mask = np.asarray(self.valid)
            _require(mask.shape == shape and mask.dtype == np.bool_,
                     f"valid must be a boolean array of shape {shape}")
            self.valid = mask & finite
        if self.metadata is None:
            self.metadata = {}
        _require(isinstance(self.metadata, dict), "metadata must be an object")


def read_c3d(path, *, cmu=False):
    """Read native labeled C3D points; analog channels are not force labels here."""
    try:
        import c3d
    except ImportError as error:
        raise ImportValidationError("Install python/requirements-mocap.txt to read C3D") from error
    class MarkerReader(c3d.Reader):
        def _check_metadata(self):
            self.ignored_empty_analog_subframes = None
            header = self.header
            # Some files declare one analog subframe despite having no analog
            # channels or bytes. Normalize only for validation, never on disk.
            if (self.get("ANALOG:USED") is not None and
                    self.get("ANALOG:RATE") is not None and
                    self.analog_used == 0 and header.analog_count == 0 and
                    self.analog_rate == 0 and header.analog_per_frame != 0):
                original = header.analog_per_frame
                header.analog_per_frame = 0
                try:
                    super()._check_metadata()
                finally:
                    header.analog_per_frame = original
                self.ignored_empty_analog_subframes = original
            else:
                super()._check_metadata()

    with Path(path).open("rb") as stream, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            reader = MarkerReader(stream)
        except AssertionError as error:
            raise ImportValidationError(f"Inconsistent C3D metadata: {error}") from error
        _require(reader.get("POINT:RATE") is not None, "C3D POINT:RATE missing; rate will not be guessed")
        rate = float(reader.point_rate)
        _require(np.isfinite(rate) and rate > 0, "C3D POINT:RATE must be positive and finite")
        unit_param = reader.get("POINT:UNITS")
        _require(unit_param is not None, "C3D POINT:UNITS is missing; units will not be guessed")
        units_raw = unit_param.string_value
        units = units_raw.strip().lower()
        labels = [str(label).strip() for label in reader.point_labels]
        _require(len(labels) == reader.point_used, "C3D POINT:LABELS count does not match POINT:USED")
        first, last = int(reader.first_frame), int(reader.last_frame)
        _require(first >= 1 and last >= first, "Invalid C3D frame range")
        # c3d 0.6.0 has a high-word bug in first_frame. Never silently shift an
        # extended (>65535) starting frame; reject unsupported reader semantics.
        start_param = reader.get("TRIAL:ACTUAL_START_FIELD")
        if start_param is not None:
            words = np.asarray(start_param.uint16_array).ravel()
            _require(len(words) == 2, "C3D ACTUAL_START_FIELD must contain two 16-bit words")
            _require(first == int(words[0]) + (int(words[1]) << 16),
                     "C3D extended starting frame is unsupported by c3d 0.6.0; "
                     "refusing to shift native timing")
        count_param = reader.get("POINT:FRAMES")
        if count_param is not None:
            count = float(count_param.float_value if count_param.bytes_per_element == 4
                          else count_param.uint16_value)
            _require(np.isfinite(count) and count >= 1 and count.is_integer(), "Invalid C3D POINT:FRAMES")
            if count < 65535:  # 65535 may be a saturated legacy count in long trials.
                _require(count == last - first + 1, "C3D POINT:FRAMES disagrees with native frame range")
        frames, points = [], []
        # No NaN replacement, no smoothing, and no assumption that camera_count=0
        # proves invalidity: C3D may contain processed or synthetic valid samples.
        for frame, point, _analog in reader.read_frames(copy=True, check_nan=False):
            frames.append(frame)
            points.append(point)
        _require(frames == list(range(first, last + 1)), "C3D truncated or frame range inconsistent")
        data = np.asarray(points, dtype=np.float64)
        screen_axes = {}
        for name in ("X_SCREEN", "Y_SCREEN"):
            param = reader.get("POINT:" + name)
            screen_axes[name] = param.string_value.strip() if param is not None else None
        source = {"format": "c3d", "source_id": Path(path).stem,
                  "file": file_record(path), "reader": "c3d " + c3d.__version__,
                  "units_parameter": units_raw, "screen_axes": screen_axes,
                  "point_count": int(reader.point_used), "analog_channels": int(reader.analog_used),
                  "ignored_empty_analog_subframes": reader.ignored_empty_analog_subframes,
                  "zero_residual_samples": int(np.sum(data[:, :, 3] == 0)),
                  "max_residual_in_source_units": float(np.max(data[:, :, 3])),
                  "processing_caution": "Labeled C3D is not proof of unprocessed/raw observations. "
                  "Zero residual can indicate modeled, filtered or interpolated data. "
                  "No force-platform or analog data is used by this importer."}
        native_timing = {"native_rate_hz": rate, "first_frame": first, "last_frame": last,
                         "frame_count": len(frames), "basis": "(C3D frame_number - 1) / POINT:RATE; "
                         "frame 1 has t=0; not acquisition wall-clock time", "resampled": False}
    source["reader_warnings"] = [str(item.message) for item in caught]
    for warning in source["reader_warnings"]:
        _require(not any(text in warning.lower() for text in ("inconsistent", "incomplete reading", "reached end of file")),
                 f"C3D reader could not verify all data: {warning}")
    if cmu:
        match = re.fullmatch(r"(\d{2,3})_(\d{2,3})", Path(path).stem)
        _require(match is not None, "CMU adapter requires a subject_trial.c3d filename, e.g. 07_01.c3d")
        subject, trial = match.groups()
        source.update({"dataset": "CMU Graphics Lab Motion Capture Database",
                       "subject_id": subject, "trial_id": trial,
                       "source_id": f"CMU:{subject}_{trial}",
                       "url": f"{CMU_HOME}subjects/{subject}/{subject}_{trial}.c3d",
                       "catalog_url": f"{CMU_HOME}search.php?subjectnumber={int(subject)}",
                       "marker_evidence": [CMU_HOME + "faqs.php", CMU_HOME + "markerPlacementGuide.pdf"],
                       "terms_url": CMU_HOME,
                       "terms_summary": "CMU permits use/copy/modification/redistribution; "
                       "do not directly resell data or converted data. Acknowledge CMU when publishing. "
                       "See the source site for authoritative terms; quality is not guaranteed.",
                       "processing_caution": source["processing_caution"] + " CMU warns BodyBuilder "
                       "can add synthetic points; mapping must select documented capture markers explicitly."})
    valid = (data[:, :, 3] >= 0) & np.isfinite(data[:, :, 3])
    return LabeledPoints(labels, (np.asarray(frames, dtype=float) - 1.0) / rate,
                         data[:, :, :3], units, valid,
                         {"source": source, "timing": native_timing})


def read_input(path, format):
    _require(format in ("c3d", "cmu-c3d"), f"Unsupported adapter {format}")
    return read_c3d(path, cmu=format == "cmu-c3d")


def validate_mapping(mapping):
    _keys(mapping, ("schema_version", "description", "label_prefix", "expected_units",
                    "source_axes", "world_from_source_m", "targets", "limitations"), context="mapping")
    _require(type(mapping["schema_version"]) is int and mapping["schema_version"] == 1,
             "Unsupported mapping schema_version")
    _text(mapping["description"], "mapping.description")
    _require(isinstance(mapping["label_prefix"], str), "label_prefix must be a string (possibly empty)")
    _require(isinstance(mapping["expected_units"], str) and mapping["expected_units"] in UNIT_TO_M,
             "mapping.expected_units is unsupported")
    axes = mapping["source_axes"]
    _keys(axes, ("description", "evidence"), context="source_axes")
    _text(axes["description"], "source_axes.description")
    _strings(axes["evidence"], "source_axes.evidence", nonempty=True)
    transform = mapping["world_from_source_m"]
    _keys(transform, ("rotation", "translation_m", "ground_assumption", "evidence"), context="transform")
    rotation = _numeric(transform["rotation"], "rotation")
    translation = _numeric(transform["translation_m"], "translation_m")
    _require(rotation.shape == (3, 3) and np.isfinite(rotation).all(), "rotation must be a finite 3x3 matrix")
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
             and abs(np.linalg.det(rotation) - 1) < 1e-9,
             "rotation must be orthonormal with det +1; scaling/reflection is forbidden")
    _require(translation.shape == (3,) and np.isfinite(translation).all(), "translation_m must be finite [3]")
    _text(transform["ground_assumption"], "ground_assumption")
    _strings(transform["evidence"], "ground evidence", nonempty=True)
    _strings(mapping["limitations"], "mapping.limitations", nonempty=True)
    _keys(mapping["targets"], TARGET_NAMES, context="mapping.targets")
    for name, spec in mapping["targets"].items():
        _require(isinstance(spec, dict), f"target {name} must be an object")
        kind = spec.get("kind")
        _require(kind in ("marker", "weighted_mean"), f"Unsupported derivation for {name}: {kind}")
        fields = ("label",) if kind == "marker" else ("labels", "weights")
        _keys(spec, ("kind", "anatomical_meaning", "evidence", "limitations", *fields), context=name)
        _text(spec["anatomical_meaning"], f"{name}.anatomical_meaning")
        _strings(spec["evidence"], f"{name}.evidence", nonempty=True)
        _strings(spec["limitations"], f"{name}.limitations", nonempty=True)
        if kind == "marker":
            _text(spec["label"], f"{name}.label")
        else:
            _strings(spec["labels"], f"{name}.labels", nonempty=True)
            _require(len(spec["labels"]) >= 2 and len(spec["labels"]) == len(set(spec["labels"])),
                     f"{name}: weighted_mean needs at least two distinct source labels")
            weights = _numeric(spec["weights"], f"{name}.weights")
            _require(weights.shape == (len(spec["labels"]),) and np.isfinite(weights).all()
                     and np.all(weights >= 0) and abs(weights.sum() - 1) < 1e-12,
                     f"{name}: weights must be finite, nonnegative and sum to one")
    return rotation, translation


def inspect_points(points):
    """JSON-safe label inventory; unused invalid observations are visible here."""
    labels = []
    for index, label in enumerate(points.labels):
        valid = points.valid[:, index]
        observed = points.positions[valid, index]
        labels.append({"label": label, "invalid_samples": int((~valid).sum()),
                       "min": observed.min(axis=0).tolist() if len(observed) else None,
                       "max": observed.max(axis=0).tolist() if len(observed) else None})
    return {"frames": len(points.time_s), "duration_s": float(points.time_s[-1] - points.time_s[0]),
            "time_offset_s": float(points.time_s[0]), "units": points.units,
            "interval_s": {"min": float(np.diff(points.time_s).min()),
                           "max": float(np.diff(points.time_s).max())},
            "labels": labels, "metadata": points.metadata}


def reference_diagnostics(reference):
    positions = np.asarray(reference["positions_m"])
    return {"frames": len(positions), "duration_s": reference["time_s"][-1],
            "bounds_m": {"min": positions.min(axis=(0, 1)).tolist(),
                         "max": positions.max(axis=(0, 1)).tolist()},
            "targets": {name: {"min_z_m": float(positions[:, i, 2].min()),
                               "max_z_m": float(positions[:, i, 2].max()),
                               "samples_below_z0": int(np.sum(positions[:, i, 2] < 0))}
                        for i, name in enumerate(TARGET_NAMES)}}


def convert_points(points, mapping, *, mapping_record=None, label_prefix=None):
    effective = deepcopy(mapping)
    overrides = {}
    if label_prefix is not None:
        effective["label_prefix"] = label_prefix
        overrides["label_prefix"] = label_prefix
    rotation, translation = validate_mapping(effective)
    _require(effective["expected_units"] == points.units,
             f"Mapping expects {effective['expected_units']}; input declares {points.units}")
    lookup = {name: i for i, name in enumerate(points.labels)}
    targets, resolved_labels = [], {}
    for name in TARGET_NAMES:
        spec = effective["targets"][name]
        labels = [spec["label"]] if spec["kind"] == "marker" else spec["labels"]
        full_labels = [effective["label_prefix"] + label for label in labels]
        missing = [label for label in full_labels if label not in lookup]
        _require(not missing, f"{name}: missing exact source labels {missing}; inspect input, "
                 "then set label_prefix/mapping explicitly (no automatic aliases)")
        indices = [lookup[label] for label in full_labels]
        bad = np.argwhere(~points.valid[:, indices])
        if len(bad):
            frame, label_index = bad[0]
            raise ImportValidationError(f"{name}: invalid selected point {full_labels[label_index]!r} "
                                        f"at sample {frame} (source t={points.time_s[frame]:.9g}s); "
                                        f"{len(bad)} invalid selected observations; no gap filling")
        weights = [1.0] if spec["kind"] == "marker" else spec["weights"]
        targets.append(np.einsum("tnc,n->tc", points.positions[:, indices, :], weights))
        resolved_labels[name] = full_labels
    scale = UNIT_TO_M[points.units]
    positions_m = (np.stack(targets, axis=1) * scale) @ rotation.T + translation
    _require(np.isfinite(positions_m).all(), "Conversion produced nonfinite positions")
    reference = {
        "schema_version": 1,
        "time_s": (points.time_s - points.time_s[0]).tolist(),
        "positions_m": positions_m.tolist(),
        "target_names": list(TARGET_NAMES),
        "metadata": {
            **deepcopy(points.metadata),
            "time_offset_s": float(points.time_s[0]),
            "units": {"source": points.units, "output": "m", "scale_to_m": scale},
            "axes": {"source": effective["source_axes"],
                     "output": "Right-handed XYZ; +Z up; proposed ground plane Z=0. "
                     "Horizontal axes follow the explicit rotation, not inferred walking direction."},
            "ground_transform": deepcopy(effective["world_from_source_m"]),
            "mapping": {"file": mapping_record, "configuration": effective,
                        "overrides": overrides, "resolved_labels": resolved_labels},
            "processing": {"resampling": "none", "gap_filling": "none", "smoothing": "none",
                           "body_scaling": "none", "automatic_ground_fit": "none",
                           "anatomical_joint_center_inference": "none"},
            "invalid_source_observations": int((~points.valid).sum()),
        },
    }
    validate_reference(reference)
    reference["metadata"]["diagnostics"] = reference_diagnostics(reference)
    return reference


def validate_reference(reference):
    _keys(reference, ("schema_version", "time_s", "positions_m", "target_names", "metadata"),
          context="canonical reference")
    _require(type(reference["schema_version"]) is int and reference["schema_version"] == 1,
             "Unsupported reference schema_version")
    _require(reference["target_names"] == list(TARGET_NAMES), "Incorrect 16-target order (rhip is last)")
    time = _times(reference["time_s"])
    _require(time[0] == 0, "Canonical time_s must start at zero")
    positions = _numeric(reference["positions_m"], "positions_m")
    _require(positions.shape == (len(time), 16, 3) and np.isfinite(positions).all(),
             "positions_m must be finite [T][16][3]")
    _require(isinstance(reference["metadata"], dict), "reference metadata must be an object")


def write_json(path, obj, *, force=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    with path.open("w" if force else "x", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "convert"):
        part = sub.add_parser(command)
        part.add_argument("input", type=Path)
        part.add_argument("--format", required=True, choices=("c3d", "cmu-c3d"))
        part.add_argument("--output", type=Path, required=command == "convert")
        part.add_argument("--force", action="store_true", help="Explicitly replace existing outputs")
        if command == "convert":
            part.add_argument("--mapping", required=True, type=Path)
            part.add_argument("--label-prefix", help="Explicit namespace override, recorded in provenance")
            part.add_argument("--source-id", help="Explicit source identifier (CMU auto-records subject/trial)")
            part.add_argument("--source-url", help="Provenance URL for a general source")
            part.add_argument("--preview", type=Path, help="Also generate standalone offline HTML; no GUI opens")
    args = parser.parse_args(argv)
    try:
        protected = {args.input.resolve()}
        if args.command == "convert":
            protected.add(args.mapping.resolve())
        outputs = [path for path in (args.output, getattr(args, "preview", None)) if path is not None]
        _require(len({path.resolve() for path in outputs}) == len(outputs), "Outputs must be distinct")
        for path in outputs:
            _require(path.resolve() not in protected, "Output cannot overwrite input or mapping")
            _require(args.force or not path.exists(), f"Output exists: {path}; use --force to replace it")
        points = read_input(args.input, args.format)
        if args.command == "inspect":
            result = inspect_points(points)
        else:
            if args.source_id:
                points.metadata["source"]["source_id"] = args.source_id
            if args.source_url:
                points.metadata["source"]["url"] = args.source_url
            result = convert_points(points, load_json(args.mapping), mapping_record=file_record(args.mapping),
                                    label_prefix=args.label_prefix)
        if args.output:
            write_json(args.output, result, force=args.force)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "convert":
            if args.preview:
                from mocap_preview import write_preview
                write_preview(args.preview, result, force=args.force, points=points)
            print(json.dumps({"reference": str(args.output), "preview": str(args.preview) if args.preview else None,
                              **result["metadata"]["diagnostics"]}, indent=2))
    except (ImportValidationError, OSError, KeyError, ValueError) as error:
        parser.exit(2, f"mocap import: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
