"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v

No downloads or GUI needed; C3D fixtures are generated locally.
"""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

import c3d
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import mocap_import as mi
import mocap_preview as preview


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.mapping = mi.load_json(ROOT / "data/mocap/cmu/mapping.json")
        self.labels = []
        for spec in self.mapping["targets"].values():
            for label in spec.get("labels", [spec.get("label")]):
                full = "liu:" + label
                if full not in self.labels:
                    self.labels.append(full)
        self.positions = np.arange(3 * len(self.labels) * 3, dtype=float).reshape(3, len(self.labels), 3)
        self.points = mi.LabeledPoints(self.labels, [5.0, 5.01, 5.037], self.positions, "mm")

    def convert(self, **kwargs):
        return mi.convert_points(self.points, self.mapping, **kwargs)

    def assert_invalid(self, call, match=None):
        with self.assertRaisesRegex(mi.ImportValidationError, match or ".*"):
            call()

    def test_shape_order_timing_no_resampling(self):
        result = self.convert()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["target_names"], list(mi.TARGET_NAMES))
        self.assertEqual(result["target_names"][-1], "rhip")
        self.assertEqual(np.asarray(result["positions_m"]).shape, (3, 16, 3))
        np.testing.assert_allclose(result["time_s"], [0, .01, .037], atol=1e-15)
        self.assertEqual(result["metadata"]["time_offset_s"], 5)
        self.assertNotIn("review", result["metadata"])
        self.assertEqual(result["metadata"]["processing"]["body_scaling"], "none")

    def test_exact_marker_and_explicit_derived_points(self):
        result = np.asarray(self.convert()["positions_m"])
        i = self.labels.index("liu:LFWT")
        np.testing.assert_allclose(result[:, 14], self.positions[:, i] / 1000)
        pelvis = [self.labels.index("liu:" + n) for n in ("LFWT", "RFWT", "LBWT", "RBWT")]
        head = [self.labels.index("liu:" + n) for n in ("LFHD", "RFHD", "LBHD", "RBHD")]
        np.testing.assert_allclose(result[:, 0], self.positions[:, pelvis].mean(axis=1) / 1000)
        np.testing.assert_allclose(result[:, 1], self.positions[:, head].mean(axis=1) / 1000)
        for target, labels in ((8, ('LWRA', 'LWRB')), (9, ('RWRA', 'RWRB'))):
            indices = [self.labels.index('liu:' + label) for label in labels]
            np.testing.assert_allclose(result[:, target], self.positions[:, indices].mean(axis=1) / 1000)

    def test_unit_conversions(self):
        for units, factor in mi.UNIT_TO_M.items():
            with self.subTest(units=units):
                self.points.units = units
                self.mapping["expected_units"] = units
                result = np.asarray(self.convert()["positions_m"])
                np.testing.assert_allclose(result[:, 2], self.positions[:, self.labels.index("liu:LTOE")] * factor)
        self.mapping["expected_units"] = "mm"
        self.assert_invalid(self.convert, "input declares")

    def test_rigid_transform_order_and_no_scaling(self):
        original = np.asarray(self.convert()["positions_m"])
        transform = self.mapping["world_from_source_m"]
        transform["rotation"] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # Y-up -> Z-up
        transform["translation_m"] = [.1, -.2, 1.5]
        result = np.asarray(self.convert()["positions_m"])
        np.testing.assert_allclose(result, original @ np.asarray(transform["rotation"]).T + [.1, -.2, 1.5])
        np.testing.assert_allclose(np.linalg.norm(result[:, 0] - result[:, 1], axis=1),
                                   np.linalg.norm(original[:, 0] - original[:, 1], axis=1))
        np.testing.assert_array_equal(self.points.positions, self.positions)

    def test_forbid_scale_reflection_and_nonfinite_transform(self):
        for rotation in ([[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                         [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         [[1, 0, 0], [0, 1, .1], [0, 0, 1]]):
            self.mapping["world_from_source_m"]["rotation"] = rotation
            self.assert_invalid(self.convert, "orthonormal")
        self.mapping["world_from_source_m"]["rotation"] = np.eye(3).tolist()
        self.mapping["world_from_source_m"]["translation_m"] = [0, float("nan"), 0]
        self.assert_invalid(self.convert, "translation_m")

    def test_prefix_override_is_explicit_and_recorded(self):
        self.mapping["label_prefix"] = "other:"
        self.assert_invalid(self.convert, "missing exact source labels")
        result = self.convert(label_prefix="liu:", mapping_record={"sha256": "test"})
        self.assertEqual(result["metadata"]["mapping"]["overrides"], {"label_prefix": "liu:"})
        self.assertEqual(result["metadata"]["mapping"]["file"]["sha256"], "test")
        self.assertEqual(self.mapping["label_prefix"], "other:")

    def test_selected_gap_is_rejected_not_filled_or_trimmed(self):
        index = self.labels.index("liu:LTOE")
        self.points.valid[1, index] = False
        self.assert_invalid(self.convert, "invalid selected point 'liu:LTOE'.*sample 1")

    def test_nonfinite_selected_point_is_rejected(self):
        self.positions[0, 0, 2] = np.nan
        self.points = mi.LabeledPoints(self.labels, [0, 1, 2], self.positions, "mm")
        self.assert_invalid(self.convert, "invalid selected point")

    def test_unused_gap_does_not_break_mapping(self):
        points = np.concatenate((self.positions, np.full((3, 1, 3), np.nan)), axis=1)
        self.points = mi.LabeledPoints(self.labels + ["unused"], [0, 1, 2], points, "mm")
        result = self.convert()
        self.assertEqual(result["metadata"]["invalid_source_observations"], 3)
        self.assertEqual(mi.inspect_points(self.points)["labels"][-1]["min"], None)

    def test_timestamps_must_be_finite_increasing(self):
        for times in ([0, 0, 1], [0, 2, 1], [0, np.nan, 2], [np.inf, 1, 2], [0]):
            self.assert_invalid(lambda: mi.LabeledPoints(self.labels, times, self.positions, "mm"))

    def test_labels_and_dimensions_are_strict(self):
        labels = self.labels.copy()
        labels[0] = labels[1]
        self.assert_invalid(lambda: mi.LabeledPoints(labels, [0, 1, 2], self.positions, "mm"), "Duplicate")
        self.assert_invalid(lambda: mi.LabeledPoints(self.labels, [0, 1], self.positions, "mm"), "shape")
        self.assert_invalid(lambda: mi.LabeledPoints(self.labels, [0, 1, 2], self.positions, "furlong"), "units")
        self.assert_invalid(lambda: mi.LabeledPoints(self.labels, [0, 1, 2], self.positions, "mm",
                                                    np.ones(self.positions.shape[:2], dtype=int)), "boolean")
        self.assert_invalid(lambda: mi.LabeledPoints(self.labels, ["0", "1", "2"], self.positions, "mm"), "strings")

    def test_mapping_rejects_unknown_targets_fields_and_implicit_operations(self):
        for key, value in (("body_scale", 2), ("unused", None)):
            self.mapping[key] = value
            self.assert_invalid(self.convert, "unknown keys")
            del self.mapping[key]
        self.mapping["targets"]["lhip"]["offset_m"] = [0, 0, .1]
        self.assert_invalid(self.convert, "unknown keys")
        del self.mapping["targets"]["lhip"]["offset_m"]
        del self.mapping["targets"]["rhip"]
        self.assert_invalid(self.convert, "rhip")

    def test_weights_fail_closed(self):
        for weights in ([1, 1], [-.5, 1.5], [1], [np.nan, .5]):
            self.mapping["targets"]["head"]["weights"] = weights
            self.assert_invalid(self.convert, "weights")

    def test_negative_ground_coordinates_are_preserved(self):
        self.points.positions[:, self.labels.index("liu:LTOE"), 2] = -8.5
        result = self.convert()
        self.assertEqual(result["metadata"]["diagnostics"]["targets"]["ltoe"]["samples_below_z0"], 3)
        np.testing.assert_allclose(np.asarray(result["positions_m"])[:, 2, 2], -.0085)

    def test_general_c3d_adapter_and_file_checksum(self):
        path = self.write_c3d().rename(self.directory / "general_recording.c3d")
        points = mi.read_input(path, "c3d")
        self.assertEqual(points.metadata["source"]["file"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(points.metadata["source"]["source_id"], "general_recording")
        np.testing.assert_array_equal(points.positions, self.positions)
        result = mi.convert_points(points, self.mapping)
        out = self.directory / "result.json"
        mi.write_json(out, result)
        mi.validate_reference(mi.load_json(out))
        with self.assertRaises(FileExistsError):
            mi.write_json(out, result)

    def test_json_configuration_rejects_duplicate_and_nonstandard_values(self):
        path = self.directory / "bad.json"
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            path.write_text(text)
            self.assert_invalid(lambda: mi.load_json(path))

    def test_alternative_json_source_format_is_rejected(self):
        path = self.directory / "points.json"
        self.assert_invalid(lambda: mi.read_input(path, "points-json"), "Unsupported adapter")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
            mi.main(["inspect", str(path), "--format", "points-json"])
        self.assertEqual(exit.exception.code, 2)

    def test_reference_validation(self):
        result = self.convert()
        for field, value in (("target_names", list(reversed(mi.TARGET_NAMES))),
                             ("time_s", [1, 2, 3]), ("positions_m", [[[0, 0, 0]]]),
                             ("schema_version", True)):
            obj = deepcopy(result)
            obj[field] = value
            self.assert_invalid(lambda: mi.validate_reference(obj))

    def test_preview_offline_all_frames_and_script_injection_safety(self):
        result = self.convert()
        result["metadata"]["source"] = {"source_id": '</script><script>alert("bad")</script>'}
        text = preview.render_preview(result)
        self.assertNotIn('<script>alert("bad")</script>', text)
        self.assertNotIn('src="http', text)
        self.assertNotIn('fetch(', text)
        self.assertNotIn('UNAPPROVED', text)
        self.assertIn('Right-handed XYZ', text)
        self.assertIn('Feet close-up', text)
        self.assertNotIn('data-view', text)
        self.assertNotIn('◀ Sample', text)
        self.assertIn('aria-label="Previous frame">◀</button>', text)
        self.assertIn('aria-label="Next frame">▶</button>', text)
        self.assertIn('<summary>Marker mapping</summary>', text)
        self.assertIn('<summary>Import metadata (JSON)</summary>', text)
        self.assertIn('<th>Evidence</th>', text)
        embedded = re.search(r'<script id="reference-data" type="application/json">(.*?)</script>', text, re.S)
        self.assertEqual(json.loads(embedded[1]), result)

    def test_preview_raw_markers_transform_gaps_and_reference_unchanged(self):
        extra = np.array([[[100., 200., 300.]], [[np.nan, 0., 0.]], [[400., 500., 600.]]])
        points = mi.LabeledPoints(self.labels + ['unused</script>'], self.points.time_s,
                                  np.concatenate([self.positions, extra], axis=1), 'mm')
        mapping = deepcopy(self.mapping)
        mapping['world_from_source_m']['rotation'] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        mapping['world_from_source_m']['translation_m'] = [1, 2, 3]
        reference = mi.convert_points(points, mapping)
        original = deepcopy(reference)
        text = preview.render_preview(reference, points=points)
        embedded = json.loads(re.search(r'<script id="reference-data" type="application/json">(.*?)</script>', text, re.S)[1])
        raw = embedded.pop('preview_markers')
        self.assertEqual(embedded, original)
        self.assertEqual(reference, original)
        self.assertEqual(raw['labels'], points.labels)
        self.assertIsNone(raw['positions_m'][1][-1])
        np.testing.assert_allclose(raw['positions_m'][0][-1], [.8, 2.1, 3.3])
        self.assertNotIn('unused</script>', text)
        self.assertIn('<input id="raw" type="checkbox">', text)
        self.assertNotIn('<input id="raw" type="checkbox" checked>', text)
        self.assertIn('id="tracking"', text)
        points.positions[0, 0, 0] += 1
        self.assert_invalid(lambda: preview.render_preview(reference, points=points), 'does not match')

    @unittest.skipUnless(shutil.which("node"), "Node needed for JS projection regression")
    def test_preview_camera_orientation(self):
        # Execute the renderer's actual projection, not a Python reimplementation.
        function = re.search(r"function projection\(w,h\)\{.*?\n\}", preview.HTML, re.S)[0]
        script = r'''
const samples=[[[0,0,0]]], frame=0, lo=[-1,-1,-1], hi=[1,1,1], zoom=1;
const $=()=>({checked:false});
let yaw=0,elevation=0;
''' + function + r'''
function check(ok,message){if(!ok)throw Error(message);}
function basis(){const p=projection(800,600),o=p([0,0,0]);
 return [[1,0,0],[0,1,0],[0,0,1]].map(v=>{const q=p(v);return [q[0]-o[0],o[1]-q[1]];});}
elevation=Math.PI/2;
let b=basis();check(b[0][0]>0 && b[1][1]>0,'Top view must show +X right, +Y up');
yaw=-Math.PI/2;elevation=0;
b=basis();check(b[1][0]>0 && b[2][1]>0,'View from +X must show +Y right, +Z up');
yaw=-.6;elevation=.35;
b=basis();check(b[0][0]*b[1][1]-b[1][0]*b[0][1]>0,'Positive elevation must view ground from above');
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node needed for raw-marker drawing regression')
    def test_preview_raw_marker_visibility_and_selection(self):
        function = re.search(r'function drawView\(\)\{.*?\n\}', preview.HTML, re.S)[0]
        script = r'''
const arcs=[],texts=[],ctx=new Proxy({}, {get:(o,k)=>o[k]||((...a)=>{if(k==='arc')arcs.push(a);if(k==='fillText')texts.push(a[0]);})});
const canvas={},context=()=>[ctx,800,600],projection=()=>p=>p.slice(0,2),line=()=>{};
const controls={raw:{checked:true},tracking:{checked:false},labels:{checked:false},feet:{checked:false}};
const $=id=>controls[id],frame=0,samples=[[[0,0,1]]],lo=[0,0,0],hi=[1,1,2];
const names=['pelvis'],selected=0,sourceLabels=()=>['A'],specs={pelvis:{kind:'weighted_mean'}},links=[],sides=['purple'];
const raw={labels:['A','B','missing'],positions_m:[[[0,0,1],[1,0,1],null]]};let projected=[];
''' + function + r'''
function check(ok){if(!ok)throw Error('raw marker visibility/selection regression');}
drawView();check(arcs.length===2 && arcs[0][2]===9 && arcs[1][2]===3);
check(texts.includes('A') && !texts.includes('B') && projected.length===0);
arcs.length=0;controls.raw.checked=false;controls.tracking.checked=true;
drawView();check(arcs.length===0 && projected.length===1);
'''
        result = subprocess.run([shutil.which('node'), '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_convert_preview_and_overwrite_guards(self):
        path = self.write_c3d()
        output, html_path = self.directory / "ref.json", self.directory / "ref.html"
        args = ["convert", str(path), "--format", "c3d", "--mapping",
                str(ROOT / "data/mocap/cmu/mapping.json"), "--output", str(output),
                "--preview", str(html_path)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mi.main(args), 0)
        self.assertTrue(html_path.exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
            mi.main(args)
        self.assertEqual(exit.exception.code, 2)
        args[args.index("--output") + 1] = str(path)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mi.main(args + ["--force"])

    def write_c3d(self, *, point_scale=-1, first_frame=235, invalid=False):
        path = self.directory / "07_01.c3d"
        writer = c3d.Writer(point_rate=120, point_scale=point_scale, point_units="mm")
        writer.set_point_labels(self.labels)
        writer.set_start_frame(first_frame)
        writer.set_screen_axis("+X", "+Z")
        data = np.zeros((3, len(self.labels), 5))
        data[:, :, :3] = self.positions
        data[:, :, 3] = 1.0
        data[:, :, 4] = 3
        if invalid:
            data[1, self.labels.index("liu:LTOE"), 3] = -1
        writer.add_frames([(p, np.empty((0, 0))) for p in data])
        with path.open("wb") as stream, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            writer.write(stream)
        return path

    def test_c3d_native_frames_units_and_no_double_scale(self):
        # Binary-exact integer quantization avoids c3d.Writer's truncation at .1.
        for scale in (-1, .25):
            with self.subTest(point_scale=scale):
                path = self.write_c3d(point_scale=scale)
                points = mi.read_c3d(path, cmu=True)
                np.testing.assert_allclose(points.positions, self.positions)
                self.assertEqual(points.units, "mm")
                self.assertEqual(points.labels, self.labels)
                self.assertEqual(points.metadata["timing"]["first_frame"], 235)
                self.assertEqual(points.metadata["timing"]["last_frame"], 237)
                np.testing.assert_allclose(points.time_s, np.arange(234, 237) / 120)
                result = mi.convert_points(points, self.mapping)
                self.assertAlmostEqual(result["metadata"]["time_offset_s"], 1.95)
                np.testing.assert_allclose(result["time_s"], np.arange(3) / 120, atol=1e-15)
                self.assertEqual(result["metadata"]["source"]["source_id"], "CMU:07_01")
                json.dumps(result, allow_nan=False)  # no numpy scalar leakage

    def test_empty_analog_stale_subframes_preserve_points_and_file(self):
        path = self.write_c3d()
        raw = bytearray(path.read_bytes())
        struct.pack_into('<H', raw, 18, 1)  # Header analog subframes only.
        path.write_bytes(raw)
        points = mi.read_c3d(path)
        np.testing.assert_array_equal(points.positions, self.positions)
        np.testing.assert_allclose(points.time_s, np.arange(234, 237) / 120)
        self.assertEqual(points.metadata['source']['analog_channels'], 0)
        self.assertEqual(points.metadata['source']['ignored_empty_analog_subframes'], 1)
        self.assertEqual(path.read_bytes(), bytes(raw))

    def test_empty_analog_fix_does_not_bypass_other_checks(self):
        for offset, value in ((4, 1), (2, len(self.labels) + 1)):
            path = self.write_c3d()
            raw = bytearray(path.read_bytes())
            struct.pack_into('<H', raw, 18, 1)
            struct.pack_into('<H', raw, offset, value)
            path.write_bytes(raw)
            self.assert_invalid(lambda: mi.read_c3d(path), 'Inconsistent C3D metadata')

    def test_c3d_validity_flags_are_honored(self):
        points = mi.read_c3d(self.write_c3d(invalid=True))
        self.assertFalse(points.valid[1, self.labels.index("liu:LTOE")])
        self.assert_invalid(lambda: mi.convert_points(points, self.mapping), "invalid selected")

    def test_c3d_requires_explicit_unit_and_rate_parameters(self):
        for parameter in (b"UNITS", b"RATE"):
            path = self.write_c3d()
            raw = path.read_bytes()
            self.assertIn(parameter, raw)
            # Rename every occurrence: ANALOG:RATE can precede POINT:RATE.
            path.write_bytes(raw.replace(parameter, b"X" * len(parameter)))
            self.assert_invalid(lambda: mi.read_c3d(path), parameter.decode())

    def test_c3d_extended_start_cannot_silently_lose_a_frame(self):
        path = self.write_c3d(first_frame=70000)
        self.assert_invalid(lambda: mi.read_c3d(path), "extended starting frame")

    def test_truncated_c3d_rejected(self):
        path = self.write_c3d()
        path.write_bytes(path.read_bytes()[:-1000])
        self.assert_invalid(lambda: mi.read_c3d(path), "truncated")


if __name__ == "__main__":
    unittest.main()
