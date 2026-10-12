"""Load actual direct C++ NPZ exports with NumPy (never allow pickles).

No simulator, planner, GUI, or source mocap is used. C++ fixture generators
compile embedded models and save synthetic poses/labels, then the real export
API/CLI writes archives. Build test_export, test_export_offline, and data_factory
first; GRF_EXPORT_TEST_BUILD_DIR optionally selects a nondefault build directory.
Run with the NumPy-enabled mocap venv and unittest, or CTest's export_dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("GRF_EXPORT_TEST_BUILD_DIR", ROOT / "build"))
ARRAYS = {"markers_m", "mass_kg", "time_s", "contact_wrench",
          "episode_offsets", "source_id"}
BINARIES = [BUILD / name for name in ("test_export", "test_export_offline", "data_factory")]


@unittest.skipIf(np is None, "NumPy required to load direct NPZ exports")
@unittest.skipUnless(all(path.is_file() for path in BINARIES), "Build export test targets first")
class DirectNpzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="grf-npz-numpy-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.offline = cls.directory / "offline"
        cls.physical = cls.directory / "physical"
        for command in (
            [str(BUILD / "test_export_offline"), "--fixtures", str(cls.offline)],
            [str(BUILD / "test_export"), str(ROOT / "assets/winter_baseline_male.xml"),
             str(cls.physical)],
        ):
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{command[0]} failed:\n{result.stdout}\n{result.stderr}")
        cls.expected_markers = np.loadtxt(cls.offline / "recorded_markers.txt").reshape(5, 6, 3)
        cls.expected_wrench = np.loadtxt(cls.offline / "recorded_wrench.txt").reshape(5, 2, 6)

    def load(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), ARRAYS)
            arrays = {key: archive[key] for key in archive.files}
        for key, array in arrays.items():
            self.assertFalse(array.dtype.hasobject, key)
            if key not in ("source_id", "episode_offsets"):
                self.assertEqual(array.dtype, np.dtype("float64"), key)
        self.assertEqual(arrays["episode_offsets"].dtype, np.dtype("int64"))
        self.assertEqual(arrays["source_id"].dtype.kind, "U")
        return arrays

    def fixed_permutation(self, markers, original):
        order = []
        for point in markers[0]:
            matches = np.flatnonzero((original[0] == point).all(axis=1))
            self.assertEqual(len(matches), 1, "Each anonymous slot is one recorded marker")
            order.append(int(matches[0]))
        self.assertEqual(sorted(order), list(range(original.shape[1])))
        np.testing.assert_array_equal(markers, original[:, order, :])
        return order

    def assert_same_labels(self, first, second):
        for key in ARRAYS - {"markers_m"}:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def cli(self, *arguments):
        return subprocess.run([str(BUILD / "data_factory"), "export", *map(str, arguments)],
                              capture_output=True, text=True, timeout=30)

    def test_compressed_six_array_archive_shapes_and_exact_precision(self):
        path = self.offline / "recorded.npz"
        arrays = self.load(path)
        self.assertEqual(arrays["markers_m"].shape, (5, 6, 3))
        self.assertEqual(arrays["contact_wrench"].shape, (5, 2, 6))
        self.assertEqual(arrays["mass_kg"].shape, (1,))
        self.assertEqual(arrays["time_s"].shape, (5,))
        self.assertEqual(arrays["source_id"].shape, (1,))
        np.testing.assert_array_equal(arrays["mass_kg"], [5.0])  # Not summary's bogus 999 kg.
        np.testing.assert_array_equal(arrays["episode_offsets"], [0, 5])
        np.testing.assert_array_equal(arrays["time_s"], [2, 2.25, 2.5, 2.75, 3])
        np.testing.assert_array_equal(arrays["source_id"], ["synthetic-recording"])
        np.testing.assert_array_equal(arrays["contact_wrench"], self.expected_wrench)
        self.fixed_permutation(arrays["markers_m"], self.expected_markers)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), {name + ".npy" for name in ARRAYS})
            self.assertIsNone(archive.testzip(), "NPY entries have correct CRC32")
            for member in archive.infolist():
                self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)
                self.assertLess(member.compress_size, member.file_size)
        self.assertFalse(Path(str(path) + ".metadata.json").exists())

    def test_force_sign_cop_torsion_flight_and_undefined_targets(self):
        wrench = self.load(self.offline / "recorded.npz")["contact_wrench"]
        np.testing.assert_array_equal(wrench[0, 0, :3], [2, 3, 10])
        self.assertAlmostEqual(wrench[0, 0, 3], .2)  # -My/Fz.
        self.assertAlmostEqual(wrench[0, 0, 4], -.3)  # Mx/Fz.
        self.assertAlmostEqual(wrench[0, 0, 5], .5)  # Mz - (CoPx*Fy - CoPy*Fx).
        # The saved nextafter values differ from a fresh raw-moment derivation.
        self.assertEqual(wrench[0, 0, 3], np.nextafter(.2, 1.0))
        self.assertEqual(wrench[0, 0, 5], np.nextafter(1.7 - (.2 * 3 - (-.3 * 2)), 1.0))
        self.assertTrue(np.isfinite(wrench[..., :3]).all())
        np.testing.assert_array_equal(wrench[2, :, :3], np.zeros((2, 3)))  # Flight retained.
        self.assertTrue(np.isnan(wrench[2, :, 3:]).all())
        self.assertEqual(wrench[3, 0, 2], -4.5)  # Nonpositive Fz never deletes force.
        np.testing.assert_array_equal(wrench[3, 1, :3], [3, -4, 0])
        self.assertTrue(np.isnan(wrench[3, :, 3:]).all())
        self.assertTrue(np.isfinite(wrench[1, 1, 3:]).all())  # Tiny positive Fz is valid.
        self.assertEqual(wrench[4, 0, 2], np.nextafter(0., 1.))  # Preserve subnormal.
        self.assertTrue(np.isnan(wrench[4, 0, 3:]).all())  # Raw-moment arithmetic overflow.

    def test_strict_threshold_masks_only_cop_and_torsion(self):
        recorded = self.load(self.offline / "recorded.npz")
        for name, threshold in (("masked.npz", 10), ("partial_mask.npz", 2.5)):
            masked = self.load(self.offline / name)
            for key in ARRAYS - {"contact_wrench"}:
                np.testing.assert_array_equal(masked[key], recorded[key])
            np.testing.assert_array_equal(masked["contact_wrench"][..., :3],
                                          recorded["contact_wrench"][..., :3])
            expected = recorded["contact_wrench"].copy()
            expected[..., 3:][expected[..., 2] <= threshold] = np.nan
            np.testing.assert_array_equal(masked["contact_wrench"], expected)
        self.assertTrue(np.isnan(self.load(self.offline / "masked.npz")
                                 ["contact_wrench"][..., 3:]).all())

    def test_alternate_layout_saved_poses_no_label_changes(self):
        recorded = self.load(self.offline / "recorded.npz")
        same = self.load(self.offline / "same.npz")
        alternate = self.load(self.offline / "alternate.npz")
        order = self.fixed_permutation(recorded["markers_m"], self.expected_markers)
        np.testing.assert_array_equal(recorded["markers_m"], same["markers_m"])
        self.assertFalse(np.array_equal(recorded["markers_m"], alternate["markers_m"]))
        expected = np.loadtxt(self.offline / "alternate_markers.txt").reshape(5, 6, 3)
        np.testing.assert_allclose(alternate["markers_m"], expected[:, order, :],
                                   rtol=0, atol=1e-15)
        self.assert_same_labels(recorded, same)
        self.assert_same_labels(recorded, alternate)

    def test_fixed_anonymous_permutation_run_salted_and_path_independent(self):
        recorded = self.load(self.offline / "recorded.npz")
        order = self.fixed_permutation(recorded["markers_m"], self.expected_markers)
        for filename in ("shuffle.npz", "default.npz"):
            variant = self.load(self.offline / filename)
            other_order = self.fixed_permutation(variant["markers_m"], self.expected_markers)
            self.assertNotEqual(order, other_order)
            self.assert_same_labels(recorded, variant)
        modified = self.load(self.offline / "modified.npz")
        expected = np.loadtxt(self.offline / "modified_markers.txt").reshape(5, 6, 3)
        modified_order = self.fixed_permutation(modified["markers_m"], expected)
        self.assertNotEqual(order, modified_order, "Same CLI seed is salted by run contents")
        self.assert_same_labels(recorded, modified)
        for filename in ("repeat.npz", "relocated.npz"):
            self.assertEqual((self.offline / "recorded.npz").read_bytes(),
                             (self.offline / filename).read_bytes())

    def test_episode_offsets_masses_original_recording_groups_and_unicode(self):
        arrays = self.load(self.offline / "episodes.npz")
        np.testing.assert_array_equal(arrays["episode_offsets"], [0, 5, 7, 12, 17])
        np.testing.assert_array_equal(arrays["mass_kg"], [5, 7.5, 5, 5])
        np.testing.assert_array_equal(arrays["source_id"], ["synthetic-recording",
                                      "another-recording", "synthetic-recording",
                                      "synthetic-錄影-Δ-🦶"])
        self.assertEqual(arrays["markers_m"].shape, (17, 6, 3))
        self.assertEqual(arrays["contact_wrench"].shape, (17, 2, 6))
        recorded = self.load(self.offline / "recorded.npz")
        for start in (0, 7, 12):
            np.testing.assert_array_equal(arrays["time_s"][start:start + 5], recorded["time_s"])
            np.testing.assert_array_equal(arrays["contact_wrench"][start:start + 5],
                                          recorded["contact_wrench"])
        np.testing.assert_array_equal(arrays["time_s"][5:7], [0, .1])
        np.testing.assert_array_equal(arrays["markers_m"][:5], arrays["markers_m"][7:12])
        first_order = self.fixed_permutation(arrays["markers_m"][:5], self.expected_markers)
        second_order = self.fixed_permutation(arrays["markers_m"][5:7], self.expected_markers[:2])
        self.assertNotEqual(first_order, second_order)
        self.assertEqual(self.load(self.offline / "nested.npz")["source_id"][0],
                         "synthetic-錄影-Δ-🦶")
        for start, stop in zip(arrays["episode_offsets"][:-1], arrays["episode_offsets"][1:]):
            self.assertTrue((np.diff(arrays["time_s"][start:stop]) > 0).all())

    def test_archives_cross_compression_buffer_boundaries(self):
        arrays = self.load(self.offline / "long.npz")
        self.assertEqual(arrays["markers_m"].shape, (1500, 6, 3))
        np.testing.assert_array_equal(arrays["episode_offsets"], [0, 1500])
        np.testing.assert_array_equal(arrays["time_s"], np.arange(1500) / 600.)
        np.testing.assert_array_equal(arrays["contact_wrench"],
                                      np.tile(self.expected_wrench, (300, 1, 1)))
        self.fixed_permutation(arrays["markers_m"],
                               np.loadtxt(self.offline / "long_markers.txt").reshape(1500, 6, 3))
        with zipfile.ZipFile(self.offline / "long.npz") as archive:
            self.assertIsNone(archive.testzip())
            self.assertGreater(archive.getinfo("markers_m.npy").file_size, 65536)
            self.assertGreater(archive.getinfo("markers_m.npy").compress_size, 65536)
            self.assertGreater(archive.getinfo("contact_wrench.npy").file_size, 65536)

    def test_physical_model_140_markers_and_independent_layout(self):
        recorded = self.load(self.physical / "recorded.npz")
        same = self.load(self.physical / "same.npz")
        alternate = self.load(self.physical / "alternate.npz")
        self.assertEqual(recorded["markers_m"].shape, (2, 140, 3))
        self.assertGreater(recorded["mass_kg"][0], 0)
        expected = np.loadtxt(self.physical / "recorded_markers.txt").reshape(2, 140, 3)
        self.fixed_permutation(recorded["markers_m"], expected)
        np.testing.assert_array_equal(recorded["contact_wrench"],
                                      np.loadtxt(self.physical / "recorded_wrench.txt").reshape(2, 2, 6))
        np.testing.assert_array_equal(recorded["markers_m"], same["markers_m"])
        self.assertFalse(np.array_equal(recorded["markers_m"], alternate["markers_m"]))
        self.assert_same_labels(recorded, alternate)
        episodes = self.load(self.physical / "episodes.npz")
        np.testing.assert_array_equal(episodes["episode_offsets"], [0, 2, 4])
        np.testing.assert_array_equal(episodes["source_id"], ["synthetic-physical-model"] * 2)
        np.testing.assert_array_equal(episodes["markers_m"][:2], episodes["markers_m"][2:])

    def test_cli_direct_multi_input_npz_no_intermediates_or_sidecar(self):
        destination = self.directory / "direct" / "DATA.npz"
        result = self.cli(self.offline / "source.grf", "--output", destination,
                          self.offline / "second.grf", "--permutation-seed", 42,
                          self.offline / "nested.grf")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(destination.parent.iterdir()), [destination])
        arrays = self.load(destination)
        expected = self.load(self.offline / "episodes.npz")
        np.testing.assert_array_equal(arrays["episode_offsets"], [0, 5, 7, 12])
        np.testing.assert_array_equal(arrays["mass_kg"], [5, 7.5, 5])
        np.testing.assert_array_equal(arrays["source_id"], expected["source_id"][[0, 1, 3]])
        indices = np.r_[0:7, 12:17]
        for name in ("markers_m", "contact_wrench", "time_s"):
            np.testing.assert_array_equal(arrays[name], expected[name][indices])
        alternate_path = destination.parent / "alternative.npz"
        result = self.cli(self.offline / "source.grf", "--output", alternate_path,
                          "--layout-index", 11, "--permutation-seed", 42)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(alternate_path.read_bytes(), (self.offline / "alternate.npz").read_bytes())
        masked_path = destination.parent / "masked.npz"
        result = self.cli(self.offline / "source.grf", "--output", masked_path,
                          "--min-normal-force", 10, "--permutation-seed", 42)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(masked_path.read_bytes(), (self.offline / "masked.npz").read_bytes())
        default_path = destination.parent / "default.npz"
        result = self.cli(self.offline / "source.grf", "--output", default_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(default_path.read_bytes(), (self.offline / "default.npz").read_bytes())

    def test_cli_rejections_preserve_output_and_legacy_always_rejected(self):
        output = self.directory / "preserved.npz"
        original = (self.offline / "recorded.npz").read_bytes()
        output.write_bytes(original)
        for recording, options, fragment in (
            ("incomplete.grf", [], "Incomplete"),
            ("legacy.grf", [], "legacy"),
            ("legacy.grf", ["--allow-incomplete"], "legacy"),
            ("no_source.grf", [], "source_id"),
            ("bad_steps.grf", [], "physics steps"),
            ("source.grf", ["--min-normal-force", "-1"], "nonnegative"),
            ("source.grf", ["--min-normal-force", "nan"], "finite"),
            ("source.grf", ["--min-normal-force", "inf"], "finite"),
            ("source.grf", ["--layout-index", "-1"], "nonnegative"),
            ("source.grf", ["--permutation-seed", "-1"], "nonnegative"),
        ):
            with self.subTest(recording=recording, options=options):
                result = self.cli(self.offline / recording, "--output", output, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(fragment, result.stderr)
                self.assertEqual(output.read_bytes(), original)
        result = self.cli(self.offline / "source.grf", self.offline / "different_markers.grf",
                          "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Marker counts differ", result.stderr)
        self.assertEqual(output.read_bytes(), original)
        diagnostic = self.directory / "diagnostic.npz"
        result = self.cli(self.offline / "incomplete.grf", "--output", diagnostic,
                          "--allow-incomplete", "--permutation-seed", 42)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_same_labels(self.load(diagnostic), self.load(self.offline / "recorded.npz"))
        self.fixed_permutation(self.load(diagnostic)["markers_m"], self.expected_markers)
        obsolete = self.directory / "obsolete.json"
        result = self.cli(self.offline / "source.grf", "--output", obsolete)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".npz", result.stderr)
        self.assertFalse(obsolete.exists())
        for arguments in ([], ["--output", output], [self.offline / "source.grf"],
                          [self.offline / "source.grf", "--min-normal-force"],
                          [self.offline / "source.grf", "--unknown"]):
            self.assertNotEqual(self.cli(*arguments).returncode, 0)
        help_result = self.cli("--help")
        self.assertEqual(help_result.returncode, 0)
        for text in ("<dataset.npz>", "--min-normal-force", "--layout-index", "never masks GRF"):
            self.assertIn(text, help_result.stdout)


if __name__ == "__main__":
    unittest.main()
