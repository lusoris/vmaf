"""Fork-added snapshot gate for the SSIMULACRA 2 feature extractor (T3-3,
ADR-0164). Pins the CPU output of `vmaf --feature ssimulacra2` against
reference values generated on the current master.

Unlike the Netflix golden assertions (`python/test/quality_runner_test.py`
et al.), SSIMULACRA 2 is a fork-added metric — these values are NOT part
of the upstream Netflix bit-exactness contract. They pin the fork's
self-consistency; drift here means the SIMD ports (or scalar) changed
behaviour.

Tolerance is `places=4` (1e-4). Both the phase-1 `cbrtf` and the
phase-3 sRGB EOTF (`powf((x + 0.055)/1.055, 2.4)`) have been
replaced with deterministic host-independent implementations in
`libvmaf/src/feature/ssimulacra2_math.h` — a Newton–Raphson
cube root (accuracy ~7e-7) and a 1024-entry LUT for the EOTF
(accuracy ~5e-7, LUT values committed as hardcoded hex-float
literals by the `scripts/gen_ssimulacra2_eotf_lut.py` generator).
No runtime libc dependency for transcendentals, so the output is
bit-for-bit identical across glibc / musl / macOS libSystem hosts
and the tight `places=4` gate holds.
"""

import json
import os
import subprocess
import tempfile
import unittest

from vmaf import ExternalProgram
from vmaf.config import VmafConfig


class Ssimulacra2SnapshotTest(unittest.TestCase):
    """Snapshot gate for the `ssimulacra2` feature extractor (fork-added)."""

    RC_SUCCESS = 0

    def setUp(self):
        self.output_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name

    def tearDown(self):
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)

    def _run_ssimulacra2(self, ref, dis, width, height, bitdepth=8):
        """Invoke `vmaf --feature ssimulacra2` and return the parsed JSON."""
        exe = ExternalProgram.vmafexec
        cmd = (
            f"{exe} --reference {ref} --distorted {dis} "
            f"--width {width} --height {height} --pixel_format 420 "
            f"--bitdepth {bitdepth} --json --feature ssimulacra2 "
            f"--output {self.output_file_path} --quiet"
        )
        # `cmd` is built from the test's own `exe`, `ref`, `dis` (all
        # hardcoded test-fixture paths) and integer geometry/bitdepth fields.
        # No attacker-controlled string. See Research-0090, F13–F17.
        # nosemgrep: python.lang.security.audit.subprocess-shell-true.subprocess-shell-true
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, self.RC_SUCCESS, f"vmaf exited {ret}: {cmd}")
        with open(self.output_file_path) as fo:
            return json.load(fo)

    def test_ssimulacra2_src01_576x324(self):
        """src01_hrc00 vs src01_hrc01 at 576x324, 48 frames. Primary gate."""
        result = self._run_ssimulacra2(
            VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
            576,
            324,
        )
        pooled = result["pooled_metrics"]["ssimulacra2"]
        frames = result["frames"]
        self.assertEqual(len(frames), 48)
        self.assertAlmostEqual(pooled["mean"], 24.613842, places=4)
        self.assertAlmostEqual(pooled["min"], 13.816480, places=4)
        self.assertAlmostEqual(pooled["max"], 49.955009, places=4)
        self.assertAlmostEqual(pooled["harmonic_mean"], 22.904087, places=4)
        self.assertAlmostEqual(frames[0]["metrics"]["ssimulacra2"], 49.955009, places=4)
        self.assertAlmostEqual(frames[47]["metrics"]["ssimulacra2"], 37.408924, places=4)

    def test_ssimulacra2_small_160x90(self):
        """Tiny 160x90 derived fixture — exercises the <8x8 tail path."""
        ref_name = (
            "ref_test_0_1_src01_hrc00_576x324_576x324_vs_"
            "src01_hrc01_576x324_576x324_q_160x90.yuv"
        )
        dis_name = (
            "dis_test_0_1_src01_hrc00_576x324_576x324_vs_"
            "src01_hrc01_576x324_576x324_q_160x90.yuv"
        )
        result = self._run_ssimulacra2(
            VmafConfig.test_resource_path("yuv", ref_name),
            VmafConfig.test_resource_path("yuv", dis_name),
            160,
            90,
        )
        pooled = result["pooled_metrics"]["ssimulacra2"]
        frames = result["frames"]
        self.assertEqual(len(frames), 48)
        self.assertAlmostEqual(pooled["mean"], 77.693109, places=4)
        self.assertAlmostEqual(pooled["min"], 72.806309, places=4)
        self.assertAlmostEqual(pooled["max"], 86.795857, places=4)
        self.assertAlmostEqual(frames[0]["metrics"]["ssimulacra2"], 86.795857, places=4)
        self.assertAlmostEqual(frames[47]["metrics"]["ssimulacra2"], 82.603946, places=4)


if __name__ == "__main__":
    unittest.main()
