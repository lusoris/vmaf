"""Fork-added snapshot gate for the SSIMULACRA 2 feature extractor (T3-3,
ADR-0164). Pins the CPU output of `vmaf --feature ssimulacra2` against
reference values generated on the current master.

Unlike the Netflix golden assertions (`python/test/quality_runner_test.py`
et al.), SSIMULACRA 2 is a fork-added metric — these values are NOT part
of the upstream Netflix bit-exactness contract. They pin the fork's
self-consistency; drift here means the SIMD ports (or scalar) changed
behaviour.

Tolerance is `places=3` (1e-3). Although the SSIMULACRA 2 scalar
extractor + AVX2 + AVX-512 + NEON SIMD TUs all compile in dedicated
static libs with `-ffp-contract=off` (so FMA fusion is deterministic
across hosts), the phase-1 `cbrtf` + phase-3 `powf` per-lane scalar
libm calls carry a small cross-host variance (~2e-4): glibc vs musl
vs macOS libc implement these transcendentals with different
polynomial approximations. Within-host bit-exactness is preserved
(15 kernel-level unit tests under `test_ssimulacra2_simd` assert
this on AVX-512 and NEON) — only the aggregate pooled per-frame
scores drift slightly.

1e-3 tolerance gives comfortable headroom while still flagging any
material regression (an actual behaviour change in the extractor
would drift by orders of magnitude more).
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
        self.assertAlmostEqual(pooled["mean"], 24.612655, places=3)
        self.assertAlmostEqual(pooled["min"], 13.827219, places=3)
        self.assertAlmostEqual(pooled["max"], 49.941581, places=3)
        self.assertAlmostEqual(pooled["harmonic_mean"], 22.903611, places=3)
        self.assertAlmostEqual(frames[0]["metrics"]["ssimulacra2"], 49.941581, places=3)
        self.assertAlmostEqual(frames[47]["metrics"]["ssimulacra2"], 37.411922, places=3)

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
        self.assertAlmostEqual(pooled["mean"], 77.693091, places=3)
        self.assertAlmostEqual(pooled["min"], 72.809320, places=3)
        self.assertAlmostEqual(pooled["max"], 86.802974, places=3)
        self.assertAlmostEqual(frames[0]["metrics"]["ssimulacra2"], 86.802974, places=3)
        self.assertAlmostEqual(frames[47]["metrics"]["ssimulacra2"], 82.604036, places=3)


if __name__ == "__main__":
    unittest.main()
