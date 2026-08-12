from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tileops_foundry_loop.round import RoundError, validate_round


HIR = """\
@module
class Operator:
    def kernel(self, x):
        with Mesh(("cta",), (1,), ("block",)) as mesh:
            local = tf.reshard(x, (N @ mesh.block,), "rmem")
            return tf.reshard(local, (N @ mesh.block,), "gmem")
"""

RUNTIME = """\
@runtime_module(Operator)
class Runtime:
    @runtime_func
    def kernel(self, x):
        return ProductionKernel(x)
"""


class RoundValidationTests(unittest.TestCase):
    def make_round(self, root: Path) -> Path:
        (root / "artifacts").mkdir()
        (root / "authored_hir.py").write_text(HIR, encoding="utf-8")
        (root / "runtime_twin.py").write_text(RUNTIME, encoding="utf-8")
        for name, value in (
            ("check.json", {"passed": True}),
            ("analyze.json", {"result": "ok"}),
            ("schedule.json", {"result": "ok"}),
        ):
            (root / "artifacts" / name).write_text(json.dumps(value), encoding="utf-8")
        provenance = {
            "tilefoundry": {
                "commit": "a" * 40,
                "version": "0.0.2.dev1",
                "wheel_sha256": "b" * 64,
            },
            "hir": {"source": "authored_hir.py", "selector": "Operator.kernel"},
            "runtime_twin": {
                "source": "runtime_twin.py",
                "selector": "Runtime.kernel",
                "production_kernel": "src/tileops/kernels/op.py:main",
                "production_symbol": "ProductionKernel",
            },
            "tilefoundry_check": {
                "status": "passed",
                "report": "artifacts/check.json",
                "command": "tilefoundry check runtime_twin.py:Runtime.kernel --json",
            },
            "analysis": {
                "status": "passed",
                "report": "artifacts/analyze.json",
                "command": "tilefoundry analyze authored_hir.py:Operator.kernel --compute-cost --memory --roofline --timeline --json",
            },
            "schedule": {
                "status": "passed",
                "report": "artifacts/schedule.json",
                "command": "tilefoundry schedule authored_hir.py:Operator.kernel --json",
            },
            "decisions": [
                {
                    "tilefoundry_evidence": "memory pressure",
                    "hir_choice": "rmem shard",
                    "kernel_site": "main",
                }
            ],
            "primitive_experiments": [
                {
                    "bottleneck": "copy",
                    "primitive": "T.copy",
                    "hypothesis": "vectorize movement",
                    "control_ms": 2.0,
                    "candidate_ms": 1.0,
                    "verdict": "kept",
                }
            ],
        }
        (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (root / "findings.json").write_text('{"findings": []}', encoding="utf-8")
        return root

    def test_complete_round_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIn("runtime_twin", validate_round(self.make_round(Path(temporary))))

    def test_hir_without_reshard_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_round(Path(temporary))
            (root / "authored_hir.py").write_text(HIR.replace("tf.reshard", "tf.copy"), encoding="utf-8")
            with self.assertRaisesRegex(RoundError, "reshard"):
                validate_round(root)

    def test_placeholder_runtime_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_round(Path(temporary))
            (root / "runtime_twin.py").write_text(
                RUNTIME.replace("return ProductionKernel(x)", "raise NotImplementedError"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RoundError, "placeholder"):
                validate_round(root)


if __name__ == "__main__":
    unittest.main()
