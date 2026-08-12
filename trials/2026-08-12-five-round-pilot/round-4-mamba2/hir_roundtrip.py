"""Emit and re-import canonical source for the authored one-step module."""

import importlib.util
from pathlib import Path

from tilefoundry.inspection import as_script


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("round4_authored_hir", HERE / "authored_hir.py")
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

rendered = as_script(MOD.Mamba2Step, module="Mamba2StepRoundTrip")
(HERE / "authored_hir_roundtrip.py").write_text(rendered, encoding="ascii")

roundtrip_spec = importlib.util.spec_from_file_location(
    "round4_authored_hir_roundtrip", HERE / "authored_hir_roundtrip.py"
)
assert roundtrip_spec is not None and roundtrip_spec.loader is not None
roundtrip_mod = importlib.util.module_from_spec(roundtrip_spec)
roundtrip_spec.loader.exec_module(roundtrip_mod)
print(roundtrip_mod.Mamba2StepRoundTrip)
