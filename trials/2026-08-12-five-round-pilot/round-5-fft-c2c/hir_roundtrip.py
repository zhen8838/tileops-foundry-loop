"""Exercise canonical authored-HIR source emission and re-import."""

import importlib.util
from pathlib import Path

from tilefoundry.inspection import as_script

from authored_hir import FFTPairF32


root = Path(__file__).resolve().parent
out = root / "authored_hir_roundtrip.py"
out.write_text(as_script(FFTPairF32) + "\n", encoding="utf-8")
spec = importlib.util.spec_from_file_location("authored_hir_roundtrip", out)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(f"roundtrip={out} module={module.FFTPairF32.name}")
