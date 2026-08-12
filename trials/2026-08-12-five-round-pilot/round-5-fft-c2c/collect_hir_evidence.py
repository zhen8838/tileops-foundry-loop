"""Run and persist the blind authored-HIR CLI evidence."""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TF = "$HOST_HOME/TileFoundry/.venv/bin/tilefoundry"
CASES = (
    (
        "complex64-dtype",
        [TF, "check", f"{ROOT / 'complex64_dtype_repro.py'}:Complex64Identity.identity", "--inputs", "random", "--out", "output", "--fn", "nan_inf", "--json"],
        1,
    ),
    (
        "complex128-dtype",
        [TF, "check", f"{ROOT / 'complex128_dtype_repro.py'}:Complex128Identity.identity", "--inputs", "random", "--out", "output", "--fn", "nan_inf", "--json"],
        1,
    ),
    (
        "f64-pair",
        [TF, "check", f"{ROOT / 'f64_pair_repro.py'}:F64Pair.pair_add", "--inputs", "random", "--out", "output[0]", "--fn", "nan_inf", "--out", "output[1]", "--fn", "nan_inf", "--json"],
        1,
    ),
    (
        "authored-hir-check",
        [TF, "check", f"{ROOT / 'authored_hir.py'}:FFTPairF32.dft_pair_f32", "--inputs", "random", "--out", "output[0]", "--fn", "nan_inf", "--out", "output[1]", "--fn", "nan_inf", "--json"],
        0,
    ),
    (
        "authored-hir-analyze",
        [TF, "analyze", f"{ROOT / 'authored_hir.py'}:FFTPairF32", "--compute-cost", "--memory", "--roofline", "--timeline", "--json"],
        0,
    ),
    (
        "authored-hir-schedule",
        [TF, "schedule", f"{ROOT / 'authored_hir.py'}:FFTPairF32", "--topology", "cta", "--solver-workers", "1", "--first-plan", "--json"],
        0,
    ),
)


def main() -> None:
    suite = ET.Element("testsuite", name="tilefoundry_cli", tests=str(len(CASES)))
    failures = 0
    for name, command, expected_rc in CASES:
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        suffix = "json" if name.startswith("authored-hir") else "log"
        (ROOT / f"{name}.{suffix}").write_text(run.stdout, encoding="utf-8")
        (ROOT / f"{name}.stderr").write_text(run.stderr, encoding="utf-8")
        case = ET.SubElement(suite, "testcase", name=name)
        ET.SubElement(case, "system-out").text = run.stdout
        ET.SubElement(case, "system-err").text = run.stderr
        if run.returncode != expected_rc:
            failures += 1
            ET.SubElement(
                case,
                "failure",
                message=f"exit {run.returncode}, expected {expected_rc}",
            )
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(
        ROOT / "tilefoundry-cli.xml", encoding="utf-8", xml_declaration=True
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
