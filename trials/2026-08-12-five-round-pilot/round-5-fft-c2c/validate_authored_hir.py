"""Validate the f32 real-pair HIR against literal DFT and butterfly oracles."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from authored_hir import FFTPairF32


ROOT = Path(__file__).resolve().parent


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    expected = expected.to(actual.device)
    delta = (actual - expected).abs()
    passed = True
    error = None
    try:
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    except AssertionError as exc:
        passed = False
        error = str(exc)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "assert_close": passed,
        "assertion": error,
    }


def main() -> None:
    torch.manual_seed(17)
    x = torch.randn(2, 4, dtype=torch.complex64)
    t = torch.arange(4, dtype=torch.float32)[:, None]
    k = torch.arange(4, dtype=torch.float32)[None, :]
    theta = -2.0 * math.pi * t * k / 4
    w_r, w_i = torch.cos(theta), torch.sin(theta)

    y_r, y_i = FFTPairF32.dft_pair_f32(x.real, x.imag, w_r, w_i)
    expected_r = x.real @ w_r - x.imag @ w_i
    expected_i = x.real @ w_i + x.imag @ w_r

    even, odd = x[:, :2], x[:, 2:]
    tw = torch.polar(torch.ones(2), torch.tensor([0.0, -math.pi / 2]))
    b_r, b_i = FFTPairF32.complex_butterfly(
        even.real,
        even.imag,
        odd.real,
        odd.imag,
        tw.real,
        tw.imag,
    )
    prod = odd * tw
    expected_b = torch.cat((even + prod, even - prod), dim=1)

    payload = {
        "dft_real": _errors(y_r, expected_r),
        "dft_imag": _errors(y_i, expected_i),
        "butterfly_real": _errors(b_r, expected_b.real),
        "butterfly_imag": _errors(b_i, expected_b.imag),
    }
    (ROOT / "authored-hir-oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    suite = ET.Element("testsuite", name="authored_hir", tests="4")
    failures = 0
    for name, record in payload.items():
        case = ET.SubElement(suite, "testcase", name=name)
        if not record["assert_close"]:
            failures += 1
            ET.SubElement(case, "failure", message="assert_close failed").text = record[
                "assertion"
            ]
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(
        ROOT / "authored-hir-oracle.xml", encoding="utf-8", xml_declaration=True
    )
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
