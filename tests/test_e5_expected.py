"""Expected outcome of E5 (configs/e5_expected.yaml) against the audits of runs/e5/openacc_acc (docs/m4_contract.md,
3.4). The comparison is skipped while the runs or the expected values are missing."""

from typing import Any

import numpy as np
import pytest
import yaml

from cf_stability.utils import REPO_ROOT, read_json, resolve_path

EXPECTED = REPO_ROOT / "configs" / "e5_expected.yaml"
RUNS_ROOT = "runs"  # the results of the experiments, relative to the repository root


def expected() -> dict[str, Any]:
    return yaml.safe_load(EXPECTED.read_text(encoding="utf-8"))


def test_expected_outcome_is_well_formed():
    e = expected()
    assert set(e) == {"experiment", "data", "share", "rule", "min_share_unstable"}
    assert e["share"] in ("band_numerical", "grid_numerical") and e["rule"] in ("each", "mean")
    assert e["min_share_unstable"] and all(v is None or 0.0 <= v <= 1.0 for v in e["min_share_unstable"].values())


@pytest.mark.parametrize("model", sorted(expected()["min_share_unstable"]))
def test_e5_expected_outcome(model):
    e = expected()
    value = e["min_share_unstable"][model]
    runs = resolve_path(RUNS_ROOT) / e["experiment"] / e["data"] / model
    audits = sorted(runs.glob("*/stability.json"))
    if value is None:
        pytest.skip(f"no expected value for {model} in {EXPECTED.name}")
    if not audits:
        pytest.skip(f"no audited runs under {runs}")
    shares = {}
    for path in audits:
        result = read_json(path)
        assert "error" not in result, f"{path}: {result.get('error')}"
        support = result["audit"]["summary"][e["share"]]["support"]
        assert support is not None, f"{path}: no {e['share']} share in support (no band or no support)"
        shares[path.parent.name] = support["unstable"]
    if e["rule"] == "each":
        below = {run: share for run, share in shares.items() if share < value}
        assert not below, f"{model}: runs with fewer than {value:.0%} of the speeds flagged unstable: {below}"
    else:
        assert np.mean(list(shares.values())) >= value, f"{model}: mean share {np.mean(list(shares.values())):.3f}"
