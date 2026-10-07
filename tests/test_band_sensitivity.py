"""Band sensitivity of the audit (scripts/analysis/band_sensitivity.py, D119): the table from the audits of the
three bands, on hand-made audit files."""

import importlib.util
import os
import time

import numpy as np
import pandas as pd
import pytest

from cf_stability.eval.stats import bootstrap_ci
from cf_stability.utils import REPO_ROOT, write_json

SHARES = ("stable", "unstable", "outside", "none", "indifferent", "undefined")


def module():
    path = REPO_ROOT / "scripts" / "analysis" / "band_sensitivity.py"
    spec = importlib.util.spec_from_file_location("band_sensitivity", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def audit(stable: float, unstable: float, outside: float, unstable_eq: float, quantiles=None) -> dict:
    shares = {key: 0.0 for key in SHARES} | {"stable": stable, "unstable": unstable, "outside": outside}
    shares["none"] = 1.0 - stable - unstable - outside
    summary = {"band_numerical": {"all": shares, "support": shares},
               "share_unstable_numerical": {"all": unstable_eq, "support": unstable_eq}}  # fmt: skip
    payload = {"model": "mlp", "audit": {"summary": summary}}
    if quantiles is not None:
        payload["band_override"] = {"quantiles": list(quantiles)}
    return payload


def test_table_of_the_three_bands(tmp_path):
    bs = module()
    assert list(bs.BANDS) == ["1/99", "5/95", "10/90"]
    files = [name for name, _ in bs.BANDS.values()]
    assert files == ["stability_q01_99.json", "stability.json", "stability_q10_90.json"]
    rng = np.random.default_rng(0)
    root, expected = tmp_path / "runs", {}
    old = time.time() - 100.0
    for k in range(6):  # six runs of the gru: the 1/99 band moves outside equilibria into the band
        run = root / "e1" / "follownet_highd" / "gru" / f"driver_fold{k % 5}_seed{k // 5}"
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"fake")
        os.utime(run / "model.pt", (old, old))
        unstable_eq = 0.8 + 0.03 * k
        outside = float(rng.uniform(0.2, 0.4))
        values = {
            "5/95": audit(0.1, 0.9 - outside - 0.05, outside, unstable_eq),
            "1/99": audit(0.1, 0.9 - 0.05, 0.0, unstable_eq + 0.05, (0.01, 0.5, 0.99)),
            "10/90": audit(0.05, 0.5, 0.4, unstable_eq - 0.1, (0.1, 0.5, 0.9)),
        }
        for band, payload in values.items():
            if band == "10/90" and k == 5:
                continue  # one audit missing
            write_json(run / bs.BANDS[band][0], payload)
        expected[run.name] = values
    write_json(root / "e1" / "follownet_highd" / "gru" / "driver_fold0_seed0" / "stability_q10_90.json",
               audit(0.0, 0.0, 0.0, 0.5, (0.05, 0.5, 0.95)))  # fmt: skip  # an audit of the wrong band
    runs, problems = bs.collect(root, "e1", "follownet_highd", ["gru"])
    assert len(runs) == 6 + 6 + 4 and len(problems) == 2
    assert any("quantiles [0.05, 0.5, 0.95], expected [0.1, 0.5, 0.9]" in p for p in problems)
    assert any("stability_q10_90.json missing" in p for p in problems)
    table = bs.summarise(runs, ["gru"], {"gru": 6}).set_index("band")
    assert list(table.index) == ["1/99", "5/95", "10/90"] and list(table.runs) == [6, 6, 4]
    summaries = [v["5/95"]["audit"]["summary"] for v in expected.values()]
    reference = np.array([s["share_unstable_numerical"]["support"] for s in summaries])
    ci = bootstrap_ci(reference, list(expected), n_resamples=1000, seed=0)  # resampled by run: any row order
    assert table.loc["5/95", "unstable_eq"] == pytest.approx(reference.mean())
    assert (table.loc["5/95", "unstable_eq_low"], table.loc["5/95", "unstable_eq_high"]) == (ci["low"], ci["high"])
    assert table.loc["1/99", "unstable_eq_change"] == pytest.approx(0.05) and table.loc["1/99", "pairs"] == 6
    assert table.loc["10/90", "unstable_eq_change"] == pytest.approx(-0.1) and table.loc["10/90", "pairs"] == 4
    assert table.loc["1/99", "outside"] == 0.0 and table.loc["10/90", "outside"] == pytest.approx(0.4)
    assert (table.h1_1_share_rule == "met").all()  # >= 0.5 with the lower end above 0.3
    md = bs.markdown(table.reset_index(), problems, "e1", "follownet_highd")
    assert md.startswith("# Band sensitivity of the audit (D119)") and md.count("\n| gru |") == 3
    assert "Missing or unusable audits: 2" in md

    out = tmp_path / "tables"
    assert bs.main(["--runs-root", str(root), "--architectures", "gru", "--out", str(out)]) == 0
    written = pd.read_csv(out / "band_sensitivity.csv")
    assert list(written.band) == ["1/99", "5/95", "10/90"] and (out / "band_sensitivity.md").is_file()
