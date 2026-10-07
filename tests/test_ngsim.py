"""NGSIM loader on synthetic raw data in the original format (two periods, restarting ids)."""

from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cf_stability.data import ngsim
from cf_stability.data.extraction import ExtractionConfig
from cf_stability.data.ngsim import FT, build_events, build_pairs, build_tracks, load_raw, validate_reconstruction
from cf_stability.data.reconstruction import ReconstructionConfig, derivatives, reconstruct_all
from test_reconstruction import DT, add_noise, assert_feasible, simulate_platoon

MS = {p: pd.Timestamp(s, tz="America/Los_Angeles").value // 10**6 for p, s in ((0, "2005-04-13 16:00"), (1, "2005-04-13 17:00"))}
DATA_CFG = {"name": "ngsim_i80", "loader": "ngsim", "location": "i-80", "site": "i80", "reconstruct": True,
            "reconstruction": {"s_min": 0.5}, "lanes": [1, 2, 3, 4, 5, 6]}
N = 601
DROPOUT = slice(200, 260)  # vehicle 3 of period 0 has Preceding = 0 there
GAP = slice(400, 405)  # frames missing from vehicle 5 of period 0


def period_rows(x_true, lanes, lengths, classes, rng, start_ms, frame0) -> pd.DataFrame:
    """Rows in the original NGSIM format; vehicles in one lane follow each other in index order."""
    noisy = add_noise(x_true, rng)
    ids = np.arange(1, len(x_true) + 1)
    frames = []
    for i, lane in enumerate(lanes):
        same = [j for j in range(len(lanes)) if lanes[j] == lane]
        pos = same.index(i)
        lead = same[pos - 1] if pos else None
        foll = same[pos + 1] if pos + 1 < len(same) else None
        v, a, _ = derivatives(x_true[i], DT)
        headway = (noisy[lead] - noisy[i]) / FT if lead is not None else np.zeros(N)
        frames.append(pd.DataFrame({
            "Vehicle_ID": ids[i], "Frame_ID": frame0 + np.arange(N), "Total_Frames": N,
            "Global_Time": start_ms + 100 * np.arange(N), "Local_X": 12.0 * (lane - 0.5), "Local_Y": noisy[i] / FT,
            "Global_X": 6042000.0 + 12.0 * lane, "Global_Y": 2133000.0 + noisy[i] / FT, "v_length": lengths[i] / FT,
            "v_Width": 6.0, "v_Class": classes[i], "v_Vel": v / FT, "v_Acc": a / FT, "Lane_ID": lane,
            "O_Zone": np.nan, "D_Zone": np.nan, "Int_ID": np.nan, "Section_ID": np.nan, "Direction": np.nan,
            "Movement": np.nan, "Preceding": ids[lead] if lead is not None else 0,
            "Following": ids[foll] if foll is not None else 0, "Space_Headway": headway,
            "Time_Headway": np.where(v > 0, headway * FT / np.maximum(v, 1e-9), 9999.99), "Location": "i-80",
        }))
    return pd.concat(frames, ignore_index=True)


def make_raw() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # period 0: single-lane platoon with a truck; the first vehicle's length is overstated by 2.5 m,
    # so the raw net gap of its follower is negative at standstill
    lengths0 = np.array([4.5, 4.5, 12.0, 4.5, 4.5, 4.5])
    _, x0 = simulate_platoon(lengths0)
    stated0 = lengths0 + np.r_[2.5, np.zeros(5)]
    p0 = period_rows(x0, [2] * 6, stated0, [2, 2, 3, 2, 2, 2], rng, MS[0] - 3000, frame0=1)
    p0.loc[(p0.Vehicle_ID == 3) & p0.Frame_ID.isin(np.arange(N)[DROPOUT] + 1), "Preceding"] = 0
    p0 = p0[~((p0.Vehicle_ID == 5) & p0.Frame_ID.isin(np.arange(N)[GAP] + 1))]
    # period 1: two-lane platoon, vehicle and frame ids restart
    _, xa = simulate_platoon(np.full(5, 4.5), x0=300.0)
    _, xb = simulate_platoon(np.full(5, 4.5), phase=5.0, x0=280.0)
    p1 = period_rows(np.vstack([xa, xb]), [3] * 5 + [4] * 5, np.full(10, 4.5), [2] * 10, rng, MS[1] + 20_000, frame0=201)
    df = pd.concat([p0, p1], ignore_index=True)
    df = pd.concat([df, df.sample(40, random_state=0)], ignore_index=True)  # exact duplicates
    return df.sample(frac=1.0, random_state=1, ignore_index=True)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    df = make_raw()
    path = tmp_path_factory.mktemp("ngsim") / "ngsim_i-80.csv"
    df.to_csv(path, index=False)
    return path, df


@pytest.fixture(scope="module")
def tracks(raw):
    return build_tracks(load_raw(raw[0]))


@pytest.fixture(scope="module")
def reconstructed(tracks):
    result = reconstruct_all(tracks, ReconstructionConfig())
    return result, {tid: dataclasses.replace(tr, rec=result["tracks"][tid]) for tid, tr in tracks.items()}


def test_load_raw_cleans_units_periods_and_ids(raw, tmp_path):
    path, df_raw = raw
    df = load_raw(path)
    stats = df.attrs["load_stats"]
    assert stats["raw_duplicates"] == 40 and len(df) == len(df_raw) - 40
    assert df.groupby("period").size().to_dict() == {0: 6 * N - 5, 1: 10 * N}
    # t counts from frame 0 of the first period's clock (global_time - 100 * frame_id)
    assert df.groupby("period")["t"].min().round(6).to_dict() == {0: 0.1, 1: 3623.1}
    expected = {f"p0_{i}" for i in range(1, 7)} | {"p0_5_1"} | {f"p1_{i}" for i in range(1, 11)}
    assert set(df["track_id"].cat.categories) == expected and stats["tracks_split"] == 1

    merged = df.merge(df_raw.drop_duplicates().add_prefix("raw_"), left_on=["vehicle_id", "frame_id", "global_time"],
                      right_on=["raw_Vehicle_ID", "raw_Frame_ID", "raw_Global_Time"], validate="one_to_one")
    feet = (("local_y", "Local_Y"), ("v_length", "v_length"), ("v_vel", "v_Vel"), ("v_acc", "v_Acc"), ("space_headway", "Space_Headway"))
    for si, orig in feet:
        np.testing.assert_allclose(merged[si], merged["raw_" + orig] * FT, rtol=1e-12)

    # parquet with API (lower-case) names; zone/intersection columns as text like scripts/download_data.py
    parquet = tmp_path / "raw.parquet"
    text = ["o_zone", "d_zone", "int_id", "section_id", "direction", "movement"]
    df_raw.rename(columns=str.lower).assign(**{c: "" for c in text}).to_parquet(parquet)
    pd.testing.assert_frame_equal(load_raw(parquet), df)


def test_tracks_leaders_and_rear_bumper_convention(raw, tracks):
    _, df_raw = raw
    for tr in tracks.values():
        assert np.all(np.diff(tr.frame) == 1) and np.allclose(np.diff(tr.t), DT)
    assert all(p is None for p in tracks["p0_1"].preceding)
    assert all(p == "p0_1" for p in tracks["p0_2"].preceding)
    prec3 = tracks["p0_3"].preceding
    assert all(p is None for p in prec3[DROPOUT]) and all(p == "p0_2" for p in np.delete(prec3, np.arange(N)[DROPOUT]))
    assert set(tracks["p0_6"].preceding) == {"p0_5", None, "p0_5_1"}
    assert all(p == "p1_6" for p in tracks["p1_7"].preceding)
    assert tracks["p0_3"].v_class == 3 and tracks["p0_3"].length == pytest.approx(12.0)

    pairs = {p.follower_id: p for p in build_pairs(tracks, "ngsim_i80", "i80", positions="raw")}
    assert "ngsim_i80/i80/p0_1" not in pairs and "ngsim_i80/i80/p1_6" not in pairs
    pair = pairs["ngsim_i80/i80/p0_2"]
    assert set(pair.leader_id) == {"ngsim_i80/i80/p0_1"}
    rows = df_raw[(df_raw.Vehicle_ID == 2) & (df_raw.Global_Time < MS[1] - 600_000)].drop_duplicates().sort_values("Frame_ID")
    s = pair.x_lead - pair.x_follower
    np.testing.assert_allclose(s, rows["Space_Headway"].to_numpy() * FT - 7.0, atol=1e-9)
    pair3 = pairs["ngsim_i80/i80/p0_3"]
    assert all(p is None for p in pair3.leader_id[DROPOUT]) and np.isnan(pair3.x_lead[DROPOUT]).all()


def test_reconstructed_platoons_are_consistent(reconstructed):
    result, rec_tracks = reconstructed
    for tr in rec_tracks.values():
        assert_feasible(tr.rec.x)
    assert result["report"]["gaps_before"]["share_gap_nonpositive"] > 0.0
    for pair in build_pairs(rec_tracks, "ngsim_i80", "i80"):
        has = np.array([p is not None for p in pair.leader_id])
        assert (pair.x_lead - pair.x_follower)[has].min() >= 0.5 - 0.05


def test_macro_agreement_raw_vs_reconstructed(tracks, reconstructed):
    result, _ = reconstructed
    report = validate_reconstruction(tracks, result, DATA_CFG)
    total = report["macro"]["total"]
    assert report["n_tracks"] == len(tracks) and total["n_cells"] >= 10
    assert abs(total["rel_diff_distance"]) < 0.005 and abs(total["rel_diff_time"]) < 0.005
    assert total["speed_corr"] > 0.98


def test_build_events_and_reconstruction_cache(raw, monkeypatch):
    cfg = {**DATA_CFG, "raw_path": str(raw[0])}
    # independent of the free space of the test machine (the cache written here is < 1 MB)
    monkeypatch.setattr(ngsim.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**12))
    stats: Counter = Counter()  # as in scripts/extract_events.py
    try:
        events = build_events(cfg, ExtractionConfig(), stats)
    except NotImplementedError:
        pytest.skip("extract_events is not implemented yet")
    assert len(events) > 0
    for ev in events:
        ev.validate()
        assert {"period", "lane", "leader_length", "follower_length", "v_class"} <= set(ev.meta)
    assert {ev.meta["period"] for ev in events} == {0, 1}
    assert {ev.meta["lane"] for ev in events} == {2, 3, 4}
    lead_p02 = [ev for ev in events if ev.follower_id == "ngsim_i80/i80/p0_2"]
    assert lead_p02 and all(ev.meta["leader_length"] == pytest.approx(7.0) for ev in lead_p02)
    assert stats["reconstruction_report"]["macro"]["total"]["speed_corr"] > 0.98
    assert stats["raw_duplicates"] == 40 and stats["events"] == len(events)

    cache = raw[0].parent / "cache" / "ngsim_i-80_reconstructed.parquet"
    assert cache.exists()
    monkeypatch.setattr(ngsim, "reconstruct_all", lambda *a, **k: pytest.fail("cache not used"))
    again_stats: Counter = Counter()
    again = build_events(cfg, ExtractionConfig(), again_stats)
    assert [ev.event_id for ev in again] == [ev.event_id for ev in events]
    for a, b in zip(again, events):
        np.testing.assert_array_equal(a.x_follower, b.x_follower)
        assert a.meta == b.meta
    assert again_stats["reconstruction_report"] == stats["reconstruction_report"]

    # other reconstruction options do not match the cache; on a full disk the cache is left
    # as it is and the events are built from the result in memory
    monkeypatch.setattr(ngsim, "reconstruct_all", reconstruct_all)
    monkeypatch.setattr(ngsim.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    cache = raw[0].parent / "cache" / "ngsim_i-80_reconstructed.parquet"
    before = cache.stat().st_mtime_ns
    assert len(build_events({**cfg, "reconstruction": {"s_min": 0.6}}, ExtractionConfig())) > 0
    assert cache.stat().st_mtime_ns == before
