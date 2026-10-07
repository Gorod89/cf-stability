"""kNN baseline: exact neighbours, chunking, sub-sampling, checkpoints."""

import dataclasses

import numpy as np
import pytest
import torch

from cf_stability.data.schema import DT, Event
from cf_stability.models import KNN, load_model, save_model
from cf_stability.models.base import ModelContext

CONTEXT = ModelContext(center=(30.0, 0.0, 15.0), scale=(15.0, 1.5, 6.0), seed=0)


def random_events(n_events: int = 3, n: int = 200, seed: int = 0) -> list[Event]:
    """Events with independent random states and accelerations (the kNN does not care about dynamics)."""
    rng = np.random.default_rng(seed)
    events = []
    for i in range(n_events):
        s, dv, v, a = rng.uniform(2, 80, n), rng.normal(0, 1.5, n), rng.uniform(0, 30, n), rng.normal(0, 1, n)
        events.append(
            Event(
                event_id=f"e{i}", dataset="syn", site="a", follower_id=f"f{i}", leader_id=f"l{i}", t=DT * np.arange(n),
                s=s, dv=dv, v=v, a=a, v_lead=v - dv, x_lead=s, x_follower=np.zeros(n),
            )
        )  # fmt: skip
    return events


def samples(events: list[Event]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([np.stack((e.s, e.dv, e.v), -1) for e in events]), np.concatenate([e.a for e in events])


def queries(n: int = 50, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack((rng.uniform(2, 80, n), rng.normal(0, 1.5, n), rng.uniform(0, 30, n)), -1)


def test_matches_numpy_neighbours():
    events = random_events()
    model = KNN(CONTEXT, k=7)
    assert model.fit(events, CONTEXT) == {"n_samples": 600, "n_points": 600}
    assert model.points.dtype == torch.float32 and model.points.shape == (600, 3)
    states, acc = samples(events)
    center, scale = np.array(CONTEXT.center), np.array(CONTEXT.scale)
    q = queries()
    dist = (((q[:, None, :] - states[None]) / scale) ** 2).sum(-1)
    expected = acc[np.argsort(dist, axis=1)[:, :7]].mean(1)
    out = model(torch.as_tensor(q)[:, None, :])
    assert out.shape == (50,) and np.allclose(out.numpy(), expected, rtol=0, atol=1e-6)
    stored = torch.as_tensor(states)[:, None, :]  # a stored sample is its own nearest neighbour
    nearest = KNN(CONTEXT, k=1)
    nearest.fit(events, CONTEXT)
    assert np.allclose(nearest(stored).numpy(), acc, rtol=0, atol=1e-6)
    assert np.allclose(model.points.numpy(), (states - center) / scale, rtol=0, atol=1e-6)


def test_chunking_and_dtype_do_not_change_the_result():
    events = random_events()
    x = torch.as_tensor(queries(300))[:, None, :]
    outputs = []
    for chunk in (7, 64, 1024):
        model = KNN(CONTEXT, chunk=chunk)
        model.fit(events, CONTEXT)
        outputs.append(model(x))
    for out in outputs[1:]:
        assert torch.allclose(out, outputs[0], rtol=0, atol=1e-6)
    double = model.double()
    assert double.points.dtype == torch.float64 and double(x).dtype == torch.float64
    assert torch.allclose(double(x).float(), outputs[0], rtol=0, atol=1e-5)


def test_subsampling_is_reproducible():
    events = random_events()
    full = KNN(CONTEXT)
    full.fit(events, CONTEXT)
    runs = []
    for seed in (0, 0, 1):
        model = KNN(CONTEXT, max_points=100)
        assert model.fit(events, dataclasses.replace(CONTEXT, seed=seed)) == {"n_samples": 600, "n_points": 100}
        runs.append(model)
    assert torch.equal(runs[0].points, runs[1].points) and torch.equal(runs[0].targets, runs[1].targets)
    assert not torch.equal(runs[0].points, runs[2].points)
    # the kept samples are training samples, with their own accelerations
    match = (runs[0].points[:, None, :] == full.points[None]).all(-1)
    assert (match.sum(1) == 1).all()
    assert torch.equal(runs[0].targets, full.targets[match.int().argmax(1)])


def test_checkpoint_restores_the_fitted_points(tmp_path):
    model = KNN(CONTEXT, k=5, max_points=150)
    with pytest.raises(RuntimeError, match="fitted points"):
        model(torch.zeros(2, 1, 3))
    model.fit(random_events(), CONTEXT)
    x = torch.as_tensor(queries())[:, None, :]
    save_model(model, tmp_path / "knn.pt")
    loaded = load_model(tmp_path / "knn.pt")
    assert loaded.points.shape == (150, 3) and loaded.config() == model.config()
    assert torch.equal(loaded(x), model(x))
    fresh = KNN(k=5)  # hyper-parameters come from the config, points and scaler from the state
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(fresh(x), model(x))
