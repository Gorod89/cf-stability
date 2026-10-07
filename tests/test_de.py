"""Batched differential evolution: correctness, bounds, reproducibility, agreement with SciPy."""

import numpy as np
import pytest
import torch
from scipy.optimize import differential_evolution

from cf_stability.train.de import differential_evolution_batched

N_RUNS = 16  # independent problems (torch) and seeds (SciPy) for the statistical comparison


def sphere(center: torch.Tensor, offset: float = 0.0):
    return lambda pop: offset + ((pop - center[:, None, :]) ** 2).sum(-1)


def rosenbrock(center: torch.Tensor, offset: float = 0.0):
    def f(pop: torch.Tensor) -> torch.Tensor:
        y = pop - center[:, None, :] + 1.0  # minimum at pop == center
        return offset + (100.0 * (y[..., 1:] - y[..., :-1] ** 2) ** 2 + (1.0 - y[..., :-1]) ** 2).sum(-1)

    return f


FUNCTIONS = {"sphere": sphere, "rosenbrock": rosenbrock}


def centers(n: int, d: int, seed: int = 0) -> torch.Tensor:
    return torch.as_tensor(np.random.default_rng(seed).uniform(-2.0, 2.0, size=(n, d)))


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("name", ["sphere", "rosenbrock"])
def test_many_problems_at_once(name, d):
    c = centers(8, d, seed=d)
    res = differential_evolution_batched(FUNCTIONS[name](c), [-5.0] * d, [5.0] * d, 8, maxiter=1000, seed=d)
    assert res.x.shape == (8, d) and res.fun.shape == (8,)
    assert (res.fun < 1e-8).all()
    assert torch.allclose(res.x, c, rtol=0, atol=1e-3)


def test_convergence_is_per_problem():
    c = centers(8, 3)
    offsets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    objective = lambda pop: sphere(c)(pop) + offsets[:, None]  # noqa: E731
    res = differential_evolution_batched(objective, [-5.0] * 3, [5.0] * 3, 8, maxiter=150)
    # with a zero minimum the relative tolerance is never met; with an offset it is met early
    assert res.converged[:4].all() and not res.converged[4:].any()
    assert (res.n_generations[:4] < 60).all() and (res.n_generations[4:] == 150).all()
    assert len(set(res.n_generations[:4].tolist())) > 1


def test_bounds_respected():
    lower, upper = torch.tensor([-5.0, 0.0, 2.0]), torch.tensor([5.0, 1.0, 3.0])
    c = torch.tensor([[7.0, 0.5, 2.5], [0.0, -3.0, 2.2]], dtype=torch.float64)  # partly outside the box

    def objective(pop: torch.Tensor) -> torch.Tensor:
        assert (pop >= lower.double()).all() and (pop <= upper.double()).all()
        return sphere(c)(pop)

    res = differential_evolution_batched(objective, lower, upper, 2, maxiter=200, tol=0.0)
    assert torch.allclose(res.x, torch.tensor([[5.0, 0.5, 2.5], [0.0, 0.0, 2.2]], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(res.fun, torch.tensor([4.0, 9.0], dtype=torch.float64), atol=1e-6)


def test_same_seed_same_result():
    c = centers(5, 4)
    def run(seed: int):
        return differential_evolution_batched(rosenbrock(c, 1.0), [-5.0] * 4, [5.0] * 4, 5, seed=seed)

    a, b, other = run(3), run(3), run(4)
    for field in ("x", "fun", "n_generations", "converged"):
        assert torch.equal(getattr(a, field), getattr(b, field))
    assert not torch.equal(a.x, other.x)


def test_callback_can_stop():
    seen = []
    res = differential_evolution_batched(
        sphere(centers(3, 2)), [-5.0] * 2, [5.0] * 2, 3, callback=lambda r: seen.append(r.fun) or len(seen) == 7
    )
    assert len(seen) == 7 and (res.n_generations == 7).all()


def _scipy_runs(name: str, c: np.ndarray, offset: float, maxiter: int) -> tuple[np.ndarray, np.ndarray]:
    f_torch = FUNCTIONS[name](torch.as_tensor(c[None]), offset)

    def f(x: np.ndarray) -> np.ndarray:  # vectorized: x is (D, S)
        return f_torch(torch.as_tensor(x.T[None]))[0].numpy()

    runs = [
        differential_evolution(
            f, [(-5.0, 5.0)] * len(c), maxiter=maxiter, polish=False, rng=100 + i, updating="deferred", vectorized=True
        )
        for i in range(N_RUNS)
    ]
    return np.array([r.fun for r in runs]) - offset, np.array([r.nit for r in runs])


@pytest.mark.parametrize(("name", "d"), [("sphere", 3), ("sphere", 5), ("rosenbrock", 2), ("rosenbrock", 3)])
def test_agrees_with_scipy(name, d):
    """Same algorithm and settings: generations to convergence and progress rate agree with SciPy."""
    c = np.random.default_rng(d).uniform(-2.0, 2.0, size=d)
    batched_c = torch.as_tensor(np.repeat(c[None], N_RUNS, axis=0))
    # offset minimum: the relative tolerance terminates the runs
    res = differential_evolution_batched(FUNCTIONS[name](batched_c, 1.0), [-5.0] * d, [5.0] * d, N_RUNS, maxiter=1000)
    err_scipy, nit_scipy = _scipy_runs(name, c, 1.0, 1000)
    assert res.converged.all()
    assert res.n_generations.double().mean().item() == pytest.approx(nit_scipy.mean(), rel=0.2)
    err_torch = res.fun.numpy() - 1.0
    assert abs(np.median(np.log10(err_torch)) - np.median(np.log10(err_scipy))) < 0.5
    # zero minimum: fixed budget, compare the reached accuracy (orders of magnitude)
    res = differential_evolution_batched(FUNCTIONS[name](batched_c), [-5.0] * d, [5.0] * d, N_RUNS, maxiter=100)
    log_torch = np.median(np.log10(res.fun.numpy()))
    log_scipy = np.median(np.log10(_scipy_runs(name, c, 0.0, 100)[0]))
    assert log_torch == pytest.approx(log_scipy, rel=0.15)
