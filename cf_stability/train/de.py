"""Differential evolution batched over independent problems (torch, GPU-friendly).

Follows ``scipy.optimize.differential_evolution`` with ``strategy="best1bin"``,
``updating="deferred"``, ``init="latinhypercube"`` and ``polish=False``: the population
lives in the unit cube, trial components outside it are re-drawn uniformly, selection
is greedy (``trial <= current``), and a problem converges when
``std(E) <= atol + tol * |mean(E)|``. Converged problems are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor


@dataclass
class DEResult:
    x: Tensor  # [n_problems, D] best member
    fun: Tensor  # [n_problems] its energy
    n_generations: Tensor  # [n_problems] generations run (int64)
    converged: Tensor  # [n_problems] tolerance reached before maxiter


def _gather_members(pop: Tensor, idx: Tensor) -> Tensor:
    return torch.gather(pop, 1, idx.unsqueeze(-1).expand(-1, -1, pop.shape[-1]))


def differential_evolution_batched(
    objective: Callable[[Tensor], Tensor],
    lower: Sequence[float] | Tensor,
    upper: Sequence[float] | Tensor,
    n_problems: int,
    *,
    popsize: int = 15,
    maxiter: int = 300,
    tol: float = 0.01,
    atol: float = 0.0,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    callback: Callable[[DEResult], bool | None] | None = None,
) -> DEResult:
    """Minimise ``n_problems`` problems at once.

    ``objective`` maps parameters ``[n_problems, P, D]`` to energies ``[n_problems, P]``
    with ``P = max(5, popsize * D)``. ``callback`` receives the current result after every
    generation and stops the optimisation by returning ``True``.
    """
    device = torch.device(device if device is not None else "cpu")
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    n, d = n_problems, lower.numel()
    p = max(5, popsize * d)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.rand(shape, generator=gen, dtype=dtype, device=device)

    def randint(high: int, *shape: int) -> Tensor:
        return torch.randint(high, shape, generator=gen, device=device)

    def to_params(u: Tensor) -> Tensor:
        return torch.minimum(torch.maximum(lower + u * (upper - lower), lower), upper)

    rows = torch.arange(n, device=device)
    member = torch.arange(p, device=device)

    def result(pop: Tensor, energy: Tensor, n_gen: Tensor, converged: Tensor) -> DEResult:
        best = energy.argmin(dim=1)
        return DEResult(to_params(pop[rows, best]), energy[rows, best], n_gen.clone(), converged.clone())

    # Latin hypercube: one sample per stratum, strata permuted per problem and dimension
    strata = (rand(n, p, d) + torch.arange(p, dtype=dtype, device=device)[:, None]) / p
    pop = torch.gather(strata, 1, torch.argsort(rand(n, p, d), dim=1))
    energy = objective(to_params(pop))
    n_gen = torch.zeros(n, dtype=torch.int64, device=device)
    converged = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(maxiter):
        if isinstance(mutation, (int, float)):
            scale: Tensor | float = mutation
        else:
            scale = mutation[0] + (mutation[1] - mutation[0]) * rand(n, 1, 1)
        # r0, r1 uniform over the members, pairwise distinct and different from the target
        r0 = randint(p - 1, n, p)
        r0 = r0 + (r0 >= member)
        r1 = randint(p - 2, n, p)
        r1 = r1 + (r1 >= torch.minimum(r0, member))
        r1 = r1 + (r1 >= torch.maximum(r0, member))
        best = pop[rows, energy.argmin(dim=1)]
        mutant = best[:, None, :] + scale * (_gather_members(pop, r0) - _gather_members(pop, r1))
        cross = (rand(n, p, d) < recombination) | (torch.arange(d, device=device) == randint(d, n, p)[..., None])
        trial = torch.where(cross, mutant, pop)
        trial = torch.where((trial < 0.0) | (trial > 1.0), rand(n, p, d), trial)
        trial_energy = objective(to_params(trial))
        accept = (trial_energy <= energy) & ~converged[:, None]
        pop = torch.where(accept[..., None], trial, pop)
        energy = torch.where(accept, trial_energy, energy)
        n_gen += ~converged
        converged |= energy.std(dim=1, correction=0) <= atol + tol * energy.mean(dim=1).abs()
        if callback is not None and callback(result(pop, energy, n_gen, converged)):
            break
        if bool(converged.all()):
            break
    return result(pop, energy, n_gen, converged)
