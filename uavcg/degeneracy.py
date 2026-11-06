"""Measuring the optimal dual face of a seed -- the quantity that decides
whether column generation can make progress at all.

Why this is a library module and not a diagnostic script
--------------------------------------------------------
The v3.4 seed decision (spec §3.3) rests on one measurement: the seed's first
RMP has an optimal dual *face*, and CG can only descend when that face is a
single point.  A high-dimensional face means the solver hands back an arbitrary
vertex, the pricer finds columns that violate that vertex while leaving the rest
of the face intact, and the bound never moves.  That is not a theory that can be
left to a one-off script: it is the property that makes the whole baseline
meaningful, so it is asserted in `scripts/selftest.py` on every run of the
suite, and the assertion shares this code with the standalone diagnostic
(`scripts/_diag_degeneracy_face.py`).  Two implementations of a load-bearing
measurement is how a diagnosis and its regression test drift apart.

The measurement
---------------
For the covering LP `min c'x s.t. Ax >= 1, x >= 0`, the dual is
`max 1'pi s.t. A'pi <= c, pi >= 0`.  Write `v*` for its optimal value.  The
optimal dual face is

    F = { pi >= 0 : A'pi <= c, 1'pi = v* }

and the diagnostic question is whether `F` is a single point.  For each customer
coordinate `j` this module maximises and minimises `pi_j` over `F` with two small
LPs; if every coordinate has `hi_j == lo_j` the face is a point.

The singletons seed has a closed-form answer that is worth stating, because it
is what makes the assertion meaningful rather than lucky: every singleton column
`(i,)` is tight at the seed optimum, so `pi_i <= 2*d(0,i)` and
`sum_i pi_i = sum_i 2*d(0,i)`.  Equality forces `pi_i = 2*d(0,i)` for every `i`,
so the face is a point by algebra, not by solver luck.  `singleton_identity_gap`
below checks that identity numerically so a change to the seed or to
`evaluate_route` cannot break it silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from .instance import Instance, evaluate_route

#: Spec v3.4 §8: the pipeline self-check asserts the first-round dual face has
#: max coordinate span below this.
FACE_TOL = 1e-6


@dataclass(frozen=True)
class DualFace:
    """The optimal dual face of a master LP, summarised."""

    objective: float
    lo: np.ndarray
    hi: np.ndarray

    @property
    def span(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def max_span(self) -> float:
        return float(np.max(self.span)) if self.span.size else 0.0

    @property
    def free_coordinates(self) -> int:
        """Coordinates that can move: the face's dimension, counted coarsely."""
        return int(np.sum(self.span > 1e-6))

    @property
    def is_point(self) -> bool:
        """True when the optimal dual is unique (max span below FACE_TOL)."""
        return self.max_span < FACE_TOL

    def describe(self) -> str:
        return (f"value={self.objective:.10f} "
                f"free={self.free_coordinates}/{self.span.size} "
                f"max_span={self.max_span:.6f}")


def cover_matrix(inst: Instance, routes: Sequence[Sequence[int]]) -> np.ndarray:
    """Dense customer-by-column cover matrix, built from the routes."""
    A = np.zeros((inst.n, len(routes)), dtype=float)
    for j, r in enumerate(routes):
        for i in r:
            A[int(i) - 1, j] = 1.0
    return A


def optimal_dual_face(inst: Instance, routes: Sequence[Sequence[int]],
                      costs: Sequence[float]) -> DualFace:
    """Solve the dual, then bound every coordinate over the optimal face."""
    A = cover_matrix(inst, routes)
    c = np.asarray(costs, dtype=float)
    n = inst.n

    res = linprog(-np.ones(n), A_ub=A.T, b_ub=c,
                  bounds=[(0.0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(f"seed dual LP failed: {res.message}")
    v_star = -float(res.fun)

    A_ub = np.vstack([A.T, np.ones((1, n)), -np.ones((1, n))])
    b_ub = np.concatenate([c, [v_star], [-v_star]])
    lo = np.empty(n)
    hi = np.empty(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        r = linprog(e, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n,
                    method="highs")
        lo[j] = float(r.fun) if r.success else np.nan
        r = linprog(-e, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n,
                    method="highs")
        hi[j] = -float(r.fun) if r.success else np.nan
    return DualFace(objective=v_star, lo=lo, hi=hi)


def seed_dual_face(inst: Instance, routes: Sequence[Sequence[int]]) -> DualFace:
    """Convenience wrapper: cost the routes from the instance, then measure."""
    kept = [tuple(int(v) for v in r) for r in routes
            if evaluate_route(inst, r).feasible]
    if not kept:
        raise RuntimeError("seed has no feasible route")
    costs = [evaluate_route(inst, r).cost for r in kept]
    return optimal_dual_face(inst, kept, costs)


def singleton_identity_gap(inst: Instance) -> float:
    """|sum_i 2*d(0,i) - (the singleton seed's LP value)|.

    Zero means the algebraic argument for uniqueness applies: the seed LP value
    equals the sum of the per-customer dual caps, so every cap is tight and the
    optimal dual is pinned.  A non-zero value here means the uniqueness claim
    does not hold for this instance and the seed decision needs revisiting.
    """
    from .heuristics import singleton_routes

    routes = singleton_routes(inst)
    face = seed_dual_face(inst, routes)
    caps = 2.0 * np.asarray(inst.dist[0, 1:], dtype=float)
    return float(abs(np.sum(caps) - face.objective))
