"""Restricted master problem: set covering over routes, solved with HiGHS.

Formulation
-----------
    min   sum_r  (c_r + V) x_r
    s.t.  sum_{r containing i} x_r  >= 1      for every customer i   (dual pi_i >= 0)
          sum_r x_r                 <= K      if max_vehicles given  (dual sigma <= 0)
          x_r >= 0

V is an optional fixed cost per route (a drone sortie). Set covering rather
than set partitioning is used because it is numerically friendlier and the
duals stay non-negative, which matters for the pricing heuristics.

Reduced cost of column r is therefore
    c_r + V - sum_{i in r} pi_i - sigma
and the pricing subproblem must find a route minimising
    sum_(i,j) in r  dist_ij  -  sum_{i in r} pi_i
to compare against (sigma - V).

Dual handling
-------------
Duals are read from the primal's inequality marginals and then verified
against the three conditions of LP duality (sign, dual feasibility, strong
duality). If verification fails the dual LP is solved explicitly instead.
This costs a little code but removes any dependence on a solver's marginals
sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from .instance import Instance, evaluate_route

EPS = 1e-6


@dataclass
class LpSolution:
    objective: float
    pi: np.ndarray
    sigma: float
    x: np.ndarray
    dual_source: str = "primal-marginals"
    #: False when neither the solver's marginals nor the explicitly solved dual
    #: LP passed the duality checks.  The vector is still returned -- it is the
    #: best available pricing signal and pricing with a slightly-off dual only
    #: costs iterations -- but it must never be used to *certify* anything.  The
    #: distinction matters because an unverified dual can look dual-feasible
    #: while violating a column's constraint, and a certificate read off it
    #: would declare an LP optimal that was never proved.
    dual_verified: bool = True


@dataclass
class IpSolution:
    objective: float
    chosen: List[int]
    x: np.ndarray


class MasterProblem:
    """Set covering master problem with incremental column addition."""

    def __init__(
        self,
        inst: Instance,
        max_vehicles: Optional[int] = None,
        vehicle_fixed_cost: float = 0.0,
    ) -> None:
        self.inst = inst
        self.max_vehicles = max_vehicles
        self.vehicle_fixed_cost = float(vehicle_fixed_cost)

        self.routes: List[Tuple[int, ...]] = []
        self.cover: List[Tuple[int, ...]] = []
        self.base_cost: List[float] = []
        self._seen: Set[Tuple[int, ...]] = set()
        #: cached (nodes, columns) sparse structure of the cover matrix.
        #: Rebuilding it from scratch on every LP solve is O(columns * nodes)
        #: in pure Python, which dominates the whole run once the pool holds a
        #: few thousand columns - the exact situation the learned pricer is
        #: meant to create.
        self._cover_rows: Optional[np.ndarray] = None
        self._cover_cols: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.routes)

    @property
    def cost_vector(self) -> np.ndarray:
        return np.asarray(self.base_cost, dtype=float) + self.vehicle_fixed_cost

    def add(self, nodes: Sequence[int], cost: float) -> bool:
        """Add a column. Returns False when it is already present.

        `cost` is the route's real travel cost. It is NOT a reduced cost: a
        pricer works in reduced-cost space and those values are frequently
        negative, and a column with negative cost inside a covering LP makes
        the problem unbounded. The guard below exists so that mistake fails
        loudly at the point it is made instead of surfacing as a confusing
        "model is unbounded" from the solver.
        """
        key = tuple(int(v) for v in nodes)
        if not key or key in self._seen:
            return False
        if not np.isfinite(cost) or cost < -EPS:
            raise ValueError(
                f"column cost must be finite and non-negative, got {cost!r} "
                f"for route {key}. Did a reduced cost leak in?"
            )
        self._seen.add(key)
        self.routes.append(key)
        self.cover.append(key)
        self.base_cost.append(float(cost))
        # column count changed: the cached cover structure is stale.  Appending
        # one column would be cheaper still, but the matrix is rebuilt at most
        # once per LP solve and correctness beats a micro-optimisation here.
        self._cover_rows = None
        self._cover_cols = None
        return True

    def add_many(self, columns: Sequence[Tuple[Sequence[int], float]]) -> int:
        """Add pricer output, which is (route, reduced_cost) pairs.

        Only the route is taken from the pair. The column cost is recomputed
        from the instance, because a reduced cost describes how attractive a
        column is relative to the current duals, not what the column costs.
        """
        added = 0
        for nodes, _red_cost in columns:
            ev = evaluate_route(self.inst, nodes)
            if not ev.feasible:
                continue
            if self.add(nodes, ev.cost):
                added += 1
        return added

    def clone(self) -> "MasterProblem":
        """An independent copy holding the same columns.

        Used by the diagnostics that ask "what would one more pricing round
        have been worth?": they add columns to a throw-away copy and re-solve,
        leaving the live master untouched.
        """
        c = MasterProblem(
            self.inst, max_vehicles=self.max_vehicles,
            vehicle_fixed_cost=self.vehicle_fixed_cost,
        )
        for nodes, cost in zip(self.routes, self.base_cost):
            c.add(nodes, cost)
        return c

    def _cover_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """(row, column) index pairs of the cover matrix, cached across solves."""
        if self._cover_rows is None:
            m = len(self.routes)
            if m == 0:
                self._cover_rows = np.zeros(0, dtype=np.int64)
                self._cover_cols = np.zeros(0, dtype=np.int64)
            else:
                lengths = np.fromiter(
                    (len(r) for r in self.routes), dtype=np.int64, count=m
                )
                total = int(lengths.sum())
                rows = np.empty(total, dtype=np.int64)
                cols = np.empty(total, dtype=np.int64)
                pos = 0
                for j, nodes in enumerate(self.routes):
                    k = len(nodes)
                    rows[pos:pos + k] = np.fromiter(nodes, dtype=np.int64, count=k)
                    cols[pos:pos + k] = j
                    pos += k
                self._cover_rows = rows - 1
                self._cover_cols = cols
        return self._cover_rows, self._cover_cols

    def _cover_matrix(self) -> np.ndarray:
        m = len(self.routes)
        n = self.inst.n
        A = np.zeros((n, m), dtype=float)
        rows, cols = self._cover_index()
        if rows.size:
            A[rows, cols] = 1.0
        return A

    # ------------------------------------------------------------------
    # LP
    # ------------------------------------------------------------------
    def _primal_lp(self):
        n = self.inst.n
        m = len(self.routes)
        if m == 0:
            raise ValueError("master problem has no columns")

        A_cover = self._cover_matrix()
        rows = [-A_cover]
        rhs = [-np.ones(n)]
        if self.max_vehicles is not None:
            rows.append(np.ones((1, m)))
            rhs.append(np.array([float(self.max_vehicles)]))

        A_ub = np.vstack(rows)
        b_ub = np.concatenate(rhs)
        res = linprog(
            self.cost_vector,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0.0, None)] * m,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"master LP failed: {res.message}")
        return res, A_cover, A_ub, b_ub

    def _duals_from_marginals(self, res, A_cover, n) -> Tuple[np.ndarray, float, str]:
        marg = np.asarray(res.ineqlin.marginals, dtype=float)
        pi = -marg[:n]
        sigma = float(marg[n]) if self.max_vehicles is not None else 0.0
        return pi, sigma, "primal-marginals"

    def _solve_explicit_dual(self) -> Tuple[np.ndarray, float, float]:
        """max 1'pi + K sigma  s.t.  sum_{i in r} pi_i + sigma <= c_r, pi>=0, sigma<=0."""
        n = self.inst.n
        m = len(self.routes)
        A = np.zeros((m, n + 1), dtype=float)
        for j, nodes in enumerate(self.routes):
            for i in nodes:
                A[j, i - 1] = 1.0
        A[:, n] = 1.0

        c = np.concatenate([-np.ones(n), [-float(self.max_vehicles or 0)]])
        bounds = [(0.0, None)] * n
        bounds.append((None, 0.0))
        res = linprog(c, A_ub=A, b_ub=self.cost_vector, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"dual LP failed: {res.message}")
        pi = np.asarray(res.x[:n], dtype=float)
        sigma = float(res.x[n])
        return pi, sigma, float(-res.fun)

    def _verify_duals(self, pi, sigma, obj, tol=1e-4) -> bool:
        n = self.inst.n
        if np.any(pi < -tol):
            return False
        if sigma > tol:
            return False
        rows, cols = self._cover_index()
        lhs = np.zeros(len(self.routes), dtype=float)
        if rows.size:
            np.add.at(lhs, cols, np.asarray(pi, dtype=float)[rows])
        lhs += sigma
        if np.any(lhs > self.cost_vector + tol):
            return False
        expect = float(pi.sum()) + float(self.max_vehicles or 0) * sigma
        return abs(expect - obj) <= max(tol, abs(obj) * 1e-6)

    def solve_lp(self) -> LpSolution:
        res, A_cover, _, _ = self._primal_lp()
        pi, sigma, source = self._duals_from_marginals(res, A_cover, self.inst.n)
        verified = self._verify_duals(pi, sigma, res.fun)
        if not verified:
            try:
                pi, sigma, dval = self._solve_explicit_dual()
                verified = self._verify_duals(pi, sigma, dval)
                source = "explicit-dual-lp"
            except RuntimeError:
                verified = False
        if not verified:
            # Do not raise.  A degenerate LP can leave both routes unverifiable
            # within tolerance, and a run that dies here cannot say anything
            # about anything -- whereas a run that continues with a slightly-off
            # dual still generates columns, still improves the primal bound, and
            # can still be certified through the primal route (see
            # `CertificateByIncumbent`).  The one thing it must not do is claim
            # a certificate, and `dual_verified=False` is what stops that.
            source = "unverified"
        return LpSolution(float(res.fun), pi, sigma, np.asarray(res.x), source,
                          dual_verified=verified)

    def solve_ip(self) -> IpSolution:
        m = len(self.routes)
        if m == 0:
            raise ValueError("master problem has no columns")
        A_cover = self._cover_matrix()
        cons = [LinearConstraint(A_cover, lb=np.ones(self.inst.n), ub=np.inf)]
        if self.max_vehicles is not None:
            cons.append(
                LinearConstraint(np.ones((1, m)), lb=-np.inf, ub=float(self.max_vehicles))
            )
        res = milp(
            c=self.cost_vector,
            constraints=cons,
            integrality=np.ones(m),
            bounds=Bounds(0, 1),
        )
        if not res.success:
            raise RuntimeError(f"master IP failed: {res.message}")
        x = np.asarray(res.x)
        chosen = [j for j in range(m) if x[j] > 0.5]
        return IpSolution(float(res.fun), chosen, x)

    # ------------------------------------------------------------------
    def describe(self, chosen: Sequence[int]) -> Dict[str, object]:
        ev = []
        for j in chosen:
            nodes = self.routes[j]
            ev.append((list(nodes), float(self.base_cost[j])))
        return {
            "num_routes": len(chosen),
            "total_base_cost": float(sum(c for _, c in ev)),
            "routes": ev,
        }
