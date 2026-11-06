"""Helpers for tests and for the scaling experiments.

Nothing here is part of the algorithm; it exists so that correctness can be
checked against brute force on small instances, which is the only way to be
sure the labelling pricer and the master duals are right before any learning
is layered on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .instance import DEPOT, EPS, Instance, UAVParams, build_instance, evaluate_route


def random_instance(
    n: int,
    seed: int = 0,
    grid: float = 100.0,
    capacity: Optional[float] = None,
    tw_slack: float = 60.0,
    ready_spread: float = 120.0,
    service_time: float = 5.0,
    num_vehicles: Optional[int] = None,
    energy_capacity: Optional[float] = None,
    load_energy_factor: float = 1.0,
    base_energy_rate: float = 1.0,
) -> Instance:
    """A random Euclidean instance whose singleton routes are feasible.

    Time windows are built around each customer's depot distance so that a
    direct depot -> customer -> depot round trip always fits; without that
    guarantee the master problem could be infeasible for trivial reasons and
    every test would be measuring the wrong thing. The window is wide enough
    (2 * depot distance plus slack) that multi-customer routes exist, which is
    what makes the instance interesting for column generation at all.
    """
    rng = np.random.default_rng(seed)
    coords = np.zeros((n + 1, 2))
    coords[0] = (grid / 2.0, grid / 2.0)
    coords[1:] = rng.uniform(0.0, grid, size=(n, 2))

    depot_dist = np.sqrt(((coords[1:] - coords[0]) ** 2).sum(axis=1))
    demand = np.zeros(n + 1)
    demand[1:] = rng.integers(1, 11, size=n).astype(float)
    cap = float(capacity) if capacity is not None else float(max(20.0, demand[1:].max() * 6.0))

    width = tw_slack + 2.0 * depot_dist + service_time
    ready = np.zeros(n + 1)
    ready[1:] = depot_dist + rng.uniform(0.0, ready_spread, size=n)
    due = np.zeros(n + 1)
    due[1:] = ready[1:] + width
    service = np.zeros(n + 1)
    service[1:] = service_time
    horizon = float(due[1:].max() + depot_dist.max() + service_time + 10.0)
    ready[DEPOT] = 0.0
    due[DEPOT] = horizon

    params = UAVParams(
        capacity=cap,
        energy_capacity=energy_capacity,
        base_energy_rate=base_energy_rate,
        load_energy_factor=load_energy_factor,
    )
    return build_instance(
        name=f"rand-n{n}-s{seed}",
        coords=coords,
        demand=demand,
        ready=ready,
        due=due,
        service=service,
        num_vehicles=int(num_vehicles if num_vehicles is not None else n),
        params=params,
    )


def enumerate_routes(
    inst: Instance,
    max_length: Optional[int] = None,
    limit: int = 2_000_000,
) -> List[Tuple[int, ...]]:
    """Every feasible route, by depth-first enumeration. Small n only."""
    n = inst.n
    cap = inst.params.capacity
    out: List[Tuple[int, ...]] = []
    cap_len = max_length if max_length is not None else n

    def dfs(path: List[int], visited: int, node: int, time: float, load: float) -> None:
        if len(out) >= limit:
            return
        if path and evaluate_route(inst, tuple(path)).feasible:
            out.append(tuple(path))
        if len(path) >= cap_len:
            return
        for j in range(1, n + 1):
            if (visited >> j) & 1:
                continue
            if load + inst.demand[j] > cap + EPS:
                continue
            arr = time + inst.service[node] + inst.travel_time[node, j]
            if arr > inst.due[j] + EPS:
                continue
            start = arr if arr > inst.ready[j] else inst.ready[j]
            path.append(j)
            dfs(path, visited | (1 << j), j, start, load + float(inst.demand[j]))
            path.pop()

    dfs([], 1, DEPOT, 0.0, 0.0)
    return out


def write_solomon(inst: Instance, path: str | Path) -> None:
    """Serialise an instance back into Solomon format (round-trip testing)."""
    lines = [inst.name, "", "VEHICLE", "NUMBER     CAPACITY",
             f"{inst.num_vehicles:5d} {inst.params.capacity:10.0f}", "", "CUSTOMER",
             "CUST NO.   XCOORD.    YCOORD.    DEMAND    READY TIME   DUE DATE   SERVICE TIME", ""]
    for i in range(inst.n + 1):
        lines.append(
            f"{i:5d} {inst.coords[i,0]:10.4f} {inst.coords[i,1]:10.4f} "
            f"{inst.demand[i]:9.4f} {inst.ready[i]:12.4f} {inst.due[i]:11.4f} "
            f"{inst.service[i]:12.4f}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_mask(inst: Instance, path: Sequence[int]) -> Tuple[bool, int]:
    """Check that the rollout feasibility mask never blocks a known route.

    The mask is built from necessary conditions only, so it must accept every
    arc of every feasible route.  This walks a route that is *already known to
    be feasible*, rebuilds the incremental state exactly the way the rollout
    does, and reports the first step at which the mask would have removed the
    route's own next customer.

    Returns (ok, step).  `ok=False` means the mask is unsound and the search
    it drives can never find that route - a silent loss of optimal columns,
    which is why this is worth an explicit test rather than a comment.
    """
    from .features import RolloutState, FeatureContext

    ctx = FeatureContext(inst)
    n = inst.n
    node, time, load = DEPOT, 0.0, 0.0
    dist_so_far, pi_sum, energy_lb = 0.0, 0.0, 0.0
    visited = np.zeros(n + 1, dtype=bool)
    visited[DEPOT] = True
    zeros = np.zeros(n)

    for step, j in enumerate(path):
        st = RolloutState(
            node=node, time=time, load=load, dist_so_far=dist_so_far,
            pi_sum=pi_sum, energy_lb=energy_lb, step=step,
            path=tuple(path[:step]), visited=visited.copy(),
        )
        cand = ctx.mask(st)
        if j not in cand:
            return False, step

        arr = time + inst.service[node] + inst.travel_time[node, j]
        start = arr if arr > inst.ready[j] else float(inst.ready[j])
        rate_lb = ctx.base_rate * (1.0 + ctx.load_factor * float(inst.demand[j]) / ctx.cap)
        energy_lb += rate_lb * float(inst.dist[node, j])
        dist_so_far += float(inst.dist[node, j])
        pi_sum += float(zeros[j - 1])
        load += float(inst.demand[j])
        visited[j] = True
        node, time = j, start

    return True, len(path)
