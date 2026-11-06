"""Problem data for the multi-drone routing study.

One model, two modes
--------------------
    energy_capacity is None -> plain VRPTW   (comparable with Solomon BKS)
    energy_capacity is set  -> UAV mode, energy depends on the load carried

Load convention (UAV mode)
--------------------------
A drone lifts the whole payload of its route at the depot and gets lighter
as it delivers. So the load carried on an arc is the demand *still to be
delivered* at that moment, not the demand already delivered.

That quantity is not knowable while a route is built forward, which has a
direct consequence for the pricing problem: energy cannot be used as an
incremental pruning resource. It is therefore evaluated as a final check on
completed routes only, while (time, load) remain the pruning resources.
This keeps the labelling algorithm exact - it just prunes less. See
pricers/exact.py for the argument in full.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEPOT = 0
EPS = 1e-6

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    """Pull every numeric token out of a line."""
    return [float(tok) for tok in _NUM.findall(text)]


@dataclass(frozen=True)
class UAVParams:
    """Drone-side parameters.

    base_energy_rate / load_energy_factor define the energy model
        energy(arc, carried) = base_energy_rate * dist * (1 + load_energy_factor * carried / capacity)
    with load_energy_factor = 0 the model degenerates to energy proportional
    to distance, which is a useful ablation.
    """

    capacity: float
    energy_capacity: Optional[float] = None
    base_energy_rate: float = 1.0
    load_energy_factor: float = 1.0

    @property
    def energy_enabled(self) -> bool:
        return self.energy_capacity is not None


@dataclass
class Instance:
    name: str
    num_vehicles: int
    params: UAVParams
    coords: np.ndarray          # (n+1, 2), row 0 is the depot
    demand: np.ndarray          # (n+1,)
    ready: np.ndarray           # (n+1,)
    due: np.ndarray             # (n+1,)
    service: np.ndarray         # (n+1,)
    dist: np.ndarray            # (n+1, n+1)
    travel_time: np.ndarray     # (n+1, n+1)

    @property
    def n(self) -> int:
        return int(self.coords.shape[0]) - 1

    @property
    def customers(self) -> range:
        return range(1, self.n + 1)

    def arc_energy(self, i: int, j: int, carried: float) -> float:
        """Energy burned on arc (i, j) while carrying `carried` units of payload."""
        p = self.params
        rate = p.base_energy_rate
        if p.capacity > 0:
            rate *= 1.0 + p.load_energy_factor * carried / p.capacity
        return rate * float(self.dist[i, j])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mode = "UAV" if self.params.energy_enabled else "VRPTW"
        return f"<Instance {self.name} n={self.n} mode={mode} cap={self.params.capacity:g}>"


@dataclass
class RouteEval:
    feasible: bool
    cost: float = math.inf
    load: float = 0.0
    energy: float = 0.0
    finish: float = 0.0
    reason: str = ""


def evaluate_route(inst: Instance, nodes: Sequence[int], eps: float = EPS) -> RouteEval:
    """Exact feasibility and cost of the closed route 0 -> nodes... -> 0."""
    p = inst.params
    total = float(sum(inst.demand[j] for j in nodes))
    if total > p.capacity + eps:
        return RouteEval(False, reason="capacity")

    delivered = 0.0
    t = 0.0            # start of service at the depot
    prev = DEPOT
    cost = 0.0
    energy = 0.0

    for j in nodes:
        arr = t + inst.service[prev] + inst.travel_time[prev, j]
        if arr > inst.due[j] + eps:
            return RouteEval(False, reason=f"time_window@{j}")
        start = arr if arr > inst.ready[j] else inst.ready[j]
        energy += inst.arc_energy(prev, j, total - delivered)
        cost += float(inst.dist[prev, j])
        delivered += float(inst.demand[j])
        t = start
        prev = j

    arr = t + inst.service[prev] + inst.travel_time[prev, DEPOT]
    if arr > inst.due[DEPOT] + eps:
        return RouteEval(False, reason="time_window@depot")
    energy += inst.arc_energy(prev, DEPOT, total - delivered)
    cost += float(inst.dist[prev, DEPOT])

    if p.energy_enabled and energy > p.energy_capacity + eps:
        return RouteEval(False, reason="energy")

    finish = arr if arr > inst.ready[DEPOT] else inst.ready[DEPOT]
    return RouteEval(True, cost=cost, load=total, energy=energy, finish=finish)


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------

def build_instance(
    name: str,
    coords: np.ndarray,
    demand: np.ndarray,
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    num_vehicles: int,
    params: UAVParams,
    speed: float = 1.0,
) -> Instance:
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    travel_time = dist / float(speed)
    return Instance(
        name=name,
        num_vehicles=int(num_vehicles),
        params=params,
        coords=coords,
        demand=np.asarray(demand, dtype=float),
        ready=np.asarray(ready, dtype=float),
        due=np.asarray(due, dtype=float),
        service=np.asarray(service, dtype=float),
        dist=dist,
        travel_time=travel_time,
    )


def with_rounded_distances(inst: Instance) -> Instance:
    """Return `inst` with integer Euclidean distances, as Solomon states them.

    Why this exists at all
    ----------------------
    Every published Solomon / Gehring-Homberger optimum -- and therefore every
    "best known solution" a red-line check would compare against -- is stated in
    the benchmark's own convention, which rounds each arc to the nearest
    integer.  `build_instance` does not: it keeps `sqrt(dx^2 + dy^2)` exactly,
    which is the right choice for the solver (rounding is a modelling artefact,
    not physics) and the wrong one for comparing against the literature.

    Measured on the 100-customer Solomon set, the gap is 0.223 in absolute value
    per arc on average, at most 0.5.  A ten-route solution uses about 110 arcs,
    so the two conventions put the same solution's objective apart by order 1-25
    -- larger than the differences a red-line check is meant to detect.  In
    other words the comparison is only meaningful after this transform.

    `np.rint` is safe here, and the tie-break it uses (half to even) differs
    from the benchmark's `int(x + 0.5)` only at exact halves.  A tie needs
    `sqrt(k) = m + 0.5`, i.e. `k = m^2 + m + 0.25`, which is never an integer,
    and every squared distance here is an integer -- so no tie can occur and the
    two rules agree on every arc.

    Scope: used by the §8 red-line anchor and nothing else.  All main
    experiments keep the exact distances, so this must never be wired into
    `load_instance`'s default path.
    """
    dist = np.rint(inst.dist)
    return replace(inst, dist=dist, travel_time=np.rint(inst.travel_time))


def parse_solomon(path: str | Path) -> Tuple[Instance, dict]:
    """Read a Solomon / Gehring-Homberger formatted instance.

    Returns the instance in plain VRPTW mode plus a dict of raw instance
    attributes, so a caller can decide how to turn it into a UAV instance.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    head = next((ln.strip() for ln in text if ln.strip()), "unnamed")

    num_vehicles, capacity = 0, 0.0
    for idx, ln in enumerate(text):
        if ln.strip().upper().startswith("NUMBER"):
            for j in range(idx + 1, min(idx + 4, len(text))):
                nums = _numbers(text[j])
                if len(nums) >= 2:
                    num_vehicles, capacity = int(nums[0]), float(nums[1])
                    break
            break

    start = 0
    for idx, ln in enumerate(text):
        if ln.strip().upper().startswith("CUSTOMER"):
            start = idx + 1
            break

    rows: List[List[float]] = []
    for ln in text[start:]:
        nums = _numbers(ln)
        if len(nums) >= 7:
            rows.append(nums[:7])
    if not rows:
        raise ValueError(f"no customer rows found in {path}")

    arr = np.array(rows, dtype=float)
    coords = arr[:, 1:3]
    demand = arr[:, 3]
    ready = arr[:, 4]
    due = arr[:, 5]
    service = arr[:, 6]

    params = UAVParams(capacity=capacity, energy_capacity=None)
    inst = build_instance(
        name=head, coords=coords, demand=demand, ready=ready, due=due,
        service=service, num_vehicles=num_vehicles, params=params,
    )
    meta = {
        "source": str(path),
        "capacity": capacity,
        "num_vehicles": num_vehicles,
        "depot_horizon": float(due[0]),
    }
    return inst, meta


def as_uav(
    inst: Instance,
    energy_capacity: float,
    load_energy_factor: float = 1.0,
    base_energy_rate: float = 1.0,
    capacity: Optional[float] = None,
) -> Instance:
    """Return a copy of `inst` switched into UAV mode.

    `energy_capacity` is expressed in the same units as dist * base_energy_rate.
    A common way to pick a value: take the length of a full depot -> depot
    round trip and multiply by a slack factor.
    """
    params = UAVParams(
        capacity=float(capacity if capacity is not None else inst.params.capacity),
        energy_capacity=float(energy_capacity),
        base_energy_rate=base_energy_rate,
        load_energy_factor=load_energy_factor,
    )
    return Instance(
        name=f"{inst.name}-uav",
        num_vehicles=inst.num_vehicles,
        params=params,
        coords=inst.coords,
        demand=inst.demand,
        ready=inst.ready,
        due=inst.due,
        service=inst.service,
        dist=inst.dist,
        travel_time=inst.travel_time,
    )


def truncate(inst: Instance, n_customers: int) -> Instance:
    """Keep the depot and the first `n_customers` customers (for scaling tests)."""
    m = n_customers + 1
    return Instance(
        name=f"{inst.name}-n{n_customers}",
        num_vehicles=inst.num_vehicles,
        params=inst.params,
        coords=inst.coords[:m],
        demand=inst.demand[:m],
        ready=inst.ready[:m],
        due=inst.due[:m],
        service=inst.service[:m],
        dist=inst.dist[:m, :m],
        travel_time=inst.travel_time[:m, :m],
    )


def without_time_windows(inst: Instance) -> Instance:
    """Open every time window up to the depot horizon.

    Time windows are the fiddliest part of this study: they interact with the
    depot closing time, they make the feasibility mask harder to reason about,
    and on Solomon instances they prune so aggressively that a weak pricer can
    look strong by accident.  Opening them turns the problem into a CVRP (or a
    CVRP with an energy budget), which is the natural fallback if the VRPTW
    version turns out to be too hard to read results from.

    Only the data changes - every code path stays identical - so an experiment
    can be re-run with `--no-tw` and compared directly.
    """
    ready = np.zeros_like(inst.ready)
    due = np.full_like(inst.due, float(inst.due[DEPOT]))
    return Instance(
        name=f"{inst.name}-notw",
        num_vehicles=inst.num_vehicles,
        params=inst.params,
        coords=inst.coords,
        demand=inst.demand,
        ready=ready,
        due=due,
        service=inst.service,
        dist=inst.dist,
        travel_time=inst.travel_time,
    )


def narrow_time_windows(
    inst: Instance, scale: float
) -> Tuple[Instance, dict]:
    """Shrink every customer time window towards its midpoint by `scale`.

    The reference paper narrows the time windows of its 200/400-client instances
    ("the width of the time windows has been slightly reduced") to bring the
    running time down.  The same lever is the only thing that makes Solomon
    category-2 affordable here: at n=60, four of the six planned instances take
    more than 70 s on a single pricing call with the windows left untouched,
    while the ones that do run sit at 6-12% of the label cap.  So the cap is not
    what blocks the problem - the labelling work is - and narrowing is the
    cheapest way to stay inside the time budget without changing any code path.

    Two rules keep the narrowing from breaking the problem instead of easing it:

    1. **The depot window is never touched.**  `due[0]` is the vehicle horizon
       and `evaluate_route` enforces it on every route, so shrinking it silently
       removes working time from all of them; shrinking its `ready` would delay
       every departure.
    2. **Every customer must stay reachable by a direct round trip.**  A
       narrowed due date can land before the earliest possible arrival, which
       makes that customer unservable and turns the master problem infeasible --
       surfacing as "no column exists" rather than as a data error.  Such a
       window is clamped back to the tightest value that still admits the round
       trip, which always stays inside the original window.  A customer that
       cannot be served by a direct round trip even *before* narrowing is a
       malformed instance and raises: clamping that one would widen the window
       past the original and relax the problem instead of tightening it.

    Returns the narrowed instance and a small report of how much clamping was
    needed, so a caller can see how close to the edge a given scale is.

    `scale=1.0` is an *exact* identity on the data (see the `shrink` note
    below), which is what lets it serve as the untouched control arm.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")

    idx = np.arange(1, inst.n + 1)

    # Rule 2, precondition.  A customer whose direct depot round trip cannot fit
    # is a data error, not something to narrow: clamping it back would *widen*
    # the window past the original and quietly relax the problem in the other
    # direction.
    earliest_arrival = float(inst.service[DEPOT]) + inst.travel_time[DEPOT, idx]
    latest_start = float(inst.due[DEPOT]) - inst.service[idx] - inst.travel_time[idx, DEPOT]
    bad = (earliest_arrival > inst.due[idx] + EPS) | (inst.ready[idx] > latest_start + EPS)
    if bad.any():
        who = [int(j) for j in idx[bad]]
        raise ValueError(
            f"instance {inst.name!r} is not well-formed: a direct depot round trip "
            f"cannot serve customer(s) {who[:8]}" + (" ..." if len(who) > 8 else "")
        )

    # Shrink each side by `shrink`.  Written as `r + shrink` / `d - shrink`
    # rather than as the algebraically identical `mid -/+ half`, because the
    # latter rounds: measured on a random 6-customer instance it reproduced the
    # original `ready` only to 2.84e-14, and "scale=1.0 is the untouched control
    # arm" is a promise this function has to keep exactly, not approximately.
    # With this form `shrink` is exactly 0.0 at scale=1.0, so the identity is
    # structural and cannot drift.
    shrink = 0.5 * (1.0 - float(scale)) * (inst.due[idx] - inst.ready[idx])
    new_ready = inst.ready[idx] + shrink
    new_due = inst.due[idx] - shrink

    # Rule 2, clamp.  Narrowing alone can still put a customer out of reach --
    # with scale=0.1 a due date of 100 against an earliest arrival of 90 lands
    # at 59.5 -- so each side is pulled back to the tightest value that still
    # admits the round trip.  Both clamps move *inside* the original window
    # (earliest_arrival <= due[j] and latest_start >= ready[j], by the check
    # above), so the result is never wider than what we started from.
    due_clamped = new_due < earliest_arrival - EPS
    ready_clamped = new_ready > latest_start + EPS
    new_due = np.maximum(new_due, earliest_arrival)
    new_ready = np.minimum(new_ready, latest_start)
    # new_ready < new_due by construction and both clamps only shrink, so the
    # window cannot collapse.  Asserted rather than assumed: a collapsed window
    # would make a customer unservable and the master problem infeasible, which
    # surfaces as "no column exists" rather than as a data error.
    assert bool(np.all(new_ready <= new_due + EPS)), (
        f"narrowing collapsed a window at scale={scale}"
    )

    ready = np.array(inst.ready, dtype=float, copy=True)
    due = np.array(inst.due, dtype=float, copy=True)
    ready[idx] = new_ready
    due[idx] = new_due
    # Rule 1, stated explicitly so a later edit cannot quietly drop it.
    ready[DEPOT] = inst.ready[DEPOT]
    due[DEPOT] = inst.due[DEPOT]

    report = {
        "scale": float(scale),
        "n_customers": int(inst.n),
        "mean_width_before": float(np.mean(inst.due[idx] - inst.ready[idx])),
        "mean_width_after": float(np.mean(new_due - new_ready)),
        "due_clamped": int(due_clamped.sum()),
        "ready_clamped": int(ready_clamped.sum()),
    }
    return (
        Instance(
            name=f"{inst.name}-tw{scale:g}",
            num_vehicles=inst.num_vehicles,
            params=inst.params,
            coords=inst.coords,
            demand=inst.demand,
            ready=ready,
            due=due,
            service=inst.service,
            dist=inst.dist,
            travel_time=inst.travel_time,
        ),
        report,
    )
