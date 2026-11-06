"""Benchmark instance loading, shared by every experiment script.

Keeping this in one place matters more than it looks: the UAV energy budget
has to be derived from the instance itself, and getting it wrong makes the
energy constraint silently inactive, at which point every "drone" result is
really a VRPTW result.  One definition, used everywhere, removes that risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .heuristics import cheapest_insertion
from .instance import (
    Instance,
    as_uav,
    evaluate_route,
    narrow_time_windows,
    parse_solomon,
    truncate,
    without_time_windows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "solomon100"


def make_uav(inst: Instance, factor: float, load_factor: float = 1.0) -> Instance:
    """Energy budget derived from the seed solution, so the constraint bites.

    `factor` multiplies the largest energy of any route in a cheapest-insertion
    solution.  Measured on Solomon C101 n=25: factor 1.0 leaves the constraint
    inactive (LP identical to VRPTW), 0.7 makes it bind, 0.35 saturates.  A
    floor of 1.05x the cheapest single-customer round trip keeps the master
    problem feasible, since every customer must still be servable somehow.

    NOTE (v3.4): the `cheapest_insertion` call below is *instance definition*
    (it sets the energy budget), not the column-generation seed.  The v3.4
    singleton-seed decision (`uavcg/heuristics.py::singleton_routes`) governs
    which columns seed the RMP and deliberately does not touch this: changing
    it would silently redefine the instance and move every recorded number.
    """
    routes = cheapest_insertion(inst)
    seed_max = max(evaluate_route(inst, r).energy for r in routes)
    floor = max(evaluate_route(inst, (i,)).energy * 1.05 for i in inst.customers)
    return as_uav(
        inst,
        energy_capacity=max(factor * seed_max, floor),
        load_energy_factor=load_factor,
    )


def load_instance(
    name: str,
    n: int = 0,
    uav: float = 0.0,
    load_factor: float = 1.0,
    no_tw: bool = False,
    data_dir: Optional[Path] = None,
    capacity_factor: float = 1.0,
    tw_scale: float = 1.0,
) -> Instance:
    """Solomon file -> truncation -> TW narrowing -> TW relaxation -> UAV.

    Order matters.  Narrowing is applied *after* truncation so the
    direct-round-trip feasibility check sees the geometry the solver will
    actually work on, and *before* the UAV budget so the budget is calibrated
    on the instance that gets solved.  `tw_scale < 1` activates the protocol in
    spec DynML-S §3.3; `1.0` is the untouched instance and costs nothing.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA
    path = data_dir / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"instance not found: {path}\nrun scripts/fetch_instances.py first"
        )
    inst, _meta = parse_solomon(path)
    if n and n < inst.n:
        inst = truncate(inst, n)
    if tw_scale != 1.0:
        inst, _rep = narrow_time_windows(inst, tw_scale)
    if capacity_factor != 1.0:
        from .instance import Instance as _I
        from .instance import UAVParams as _P

        inst = _I(
            name=inst.name, num_vehicles=inst.num_vehicles,
            params=_P(capacity=inst.params.capacity * capacity_factor,
                      energy_capacity=inst.params.energy_capacity,
                      base_energy_rate=inst.params.base_energy_rate,
                      load_energy_factor=inst.params.load_energy_factor),
            coords=inst.coords, demand=inst.demand, ready=inst.ready,
            due=inst.due, service=inst.service, dist=inst.dist,
            travel_time=inst.travel_time,
        )
    if no_tw:
        inst = without_time_windows(inst)
    if uav > 0:
        inst = make_uav(inst, uav, load_factor)
    return inst
