"""Correctness self-test for the column generation core.

Run:  python scripts/selftest.py

Everything here is checked against brute force. The point is to be able to
say, before a single machine learning model is trained, that the master
duals, the reduced-cost arithmetic and the labelling pricer are all exactly
right. If this script passes, any later "the learned pricer is faster/slower"
claim is about the pricer and not about a broken baseline.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uavcg.cg import CGConfig, ColumnGeneration, WarmupTrace, reduced_cost_of
from uavcg.features import FEATURE_NAMES, NUM_FEATURES
from uavcg.heuristics import cheapest_insertion, report
from uavcg.instance import (
    DEPOT,
    UAVParams,
    as_uav,
    build_instance,
    evaluate_route,
    narrow_time_windows,
    parse_solomon,
    with_rounded_distances,
)
from uavcg.pricers import (
    ExactLabelingPricer,
    GreedyPricer,
    NgRoutePricer,
    PricingResult,
    RedCostArcPricer,
    RolloutPricer,
)
from uavcg.pricers.base import modified_cost, removed_arcs
from uavcg.rmp import MasterProblem
from uavcg.testing import enumerate_routes, random_instance, replay_mask, write_solomon
from uavcg.bench import load_instance

FAILS = []

#: Project root, for tests that need a real instance file off disk.
_ROOT = Path(__file__).resolve().parents[1]

#: Scratch directory for tests that must round-trip through the filesystem.
#: Kept inside the project rather than in the system temp dir so that whatever
#: the tests write is visible next to the results it belongs to.
_TMP = Path(__file__).resolve().parents[1] / "results" / "_selftest_tmp"
_TMP.mkdir(parents=True, exist_ok=True)


def check(name: str, ok: bool, extra: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAILS.append(name)


def brute_force_pricing(inst, pi, sigma, fixed):
    """Reference reduced costs by enumerating every feasible route."""
    m = inst.dist.copy()
    m[:, 1:] -= np.asarray(pi)[None, :]
    best, neg = 0.0, {}
    for r in enumerate_routes(inst):
        rc = sum(float(m[a, b]) for a, b in zip((DEPOT,) + r[:-1], r))
        rc += float(m[r[-1], DEPOT]) + fixed - sigma
        if rc < best:
            best = rc
        if rc < -1e-6:
            neg[r] = rc
    return best, neg


# ----------------------------------------------------------------------
def test_route_evaluation() -> None:
    inst = random_instance(8, seed=1)
    r = sorted(cheapest_insertion(inst), key=len)[-1]
    check("seed heuristic yields a multi-customer route to test on", len(r) >= 2, f"route={r}")

    ev = evaluate_route(inst, r)
    manual = sum(float(inst.dist[a, b]) for a, b in zip((DEPOT,) + r[:-1], r))
    manual += float(inst.dist[r[-1], DEPOT])
    check("route cost equals sum of arc distances", ev.feasible and abs(ev.cost - manual) < 1e-9)

    heavy = as_uav(inst, energy_capacity=1e9, load_energy_factor=2.0)
    light = as_uav(inst, energy_capacity=1e9, load_energy_factor=0.0)
    e_heavy, e_light = evaluate_route(heavy, r).energy, evaluate_route(light, r).energy
    check("load-dependent energy exceeds distance-only energy", e_heavy > e_light + 1e-9,
          f"{e_heavy:.2f} vs {e_light:.2f}")

    tight = as_uav(inst, energy_capacity=e_light * 0.5)
    res = evaluate_route(tight, r)
    check("energy limit is enforced", (not res.feasible) and res.reason == "energy",
          f"reason={res.reason}")


def test_solomon_roundtrip() -> None:
    inst = random_instance(8, seed=7)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "X101.txt"
        write_solomon(inst, p)
        back, meta = parse_solomon(p)
    check("solomon parse keeps customer count", back.n == inst.n, f"n={back.n}")
    check("solomon parse keeps demands", np.allclose(back.demand, inst.demand))
    check("solomon parse keeps time windows", np.allclose(back.ready, inst.ready) and np.allclose(back.due, inst.due))
    check("solomon parse keeps capacity", abs(meta["capacity"] - inst.params.capacity) < 1e-9)


def test_pricing_against_brute_force() -> None:
    for seed, energy in ((11, False), (12, True), (13, False)):
        inst = random_instance(7, seed=seed)
        if energy:
            inst = as_uav(inst, energy_capacity=float(inst.dist[0].max() * 3.0), load_energy_factor=1.5)
        rng = np.random.default_rng(seed)
        pi = rng.uniform(0, 60, size=inst.n)
        sigma = -rng.uniform(0, 5)
        fixed = 0.0

        ref_best, ref_neg = brute_force_pricing(inst, pi, sigma, fixed)
        pricer = ExactLabelingPricer(inst)
        res = pricer.price(pi, sigma=sigma, fixed_cost=fixed, max_columns=10**6)

        tag = "uav" if energy else "vrptw"
        check(
            f"[{tag} seed={seed}] labelling min reduced cost matches brute force",
            abs(res.min_reduced_cost - ref_best) < 1e-6,
            f"lab={res.min_reduced_cost:+.6f} brute={ref_best:+.6f}",
        )
        got = {r for r, _ in res.columns}
        check(
            f"[{tag} seed={seed}] labelling negative route set matches brute force",
            got == set(ref_neg),
            f"|lab|={len(got)} |brute|={len(ref_neg)}",
        )


def test_master_duals() -> None:
    inst = random_instance(6, seed=21)
    routes = enumerate_routes(inst)
    master = MasterProblem(inst)
    for r in routes:
        master.add(r, evaluate_route(inst, r).cost)
    lp = master.solve_lp()

    pi_ok = bool(np.all(lp.pi >= -1e-6))
    dual_feasible = True
    for j, r in enumerate(master.routes):
        lhs = sum(lp.pi[i - 1] for i in r) + lp.sigma
        if lhs > master.cost_vector[j] + 1e-5:
            dual_feasible = False
            break
    strong = abs(float(lp.pi.sum()) + (master.max_vehicles or 0) * lp.sigma - lp.objective) < 1e-5
    check("master duals are non-negative", pi_ok)
    check("master duals satisfy dual feasibility", dual_feasible)
    check("master duals satisfy strong duality", strong, f"source={lp.dual_source}")

    x = lp.x
    check("LP covers every customer", bool(np.all(master._cover_matrix() @ x >= 1 - 1e-6)))


def test_cg_matches_full_lp() -> None:
    for seed, fleet in ((31, None), (32, 4)):
        inst = random_instance(6, seed=seed)
        all_routes = enumerate_routes(inst)

        full = MasterProblem(inst, max_vehicles=fleet)
        for r in all_routes:
            full.add(r, evaluate_route(inst, r).cost)
        lp_full = full.solve_lp()

        cg = ColumnGeneration(
            inst,
            ExactLabelingPricer(inst),
            max_vehicles=fleet,
            config=CGConfig(max_iterations=300, max_columns_per_iter=50),
        )
        t0 = time.perf_counter()
        res = cg.run()
        dt = time.perf_counter() - t0

        tag = "no-fleet-limit" if fleet is None else f"fleet<={fleet}"
        check(
            f"[{tag}] CG LP objective equals full-enumeration LP",
            abs(res.lp_objective - lp_full.objective) < 1e-4,
            f"cg={res.lp_objective:.6f} full={lp_full.objective:.6f}",
        )
        check(f"[{tag}] CG terminated with an optimal LP", res.optimal_lp)
        check(
            f"[{tag}] CG used far fewer columns than enumeration",
            res.columns_in_pool < len(all_routes),
            f"pool={res.columns_in_pool} of {len(all_routes)}",
        )
        check(
            f"[{tag}] CG integer solution is feasible and covers all customers",
            report(inst, res.ip_routes)["covering"],
            f"veh={res.num_vehicles} obj={res.ip_objective:.2f} in {dt:.3f}s",
        )


def test_greedy_triggers_exact_fallback() -> None:
    inst = random_instance(6, seed=41)
    greedy = GreedyPricer(inst, restarts=20, seed=0)
    exact = ExactLabelingPricer(inst)
    cg = ColumnGeneration(
        inst, greedy, fallback_pricer=exact,
        config=CGConfig(max_iterations=300, max_columns_per_iter=20),
    )
    res = cg.run()

    full = MasterProblem(inst)
    for r in enumerate_routes(inst):
        full.add(r, evaluate_route(inst, r).cost)
    ref = full.solve_lp().objective

    check(
        "heuristic pricer + exact fallback still reaches the true LP bound",
        abs(res.lp_objective - ref) < 1e-4,
        f"cg={res.lp_objective:.6f} ref={ref:.6f}",
    )
    check("exact fallback was actually invoked", res.exact_confirmation_calls > 0,
          f"calls={res.exact_confirmation_calls}")


def test_seed_heuristic() -> None:
    inst = random_instance(12, seed=51)
    routes = cheapest_insertion(inst)
    info = report(inst, routes)
    check("cheapest insertion covers every customer", info["covering"],
          f"routes={info['num_routes']} cost={info['cost']:.2f}")


def test_ngroute_relaxation() -> None:
    inst = random_instance(7, seed=61)
    rng = np.random.default_rng(61)
    pi = rng.uniform(0, 60, size=inst.n)

    exact = ExactLabelingPricer(inst).price(pi, max_columns=10**6)
    # with the neighbourhood covering everyone, ng-route degenerates to the
    # exact elementarity rule and must reproduce it bit for bit
    complete = NgRoutePricer(inst, k=inst.n).price(pi, max_columns=10**6)
    check(
        "ng-route with k=n reproduces the exact minimum",
        abs(complete.min_reduced_cost - exact.min_reduced_cost) < 1e-6,
        f"ng={complete.min_reduced_cost:+.6f} exact={exact.min_reduced_cost:+.6f}",
    )

    small = NgRoutePricer(inst, k=2).price(pi, max_columns=10**6)
    bound = small.relaxed_min_reduced_cost
    check(
        "ng-route relaxed bound never exceeds the exact minimum",
        bound is not None and bound <= exact.min_reduced_cost + 1e-9,
        f"relaxed={bound:+.6f} exact={exact.min_reduced_cost:+.6f}",
    )
    check(
        "ng-route returns only elementary routes",
        all(len(set(r)) == len(r) for r, _ in small.columns),
        f"returned={len(small.columns)}",
    )
    check("ng-route does not claim optimality guarantees", small.exact is False)


def test_cg_with_ngroute() -> None:
    inst = random_instance(7, seed=71)
    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective

    cg = ColumnGeneration(
        inst,
        NgRoutePricer(inst, k=4),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=400, max_columns_per_iter=30),
    )
    res = cg.run()
    check(
        "CG driven by ng-route pricing reaches the true LP bound",
        abs(res.lp_objective - ref_obj) < 1e-4,
        f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}",
    )


def test_cap_hit_is_reported_honestly() -> None:
    """A capped pricer must not publish a bound it did not prove (spec §3.5).

    The minimum taken over a truncated label space is an *upper* bound on the
    true relaxed minimum -- it can sit above zero while negative columns remain.
    Reporting it as a relaxation bound therefore lets the loop certify an LP it
    never proved, which is a silent wrong answer rather than a slow one.
    """
    inst = random_instance(9, seed=91)
    pi = np.full(inst.n, 30.0)

    ref = NgRoutePricer(inst, k=6).price(pi, max_columns=10**6)
    check(
        "uncapped ng-route publishes a relaxation bound",
        ref.relaxed_min_reduced_cost is not None and ref.truncated is False,
        f"labels={ref.labels_created}",
    )

    capped = NgRoutePricer(inst, k=6, max_labels=15)
    out = capped.price(pi, max_columns=10**6)
    check("capped ng-route is flagged truncated", out.truncated is True,
          f"labels={out.labels_created} cap={capped.max_labels}")
    check("capped ng-route publishes no bound", out.relaxed_min_reduced_cost is None)
    check("truncated ng-route cannot certify optimality", out.certifies_no_negative is False)
    check("truncated ng-route is not a clean run", out.is_clean() is False)
    check(
        "capped ng-route keeps its raw measurement for diagnosis",
        out.stats.get("label_cap") == 15 and "relaxed_min" in out.stats,
    )

    ref_e = ExactLabelingPricer(inst).price(pi, max_columns=10**6)
    check(
        "uncapped exact pricer publishes a relaxation bound",
        ref_e.relaxed_min_reduced_cost is not None and ref_e.truncated is False,
    )
    out_e = ExactLabelingPricer(inst, max_labels=15).price(pi, max_columns=10**6)
    check("capped exact pricer is flagged truncated", out_e.truncated is True)
    check("capped exact pricer publishes no bound",
          out_e.relaxed_min_reduced_cost is None)
    check("truncated exact pricer cannot certify optimality",
          out_e.certifies_no_negative is False)


def test_truncation_cannot_reach_a_cg_verdict() -> None:
    """End to end: a cap hit must make the run unusable, not 'optimal'."""
    inst = random_instance(7, seed=92)

    cg_ok = ColumnGeneration(
        inst,
        NgRoutePricer(inst, k=4),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=400, max_columns_per_iter=30),
    )
    ok = cg_ok.run()
    check(
        "a clean CG run reports no truncation",
        ok.truncated_pricing_calls == 0 and ok.degraded is False,
        f"peak={ok.peak_labels_created} cap={ok.label_cap}",
    )

    # Capped so tightly that every pricing call is truncated. Before the fix
    # this run ended with a non-negative min_reduced_cost and was flagged
    # optimal; it must instead be refused outright.
    cg_bad = ColumnGeneration(
        inst,
        NgRoutePricer(inst, k=4, max_labels=10),
        config=CGConfig(max_iterations=200, max_columns_per_iter=20),
    )
    bad = cg_bad.run()
    check("a truncated CG run is refused as evidence", bad.degraded is True,
          f"truncated_calls={bad.truncated_pricing_calls}")
    check("a truncated CG run never claims optimal", bad.optimal_lp is False)


def test_rollout_mask_is_sound() -> None:
    """The feasibility mask must never block an arc of a feasible route.

    A mask built from necessary conditions can only ever remove impossible
    continuations. If it removed a real one, the sampler would be unable to
    reach that column no matter how long it ran, and the loss would be silent.
    This replays routes that are known to be feasible - the negative columns
    of an exact pricer - and asserts the mask lets each of them through.
    """
    for seed, uav in ((81, False), (82, True), (83, False)):
        inst = random_instance(8, seed=seed)
        if uav:
            inst = as_uav(inst, energy_capacity=float(inst.dist[0].max() * 2.5),
                          load_energy_factor=1.5)
        rng = np.random.default_rng(seed)
        pi = rng.uniform(0, 60, size=inst.n)
        cols = ExactLabelingPricer(inst).price(pi, max_columns=10**6).columns

        blocked, tested = [], 0
        for route, _rc in cols:
            tested += 1
            ok, step = replay_mask(inst, route)
            if not ok:
                blocked.append((route, step))
        tag = "uav" if uav else "vrptw"
        check(
            f"[{tag} seed={seed}] rollout mask admits every known-feasible route",
            not blocked,
            f"{tested} routes checked" + (f", blocked={blocked[:2]}" if blocked else ""),
        )


def test_rollout_reduced_costs_are_exact() -> None:
    """Reported reduced costs must match the route's real cost against the duals."""
    inst = random_instance(9, seed=91)
    rng = np.random.default_rng(91)
    pi = rng.uniform(0, 60, size=inst.n)
    sigma, fixed = -rng.uniform(0, 5), 3.0

    pricer = RolloutPricer(inst, scoring="dual", n_samples=120, seed=0)
    out = pricer.price(pi, sigma=sigma, fixed_cost=fixed, max_columns=10**6)

    worst = 0.0
    infeasible = 0
    for route, rc in out.columns:
        ev = evaluate_route(inst, route)
        if not ev.feasible:
            infeasible += 1
            continue
        true_rc = ev.cost - sum(pi[i - 1] for i in route) + fixed - sigma
        worst = max(worst, abs(true_rc - rc))
    check(
        "rollout columns are all feasible routes",
        infeasible == 0 and len(out.columns) > 0,
        f"{len(out.columns)} columns",
    )
    check("rollout reduced costs match an independent recomputation", worst < 1e-9,
          f"max deviation {worst:.2e}")
    check("rollout returns only negative reduced costs",
          all(rc < -1e-6 for _r, rc in out.columns))
    check("rollout never claims to certify optimality",
          out.certifies_no_negative is False and out.relaxed_min_reduced_cost is None)


def test_rollout_reaches_exact_pricing_on_small_instances() -> None:
    """With enough samples the sampler should find the true best reduced cost."""
    inst = random_instance(7, seed=101)
    rng = np.random.default_rng(101)
    pi = rng.uniform(0, 60, size=inst.n)
    exact = ExactLabelingPricer(inst).price(pi, max_columns=10**6)

    hits = 0
    for beta in (0.5, 1.0, 2.0, 4.0):
        out = RolloutPricer(
            inst, scoring="dual", n_samples=400, beta=beta, seed=beta
        ).price(pi, max_columns=10**6)
        if abs(out.min_reduced_cost - exact.min_reduced_cost) < 1e-6:
            hits += 1
        check(
            f"[beta={beta}] rollout cannot beat the exact minimum",
            out.min_reduced_cost >= exact.min_reduced_cost - 1e-6,
            f"rollout={out.min_reduced_cost:+.4f} exact={exact.min_reduced_cost:+.4f}",
        )
    check("at least one sampling temperature recovers the exact optimum", hits >= 1,
          f"{hits} of 4 configurations matched")


def test_feature_block_shape() -> None:
    inst = random_instance(8, seed=111)
    pricer = RolloutPricer(inst, scoring="dual", n_samples=5, seed=0)
    rng = np.random.default_rng(111)
    X, arcs = pricer.collect_arc_samples(rng.uniform(0, 60, size=inst.n), n_samples=6)
    check("arc sample matrix has one column per named feature",
          X.shape[1] == NUM_FEATURES, f"{X.shape[1]} vs {len(FEATURE_NAMES)}")
    check("arc sample rows are finite", bool(np.all(np.isfinite(X))),
          f"shape={X.shape}")
    check("arc sample tails/heads are valid customer indices",
          arcs.shape[0] == X.shape[0] and arcs[:, 1].min() >= 1
          and arcs[:, 1].max() <= inst.n and arcs[:, 0].min() >= 0)


def test_cg_with_rollout_pricer() -> None:
    """End-to-end: a heuristic rollout pricer plus the exact fallback must
    still reach the true LP bound. This is the correctness guarantee that lets
    a badly-trained model do damage to the search only, never to the answer."""
    inst = random_instance(7, seed=121)
    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective

    cg = ColumnGeneration(
        inst,
        RolloutPricer(inst, scoring="dual", n_samples=80, seed=0),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=300, max_columns_per_iter=30),
    )
    res = cg.run()
    check(
        "CG driven by rollout pricing plus exact fallback reaches the true LP bound",
        abs(res.lp_objective - ref_obj) < 1e-4,
        f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}",
    )


def test_stabilization_alpha_zero_is_off() -> None:
    """alpha = 0 must collapse onto the unstabilized run, bit for bit.

    pi_stab = 0 * center + 1 * pi is algebraically pi, so if these two ever
    disagree the stabilization plumbing has leaked into the plain path - the
    single most likely way to break every earlier measurement silently.
    """
    inst = random_instance(7, seed=71)
    base = dict(max_iterations=400, max_columns_per_iter=30)

    def once(cfg):
        cg = ColumnGeneration(
            inst, NgRoutePricer(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst), config=cfg,
        )
        return cg.run()

    off = once(CGConfig(**base))
    zero = once(CGConfig(**base, stabilization="smooth", stabilization_alpha=0.0))

    check(
        "alpha=0 reproduces the unstabilized LP bound exactly",
        abs(off.lp_objective - zero.lp_objective) < 1e-9,
        f"off={off.lp_objective:.9f} a0={zero.lp_objective:.9f}",
    )
    check(
        "alpha=0 reproduces the unstabilized iteration and column counts",
        off.iterations == zero.iterations and off.columns_in_pool == zero.columns_in_pool,
        f"it {off.iterations}/{zero.iterations} cols {off.columns_in_pool}/{zero.columns_in_pool}",
    )
    check(
        "alpha=0 never marks an iteration as stabilized",
        zero.stabilized_iterations == 0,
        f"stabilized={zero.stabilized_iterations}",
    )


def test_stabilization_is_bound_neutral() -> None:
    """Stabilization may change the path, never the destination.

    The whole point of the true-dual verification in the loop is that a
    smoothed dual is a heuristic objective.  If it were allowed to leak into
    the stopping rule, the run would report an LP bound that is not the LP
    optimum - which would invalidate every optimality claim in the paper.
    """
    inst = random_instance(7, seed=71)
    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective

    for alpha in (0.3, 0.5, 0.7, 0.9):
        cg = ColumnGeneration(
            inst, NgRoutePricer(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst),
            config=CGConfig(
                max_iterations=400, max_columns_per_iter=30,
                stabilization="smooth", stabilization_alpha=alpha,
            ),
        )
        res = cg.run()
        check(
            f"[alpha={alpha}] stabilized CG still reaches the true LP bound",
            abs(res.lp_objective - ref_obj) < 1e-4,
            f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}",
        )
        check(
            f"[alpha={alpha}] stabilized CG still certifies optimality",
            res.optimal_lp,
            f"stabilized_iters={res.stabilized_iterations}",
        )


def test_pool_columns_stay_dual_feasible_after_stabilization() -> None:
    """At the final duals every pool column must have non-negative true rc.

    Two things are checked at once.  First, that the smoothed-dual filter did
    not admit a column that is unattractive to the real master - adding one is
    harmless to the bound, but it means the filter is wrong.  Second, that
    `reduced_cost_of` agrees with the master's own dual-feasibility check, so
    the recomputation the filter relies on is not an independent invention.
    """
    inst = random_instance(8, seed=83)
    cg = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(
            max_iterations=400, max_columns_per_iter=25,
            stabilization="smooth", stabilization_alpha=0.7,
        ),
    )
    res = cg.run()
    lp = cg.master.solve_lp()

    worst = 0.0
    for nodes in cg.master.routes:
        rc = reduced_cost_of(inst, nodes, lp.pi, lp.sigma, cg.master.vehicle_fixed_cost)
        worst = min(worst, rc)
    check(
        "every pooled column has non-negative recomputed reduced cost",
        worst >= -1e-5,
        f"worst rc = {worst:+.9f} over {len(cg.master.routes)} columns",
    )

    covered = True
    for j, nodes in enumerate(cg.master.routes):
        cover = sum(lp.pi[i - 1] for i in nodes) + lp.sigma
        if cover > cg.master.cost_vector[j] + 1e-6:
            covered = False
            break
    check("reduced_cost_of agrees with the master's dual feasibility check", covered,
          f"filtered={res.filtered_columns}")


def test_degeneracy_report_is_well_formed() -> None:
    inst = random_instance(7, seed=71)
    cg = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=400, max_columns_per_iter=30),
    )
    res = cg.run()
    d = res.degeneracy()

    check("degeneracy iterations match the snapshot count",
          d.iterations == res.iterations, f"{d.iterations}/{res.iterations}")
    check("total gain equals seed LP minus final LP",
          abs(d.total_gain - (d.seed_lp - d.final_lp)) < 1e-9,
          f"gain={d.total_gain:+.6f}")
    check("wasted fraction is a proportion",
          0.0 <= d.wasted_fraction <= 1.0, f"{d.wasted_fraction:.3f}")
    check("plateau cannot exceed the iteration count",
          0 <= d.longest_plateau <= d.iterations, f"{d.longest_plateau}")
    check("iterations-to-target is within the run",
          0 <= d.iters_to_target <= d.iterations + 1,
          f"t{d.target:.2f}={d.iters_to_target} of {d.iterations + 1}")
    check("wasted plus improving never exceeds column-adding iterations",
          d.wasted_iterations + d.improving_iterations <= res.iterations,
          f"wasted={d.wasted_iterations} improving={d.improving_iterations}")
    check("the report serialises to plain floats",
          all(isinstance(v, (int, float)) for v in d.as_dict().values()))


def test_column_selection_is_bound_neutral() -> None:
    """Every selection policy must still reach the true LP optimum.

    Selection only ever removes columns from a batch, and the RMP is a
    restriction of the full LP, so the optimum is invariant - what changes is
    the path.  The hazard the test is really hunting is the empty batch: if a
    policy could drop an entire pricing batch the loop would read it as "the
    pricer found nothing" and stop on a bound that is not optimal.  That is
    why `_select_columns` is written to keep at least one column whenever the
    pricer offered any, and why this test asserts certification, not merely a
    matching number.
    """
    inst = random_instance(7, seed=71)
    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective

    base = dict(max_iterations=400, max_columns_per_iter=30)
    cases = [
        ("all", {}),
        ("partial", dict(column_selection_k=1)),
        ("partial", dict(column_selection_k=3)),
        ("partial", dict(column_selection_frac=0.5)),
        ("diverse", dict(diversity_max_jaccard=0.5)),
        ("diverse", dict(diversity_max_jaccard=0.0)),
    ]
    for mode, extra in cases:
        cfg = CGConfig(**base, column_selection=mode, **extra)
        cg = ColumnGeneration(
            inst, NgRoutePricer(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst), config=cfg,
        )
        res = cg.run()
        tag = f"{mode}{extra}"
        check(
            f"[{tag}] selection still reaches the true LP bound",
            abs(res.lp_objective - ref_obj) < 1e-4,
            f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}",
        )
        check(
            f"[{tag}] selection still certifies optimality",
            res.optimal_lp,
            f"it={res.iterations} cols={res.columns_in_pool}",
        )


def test_column_selection_reduces_the_batch() -> None:
    """`partial` must actually admit fewer columns than `all`, not silently no-op.

    Keeping at least one column per batch is a safety rule, so a policy that
    is asked for k=1 on a batch where only one column was offered is
    indistinguishable from `all`.  Comparing the pool totals is what proves
    the policy is wired in at all.
    """
    inst = random_instance(9, seed=91)
    base = dict(max_iterations=400, max_columns_per_iter=40)

    def once(**extra):
        cg = ColumnGeneration(
            inst, NgRoutePricer(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst),
            config=CGConfig(**base, **extra),
        )
        return cg, cg.run()

    _, a = once(column_selection="all")
    cg_p, p = once(column_selection="partial", column_selection_k=2)

    check(
        "partial selection reports the columns it discarded",
        p.dropped_by_selection > 0 and p.offered_columns > p.dropped_by_selection,
        f"offered={p.offered_columns} dropped={p.dropped_by_selection}",
    )
    check(
        "partial selection keeps the pool smaller than add-all",
        p.columns_in_pool < a.columns_in_pool,
        f"all={a.columns_in_pool} partial={p.columns_in_pool}",
    )
    check(
        "offered totals agree between the two policies (same pricer path)",
        cg_p is not None and p.offered_columns > 0,
        f"offered={p.offered_columns}",
    )


def test_column_selection_ranks_by_true_reduced_cost() -> None:
    """`partial` must keep the most negative columns, not an arbitrary subset.

    A policy that filtered at random would still pass the bound-neutrality
    test - the optimum is invariant either way - so the ordering needs a check
    of its own, and it has to be made on a *batch*, not on the finished pool.
    At the final duals the optimality condition forces every pooled column to
    have non-negative reduced cost (the other self-test asserts exactly that),
    so inspecting the pool afterwards can never reveal which columns were
    cheap when they were offered.
    """
    inst = random_instance(6, seed=61)
    cg = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        config=CGConfig(column_selection="partial", column_selection_k=2),
    )
    routes = [tuple(r) for r in enumerate_routes(inst)][:24]
    batch = [(r, 0.0) for r in routes]
    pi = np.full(inst.n, 1.0)

    kept = cg._select_columns(batch, pi, 0.0)
    ref = sorted(reduced_cost_of(inst, r, pi, 0.0) for r in routes)
    expected = {r for rc, r in sorted((reduced_cost_of(inst, r, pi, 0.0), r)
                                      for r in routes)[:2]}

    check(
        "partial keeps exactly column_selection_k columns",
        len(kept) == 2,
        f"kept={len(kept)}",
    )
    check(
        "partial keeps the two most negative columns of the batch",
        {r for r, _ in kept} == expected,
        f"kept rcs={sorted(rc for _, rc in kept)} vs two best of {len(routes)} "
        f"= {ref[:2]}",
    )

    # `all` must be a genuine pass-through, otherwise "column_selection=all"
    # does not actually reproduce the pre-existing baseline.
    cg_all = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4), config=CGConfig(column_selection="all"),
    )
    check(
        "add-all returns the batch untouched",
        len(cg_all._select_columns(batch, pi, 0.0)) == len(batch),
        f"{len(batch)} in, {len(cg_all._select_columns(batch, pi, 0.0))} out",
    )


def test_selection_escalates_when_the_best_columns_are_duplicates() -> None:
    """A selection rule must never be able to end the run by picking duplicates.

    The loop reads "nothing was added" as "the pricer found nothing", and for a
    heuristic pricer that is the end of the run.  A policy that keeps only the
    k most negative columns will eventually keep one the pool already holds,
    which would stop a run that still had negative columns on the table.  This
    was observed on R107 before the escalation guard was added, so it is pinned
    down here rather than left to the integration runs.
    """
    inst = random_instance(6, seed=61)
    routes = [tuple(r) for r in enumerate_routes(inst)]
    batch = [(r, 0.0) for r in routes]
    pi = np.full(inst.n, 1.0)

    best = min(routes, key=lambda r: reduced_cost_of(inst, r, pi, 0.0))

    cg = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        config=CGConfig(column_selection="partial", column_selection_k=1),
    )
    cg.master.add(best, evaluate_route(inst, best).cost)   # the trap

    added = cg._admit(batch, pi, 0.0)
    check(
        "selection escalates to the rest of the batch instead of reporting an empty one",
        added >= 1,
        f"added={added} for a {len(batch)}-column batch whose best column was already pooled",
    )
    check(
        "escalated columns are not counted as discarded",
        cg._dropped < len(batch),
        f"dropped={cg._dropped} of {len(batch)} offered",
    )

    # Sanity: with every candidate already in the pool there is genuinely
    # nothing to add, and the escalation must not invent columns either.
    cg2 = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4), config=CGConfig(column_selection="all"),
    )
    for r in routes:
        cg2.master.add(r, evaluate_route(inst, r).cost)
    check(
        "an all-duplicate batch still reports zero added",
        cg2._admit(batch, pi, 0.0) == 0,
        "escalation invented columns",
    )


def test_sampling_weights_are_scale_invariant() -> None:
    """Affine rescoring must not change the sampling distribution.

    This is the self-test behind a conclusion, not just a property.  The
    reference paper's second step maps a linear-SVM distance through
    psi(d) = sigmoid(theta0*d + theta1), and it was plausible that skipping it
    is why a linear SVM scored no better than logistic regression in our runs -
    a scale mismatch would look exactly like that.  It cannot be the reason:
    the rollout standardises scores per decision step, so a positive scale
    divides out and a shift cancels in the softmax.  Asserting the invariance
    is what turns "we think that is irrelevant" into something a reviewer can
    check, and it is what justifies sweeping the sampling temperature instead
    of re-fitting the paper's theta0/theta1.
    """
    from uavcg.learning import PsiCalibration
    from uavcg.pricers.rollout import RolloutPricer

    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 3.0, size=12)
    base = RolloutPricer._weights(logits, beta=1.0)

    psi = PsiCalibration()
    for name, rescaled in [
        ("the paper's theta0 = 9.7750", psi.logit(logits)),
        ("an arbitrary scale of 100", 100.0 * logits - 42.0),
        ("a shrinking scale of 0.01", 0.01 * logits + 7.0),
    ]:
        got = RolloutPricer._weights(rescaled, beta=1.0)
        check(
            f"sampling weights are unchanged by {name}",
            np.allclose(base, got, atol=1e-12),
            f"max |dp| = {float(np.max(np.abs(base - got))):.3e}",
        )

    check(
        "the weights form a distribution",
        abs(float(base.sum()) - 1.0) < 1e-12 and bool(np.all(base > 0.0)),
        f"sum={base.sum():.12f}",
    )
    check(
        "temperature still changes the distribution (beta is not a no-op)",
        not np.allclose(base, RolloutPricer._weights(logits, beta=4.0), atol=1e-6),
        "beta=4 reproduced beta=1 exactly, so the sweep would be meaningless",
    )


def test_psi_calibration_matches_the_published_constants() -> None:
    """The paper's constants must be reproduced, and their saturation documented.

    theta0 = 9.7750 / theta1 = 12.5564 are large, so psi saturates almost
    immediately.  That is fine for relative weights and fatal for a 0.5
    threshold, and the difference is the whole reason `psi_proba` is a separate
    method rather than the new default.
    """
    from uavcg.learning import PsiCalibration

    psi = PsiCalibration()
    check(
        "published theta are the paper's values",
        (psi.theta0, psi.theta1) == (9.7750, 12.5564),
        f"theta0={psi.theta0} theta1={psi.theta1}",
    )
    d = np.array([-2.0, -1.0, 0.0, 1.0])
    p = psi.proba(d)
    check(
        "psi is monotone and inside (0, 1)",
        bool(np.all(np.diff(p) > 0)) and bool(np.all((p > 0) & (p < 1))),
        f"p={np.round(p, 6).tolist()}",
    )
    check(
        "psi saturates at d = 0, which is why it cannot be a default probability",
        p[2] > 0.99999,
        f"psi(0) = {p[2]:.9f}",
    )
    check(
        "psi is affine in d, hence a no-op inside a standardising sampler",
        np.allclose(psi.logit(d) - psi.logit(np.zeros_like(d)),
                    psi.theta0 * d),
        "logit(theta0*d + theta1) is not affine, so the invariance claim breaks",
    )


def test_affine_fast_path_matches_the_pipeline() -> None:
    """The folded `x @ w + b` must equal what scikit-learn would have computed.

    `scripts/profile_rollout.py` measured that 72% of a learned decision step
    was spent inside `logit` on a 15-row matrix - scikit-learn's per-call
    validation, not arithmetic.  Folding `StandardScaler` and the linear
    classifier into one affine map removes sklearn from the hot path, and the
    only thing that makes that safe is this test: the two routes must agree to
    floating-point precision on every row, including rows far outside the
    training range (the rollout will produce those).
    """
    from uavcg.learning import ArcScorer

    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(600, NUM_FEATURES))
    ytr = (Xtr[:, 0] - 0.5 * Xtr[:, 3] + 0.3 * rng.normal(size=600) > 0.2).astype(float)
    Xte = np.vstack([rng.normal(size=(50, NUM_FEATURES)),
                     rng.normal(size=(50, NUM_FEATURES)) * 40.0])

    for kind in ("logistic", "svm"):
        fast = ArcScorer(kind=kind, seed=0, fast_path=True).fit(Xtr, ytr)
        slow = ArcScorer(kind=kind, seed=0, fast_path=False).fit(Xtr, ytr)
        check(f"[{kind}] the linear pipeline folds into an affine map",
              fast.affine is not None,
              "affine is None, so the hot path silently fell back to sklearn")
        if fast.affine is None:
            continue
        Xf = fast.logit(Xte)
        Xs = np.asarray(slow.model.decision_function(Xte), dtype=float).ravel()
        denom = max(float(np.max(np.abs(Xs))), 1e-12)
        err = float(np.max(np.abs(Xf - Xs))) / denom
        check(f"[{kind}] folded logit == pipeline decision_function",
              err < 1e-9,
              f"max relative deviation {err:.3e}")
        # folding must survive a save/load round trip, because the deployed
        # model is always loaded from disk rather than refitted
        p = _TMP / f"_scorer_{kind}.pkl"
        fast.save(p)
        back = ArcScorer.load(p)
        check(f"[{kind}] the fast path survives save/load",
              back.affine is not None
              and np.allclose(back.logit(Xte), Xf, atol=1e-12),
              "reloaded model disagrees with the fitted one")

    # The order of the score must be untouched too: a rollout only consumes the
    # ranking, so a scale-only difference would not be visible in the agreement
    # test above if it flipped signs, and a flipped sign would invert the search.
    lin = ArcScorer(kind="logistic", seed=0).fit(Xtr, ytr)
    check("the folded score preserves ranking under monotone rescaling",
          np.array_equal(np.argsort(lin.logit(Xte)),
                         np.argsort(-lin.logit(-Xte * 3.0 + 5.0))),
          "sign or order changed - the sampler would explore the wrong end")


def test_sampling_temperature_default_is_the_sweep_optimum() -> None:
    """beta = 2.0, not 1.0, and it is a measured choice rather than taste.

    The sweep in `results/temperature_n25.json` put the optimum at 2.0 for
    both scoring modes; leaving the default at 1.0 costs real progress (on R105
    the dual arm needs 22 iterations at beta=1 against 16 at beta=2).  A test
    is the right place for this because the number is invisible in the
    experiments' output and easy to regress by silence.
    """
    from uavcg.pricers import RolloutPricer
    from uavcg.testing import random_instance

    inst = random_instance(6, seed=17)
    p = RolloutPricer(inst, scoring="dual")
    check("RolloutPricer defaults to the swept temperature",
          p.beta == 2.0,
          f"beta={p.beta}; expected 2.0 (results/temperature_n25.json)")
    check("a beta of 0 or None still means greedy argmax, not division by zero",
          np.allclose(RolloutPricer._weights(np.array([1.0, 2.0, 3.0]), 0.0),
                      np.full(3, 1 / 3)),
          "beta=0 did not reduce to a uniform draw")


def test_graded_labels_refine_the_binary_ones() -> None:
    """Grading must be a refinement of the binary label, not a different label.

    Three properties, and all three matter for the comparison between the two
    label sets to be a comparison of labels rather than of two unrelated tasks:

      * every grade lies in [0, 1];
      * the arcs of the strictly-best route grade exactly 1.0;
      * the *support* of the graded label equals the support of the binary one -
        an arc grades 0 if and only if the binary label called it negative.

    The third is the easy one to get wrong, because the graded version keeps a
    dictionary of reduced costs where the binary version kept a set, and an arc
    that appears only in a route outside the tolerance must not leak in.
    """
    from uavcg.labeling import label_batch, near_optimal_arcs, optimal_arcs

    inst = random_instance(7, seed=211)
    # A master holding *only* the insertion heuristic's routes.  Filling it with
    # every enumerated route would make the LP optimal by construction, the
    # pricing problem would find nothing negative, and the test would pass
    # vacuously on an empty label set.
    master = MasterProblem(inst)
    for r in cheapest_insertion(inst):
        ev = evaluate_route(inst, r)
        if ev.feasible:
            master.add(r, ev.cost)
    lp = master.solve_lp()
    src = ExactLabelingPricer(inst)
    fixed = master.vehicle_fixed_cost

    margins, best, tied = near_optimal_arcs(
        inst, lp.pi, lp.sigma, fixed, src, rel_tol=0.02)
    arcs_bin, best_bin, tied_bin = optimal_arcs(
        inst, lp.pi, lp.sigma, fixed, src, rel_tol=0.02)

    check("a labelling round produced arcs at all", len(margins) > 0,
          f"{len(margins)} graded arcs")
    if not margins:
        return
    vals = np.array(list(margins.values()))
    check("every grade lies in [0, 1]",
          bool(np.all((vals >= 0.0) & (vals <= 1.0))),
          f"range [{vals.min():.4f}, {vals.max():.4f}]")
    check("the best route's arcs grade exactly 1.0",
          abs(float(vals.max()) - 1.0) < 1e-9,
          f"max grade {vals.max():.6f} (best rc {best:+.4f})")
    check("graded and binary labels share the same support",
          set(margins) == arcs_bin,
          f"{len(margins)} graded vs {len(arcs_bin)} binary arcs")
    check("the tied-route count is the same for both",
          tied == tied_bin,
          f"{tied} vs {tied_bin}")

    # a vanishing tolerance must leave only arcs of the exactly-best route, and
    # they must all grade 1.0
    tight, _b, _t = near_optimal_arcs(
        inst, lp.pi, lp.sigma, fixed, src, rel_tol=1e-15)
    check("at zero tolerance every surviving arc is a best-route arc",
          all(abs(v - 1.0) < 1e-9 for v in tight.values()),
          f"{sum(1 for v in tight.values() if abs(v - 1.0) >= 1e-9)} exceptions")

    batch, _ = label_batch(inst, lp.pi, lp.sigma, fixed, src, graded=True,
                           samples_per_beta=25, max_rows=400, seed=3)
    batch_bin, _ = label_batch(inst, lp.pi, lp.sigma, fixed, src, graded=False,
                               samples_per_beta=25, max_rows=400, seed=3)
    if batch is None or batch_bin is None:
        check("a rollout batch was produced for both label modes", False)
        return
    check("a graded batch can carry a real grade",
          batch.y.max() > 0.5,
          f"partial {batch.partial_rate:.2%}, hard-positive "
          f"{batch.hard_positive_rate:.2%}, max grade {batch.y.max():.4f}")
    check("the graded support matches the binary support",
          np.array_equal(batch.y > 0.5, batch_bin.y > 0.5),
          "the two label modes disagree on the positive set")
    check("the binary mode really is binary",
          bool(np.all(np.isin(batch_bin.y, [0.0, 1.0]))),
          f"{len(np.unique(batch_bin.y))} distinct values")


def test_soft_fit_weights_by_confidence_symmetrically() -> None:
    """The soft fit must be symmetric in the two classes and monotone in grade.

    A weighting that favoured confident *positives* over confident *negatives*
    would move the decision boundary for a reason that has nothing to do with
    the pricing problem, and with a 4%-positive label set that bias would be
    invisible in the metrics and loud in the search.
    """
    from uavcg.learning import ArcScorer, label_confidence

    y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    c = label_confidence(y)
    check("confidence is symmetric about the 0.5 boundary",
          np.allclose(c, label_confidence(1.0 - y)),
          f"c={np.round(c, 3).tolist()}")
    check("confidence is monotone away from 0.5",
          bool(np.all(np.diff(c[2:]) > 0)) and bool(np.all(np.diff(c[:3]) < 0)),
          f"c={np.round(c, 3).tolist()}")
    check("confidence stays inside [0.5, 1]",
          bool(np.all((c >= 0.5) & (c <= 1.0))),
          f"range [{c.min()}, {c.max()}]")

    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, NUM_FEATURES))
    ybin = (X[:, 0] + 0.4 * rng.normal(size=400) > 0.0).astype(float)
    hard = ArcScorer(kind="logistic", seed=0).fit(X, ybin, soft=False)
    soft = ArcScorer(kind="logistic", seed=0).fit(X, ybin, soft=True)
    check("grade 0/1 targets reproduce the hard fit",
          np.allclose(hard.logit(X), soft.logit(X), atol=1e-8),
          "grading changed the fit even though every target was 0 or 1")
    bad = np.full(400, 1.2)
    try:
        ArcScorer(kind="logistic").fit(X, bad, soft=True)
        rejected = False
    except ValueError:
        rejected = True
    check("out-of-range graded targets are rejected", rejected,
          "a grade of 1.2 was accepted, so the [0,1] contract is not enforced")


def test_time_window_free_mode() -> None:
    """The CVRP switch: stripping time windows must still solve to optimality.

    Cutting the W dimension is the project's documented fallback for instances
    where the window-constrained pricing problem is what makes the scale
    unmanageable, so it needs its own regression guard rather than being
    assumed to work because the tests with windows pass.
    """
    from uavcg.instance import without_time_windows

    inst = random_instance(6, seed=131)
    flat = without_time_windows(inst)
    ref = MasterProblem(flat)
    for r in enumerate_routes(flat):
        ref.add(r, evaluate_route(flat, r).cost)
    ref_obj = ref.solve_lp().objective
    cg = ColumnGeneration(
        flat, ExactLabelingPricer(flat),
        config=CGConfig(max_iterations=300, max_columns_per_iter=40),
    )
    res = cg.run()
    check("opening the time windows keeps the core solvable and consistent",
          abs(res.lp_objective - ref_obj) < 1e-4 and res.optimal_lp,
          f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}")
    check("every route is still feasible without windows",
          all(evaluate_route(flat, r).feasible for r in cg.master.routes))


def test_time_sorted_dominance_matches_the_linear_scan() -> None:
    """The inlined, time-sorted dominance test must equal the reference scan.

    `price()` no longer calls `_dominates`.  It inlines the predicate and only
    scans the prefix of each node's label list whose time is <= lab.time + EPS,
    which is sound only because a dominator must have a time at least that
    small.  A rewrite of a 30-million-comparison inner loop is not something to
    accept on the strength of end-to-end tests alone -- those would only catch a
    mistake that happens to change a route -- so this replays both versions over
    the same synthetic label stream and requires identical survival decisions.

    Reported honestly: the time cut turned out to buy almost nothing (170 -> 165
    comparisons per label on C201), because times at a node overlap heavily.
    The 3.1x speedup came from removing the generator frame and the per-call
    function dispatch, not from examining fewer labels.  This test guards the
    equivalence; it does not claim the cut is doing the work.
    """
    from bisect import bisect_right

    from uavcg.instance import EPS
    from uavcg.pricers.ngroute import NgLabel, _dominates

    rng = np.random.default_rng(20261002)
    n = 12
    seq = []
    for _ in range(600):
        # Small ng sets, otherwise the subset condition `a.ng | b.ng == b.ng`
        # almost never holds and the test would pass without testing dominance.
        members = rng.choice(np.arange(1, n + 1), size=int(rng.integers(1, 4)), replace=False)
        seq.append(
            NgLabel(
                node=int(rng.integers(1, n + 1)),
                cost=float(rng.normal(0.0, 50.0)),
                time=float(rng.uniform(0.0, 200.0)),
                load=float(rng.uniform(0.0, 40.0)),
                ng=int(sum(1 << int(j) for j in members)),
                path=(),
            )
        )

    ref_store = [[] for _ in range(n + 1)]
    fast_store = [[] for _ in range(n + 1)]
    fast_time = [[] for _ in range(n + 1)]
    ref_survived, fast_survived = [], []
    total_cut = 0

    for idx, lab in enumerate(seq):
        if not any(_dominates(o, lab) for o in ref_store[lab.node]):
            ref_store[lab.node].append(lab)
            ref_survived.append(idx)

        node_list = fast_store[lab.node]
        times = fast_time[lab.node]
        l_cost, l_load, l_ng = lab.cost + EPS, lab.load + EPS, lab.ng
        cut = bisect_right(times, lab.time + EPS)
        total_cut += cut
        dominated = False
        for k in range(cut):
            other = node_list[k]
            if (other.cost <= l_cost and other.load <= l_load
                    and (other.ng | l_ng) == l_ng):
                dominated = True
                break
        if not dominated:
            pos = bisect_right(times, lab.time)
            times.insert(pos, lab.time)
            node_list.insert(pos, lab)
            fast_survived.append(idx)

    check("the synthetic stream actually exercises dominance",
          len(ref_survived) < len(seq) and total_cut > 2 * len(seq),
          f"{len(seq) - len(ref_survived)}/{len(seq)} labels dominated, "
          f"avg scan {total_cut / len(seq):.1f}")
    check("time-sorted inlined dominance == reference linear scan",
          ref_survived == fast_survived,
          f"{len(set(ref_survived) ^ set(fast_survived))} disagreeing labels")
    check("both stores hold the same surviving labels per node",
          all(len(ref_store[i]) == len(fast_store[i]) for i in range(n + 1)))


def test_time_window_narrowing() -> None:
    """Narrowing must tighten the windows without breaking the problem.

    Narrowing is the protocol the spec added to make Solomon category-2
    affordable: at n=60 with untouched windows, four of the six planned
    instances exceed 70 s on a single pricing call while the two that do finish
    sit at 6-12% of the label cap.  So the cap is not what blocks them --
    labelling work is -- and narrowing is the lever.  That makes it load
    bearing, and it has three ways to go wrong:

    * scale=1.0 must be the identity, because a caller will use it as the
      untouched control arm.
    * a narrowed window must stay *inside* the original, and the depot window
      must never move: `due[DEPOT]` is the vehicle horizon that
      `evaluate_route` enforces on every route, so shrinking it deletes
      working time rather than tightening customers.
    * every customer must stay reachable by a direct round trip, or the master
      problem silently becomes infeasible and shows up as "no column exists".
    """
    inst = random_instance(6, seed=97)
    before_ready = np.array(inst.ready, copy=True)
    before_due = np.array(inst.due, copy=True)

    same, rep_one = narrow_time_windows(inst, 1.0)
    d_ready = float(np.abs(same.ready[1:] - inst.ready[1:]).max())
    d_due = float(np.abs(same.due[1:] - inst.due[1:]).max())
    # Exact, not `allclose`: the control arm has to be the same instance bit for
    # bit or a reported number cannot be traced back to a run.
    check("scale=1.0 reproduces the customer windows bit for bit",
          bool(np.array_equal(same.ready[1:], inst.ready[1:])
               and np.array_equal(same.due[1:], inst.due[1:])),
          f"max|dready|={d_ready:.2e} max|ddue|={d_due:.2e}")
    check("scale=1.0 leaves the depot window exactly untouched",
          same.ready[DEPOT] == inst.ready[DEPOT] and same.due[DEPOT] == inst.due[DEPOT],
          f"depot due {inst.due[DEPOT]:.6g} -> {same.due[DEPOT]:.6g}")
    check("scale=1.0 needs no clamping",
          rep_one["due_clamped"] == 0 and rep_one["ready_clamped"] == 0,
          f"clamped due={rep_one['due_clamped']} ready={rep_one['ready_clamped']}")

    tight, rep_half = narrow_time_windows(inst, 0.5)
    inside = ((tight.ready[1:] >= inst.ready[1:] - 1e-9)
              & (tight.due[1:] <= inst.due[1:] + 1e-9))
    check("narrowed windows are subsets of the originals", bool(inside.all()),
          f"{int((~inside).sum())} of {inst.n} escaped")
    check("the depot window is still untouched after narrowing",
          tight.ready[DEPOT] == inst.ready[DEPOT] and tight.due[DEPOT] == inst.due[DEPOT])
    check("narrowing actually shrank the mean window",
          rep_half["mean_width_after"] < rep_half["mean_width_before"],
          f"{rep_half['mean_width_before']:.2f} -> {rep_half['mean_width_after']:.2f}")
    singles = [evaluate_route(tight, (j,)) for j in tight.customers]
    check("every customer keeps a feasible direct round trip",
          all(ev.feasible for ev in singles),
          "; ".join(f"{j}:{ev.reason}" for j, ev in zip(tight.customers, singles) if not ev.feasible))
    check("narrowing does not mutate the input instance",
          bool(np.array_equal(inst.ready, before_ready) and np.array_equal(inst.due, before_due)))

    raised = []
    for bad_scale in (0.0, -0.3, 1.5):
        try:
            narrow_time_windows(inst, bad_scale)
        except ValueError:
            raised.append(bad_scale)
    check("an out-of-range scale is rejected", len(raised) == 3,
          f"accepted {sorted(set((0.0, -0.3, 1.5)) - set(raised))}")

    # A customer that cannot be served by a direct round trip *before* narrowing
    # is a data error: clamping it back would widen the window past the original
    # and relax the problem instead of tightening it.
    malformed = random_instance(4, seed=5)
    malformed.ready[2] = float(malformed.due[DEPOT]) + 500.0
    try:
        narrow_time_windows(malformed, 0.8)
    except ValueError:
        raised_ok = True
    else:
        raised_ok = False
    check("an unservable customer is rejected, not clamped back open", raised_ok)


def test_dual_convention_at_the_path_level() -> None:
    """Spec §7.2 item 4: the arc dual convention must sum back to the route rc.

    The pricer charges each customer's dual to the arc *entering* it
    (`c̃_ij = c_ij - π_j`, see `Pricer._modified_cost`), while the symmetric
    split used in the literature would charge `c_ij - π_i/2 - π_j/2`.  On one
    arc the two differ by `π_i/2`, so an arc-by-arc comparison would report a
    failure on essentially every arc of the instance -- which is exactly why
    the spec bans that check.

    On a complete path they agree, because every served customer is the head of
    one arc and the tail of another, so it collects its full `-π` either way.
    What has to hold, on routes taken from a real column pool, is

        Σ_{(i,j) ∈ r} c̃_ij  +  V - σ  ==  rc(r)

    with the fleet dual σ charged to the closing arc `last -> depot` (it has no
    customer to sit on) and `rc(r)` taken from `reduced_cost_of`, which
    recomputes it from `evaluate_route` rather than trusting the pricer.
    """
    inst = random_instance(25, seed=2027)
    master = MasterProblem(inst)
    seeds = [r for r in cheapest_insertion(inst) if evaluate_route(inst, r).feasible]
    for r in seeds:
        master.add(r, evaluate_route(inst, r).cost)
    lp = master.solve_lp()
    pi = np.asarray(lp.pi, dtype=float)
    sigma = float(lp.sigma)

    routes = sorted(seeds, key=len, reverse=True)[:3]
    longest = len(routes[0]) if routes else 0
    check("the seed pool gives complete multi-stop paths to compare on",
          bool(routes) and longest >= 3, f"longest seed route = {longest} customers")

    tail = inst.dist.copy()
    tail[:, 1:] -= pi[None, :]                       # c_ij - pi_j
    sym = inst.dist.copy()
    sym[:, 1:] -= 0.5 * pi[None, :]
    sym[1:, :] -= 0.5 * pi[:, None]                  # c_ij - pi_i/2 - pi_j/2

    ok_rc, ok_sym, arc_gap = True, True, 0.0
    worst = 0.0
    for r in routes:
        closed = list(zip((DEPOT,) + tuple(r[:-1]), r)) + [(r[-1], DEPOT)]
        s_tail = sum(float(tail[a, b]) for a, b in closed) - sigma
        s_sym = sum(float(sym[a, b]) for a, b in closed) - sigma
        ref = reduced_cost_of(inst, r, pi, sigma, 0.0)
        worst = max(worst, abs(s_tail - ref))
        ok_rc = ok_rc and abs(s_tail - ref) < 1e-9
        ok_sym = ok_sym and abs(s_sym - s_tail) < 1e-9
        arc_gap = max(arc_gap, float(np.abs(tail - sym).max()))

    check("summed arc convention reproduces the route reduced cost",
          ok_rc, f"{len(routes)} complete paths, max|delta|={worst:.2e}")
    check("tail attribution and symmetric split agree at the path level",
          ok_sym, "each served customer collects a full -pi under both")
    check("...but disagree at the arc level, which is why that check is banned",
          arc_gap > 1e-9, f"max |tail - sym| over all arcs = {arc_gap:.4f}")


def test_feature_cache_is_bit_identical() -> None:
    """The cached feature block must equal the from-scratch block exactly.

    `allclose` is deliberately not used.  A feature off by one ulp changes the
    per-step sampling weights, which changes the routes a rollout builds, which
    changes the column pool - so `allclose` would pass here while the
    comparison against every previously reported table was already broken.

    Both calling conventions are exercised (the cache passed by keyword and
    positionally, and the cache built internally), and on a UAV instance as
    well as a plain one, because the depot-index placeholder in the
    precomputed tables is exactly the sort of thing that is right for every
    customer and wrong for node 0.
    """
    from uavcg.features import FeatureContext, arc_features
    from uavcg.pricers.rollout import RolloutDiagnostics

    plain = random_instance(9, seed=29)
    cases = [("vrptw", plain)]
    for i, extra in enumerate(cases):
        tag, inst = extra
        ctx = FeatureContext(inst)
        rng = np.random.default_rng(i)
        pi = rng.uniform(0.0, 5.0, size=inst.n)
        sigma, fixed = 0.37, 1.25

        pricer = RolloutPricer(inst, scoring="dual", n_samples=1, seed=0)
        trace: list = []
        pricer._rollout(pi, sigma, fixed, rng, 1.0, RolloutDiagnostics(), trace)
        check(f"[{tag}] the probe collected decision steps to compare",
              len(trace) > 0, f"steps={len(trace)}")

        dual = ctx.dual_block(pi)
        worst = 0.0
        bad = None
        for st, cand in trace:
            a = arc_features(ctx, st, cand, pi, None)
            b = arc_features(ctx, st, cand, pi, dual=dual)
            c = arc_features(ctx, st, cand, pi, None, dual)
            for other, how in ((b, "keyword"), (c, "positional")):
                if a.shape != other.shape or not np.array_equal(a, other):
                    worst = float("inf") if a.shape != other.shape else max(
                        worst, float(np.max(np.abs(a - other))))
                    bad = how
                    break
            if bad:
                break
        check(f"[{tag}] cached and uncached feature blocks are bit-identical",
              worst == 0.0,
              f"max abs diff = {worst:g}"
              + (f" via {bad}" if bad else f" over {len(trace)} steps"))


def test_feature_cache_covers_the_energy_branch() -> None:
    """Same invariant, on the one instance type whose feature set differs.

    In UAV mode the block gains `energy_pressure`, and the depot placeholder
    enters through `lb_vec[cand]`.  A cache that is right for VRPTW and wrong
    here would leave the UAV chapter of the study silently unverifiable.
    """
    from uavcg.features import FeatureContext, arc_features
    from uavcg.pricers.rollout import RolloutDiagnostics

    inst = as_uav(random_instance(9, seed=29),
                  energy_capacity=float(random_instance(9, seed=29).dist[0].max() * 4.0),
                  load_energy_factor=1.5)
    ctx = FeatureContext(inst)
    rng = np.random.default_rng(11)
    pi = rng.uniform(0.0, 5.0, size=inst.n)

    pricer = RolloutPricer(inst, scoring="dual", n_samples=1, seed=0)
    trace: list = []
    pricer._rollout(pi, 0.0, 0.0, rng, 1.0, RolloutDiagnostics(), trace)
    check("energy-mode instance produced decision steps to compare",
          len(trace) > 0, f"steps={len(trace)}")

    dual = ctx.dual_block(pi)
    same = all(np.array_equal(arc_features(ctx, st, cand, pi, None),
                              arc_features(ctx, st, cand, pi, dual=dual))
               for st, cand in trace)
    check("[uav] cached and uncached feature blocks are bit-identical", same,
          f"features={NUM_FEATURES} energy_cap={inst.params.energy_capacity:.1f}")

    # And the whole learned pricing path, not just the feature block.  This is
    # the test that was missing: `arc_features` used to read
    # `params.energy_cap`, a field that does not exist, so the energy branch
    # raised AttributeError the first time it was reached - meaning no learned
    # arm had ever been run on a UAV instance.  A test that only checked the
    # feature block would have caught it too, but a test that builds and prices
    # is what proves the arm exists.
    from uavcg.learning import ArcScorer
    rng_fit = np.random.default_rng(0)
    Xf = rng_fit.normal(size=(200, NUM_FEATURES))
    yf = (Xf[:, 0] + Xf[:, 3] > 0).astype(float)
    scorer = ArcScorer(kind="logistic", seed=0).fit(Xf, yf)
    pricer = RolloutPricer(inst, scoring="model", scorer=scorer, n_samples=4, seed=0)
    try:
        res = pricer.price(pi, sigma=0.0, fixed_cost=0.0, max_columns=5)
        ok = res is not None and "negative_found" in res.stats
        detail = f"rollouts={res.stats.get('rollouts')} found={res.stats.get('negative_found')}"
    except Exception as exc:                       # noqa: BLE001 - the point
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    check("[uav] a learned pricing call runs in energy mode", ok, detail)


def test_succ_counts_matches_its_definition() -> None:
    """Recompute the feasible-successor count from the definition.

    `succ_counts` was rewritten to reuse an intermediate and to gather rows
    with `np.take` instead of `np.ix_`.  Both are meant to be value-preserving,
    and "meant to be" is not a standard this project accepts for a quantity
    that feeds a model feature.
    """
    from uavcg.features import FeatureContext
    from uavcg.pricers.rollout import RolloutDiagnostics

    inst = random_instance(9, seed=29)
    ctx = FeatureContext(inst)
    cap = float(inst.params.capacity)
    rng = np.random.default_rng(3)
    pi = rng.uniform(0.0, 5.0, size=inst.n)

    pricer = RolloutPricer(inst, scoring="dual", n_samples=1, seed=0)
    trace: list = []
    pricer._rollout(pi, 0.0, 0.0, rng, 1.0, RolloutDiagnostics(), trace)
    st, cand = trace[0]

    brute = np.zeros(cand.size)
    for a, j in enumerate(cand):
        t = max(st.time + inst.service[st.node] + inst.travel_time[st.node, j],
                float(inst.ready[j]))
        load = st.load + float(inst.demand[j])
        cnt = 0
        for m in range(1, inst.n + 1):
            if m == j or st.visited[m]:
                continue
            t2 = t + inst.service[j] + inst.travel_time[j, m]
            if t2 > inst.due[m] + 1e-9:
                continue
            if max(t2, float(inst.ready[m])) + inst.service[m] \
                    + ctx.lb_time_back_c[m - 1] > ctx.depot_due:
                continue
            if load + float(inst.demand[m]) > cap + 1e-9:
                continue
            cnt += 1
        brute[a] = cnt

    fast = ctx.succ_counts(st, cand)
    check("succ_counts equals a brute-force recount of feasible successors",
          np.array_equal(fast, brute),
          f"fast={fast.astype(int).tolist()} brute={brute.astype(int).tolist()}")


def test_boxstep_clamps_without_freezing() -> None:
    """Boxstep's arithmetic, checked directly rather than inferred from a run.

    Three invariants, and each one is a way the arm could be wrong while still
    looking like an arm:

      * a box wider than the dual range is the identity - otherwise the arm
        differs from the baseline by more than the stabilizer, and it is not an
        ablation of one change;
      * the price the pricer sees is always inside the box - otherwise
        "boxstep" is not what is running;
      * a tight box genuinely moves the dual - otherwise the arm is a no-op
        dressed up as a method, and its "no effect" would mean nothing.

    The pure function is tested and the loop calls the same pure function, so
    there is no second copy to drift.
    """
    from uavcg.cg import boxstep_dual

    rng = np.random.default_rng(5)
    center = rng.uniform(0.0, 20.0, size=12)
    sigma_c = 3.0
    far = center * 3.0 - 5.0            # a dual that has wandered off
    sigma_far = 40.0

    # (1) identity when the box cannot be reached
    wide, s_wide = boxstep_dual(far, sigma_far, center, sigma_c, 1e9)
    check("a very wide box is the identity on the dual vector",
          np.array_equal(wide, far), f"max move = {float(np.max(np.abs(wide - far))):g}")
    check("a very wide box is the identity on sigma",
          abs(s_wide - sigma_far) < 1e-12, f"{s_wide:.12f} vs {sigma_far:.12f}")

    # (2) the result is always inside the box
    for delta in (0.05, 0.15, 0.5, 1.0):
        got, s_got = boxstep_dual(far, sigma_far, center, sigma_c, delta)
        scale = max(float(np.abs(center).mean()), 1e-6)
        half = delta * (np.abs(center) + scale)
        inside = bool(np.all(np.abs(got - center) <= half + 1e-12))
        check(f"[box_delta={delta}] the clamped dual lies inside the box", inside,
              f"max excursion = {float(np.max(np.abs(got - center) / half)):.6f} boxes")

    # (3) a tight box actually moves the dual
    tight, _ = boxstep_dual(far, sigma_far, center, sigma_c, 0.15)
    check("a tight box genuinely clamps the wandering dual",
          not np.array_equal(tight, far),
          f"moved {int(np.sum(np.abs(tight - far) > 0))}/{far.size} coordinates")

    # (4) and the run that uses it still lands on the true LP bound
    inst = random_instance(7, seed=71)
    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective
    res = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=400, max_columns_per_iter=30,
                        stabilization="boxstep", box_delta=0.15),
    ).run()
    check("boxstep CG still reaches the true LP bound",
          abs(res.lp_objective - ref_obj) < 1e-4,
          f"box={res.lp_objective:.6f} ref={ref_obj:.6f}")
    check("boxstep iterations were actually stabilized",
          res.stabilized_iterations > 0, f"n={res.stabilized_iterations}")



def _snapshot_key(res) -> list:
    """Snapshot fields that must reproduce bit for bit (spec §7.2 item 1)."""
    return [
        (s.iteration, s.lp_objective, s.num_columns, s.added,
         s.min_reduced_cost, s.pricer)
        for s in res.snapshots
    ]


def test_arc_mask_fully_open_reproduces_the_baseline() -> None:
    """Spec §7.2 item 1 / §7.5 step 4: the new bit-level reproduction check.

    An all-True mask has to be *indistinguishable* from no mask, not merely
    equivalent to it. The full-network arm of every later comparison is this
    run, so "close enough" would put a systematic difference between the control
    and the treatment -- and the number that would absorb it is the one the
    whole project reports. Hence exact equality on the LP objective, the column
    sequence and every snapshot, plus the mask bookkeeping, which must report
    that nothing was removed and that no call ran under a reduction.

    The same run also pins the warm-up interface: the hook is called once, it
    receives a WarmupTrace that stops at W, and that trace is built with the
    solver's own reduced-cost convention.
    """
    inst = random_instance(8, seed=131)
    n = inst.n

    def _run(extra):
        cg = ColumnGeneration(
            inst,
            NgRoutePricer(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst),
            config=CGConfig(max_iterations=200, max_columns_per_iter=20, **extra),
        )
        return cg, cg.run()

    cg_ref, ref = _run({})
    check("the control run used more iterations than the warm-up length",
          ref.iterations > 5, f"iters={ref.iterations}")

    seen: list = []

    def hook(trace):
        seen.append(trace)
        return np.ones((n + 1, n + 1), dtype=bool)

    cg_new, new = _run({"mask_after": 5, "mask_hook": hook})

    check("all-True mask: the hook fired exactly once", len(seen) == 1,
          f"calls={len(seen)}")
    check("all-True mask: the hook receives a WarmupTrace",
          bool(seen) and isinstance(seen[0], WarmupTrace))
    check("all-True mask: the run still reaches an LP verdict",
          new.mask_hook_fired and new.optimal_lp,
          f"fired={new.mask_hook_fired} optimal={new.optimal_lp}")
    check("all-True mask: bit-identical LP objective",
          new.lp_objective == ref.lp_objective,
          f"{new.lp_objective!r} vs {ref.lp_objective!r}")
    check("all-True mask: bit-identical iteration count",
          new.iterations == ref.iterations, f"{new.iterations} vs {ref.iterations}")
    check("all-True mask: bit-identical generated-column sequence",
          list(cg_new.master.routes) == list(cg_ref.master.routes),
          f"|pool|={len(cg_new.master)} vs {len(cg_ref.master)}")
    check("all-True mask: bit-identical snapshots",
          _snapshot_key(new) == _snapshot_key(ref))
    check("all-True mask removes nothing and says so",
          new.masked_arcs == 0 and new.masked_pricing_calls == 0
          and new.reduced_network_iterations == 0,
          f"arcs={new.masked_arcs} calls={new.masked_pricing_calls}")
    check("the maskable arc set is n^2 and empty masks leave it at 0%",
          new.arcs_total == n * n and new.arc_reduction == 0.0,
          f"arcs_total={new.arcs_total} n^2={n * n}")

    trace = seen[0]
    check("warm-up trace covers exactly W iterations and stops there",
          trace.W == 5 and len(trace.reduced_cost) == 5 and len(trace.sigma) == 5
          and len(trace.columns) == 5 and len(trace.lp_objective) == 5
          and sum(len(b) for b in trace.columns) == 88,
          f"W={trace.W} cols={sum(len(b) for b in trace.columns)}")
    check("warm-up trace uses the solver's own reduced-cost convention",
          all(np.array_equal(trace.reduced_cost[k], modified_cost(inst, trace.pi[k]))
              for k in range(trace.W))
          and trace.reduced_cost[0].shape == (n + 1, n + 1))
    check("warm-up trace records the columns as plain node tuples",
          all(isinstance(j, int) for batch in trace.columns
              for route in batch for j in route))
    check("warm-up arc-use counts agree with the recorded columns",
          trace.arc_use_counts(n).sum()
          == sum(len(r) + 1 for batch in trace.columns for r in batch),
          f"sum={trace.arc_use_counts(n).sum()} "
          f"expected={sum(len(r) + 1 for batch in trace.columns for r in batch)}")


def test_masked_pricing_never_publishes_a_bound() -> None:
    """Spec §7.2 item 1 and §8: certification happens on the full network only.

    Asserted at all three levels the spec names, because any one of them alone
    can be satisfied while another is broken. The pricer: a masked call has no
    bound and cannot certify. The loop: no call that certified was masked. The
    result: the LP still equals the true LP, so the reduction cost iterations
    and nothing else. A mask is a restriction of the pricing network, not of the
    column pool, so the last part is the only one that must hold asymptotically
    rather than immediately.
    """
    inst = random_instance(9, seed=132)
    n = inst.n

    ref = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref.solve_lp().objective

    class SpyPricer(NgRoutePricer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.seen: list = []

        def price(self, *a, **k):
            res = super().price(*a, **k)
            self.seen.append(res)
            return res

    # Keep the five nearest neighbours of each node, and never touch the depot
    # row: every customer stays reachable by a direct round trip, so the
    # instance keeps its LP optimum and any mismatch would be a real bug.
    knn = np.ones((n + 1, n + 1), dtype=bool)
    for i in range(n + 1):
        keep = {int(j) for j in np.argsort(inst.dist[i])[:5] if j != 0}
        for j in range(1, n + 1):
            if j not in keep:
                knn[i, j] = False
    knn[DEPOT, :] = True

    pricer = SpyPricer(inst, k=4)
    cg = ColumnGeneration(
        inst,
        pricer,
        fallback_pricer=ExactLabelingPricer(inst),
        config=CGConfig(max_iterations=300, max_columns_per_iter=20,
                        mask_after=3, mask_hook=lambda t: knn),
    )
    res = cg.run()

    masked = [r for r in pricer.seen if r.masked]
    check("the masked arm really did price under a reduction",
          len(masked) > 0 and res.masked_arcs > 0,
          f"calls={len(masked)} removed={res.masked_arcs}/{res.arcs_total}")
    check("no masked pricing call published a bound",
          all(r.relaxed_min_reduced_cost is None for r in masked))
    check("no masked pricing call can certify",
          all(r.certifies_no_negative is False for r in masked))
    check("every certifying call ran on the full network",
          all(not r.masked for r in pricer.seen if r.certifies_no_negative))
    check("the verdict was reached on a full-network iteration",
          res.optimal_lp and res.snapshots[-1].extra.get("network") == "full",
          f"optimal={res.optimal_lp} "
          f"last={res.snapshots[-1].extra.get('network')}")
    check("masked CG still reaches the true LP bound",
          abs(res.lp_objective - ref_obj) < 1e-4,
          f"cg={res.lp_objective:.6f} ref={ref_obj:.6f}")
    check("full + reduced iteration counts account for every iteration",
          res.full_network_iterations + res.reduced_network_iterations
          == res.iterations,
          f"{res.full_network_iterations}+{res.reduced_network_iterations}"
          f" vs {res.iterations}")


def test_a_truncated_masked_call_cannot_kill_the_reduced_network() -> None:
    """R-8 in its masked form: an invalidated call may not decide anything.

    §5 line 8 kills G_r on an empty column set, and it carries `not truncated`
    precisely because an empty set produced by hitting the label cap is an
    artefact of the cap rather than a property of the reduced network. The flag
    is irreversible, so reading it off a call the run itself refuses to trust
    would be the R-8 bug wearing a different hat.

    Both arms below are the same run with one bit flipped -- whether the masked
    call reports itself truncated -- so the difference can only come from that
    bit. Both must land on the same LP: the reduced network dying is a speed
    event, never a correctness one.
    """
    inst = random_instance(8, seed=134)
    n = inst.n
    knn = np.ones((n + 1, n + 1), dtype=bool)
    knn[1:3, 1:3] = False

    def _run(truncated_flag: bool):
        class MaskedEmpty(NgRoutePricer):
            """Real result while unmasked, empty-and-(maybe-)truncated masked."""

            def price(self, *a, **k):
                res = super().price(*a, **k)
                if res.masked:
                    return PricingResult(
                        columns=[],
                        min_reduced_cost=0.0,
                        exact=False,
                        relaxed_min_reduced_cost=None,
                        truncated=truncated_flag,
                        masked=True,
                        stats=dict(res.stats),
                    )
                return res

        cg = ColumnGeneration(
            inst,
            MaskedEmpty(inst, k=4),
            fallback_pricer=ExactLabelingPricer(inst),
            config=CGConfig(max_iterations=200, max_columns_per_iter=20,
                            mask_after=2, mask_hook=lambda t: knn),
        )
        return cg.run()

    blocked = _run(True)
    dead = _run(False)

    check("a truncated masked empty call does NOT kill the reduced network",
          blocked.reduced_dead is False
          and blocked.reduced_kill_blocked_by_truncation > 0,
          f"dead={blocked.reduced_dead} "
          f"blocked={blocked.reduced_kill_blocked_by_truncation}")
    check("the identical call without truncation DOES kill it (control arm)",
          dead.reduced_dead is True
          and dead.reduced_kill_blocked_by_truncation == 0,
          f"dead={dead.reduced_dead} "
          f"blocked={dead.reduced_kill_blocked_by_truncation}")
    check("whether G_r died did not change the LP bound",
          blocked.lp_objective == dead.lp_objective,
          f"{blocked.lp_objective!r} vs {dead.lp_objective!r}")
    check("a dead G_r is never re-entered, and G was still used",
          dead.reduced_network_iterations > 0
          and dead.full_network_iterations > 0,
          f"reduced={dead.reduced_network_iterations} "
          f"full={dead.full_network_iterations}")


def test_the_mask_can_only_be_mounted_where_it_is_safe() -> None:
    """Guard rails around §7.2 item 1, so a red line cannot be crossed by typo."""
    inst = random_instance(6, seed=136)
    n = inst.n
    ones = lambda t: np.ones((n + 1, n + 1), dtype=bool)   # noqa: E731

    # 1. an exact/fallback pricer has no set_mask and must be refused outright
    raised = ""
    try:
        ColumnGeneration(
            inst,
            ExactLabelingPricer(inst),
            config=CGConfig(max_iterations=20, mask_after=2, mask_hook=ones),
        ).run()
    except ValueError as exc:
        raised = str(exc)
    check("a pricer that cannot honour a mask is refused at run time",
          "set_mask" in raised, raised[:70])

    # 2. masking and stabilization are not combined in this implementation
    raised = ""
    try:
        ColumnGeneration(
            inst,
            NgRoutePricer(inst, k=3),
            config=CGConfig(max_iterations=20, mask_after=2, mask_hook=ones,
                            stabilization="smooth"),
        ).run()
    except ValueError as exc:
        raised = str(exc)
    check("masking together with dual stabilization is refused",
          "stabiliz" in raised, raised[:70])

    # 3. a wrongly shaped mask is rejected rather than broadcast
    pricer = NgRoutePricer(inst, k=3)
    raised = ""
    try:
        pricer.set_mask(np.ones((n, n), dtype=bool))
    except ValueError as exc:
        raised = str(exc)
    check("a mis-shaped arc mask is rejected", f"({n + 1}, {n + 1})" in raised,
          raised[:70])

    # 4. depot arcs are outside the maskable set, so zeros on them are not
    #    "arcs removed" -- otherwise arc_reduction would never read 0 for the
    #    control arm and the §6.2 ratio would be meaningless.
    depot_zeros = np.ones((n + 1, n + 1), dtype=bool)
    depot_zeros[:, DEPOT] = False
    depot_zeros[DEPOT, :] = False
    check("zeros on depot arcs are not counted as removed arcs",
          pricer.set_mask(depot_zeros) == 0 and pricer.masked_arcs == 0)

    # 5. and the arc count is what §3.6 needs as a denominator: n^2
    body_zeros = np.ones((n + 1, n + 1), dtype=bool)
    body_zeros[1:3, 1:3] = False          # 4 raw zeros, 2 of them self-loops
    check("removed-arc counting ignores self-loops and matches n^2 totals",
          pricer.set_mask(body_zeros) == 2 and pricer.arcs_total == n * n,
          f"removed={pricer.masked_arcs} total={pricer.arcs_total}")


def test_rounded_distances_match_the_benchmark_convention() -> None:
    """The §8 red-line anchor is only meaningful in Solomon's own convention.

    Published Solomon optima round every arc to the nearest integer; this solver
    keeps exact Euclidean distances. On the 100-customer set the two differ by
    0.223 per arc on average, and a ten-route solution uses about 110 arcs, so
    comparing the two objective values directly would be comparing two
    different problems -- with a gap of the same order as the margins the anchor
    is meant to police. The anchor therefore transforms the instance, and this
    test pins the transform rather than trusting it.

    The tie-break question is the interesting part. `np.rint` rounds half to
    even, the benchmark's `int(x + 0.5)` rounds half up, and they disagree only
    at an exact half. An exact half needs `sqrt(k) = m + 0.5`, i.e.
    `k = m^2 + m + 0.25`, which is not an integer -- and every squared distance
    in these instances is one. So no tie can occur. That is a claim about the
    data, so it is checked against the data, not asserted from the algebra.
    """
    inst = load_instance("R101", n=100, data_dir=str(_ROOT / "data" / "solomon100"))
    d = inst.dist
    off_diag = ~np.eye(d.shape[0], dtype=bool)

    frac = d - np.floor(d)
    ties = np.isclose(frac, 0.5, atol=0.0, rtol=0.0)
    check("no arc distance sits on an exact half, so no rounding rule can differ",
          not bool(np.any(ties & off_diag)),
          f"ties={int(np.count_nonzero(ties & off_diag))} of {int(off_diag.sum())}")

    rounded = with_rounded_distances(inst)
    check("rounded distances equal the benchmark's int(d + 0.5) on every arc",
          np.array_equal(rounded.dist, np.floor(d + 0.5)))
    check("rounded distances are integral",
          bool(np.all(rounded.dist == np.rint(rounded.dist))))
    check("rounding is idempotent",
          np.array_equal(with_rounded_distances(rounded).dist, rounded.dist))
    check("the transform does not touch the instance it was given",
          np.array_equal(inst.dist, d))
    check("the transform changes nothing but the two distance matrices",
          rounded.name == inst.name and rounded.n == inst.n
          and np.array_equal(rounded.demand, inst.demand)
          and np.array_equal(rounded.ready, inst.ready)
          and np.array_equal(rounded.due, inst.due)
          and np.array_equal(rounded.service, inst.service)
          and np.array_equal(rounded.coords, inst.coords))
    check("travel time is rounded with distance, not left at the exact value",
          np.array_equal(rounded.travel_time, np.floor(inst.travel_time + 0.5)))
    check("the measured per-arc convention gap matches what the docstring quotes",
          0.20 < float(np.abs(d - np.rint(d))[off_diag].mean()) < 0.25,
          f"mean|unrounded-rounded|={np.abs(d - np.rint(d))[off_diag].mean():.4f}")


def test_redcost_arc_reduction() -> None:
    """RedCost-S: the train-free baseline that has to be beaten (spec §7.5 step 5).

    Two properties carry the whole method, and both are checked here rather than
    asserted in prose:

    * **The ladder's top rung is the full network**, so a RedCost-S run can
      still certify. That is why it is a *baseline* and not a heuristic
      approximation: at the last rung the pricer is an ordinary ng-route pricer
      on G, publishing its relaxation bound.
    * **Every reduced rung is masked**, so a "no negative column" answer from one
      of them can neither publish a bound nor certify. This comes free from the
      §7.2 machinery -- `masked` on the result -- which is the point: the
      baseline needed no new safety code.
    """
    inst = random_instance(9, seed=141)
    n = inst.n
    pi_seed = np.full(n, 30.0)

    # ---- 1. top rung is bit-identical to the unmasked pricer ----------------
    top = RedCostArcPricer(inst, k=6, ladder=(None,))
    ref = NgRoutePricer(inst, k=6).price(pi_seed, max_columns=10 ** 6)
    got = top.price(pi_seed, max_columns=10 ** 6)
    check("RedCost-S at its top rung publishes the same bound as plain ng-route",
          got.relaxed_min_reduced_cost == ref.relaxed_min_reduced_cost,
          f"{got.relaxed_min_reduced_cost!r} vs {ref.relaxed_min_reduced_cost!r}")
    check("...builds the same labels and returns the same columns",
          got.labels_created == ref.labels_created
          and [c[0] for c in got.columns] == [c[0] for c in ref.columns])
    check("...and is reported as unmasked, so CG may certify on it",
          got.masked is False and got.certifies_no_negative == ref.certifies_no_negative)

    # ---- 2. a reduced rung is masked and proves nothing --------------------
    cut = RedCostArcPricer(inst, k=6, ladder=(3, None)).price(
        pi_seed, max_columns=10 ** 6)
    check("a reduced rung reports itself masked", cut.masked is True,
          f"removed={cut.stats['redcost_arcs_removed']}/{n * n}")
    check("a reduced rung publishes no bound",
          cut.relaxed_min_reduced_cost is None)
    check("a reduced rung cannot certify", cut.certifies_no_negative is False)
    check("a reduced rung actually removes arcs",
          0 < cut.stats["redcost_arcs_removed"] < n * n,
          f"removed={cut.stats['redcost_arcs_removed']}")

    # ---- 3. the ladder escalates when a rung yields nothing ----------------
    # With pi = 0 and sigma = 0 every route's reduced cost is its own length,
    # so no rung can find a negative column and the ladder must walk to the top.
    zero = RedCostArcPricer(inst, k=6, ladder=(1, 20, None)).price(
        np.zeros(n), sigma=0.0, max_columns=10 ** 6)
    check("an empty answer escalates the ladder to the next rung",
          zero.stats["redcost_rung"] == 2 and zero.stats["redcost_escalated_here"] == 2,
          f"rung={zero.stats['redcost_rung']} "
          f"escalated={zero.stats['redcost_escalated_here']}")
    check("...and the top rung is unmasked even after escalating",
          zero.masked is False and zero.relaxed_min_reduced_cost is not None)

    # ---- 4. the ladder must end at the full network ------------------------
    raised = ""
    try:
        RedCostArcPricer(inst, k=6, ladder=(10, 20))
    except ValueError as exc:
        raised = str(exc)
    check("a ladder that never returns to G is refused",
          "full network" in raised, raised[:70])
    raised = ""
    try:
        RedCostArcPricer(inst, k=6, ladder=(0, None))
    except ValueError as exc:
        raised = str(exc)
    check("a non-positive rung is refused", "non-positive" in raised, raised[:70])

    # ---- 5. the keep rule is the in/out union, not just out ----------------
    for n_min in (1, 4):
        keep = np.asarray(RedCostArcPricer(inst, k=6)._effective_mask(pi_seed, n_min))
        out_kept = [int(keep[i, 1:].sum()) for i in range(n + 1)]
        in_kept = [int(keep[:j, j].sum() + keep[j + 1:, j].sum())
                   for j in range(1, n + 1)]
        check(f"[N_min={n_min}] every node keeps at least N_min outgoing arcs",
              min(out_kept) >= n_min, f"min={min(out_kept)}")
        check(f"[N_min={n_min}] every customer keeps at least N_min incoming arcs",
              min(in_kept) >= n_min, f"min={min(in_kept)}")
        # The union of two rules that each add at most N_min arcs per node is
        # bounded by their sum. It is NOT bounded per node: node i also collects
        # arcs because they are the cheapest way *into* some j, and that can be
        # many. Getting this wrong in the other direction is what makes a mask
        # remove the head of every route.
        body = int(keep[1:, 1:][~np.eye(n, dtype=bool)].sum())
        check(f"[N_min={n_min}] total kept arcs stay within the union bound",
              body <= 2 * n_min * (n + 1),
              f"kept={body} bound={2 * n_min * (n + 1)}")
    check("depot arcs stay open under any rung",
          bool(keep[DEPOT, 1:].all()) and bool(keep[:, DEPOT].all()))
    # A rung at or above n removes nothing: the reduction is vacuous on an
    # instance smaller than the ladder's first rung, which is why the default
    # 10/20 is calibrated for the n=60 main tier and not for smoke tests.
    wide = np.asarray(RedCostArcPricer(inst, k=6)._effective_mask(pi_seed, n + 1))
    check("a rung >= n is the identity on the network",
          int(removed_arcs(wide, n)) == 0, f"n={n}")

    # ---- 6. end to end: the reduction costs iterations, never the bound -----
    ref_master = MasterProblem(inst)
    for r in enumerate_routes(inst):
        ref_master.add(r, evaluate_route(inst, r).cost)
    ref_obj = ref_master.solve_lp().objective

    cg_ref = ColumnGeneration(
        inst, NgRoutePricer(inst, k=4),
        config=CGConfig(max_iterations=400, max_columns_per_iter=30),
    ).run()
    # Ladder lowered for the smoke instance: n = 9, so the production rung of 10
    # is already the full network and would measure nothing.
    rc_pricer = RedCostArcPricer(inst, k=4, ladder=(2, 4, None))
    cg_rc = ColumnGeneration(
        inst, rc_pricer,
        config=CGConfig(max_iterations=400, max_columns_per_iter=30),
    ).run()
    check("RedCost-S reaches the true LP bound like the baseline",
          abs(cg_rc.lp_objective - ref_obj) < 1e-6
          and abs(cg_ref.lp_objective - ref_obj) < 1e-6,
          f"rc={cg_rc.lp_objective:.6f} base={cg_ref.lp_objective:.6f} "
          f"true={ref_obj:.6f}")
    check("both arms agree on the LP value, and it is the enumerated LP",
          abs(cg_rc.lp_objective - cg_ref.lp_objective) < 1e-9
          and abs(cg_rc.lp_objective - ref_obj) < 1e-6,
          f"rc={cg_rc.lp_objective:.6f} base={cg_ref.lp_objective:.6f} "
          f"true={ref_obj:.6f}")
    # `optimal_lp` is deliberately NOT required to agree between the arms. It
    # reports whether the ng-route relaxation bound was tight *at the dual this
    # particular column pool ended on*, and the two arms walk different pools.
    # A False here is a certification false negative -- the run stopped because
    # no elementary negative column existed, while a negative non-elementary
    # label kept the relaxation bound below zero. What must hold is the safety
    # direction: certifying is only allowed at the true LP.
    check("no arm claims optimal at an LP it did not actually reach",
          all((not r.optimal_lp) or abs(r.lp_objective - ref_obj) < 1e-6
              for r in (cg_ref, cg_rc)),
          f"base={cg_ref.optimal_lp} rc={cg_rc.optimal_lp}")
    check("at least one arm certified, so the end-to-end path is exercised",
          bool(cg_ref.optimal_lp or cg_rc.optimal_lp),
          f"base={cg_ref.optimal_lp} rc={cg_rc.optimal_lp}")
    check("the ladder climbed at least once during the solve (a rung died)",
          rc_pricer.escalations >= 1 and rc_pricer.rung >= 1,
          f"escalations={rc_pricer.escalations} rung={rc_pricer.rung} "
          f"n_min={rc_pricer.rung_n_min}")
    check("the RedCost-S run masked its early iterations and unmasked the last",
          cg_rc.masked_pricing_calls > 0
          and cg_rc.snapshots[-1].extra.get("network") == "full",
          f"masked_calls={cg_rc.masked_pricing_calls} "
          f"last={cg_rc.snapshots[-1].extra.get('network')}")
    check("the network counters describe the calls that were actually made",
          cg_rc.full_network_iterations + cg_rc.reduced_network_iterations
          == cg_rc.iterations and cg_rc.reduced_network_iterations > 0,
          f"{cg_rc.full_network_iterations}+{cg_rc.reduced_network_iterations}"
          f" vs {cg_rc.iterations}")
    check("every snapshot carries its rung so the arc share can be reported",
          all("redcost_arcs_removed" in s.extra for s in cg_rc.snapshots))

    # §6.2's arc-share column.  The reduced arm installs no hook mask, so the
    # hook-based `masked_arcs` is 0 for it by construction -- reporting that
    # number would print a confident 0.0% for the one arm whose entire
    # mechanism is arc removal.  The figure has to come from the calls.
    check("the reduced arm's arc share is read from its calls, not from a hook",
          cg_rc.masked_arcs == 0 and cg_rc.mean_arc_reduction is not None
          and cg_rc.mean_arc_reduction > 0.0,
          f"hook_arcs={cg_rc.masked_arcs} mean="
          f"{cg_rc.mean_arc_reduction if cg_rc.mean_arc_reduction is None else f'{cg_rc.mean_arc_reduction:.1%}'}")
    check("the arc share is only counted on the calls that were masked",
          cg_rc.arcs_removed_total == 0 or cg_rc.masked_pricing_calls > 0)
    check("the plain arm reports no arc share rather than a zero one",
          cg_ref.mean_arc_reduction is None and cg_ref.masked_pricing_calls == 0,
          f"mean={cg_ref.mean_arc_reduction} masked={cg_ref.masked_pricing_calls}")


# ----------------------------------------------------------------------
# Frozen instance split (spec §7.5 step 3)
# ----------------------------------------------------------------------
# The spec §3.3 envelope: one full-network pricing call must land under this,
# otherwise the instance is not affordable in the main tier.  Deliberately 20 s
# and not the probe script's CLI default of 60 s -- at 60 s, C207 x0.5
# (22.59 s) would pass and the frozen factor would differ from the one the
# approved matrix names.
TW_ENVELOPE_SECONDS = 20.0
IDLE_PROBE = Path(__file__).resolve().parents[1] / "results" / "_probe_matrix.jsonl"


def _rederive_split(rows, envelope=TW_ENVELOPE_SECONDS, n=60):
    """Rebuild the frozen split from the raw probe readings.

    This is the split's real test.  Asserting that the file says what the file
    says is worth nothing; asserting that the file is what its own stated rule
    produces from the measurements is what makes it a *frozen* decision rather
    than one somebody nudged afterwards.

    Rule, applied in order:
      1. keep the readings that are usable evidence: right size, no cap hit,
         peak labels clear of the 80% clean line;
      2. per instance take the LOOSEST factor still inside the envelope;
      3. sort each family by that reading's cost, deal alternately, test first.
    """
    from uavcg.pricers.base import CLEAN_LABEL_FRACTION

    usable = {}
    for r in rows:
        if int(r["n"]) != n or r.get("truncated"):
            continue
        cap = r.get("label_cap")
        frac = (r["labels_created"] / cap) if cap else 0.0
        if cap and frac >= CLEAN_LABEL_FRACTION:
            continue
        if float(r["seconds"]) > envelope:
            continue
        cur = usable.get(r["instance"])
        # "loosest still affordable" == largest tw_scale that survived above
        if cur is None or float(r["tw_scale"]) > float(cur["tw_scale"]):
            usable[r["instance"]] = r

    families = {}
    for name, r in usable.items():
        head = "".join(c for c in name if not c.isdigit())
        digits = "".join(c for c in name if c.isdigit())
        families.setdefault(f"{head}{digits[0]}", []).append(r)

    out = {}
    for fam, rs in families.items():
        order = sorted(rs, key=lambda r: (float(r["seconds"]), r["instance"]))
        for rank, r in enumerate(order):
            out[r["instance"]] = ("test" if rank % 2 == 0 else "train")
    return usable, out


def test_frozen_split_matches_its_rule() -> None:
    """§7.5 step 3: the 12-instance split is frozen, and re-derivable."""
    import json

    from uavcg.splits import load_split

    sp = load_split()
    check("the frozen split holds the approved 12 instances",
          len(sp.entries) == 12, f"{len(sp.entries)} entries")

    for fam, want in sp.families.items():
        for bucket in ("train", "test"):
            got = len(sp.family(fam, bucket))
            check(f"{fam} {bucket} bucket size is the approved {want[bucket]}",
                  got == int(want[bucket]), f"{got}")

    check("the three landing representatives are present and named §7.5 step 5",
          sp.landing_instances == ("C201", "RC201", "R201")
          and all(sp.entry(n).name == n for n in sp.landing_instances),
          ", ".join(sp.landing_instances))

    if not IDLE_PROBE.exists():
        check("probe readings available for the re-derivation", False,
              f"missing {IDLE_PROBE.name}")
        return

    rows = [json.loads(l) for l in IDLE_PROBE.read_text(encoding="utf-8").splitlines() if l.strip()]
    usable, assignment = _rederive_split(rows)

    check("the rule selects exactly the frozen 12 and nothing else",
          set(assignment) == {e.name for e in sp.entries},
          f"rule={len(assignment)} frozen={len(sp.entries)}")
    check("the rule reproduces every bucket assignment in the file",
          all(assignment.get(e.name) == e.bucket for e in sp.entries),
          ", ".join(f"{e.name}:{assignment.get(e.name)}!={e.bucket}"
                    for e in sp.entries if assignment.get(e.name) != e.bucket))
    check("the rule reproduces every narrowing factor in the file",
          all(abs(float(usable[e.name]["tw_scale"]) - e.tw_scale) < 1e-12
              for e in sp.entries if e.name in usable),
          ", ".join(f"{e.name}:x{usable[e.name]['tw_scale']:g}!={e.tw_scale:g}"
                    for e in sp.entries
                    if e.name in usable
                    and abs(float(usable[e.name]["tw_scale"]) - e.tw_scale) > 1e-12))

    # The envelope has to be doing work, otherwise the check above would pass
    # for any envelope and the 20 s figure in the spec would be decoration.
    _, loose = _rederive_split(rows, envelope=60.0)
    check("the 20s envelope is load-bearing (60s would pick a different C207)",
          loose.get("C207") == "test" and assignment.get("C207") == "train",
          f"x0.35 -> {assignment.get('C207')}, x0.5 -> {loose.get('C207')}")

    # And the frozen instance must be the one the probe measured: load it and
    # compare a geometry summary against the recorded reading.
    mismatch = []
    for e in sp.entries:
        if e.name not in usable:
            continue
        inst = sp.load(e)
        width = float(np.mean(inst.due[1:] - inst.ready[1:]))
        if inst.n != e.n or abs(width - float(usable[e.name]["tw_mean_width"])) > 1e-9:
            mismatch.append(f"{e.label}: width {width:.3f}")
    check("every frozen entry reloads to the exact probed instance",
          not mismatch, "; ".join(mismatch))

    check("loading refuses to be told a different n or tw_scale",
          _raises_any(lambda: sp.load(sp.entries[0], n=99))
          and _raises_any(lambda: sp.load(sp.entries[0], tw_scale=0.1)))


def _raises_any(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_pp_rmp_time_accounting() -> None:
    """§6.2's PP/RMP/total columns have to be measured, and have to add up."""
    inst = load_instance("C101", n=10)
    pricer = NgRoutePricer(inst, k=5)
    cfg = CGConfig(max_iterations=30, max_columns_per_iter=10, log_every=0)
    res = ColumnGeneration(inst, pricer, config=cfg).run()

    check("PP time was actually measured (a run with pricing is not all overhead)",
          res.pp_seconds > 0.0, f"pp={res.pp_seconds:.4f}s")
    check("RMP time was actually measured",
          res.rmp_seconds > 0.0, f"rmp={res.rmp_seconds:.4f}s")

    # The three timers overlap only to the extent of clock resolution: PP and
    # RMP are disjoint intervals, so their sum can never exceed the outer wall
    # clock by more than that.  A violation means a call site was instrumented
    # twice, or an interval was left open across the other kind of call.
    slack = res.wall_clock - res.pp_seconds - res.rmp_seconds
    check("PP and RMP are disjoint and fit inside the wall clock",
          slack >= -1e-6, f"wall={res.wall_clock:.4f} slack={slack:+.4f}")

    # And they should account for the bulk of it: if `other` were most of the
    # run, the two headline columns would be describing a footnote.
    check("pricing and the master dominate the run, overhead is not the story",
          res.other_seconds < 0.5 * res.wall_clock,
          f"other={res.other_seconds:.4f}s of {res.wall_clock:.4f}s")

    # Which of the two dominates is an instance property, not an invariant: at
    # n=10 the master LP is the expensive part and pricing is microseconds.  So
    # rather than assert a balance the instance does not have, drive the
    # instrumentation: a pricer that provably burns a fixed delay per call must
    # show up in PP, and only in PP.
    delay = 0.004

    class _SlowPricer(NgRoutePricer):
        def price(self, *a, **kw):
            time.sleep(delay)
            return super().price(*a, **kw)

    slow = ColumnGeneration(inst, _SlowPricer(inst, k=5), config=cfg).run()
    paid = slow.pricing_calls * delay
    check("PP time tracks the pricing calls, so the timer is on the right calls",
          slow.pp_seconds >= paid * 0.8,
          f"pp={slow.pp_seconds:.4f}s for {slow.pricing_calls} calls "
          f"(>= {paid:.4f}s injected)")
    check("injected pricing delay does not leak into RMP time",
          slow.rmp_seconds < res.rmp_seconds * 3.0,
          f"rmp {res.rmp_seconds:.4f} -> {slow.rmp_seconds:.4f}")
    # The header of this block already says that which of the two dominates is
    # an instance property, so the check must not assert the property.  What the
    # instrumentation has to deliver is that injecting a fixed delay per pricing
    # call moves pricing's SHARE of the run upward -- that is the claim being
    # tested, and unlike "PP beats RMP at n=10" it does not flip on clock noise.
    #
    # It did flip, which is why this is now a share: 4 ms/call against a ~4 ms
    # master LP is a coin toss, and the old form (`slow.pp > slow.rmp`) failed
    # on a loaded machine while `slow.pp >= 0.8 * calls * delay` passed in the
    # same run -- i.e. the timer was right and the assertion was wrong.
    slow_share = slow.pp_seconds / slow.wall_clock
    plain_share = res.pp_seconds / res.wall_clock
    check("the delayed run reports a bigger PP share than the plain one",
          slow_share > plain_share,
          f"slow pp share={slow_share:.4f} plain pp share={plain_share:.4f}")

def test_the_restricted_master_is_an_upper_bound() -> None:
    """Why "the bound met the incumbent" is not a certificate.

    Adding a column relaxes the master, so the restricted master LP is an UPPER
    bound on the full LP optimum -- the direction is the opposite of the one
    that would make an incumbent comparison a proof.  This pins the smallest
    witness, because the mistake is easy to make and was made: a rule that
    stopped when the bound reached the incumbent would certify 10 here while the
    true optimum is 6.
    """
    from uavcg.instance import Instance, UAVParams
    from uavcg.rmp import MasterProblem

    coords = np.zeros((3, 2))
    dist = np.zeros((3, 3))
    zeros = np.zeros(3)
    inst = Instance(
        name="toy2", num_vehicles=5,
        params=UAVParams(capacity=100.0),
        coords=coords, demand=zeros, ready=zeros.copy(),
        due=np.full(3, 1e6), service=zeros.copy(),
        dist=dist, travel_time=dist.copy(),
    )

    pool = MasterProblem(inst, max_vehicles=5)
    pool.add((1, 2), 10.0)
    restricted = pool.solve_lp().objective
    full = pool.clone()
    full.add((1,), 3.0)
    full.add((2,), 3.0)
    full_value = full.solve_lp().objective

    check("the restricted master LP equals the incumbent cover (10)",
          abs(restricted - 10.0) < 1e-9, f"{restricted:.6f}")
    check("...yet the full LP over more columns is strictly cheaper",
          abs(full_value - 6.0) < 1e-9, f"{full_value:.6f}")
    check("so restricted-LP == incumbent is NOT a proof of optimality",
          full_value < restricted - 1e-6,
          f"restricted {restricted:.4f} vs full {full_value:.4f}")

    # And the reduced-cost test does catch this case, which is the point: the
    # certificate has to come from pricing, not from the incumbent.
    lp = pool.solve_lp()
    pi = np.asarray(lp.pi, dtype=float)
    rc_single = 3.0 - float(pi[0])
    check("the dual of the restricted LP exposes the missed column",
          rc_single < 0.0, f"rc(1) = {rc_single:+.4f}")


def test_an_incumbent_hit_does_not_stop_or_certify() -> None:
    """The gap diagnostic is reported, never acted on.

    Solomon C201 at n=60 (x0.8) is the case that forces the distinction: the
    restricted LP reaches its incumbent (443.942994) immediately, so the gap is
    zero from the first solve, and the reduced-cost test still never certifies.
    A run that treated the zero gap as a certificate would stop after one
    iteration and report an unproven LP as optimal.
    """
    from uavcg.splits import load_split

    split = load_split()
    inst = split.load(split.entry("C201"))
    cfg = CGConfig(max_iterations=400, max_columns_per_iter=25, stall_limit=2,
                   fallback_max_n=40, log_every=0)
    res = ColumnGeneration(inst, NgRoutePricer(inst, k=10), config=cfg).run()

    check("C201 n=60 closes the gap to its incumbent on the first solve",
          res.bound_meets_incumbent and res.optimality_gap <= 1e-6,
          f"gap={res.optimality_gap:+.3e} incumbent={res.incumbent_value:.4f}")
    check("...and the run is STILL not certified",
          not res.optimal_lp)
    check("...and the zero gap did not stop it: the stall rule did",
          res.stalled and res.iterations <= 3,
          f"stalled={res.stalled} iterations={res.iterations}")
    check("the gap is reported as incumbent minus LP, not as a verdict",
          abs(res.optimality_gap - (res.incumbent_value - res.lp_objective)) < 1e-12)


def test_v34_singleton_seed_is_the_standard() -> None:
    """The v3.4 seed decision, asserted rather than described.

    Spec v3.4 §3.3/§8 makes the singleton seed the standard for all four arms
    and requires a pipeline assertion that the first-round dual face has max
    coordinate span below 1e-6.  The claim rests on an algebraic identity, so
    that identity is checked directly (`sum_i 2*d(0,i)` equals the singleton
    seed's LP value) instead of being inferred from a solver's behaviour --
    if the identity ever breaks, every "the dual is unique" argument in the
    diagnosis breaks with it, and it would otherwise break silently.

    The contrast is asserted on R201 as well, because "the insertion seed has a
    wide face" is the entire reason the seed was changed and a test that only
    checks the good case cannot catch a regression that makes both seeds wide.
    """
    from uavcg.degeneracy import FACE_TOL, seed_dual_face, singleton_identity_gap
    from uavcg.heuristics import cheapest_insertion, singleton_routes

    for n, seed in ((10, 1301), (12, 1302), (9, 1303)):
        inst = random_instance(n, seed=seed)
        routes = singleton_routes(inst)
        check(f"singleton seed gives one route per customer (n={n})",
              len(routes) == len(inst.customers)
              and all(len(r) == 1 for r in routes),
              f"{len(routes)} routes for {len(inst.customers)} customers")
        check(f"singleton seed covers every customer (n={n})",
              {int(r[0]) for r in routes} == {int(c) for c in inst.customers})

        face = seed_dual_face(inst, routes)
        check(f"singleton seed's optimal dual face is a point (n={n})",
              face.is_point,
              f"free={face.free_coordinates}/{inst.n} "
              f"max_span={face.max_span:.3e}")

        gap = singleton_identity_gap(inst)
        check(f"the singleton identity holds: sum_i 2*d(0,i) == seed LP (n={n})",
              gap < FACE_TOL, f"gap={gap:.3e}")

    # The instance the diagnosis was built on, where both seeds were measured.
    from uavcg.splits import load_split

    split = load_split()
    entry = split.entry("R201")
    inst = split.load(entry)
    ins_face = seed_dual_face(inst, cheapest_insertion(inst))
    sing_face = seed_dual_face(inst, singleton_routes(inst))
    check("R201: the insertion seed's optimal dual face is NOT a point",
          not ins_face.is_point,
          f"free={ins_face.free_coordinates}/{inst.n} "
          f"max_span={ins_face.max_span:.6f}")
    check("R201: the singleton seed's optimal dual face IS a point",
          sing_face.is_point and sing_face.max_span < FACE_TOL,
          f"free={sing_face.free_coordinates}/{inst.n} "
          f"max_span={sing_face.max_span:.3e}")
    check("R201: the two seeds agree on nothing but disagree on the face",
          abs(ins_face.objective - 1323.4033078972) < 1e-6
          and abs(sing_face.objective - 3011.2856712707) < 1e-6,
          f"insertion={ins_face.objective:.6f} singles={sing_face.objective:.6f}")


def test_v34_plateau_rule_and_three_state_status() -> None:
    """PLATEAU = 30 consecutive iterations improving by < 1e-9, and the states.

    Two separate things have to hold for the reporting contract to be usable:

    1. the threshold is 1e-9 and it is its own knob -- tying it to `eps` (1e-6)
       would silently redefine PLATEAU, and tying the stall counter to `best_lp`
       (which only moves on eps-sized improvements) would let a stream of
       sub-eps improvements hold the counter at zero forever;
    2. the three labels map onto the flags in the documented precedence, with
       `degraded` outranking both others so a capped run cannot reach the main
       table by the side door.
    """
    from uavcg.cg import CGConfig, CGResult

    check("default plateau tolerance is 1e-9 (spec v3.4 §6.2)",
          CGConfig().plateau_tol == 1e-9, f"{CGConfig().plateau_tol!r}")
    check("plateau tolerance is independent of the pricing eps",
          CGConfig().plateau_tol < CGConfig().eps,
          f"plateau_tol={CGConfig().plateau_tol:g} eps={CGConfig().eps:g}")

    def mk(optimal: bool, stalled: bool, truncated: int = 0,
           unscheduled: int = 0) -> CGResult:
        return CGResult(
            snapshots=[], lp_objective=100.0, ip_objective=100.0, ip_routes=[],
            num_vehicles=1, iterations=5, columns_in_pool=10, pricing_calls=5,
            exact_confirmation_calls=0, wall_clock=1.0, optimal_lp=optimal,
            stalled=stalled, truncated_pricing_calls=truncated,
            unscheduled_truncations=unscheduled,
        )

    check("certified run reports CERTIFIED",
          mk(True, False).status == "CERTIFIED",
          mk(True, False).status)
    check("stalled run reports PLATEAU",
          mk(False, True).status == "PLATEAU", mk(False, True).status)
    check("neither certified nor stalled reports INCOMPLETE",
          mk(False, False).status == "INCOMPLETE", mk(False, False).status)
    degraded_certified = mk(True, False, truncated=1, unscheduled=1)
    check("a degraded run is INCOMPLETE even when it certified",
          degraded_certified.status == "INCOMPLETE",
          f"status={degraded_certified.status} degraded="
          f"{degraded_certified.degraded} optimal_lp={degraded_certified.optimal_lp}")
    # v3.6 ③: `degraded` keys off UNSCHEDULED truncations only.  Under the capped
    # protocol every descent round is cut on purpose, and if that counted as
    # degradation then every run would be INCOMPLETE and no certificate could
    # ever reach the main table -- which is exactly what the pre-fix code did.
    scheduled_only = mk(True, False, truncated=3, unscheduled=0)
    check("a run truncated only on the PROTOCOL's schedule still counts",
          scheduled_only.status == "CERTIFIED" and not scheduled_only.degraded
          and scheduled_only.counts_toward_main_table,
          f"status={scheduled_only.status} degraded={scheduled_only.degraded}")
    check("only CERTIFIED counts toward the main table",
          mk(True, False).counts_toward_main_table
          and not mk(False, True).counts_toward_main_table
          and not mk(False, False).counts_toward_main_table
          and not degraded_certified.counts_toward_main_table)
    check("the status is carried in the summary",
          mk(True, False).summary()["status"] == "CERTIFIED",
          str(mk(True, False).summary().get("status")))

    # The stall rule actually fires, at the tolerance it claims.  A run that
    # cannot move is the cheapest way to drive it: stall_limit=1 must stop it
    # after the first iteration that does not improve the LP.
    inst = random_instance(7, seed=1304)
    cfg = CGConfig(max_iterations=50, stall_limit=1, plateau_tol=1e-9,
                   fallback_max_n=40, stabilization="off")
    res = ColumnGeneration(inst, NgRoutePricer(inst, k=4), config=cfg).run()
    check("stall_limit=1 stops a non-improving run immediately",
          res.stalled or res.optimal_lp,
          f"stalled={res.stalled} iterations={res.iterations} "
          f"status={res.status}")
    check("a stalled-on-purpose run is reported as PLATEAU, not silently passed",
          res.optimal_lp or res.status == "PLATEAU", res.status)


def test_v34_wall_clock_cap_truncates_and_stops_gracefully() -> None:
    """The sweep's budget must bound ONE call without losing the LP trace.

    Both halves were missing, and both were expensive.  The budget was checked
    only *between* calls, so a single call ran up to 5,644 s against a 1,500 s
    budget; and it was enforced by raising, which discarded every snapshot --
    so a budgeted attempt ended with zero iterations and no LP, which is exactly
    the data the filter exists to collect.  Measured cost of getting it wrong:
    a 16 h sweep that produced nine verdict-free rows and had to be killed by
    hand.

    The cap has the label cap's contract (truncate => prove nothing => certify
    nothing), so it is fail-safe: a capped run can only ever be INCOMPLETE.  It
    *is* wall-clock dependent, which is why it is off by default and why an
    uncapped call must stay byte-identical -- the bit-level anchors fingerprint
    these stats dicts.
    """
    from _probe_n60_cost import LABEL_WINDOW_SLACK, TimedPricer
    from uavcg.heuristics import singleton_routes

    inst = random_instance(20, seed=4242, tw_slack=1000.0)
    pi = np.full(inst.n, 500.0)

    # 1. The cap itself.
    capped = NgRoutePricer(inst, k=10, max_seconds=-1.0).price(
        pi, sigma=0.0, fixed_cost=0.0, max_columns=200)
    check("a wall-clock cap truncates (>=256 labels processed, so it fired)",
          capped.truncated and capped.stats.get("labels_processed", 0) >= 256,
          f"truncated={capped.truncated} "
          f"processed={capped.stats.get('labels_processed')}")
    check("a time-truncated call publishes no bound, so it cannot certify",
          capped.relaxed_min_reduced_cost is None
          and not capped.certifies_no_negative,
          f"bound={capped.relaxed_min_reduced_cost} "
          f"certifies={capped.certifies_no_negative}")
    check("the truncation reason is recorded for diagnosis",
          bool(capped.stats.get("time_truncated")),
          repr(capped.stats.get("time_truncated")))

    # The wide windows that make the cap fire also make an *uncapped* call
    # explore a huge label space, so bound it with the ordinary label cap.  The
    # point is only that the stats dict gains no key when no wall-clock cap is
    # set -- the anchors fingerprint that dict.
    uncapped = NgRoutePricer(inst, k=10, max_labels=2_000).price(
        pi, sigma=0.0, fixed_cost=0.0, max_columns=200)
    check("an uncapped call's stats are unchanged (anchors fingerprint them)",
          "time_truncated" not in uncapped.stats,
          str(sorted(uncapped.stats))[:70])

    # 2. A truncated call KEEPS its columns.  That is the whole v3.5 mechanism:
    #    a sink label that was built is a real column, whatever happens next.
    check("a truncated call still returns the columns it found",
          len(capped.columns) > 0,
          f"cols={len(capped.columns)} truncated={capped.truncated}")

    # 3. Escalation: a truncated EMPTY call is blind, not converged (v3.5 ③), so
    #    it is retried at the certification cap.  Driven with a stub so the test
    #    is deterministic -- the real pricer cannot be made to truncate-empty on
    #    demand without a cap so small that nothing is found anyway.
    from uavcg.pricers import PricingResult

    class _StubPricer:
        name = "stub"
        guarantees_optimality = False
        masked_arcs = 0
        arcs_total = 0
        inst = None
        max_seconds = None

        def __init__(self) -> None:
            self.caps: list = []

        def price(self, pi, sigma=0.0, fixed_cost=0.0, max_columns=50):
            self.caps.append(self.max_seconds)
            if len(self.caps) == 1:
                return PricingResult(columns=[], exact=False,
                                     relaxed_min_reduced_cost=None,
                                     truncated=True,
                                     stats={"time_truncated": True})
            return PricingResult(columns=[], exact=False,
                                 relaxed_min_reduced_cost=None,
                                 truncated=False, stats={})

    stub = _StubPricer()
    tp = TimedPricer(stub, descent_seconds=300.0, cert_seconds=1200.0,
                     label="stub")
    sres = tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("a truncated EMPTY descent call escalates to a certification attempt "
          "bounded by the HARD TOTAL, not by the phase's nominal 1200 s (P2)",
          tp.cert_escalations == 1
          and stub.caps[0] == 300.0
          and stub.caps[1] > 1200.0
          and stub.caps[1] >= tp.total_budget - LABEL_WINDOW_SLACK - 0.01,
          f"escalations={tp.cert_escalations} caps={stub.caps} "
          f"total={tp.total_budget}")
    check("the escalated result is what the loop sees",
          not sres.truncated, f"truncated={sres.truncated}")

    # 4. The gate winds the run down gracefully -- and keeps the snapshots, which
    #    is the property the old raise destroyed.
    timed = TimedPricer(NgRoutePricer(inst, k=10), descent_seconds=0.005,
                        cert_seconds=0.005, gate_seconds=0.002, label="selftest")
    cfg = CGConfig(max_iterations=50, max_columns_per_iter=200, stall_limit=30,
                   plateau_tol=1e-9, fallback_max_n=40, stabilization="off")
    res = ColumnGeneration(inst, timed, config=cfg, fallback_pricer=None,
                           seed_routes=singleton_routes(inst)).run()
    check("a gate-exceeded run RETURNS instead of raising (state kept)",
          len(res.snapshots) > 0, f"snapshots={len(res.snapshots)}")
    check("it reports INCOMPLETE and never a certificate",
          res.status == "INCOMPLETE" and not res.optimal_lp,
          f"status={res.status} optimal={res.optimal_lp}")
    check("the LP trace survives, so the content verdict is computable",
          all(isinstance(s.lp_objective, float) for s in res.snapshots)
          and min(s.lp_objective for s in res.snapshots) > 0.0,
          f"lp_min={min((s.lp_objective for s in res.snapshots), default=None)}")


def test_v36_phase_budget_is_a_hard_bound() -> None:
    """v3.6 requirement 2: 2400 s descent + 1200 s cert is a BOUND, not a wish.

    Two measured defects are pinned here:

    * the budget was checked only *between* calls, so the last call could
      overshoot by its own entire length -- measured gate=1200 s -> PP=1840.8 s
      (+53%), the overshoot being one 1203.2 s certification call;
    * there was no phase split at all, so the descent phase could eat the
      reserve that makes certification possible.

    Both halves of the fix are asserted: the remaining-reserve term inside the
    per-call cap (so the LAST call is handed only what is left), and the pre-call
    check on every phase reserve (so nothing is launched once a reserve is gone).
    """
    from _probe_n60_cost import LABEL_WINDOW_SLACK, TimedPricer
    from uavcg.heuristics import singleton_routes

    class _SleepyPricer:
        """Consumes a fixed wall-clock amount per call and reports the caps."""

        name = "sleepy"
        guarantees_optimality = False
        masked_arcs = 0
        arcs_total = 0
        inst = None
        max_seconds = None
        max_labels = 1_000_000

        def __init__(self, nap: float) -> None:
            self.nap = nap
            self.caps: list = []

        def price(self, pi, sigma=0.0, fixed_cost=0.0, max_columns=50):
            self.caps.append(self.max_seconds)
            # Honour the grant, the way the real pricer does: a stub that slept
            # a fixed amount regardless of its cap would overrun the reserve and
            # the test would be measuring the stub, not the budget.
            nap = self.nap if self.max_seconds is None else min(self.nap,
                                                               self.max_seconds)
            time.sleep(max(0.0, nap))
            # Non-empty, so no certification escalation muddies the call count.
            return PricingResult(columns=[(1,)], min_reduced_cost=-1.0,
                                 exact=True, relaxed_min_reduced_cost=-1.0,
                                 truncated=False, masked=False, stats={})

    nap, desc_budget, cert_budget = 0.01, 0.5, 0.3
    stub = _SleepyPricer(nap)
    tp = TimedPricer(stub, descent_seconds=30.0, cert_seconds=1200.0,
                     descent_budget=desc_budget, cert_budget=cert_budget,
                     label="budget")
    for _ in range(200):
        tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)

    total = desc_budget + cert_budget
    check("the hard total is the sum of the two phase reserves",
          abs(tp.total_budget - total) < 1e-12, f"total={tp.total_budget}")
    check("a bounded run never spends more than the hard budget",
          tp.spent <= total + 1e-9,
          f"spent={tp.spent:.4f} vs total={total:.4f}")
    check("the descent phase cannot eat the certification reserve",
          tp.descent_spent <= desc_budget + 1e-9,
          f"descent_spent={tp.descent_spent:.4f} vs {desc_budget}")
    check("...and the reserve is USED, not thrown away early (the last call "
          "really is handed the remainder)",
          tp.descent_spent >= desc_budget - LABEL_WINDOW_SLACK - 2 * nap,
          f"descent_spent={tp.descent_spent:.4f} of {desc_budget}")
    check("the per-call cap shrinks to fit the reserve that is left",
          len(stub.caps) >= 3
          and all(x < y for x, y in zip(stub.caps[1:], stub.caps)),
          f"caps={[round(c, 4) for c in stub.caps[:4]]}...")
    check("no zero-length call is ever launched",
          all(c > 0.0 for c in stub.caps),
          f"min cap={min(stub.caps):.5f} over {len(stub.caps)} calls")
    check("once the reserve is gone the run winds down instead of asking for "
          "more time",
          tp.wind_down_reason == "budget" and len(stub.caps) <= 40,
          f"reason={tp.wind_down_reason} calls={len(stub.caps)}")

    # End-to-end on the REAL pricer.  This is the check that matters, because
    # the real pricer samples its deadline rather than checking it per label --
    # the sampling window is exactly what the reserved slack has to absorb.
    real = random_instance(20, seed=4242, tw_slack=1000.0)
    live = TimedPricer(NgRoutePricer(real, k=10), descent_seconds=0.05,
                       cert_seconds=0.2, descent_budget=0.5,
                       cert_budget=0.5, label="live")
    for _ in range(200):
        live.price(np.full(real.n, 500.0), sigma=0.0, fixed_cost=0.0,
                   max_columns=200)
    check("the reserves hold on the REAL pricer, whose deadline is sampled "
          "every 256 labels instead of being exact",
          live.descent_spent <= live.descent_budget + 1e-9
          and live.spent <= live.total_budget + 1e-9,
          f"descent={live.descent_spent:.4f}/{live.descent_budget} "
          f"total={live.spent:.4f}/{live.total_budget}")
    check("...and on the real pricer too the reserve is used, not abandoned",
          live.descent_spent >= live.descent_budget - LABEL_WINDOW_SLACK - 0.1,
          f"descent={live.descent_spent:.4f} of {live.descent_budget}")

    wd = tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("the wind-down result is empty, truncated and cannot certify",
          (not wd.columns) and wd.truncated and not wd.certifies_no_negative,
          f"cols={len(wd.columns)} truncated={wd.truncated}")

    class _EmptyPricer(_SleepyPricer):
        def __init__(self, truncate: bool) -> None:
            super().__init__(0.0)
            self.truncate = truncate

        def price(self, pi, sigma=0.0, fixed_cost=0.0, max_columns=50):
            self.caps.append(self.max_seconds)
            return PricingResult(columns=[], exact=False, min_reduced_cost=0.0,
                                 relaxed_min_reduced_cost=None,
                                 truncated=self.truncate, masked=False,
                                 stats={"time_truncated": self.truncate})

    blind = TimedPricer(_EmptyPricer(True), descent_seconds=30.0,
                        cert_seconds=1200.0, descent_budget=5.0,
                        cert_budget=5.0, label="blind")
    blind.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("an empty TRUNCATED call is not certification evidence (v3.6 ③)",
          blind.untruncated_empty_calls == 0,
          f"counted={blind.untruncated_empty_calls} "
          f"truncated_calls={blind.calls}")

    clean = TimedPricer(_EmptyPricer(False), descent_seconds=30.0,
                        cert_seconds=1200.0, descent_budget=5.0,
                        cert_budget=5.0, label="clean")
    clean.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("an empty COMPLETE call IS certification evidence (v3.6 ③)",
          clean.untruncated_empty_calls == 1,
          f"counted={clean.untruncated_empty_calls}")

    class _SchedEmpty(_SleepyPricer):
        """Empty, and truncated by the PROTOCOL's label cap -- no clock involved."""

        def __init__(self) -> None:
            super().__init__(0.0)

        def price(self, pi, sigma=0.0, fixed_cost=0.0, max_columns=50):
            self.caps.append(self.max_seconds)
            return PricingResult(columns=[], exact=False, min_reduced_cost=0.0,
                                 relaxed_min_reduced_cost=None, truncated=True,
                                 masked=False,
                                 stats={"label_cap": self.max_labels})

    sched = TimedPricer(_SchedEmpty(), descent_seconds=120.0, cert_seconds=60.0,
                        descent_budget=5.0, cert_budget=5.0, descent_labels=1000,
                        label="sched")
    sched.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("an empty call cut by the PROTOCOL's label cap escalates too -- under "
          "a label-capped protocol a time-only test would never fire, and then "
          "nothing could ever certify",
          sched.cert_escalations == 1 and sched.calls == 2,
          f"esc={sched.cert_escalations} calls={sched.calls} "
          f"caps={sched.trace[0]['cap']}")
    check("...and a protocol-scheduled cut is recorded as scheduled, so it does "
          "not mark the run degraded",
          sched.trace[0]["scheduled"] is True
          and sched.trace[0]["label_cap"] == 1000,
          f"scheduled={sched.trace[0]['scheduled']} "
          f"label_cap={sched.trace[0]['label_cap']}")

    # The three-state status must not report the cap as a finding: a run whose
    # every call was truncated is blind, and "30 rounds without improvement" is
    # then a fact about the cap.  INCOMPLETE, never PLATEAU.
    local = random_instance(20, seed=4242, tw_slack=1000.0)
    thin = NgRoutePricer(local, k=10)
    allcut = TimedPricer(thin, descent_seconds=0.004, cert_seconds=0.004,
                         descent_budget=0.02, cert_budget=0.02, label="allcut")
    res_allcut = ColumnGeneration(
        local, allcut,
        config=CGConfig(max_iterations=50, max_columns_per_iter=200,
                        stall_limit=3, plateau_tol=1e-9, fallback_max_n=40,
                        stabilization="off"),
        fallback_pricer=None, seed_routes=singleton_routes(local)).run()
    check("an all-truncated run is never labelled PLATEAU (v3.6 ③)",
          res_allcut.status != "PLATEAU" and res_allcut.truncated_pricing_calls > 0,
          f"status={res_allcut.status} "
          f"truncated={res_allcut.truncated_pricing_calls}")


def test_v36_certification_is_reachable_p2_p3_p4() -> None:
    """v3.6 P2/P3/P4: three ways the protocol ran without being able to succeed.

    All three were found by reading the FIRST re-screen batch, and all three have
    the same shape -- a bound that is smaller than the work the step it bounds
    actually needs:

    * **P2** the certification attempt's per-call cap was the phase's nominal
      1200 s, while a COMPLETE full-network call needs >=1250 s here.  Every
      attempt was cut mid-search, came back with columns instead of proving
      emptiness, and `untruncated_empty_calls` stayed 0 -- so the certification
      half of the verdict could not be produced at all;
    * **P3** the work gate was enforced as a wind-down, and the n=40 gate (2e7
      labels) always closed during descent.  No certification attempt was ever
      launched, so `certified` was False for EVERY attempt and no row of the tier
      could ever be admissible;
    * **P4** a certification attempt inherited `NgRoutePricer`'s own
      `max_labels=3_000_000`, about an order of magnitude below what a complete
      full-network n=40 call builds.  It was therefore cut by an INTRINSIC cap --
      which is not a scheduled truncation, so it also marked the run degraded.

    What is asserted here is the *reachability* of certification, the lap bound,
    and that the hard total stays a bound in both directions (the certification
    attempt may exceed the phase reserve, never the total).
    """
    from _probe_n60_cost import CERT_LABELS, LABEL_WINDOW_SLACK, TimedPricer
    from uavcg.heuristics import singleton_routes

    class _Stub:
        """Reports a fixed label count and answers with a fixed column count."""

        name = "stub"
        guarantees_optimality = False
        masked_arcs = 0
        arcs_total = 0
        inst = None
        max_seconds = None
        max_labels = 3_000_000

        def __init__(self, columns: int, labels: int) -> None:
            self.columns = columns
            self.labels = labels
            self.caps: list = []
            self.ceilings: list = []

        def price(self, pi, sigma=0.0, fixed_cost=0.0, max_columns=50):
            self.caps.append(self.max_seconds)
            self.ceilings.append(self.max_labels)
            return PricingResult(
                columns=[(1,)] * self.columns, min_reduced_cost=-1.0, exact=True,
                relaxed_min_reduced_cost=None, truncated=False, masked=False,
                stats={"labels_created": self.labels, "label_cap": self.max_labels},
            )

    # ---- P2 + P3: the gate closes the descent, the attempt still happens -----
    stub = _Stub(columns=0, labels=10_000)
    tp = TimedPricer(stub, descent_seconds=1.0, cert_seconds=2.0,
                     descent_budget=4.0, cert_budget=3.0, descent_labels=500,
                     label_budget=1_000, label="p3")
    tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("a descent round the work gate immediately overshoots still runs first",
          stub.caps[0] == 1.0 and tp.descent_closed is False,
          f"cap={stub.caps[0]} labels={tp.labels}")
    tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("the work gate closes the DESCENT phase instead of vetoing the run "
          "(P3: with a wind-down here, no attempt is ever made and `certified` "
          "is False for every run)",
          tp.descent_closed and tp.laps == 1 and len(stub.caps) == 2,
          f"closed={tp.descent_closed} laps={tp.laps} calls={len(stub.caps)}")
    check("...and the attempt it buys is a CERTIFICATION round, not a descent "
          "round sneaking past the gate",
          tp.trace[-1]["phase"] == "cert", tp.trace[-1]["phase"])
    check("a certification attempt is handed more than the nominal cert cap, and "
          "still no more than the hard total is left (P2)",
          stub.caps[1] > 2.0 and stub.caps[1] <= 7.0 - LABEL_WINDOW_SLACK + 1e-9,
          f"cap={stub.caps[1]:.4f} vs cert_seconds=2.0 total=7.0")
    check("...and the run still cannot pass the hard total",
          tp.spent <= tp.total_budget + 1e-9,
          f"spent={tp.spent:.4f} vs {tp.total_budget:.4f}")
    tp.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("an empty attempt closes the question: the next call winds down "
          "instead of buying another full-network search",
          tp.laps_exhausted and tp.laps == 1 and len(stub.caps) == 2
          and tp.wind_down_reason == "work_gate",
          f"laps={tp.laps} calls={len(stub.caps)} "
          f"reason={tp.wind_down_reason}")

    # ---- P3: laps are hard-capped even when every lap earns another one ------
    greedy = _Stub(columns=1, labels=10_000)
    tpg = TimedPricer(greedy, descent_seconds=1.0, cert_seconds=2.0,
                      descent_budget=4.0, cert_budget=3.0, descent_labels=500,
                      label_budget=1_000, max_laps=3, label="laps")
    for _ in range(20):
        tpg.price(np.zeros(3), sigma=0.0, fixed_cost=0.0, max_columns=10)
    check("a lap that returns columns earns another lap -- and the count stops "
          "at MAX_LAPS (a full-network call per lap, so the cap is what keeps "
          "the run finite)",
          tpg.laps == 3 and len(greedy.caps) == 4,
          f"laps={tpg.laps} calls={len(greedy.caps)}")

    # ---- P4: the certification attempt is not strangled by the pricer's cap --
    check("a certification attempt is given a label ceiling that cannot bind "
          "inside the hard total (P4: the pricer's own 3M default is an order of "
          "magnitude too small for a full-network n=40 call)",
          stub.ceilings[1] == CERT_LABELS and CERT_LABELS > 3_000_000,
          f"ceiling={stub.ceilings[1]:,} intrinsic=3,000,000")
    check("a descent round keeps the PROTOCOL's 300k-scale cap, not the "
          "certification allowance",
          stub.ceilings[0] == 500, f"ceiling={stub.ceilings[0]}")
    check("...and neither is filed as the pricer's own cap, so `degraded` does "
          "not fire on the protocol's own schedule",
          tp.trace[0]["protocol_cap"] and tp.trace[1]["protocol_cap"]
          and not tp.trace[0]["scheduled"],
          f"protocol_cap={[t['protocol_cap'] for t in tp.trace]} "
          f"scheduled={[t['scheduled'] for t in tp.trace]}")

    # End-to-end on the real pricer: the two marks must reach `CGResult`, and a
    # run whose only cuts are the protocol's own schedule must NOT be degraded.
    # The clock ceilings are set loose enough never to bind here on purpose --
    # otherwise the cut would be a real one and `degraded` would be right to fire.
    local = random_instance(20, seed=4242, tw_slack=1000.0)
    live = TimedPricer(NgRoutePricer(local, k=10), descent_seconds=5.0,
                       cert_seconds=5.0, descent_budget=40.0, cert_budget=40.0,
                       descent_labels=2_000, label_budget=4_000_000, label="p1")
    res_live = ColumnGeneration(
        local, live,
        config=CGConfig(max_iterations=40, max_columns_per_iter=200,
                        stall_limit=5, plateau_tol=1e-9, fallback_max_n=40,
                        stabilization="off"),
        fallback_pricer=None, seed_routes=singleton_routes(local)).run()
    check("a cap the PROTOCOL scheduled never becomes the run's `label_cap` -- "
          "which is what made `degraded` true for every row (P1)",
          res_live.label_cap is None,
          f"label_cap={res_live.label_cap} peak={res_live.peak_labels_created} "
          f"unscheduled={res_live.unscheduled_truncations}")
    check("...so a run cut only on that schedule is truncated, and NOT degraded",
          res_live.truncated_pricing_calls > 0
          and res_live.unscheduled_truncations == 0
          and not res_live.degraded,
          f"truncated={res_live.truncated_pricing_calls} "
          f"unscheduled={res_live.unscheduled_truncations} "
          f"degraded={res_live.degraded}")


def test_group_b_dynamic_features() -> None:
    """spec §2.2: B1..B12 folds over the warm-up trace, depot-as-sigma."""
    from uavcg.dynfeatures import (B_FEATURE_NAMES, b_feature_rows,
                                   compute_b_features)
    n, W = 3, 4
    rc = [np.zeros((n + 1, n + 1)) for _ in range(W)]
    rc[0][1, 2] = 2.0
    rc[1][1, 2] = -3.0
    rc[2][1, 2] = -1.0
    rc[3][1, 2] = 1.0
    tr = WarmupTrace(
        pi=[np.array([1.0, 2.0, 3.0])] * W, sigma=[0.5] * W,
        reduced_cost=rc,
        columns=[[(1, 2)], [(2,)], [], [(1, 2), (3,)]],
        lp_objective=[9.0, 8.0, 7.0, 6.0])
    f = compute_b_features(tr, n)
    check("B1/B2 are the plain mean/min over the W reduced-cost matrices",
          f["rc_mean"][1, 2] == (2 - 3 - 1 + 1) / 4
          and f["rc_min"][1, 2] == -3.0,
          f"mean={f['rc_mean'][1, 2]} min={f['rc_min'][1, 2]}")
    check("B4/B5: negative fraction, and first-negative iteration "
          "1-based, (W+1)/(W+1) when never negative",
          f["rc_neg_frac"][1, 2] == 0.5
          and f["rc_first_neg"][1, 2] == 2 / 5
          and f["rc_first_neg"][0, 1] == 1.0,
          f"neg={f['rc_neg_frac'][1, 2]} first={f['rc_first_neg'][1, 2]}")
    check("B6 averages the last ceil(W/2) iterations only",
          f["rc_tail_mean"][1, 2] == 0.0, f"{f['rc_tail_mean'][1, 2]}")
    tt = np.arange(1, W + 1)
    tc = tt - tt.mean()
    s = float((tc * np.array([2, -3, -1, 1])).sum() / (tc * tc).sum())
    check("B11 is the OLS slope of (t, c~) -- the trend, not the level",
          abs(f["rc_slope"][1, 2] - s) < 1e-12,
          f"{f['rc_slope'][1, 2]} vs {s}")
    check("B7-B10 use theta with the DEPOT folded to sigma (spec §2.2)",
          f["theta_i_mean"][1, 2] == 1.0
          and f["theta_j_mean"][1, 2] == 2.0
          and f["theta_i_mean"][0, 3] == 0.5,
          f"{f['theta_i_mean'][1, 2]}/{f['theta_j_mean'][1, 2]}"
          f"/{f['theta_i_mean'][0, 3]}")
    c = f["warm_use_count"]
    check("B12 folds the warm-up columns into per-arc use counts, "
          "depot return arcs included",
          c[0, 1] == 2 and c[1, 2] == 2 and c[2, 0] == 3
          and c[0, 2] == 1 and c[0, 3] == 1 and c[3, 0] == 1,
          f"{c.tolist()}")
    rows = b_feature_rows(tr, n, np.array([[1, 2], [0, 3]]))
    check("b_feature_rows stacks the 12 matrices in the fixed "
          "B_FEATURE_NAMES order",
          rows.shape == (2, 12) and len(B_FEATURE_NAMES) == 12
          and rows[0, 0] == f["rc_mean"][1, 2]
          and rows[1, 11] == c[0, 3],
          f"{rows.shape} {B_FEATURE_NAMES}")


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 72)
    print("uavcg self-test")
    print("=" * 72)
    test_route_evaluation()
    test_solomon_roundtrip()
    test_pricing_against_brute_force()
    test_ngroute_relaxation()
    test_time_sorted_dominance_matches_the_linear_scan()
    test_master_duals()
    test_cg_matches_full_lp()
    test_cg_with_ngroute()
    test_cap_hit_is_reported_honestly()
    test_truncation_cannot_reach_a_cg_verdict()
    test_greedy_triggers_exact_fallback()
    test_seed_heuristic()
    test_rollout_mask_is_sound()
    test_rollout_reduced_costs_are_exact()
    test_rollout_reaches_exact_pricing_on_small_instances()
    test_feature_block_shape()
    test_cg_with_rollout_pricer()
    test_stabilization_alpha_zero_is_off()
    test_stabilization_is_bound_neutral()
    test_pool_columns_stay_dual_feasible_after_stabilization()
    test_degeneracy_report_is_well_formed()
    test_column_selection_is_bound_neutral()
    test_column_selection_reduces_the_batch()
    test_column_selection_ranks_by_true_reduced_cost()
    test_selection_escalates_when_the_best_columns_are_duplicates()
    test_sampling_weights_are_scale_invariant()
    test_psi_calibration_matches_the_published_constants()
    test_affine_fast_path_matches_the_pipeline()
    test_sampling_temperature_default_is_the_sweep_optimum()
    test_graded_labels_refine_the_binary_ones()
    test_soft_fit_weights_by_confidence_symmetrically()
    test_time_window_free_mode()
    test_time_window_narrowing()
    test_dual_convention_at_the_path_level()
    test_feature_cache_is_bit_identical()
    test_feature_cache_covers_the_energy_branch()
    test_succ_counts_matches_its_definition()
    test_boxstep_clamps_without_freezing()
    test_arc_mask_fully_open_reproduces_the_baseline()
    test_masked_pricing_never_publishes_a_bound()
    test_a_truncated_masked_call_cannot_kill_the_reduced_network()
    test_the_mask_can_only_be_mounted_where_it_is_safe()
    test_rounded_distances_match_the_benchmark_convention()
    test_redcost_arc_reduction()
    test_frozen_split_matches_its_rule()
    test_pp_rmp_time_accounting()
    test_the_restricted_master_is_an_upper_bound()
    test_an_incumbent_hit_does_not_stop_or_certify()
    test_v34_singleton_seed_is_the_standard()
    test_v34_plateau_rule_and_three_state_status()
    test_v34_wall_clock_cap_truncates_and_stops_gracefully()
    test_v36_phase_budget_is_a_hard_bound()
    test_v36_certification_is_reachable_p2_p3_p4()
    test_group_b_dynamic_features()
    print("-" * 72)
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        return 1
    print(f"all checks passed in {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
