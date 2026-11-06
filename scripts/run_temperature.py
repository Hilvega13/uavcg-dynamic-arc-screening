"""Sweep the sampling temperature: the experiment that reproduces the paper's step 2.

The question this answers
-------------------------
The reference paper turns a linear-SVM distance `d` into a probability in two
steps: the SVM gives a signed distance, and a logistic map

    psi(d) = 1 / (1 + exp(-(theta0 * d + theta1))),  theta0 = 9.7750, theta1 = 12.5564

turns that distance into something that can be normalised into sampling
weights.  Our scorer skipped the second step: `ArcScorer.logit` hands the raw
`decision_function` output straight to the sampler.

That was a plausible explanation for a result this project has had to live
with - a linear SVM scoring no better than a plain logistic regression
(0.764 vs 0.766 cross-instance AUC) - because an uncalibrated SVM distance is
on an arbitrary scale, and feeding an arbitrary scale into a softmax is exactly
how a sampler ends up with the wrong amount of exploration.  A scale that is
too small makes the arm nearly uniform; too large makes it a greedy argmax.
Either way the "learning does not help" conclusion could have been an artefact
of the temperature rather than a fact about the model.

It is not, and the reason is structural.  `RolloutPricer._weights` standardises
the scores within every decision step before applying the temperature, so any
*affine* rescaling of the score cancels out.  psi is affine in d, so it is a
no-op in this sampler - asserted by a self-test rather than asserted in prose.
The paper's theta0 is, functionally, a temperature; here that role belongs to
`beta`.  So the faithful way to reproduce the paper's second step is to sweep
`beta`, not to re-fit its two constants.

What is measured
----------------
Two things, because they answer different questions:

  * `mean_negative_per_call` - how many genuinely negative columns each pricing
    call delivers.  This is the pricer's direct contribution and it does not
    depend on any stopping rule, so it is the honest measure of whether the
    learned score is picking better routes.
  * the LP bound reached under a fixed iteration budget, plus whether the run
    certified.  This is what a reader will care about, and it is the number
    that can be misleading when a run is truncated.

Equal `beta` is meaningful across scoring modes precisely because of the
standardisation: one unit is always one standard deviation of that step's
scores.  Without it the two arms would be exploring at unrelated rates and the
whole comparison would be an artefact.

Usage
-----
    python scripts/run_temperature.py --names R105
    python scripts/run_temperature.py --names R105,RC107 --betas 0.25,1,4
    python scripts/run_temperature.py --from-json results/temperature.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import run_experiment as rx
from uavcg.bench import load_instance
from uavcg.learning import ArcScorer
from uavcg.pricers.rollout import RolloutPricer

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

DEFAULT_NAMES = ["R105", "RC107"]
DEFAULT_BETAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def make_args(args, beta: float, name: str):
    return argparse.Namespace(
        samples=args.samples, beta=beta, seed=args.seed, no_prefix=False,
        pool=args.pool, exact_max_n=args.exact_max_n,
        fallback_max_n=args.fallback_max_n, ng_k=args.ng_k,
        max_iters=args.max_iters, max_cols=args.max_cols, stall=args.stall,
        target_gap=args.target_gap, data=args.data, n=args.n, uav=args.uav,
        load_factor=args.load_factor, no_tw=args.no_tw,
        greedy_restarts=60, warmup_samples=args.warmup_samples,
        instances=name, model=args.model,
    )


def analyse(row: dict) -> dict:
    """Extra readings that the raw row does not carry."""
    curve = row["curve"]
    vals = [c[2] for c in curve]
    seed = vals[0] if vals else float("nan")
    final = row["lp_objective"]
    gain = seed - final
    improving = 0
    for prev, cur in zip(vals, vals[1:]):
        if cur < prev - 1e-6:
            improving += 1
    with_cols = sum(1 for c in curve if c[3] > 0)
    return {
        "seed_lp": seed,
        "gain": gain,
        "improving_iters": improving,
        "with_columns": with_cols,
        "wasted_fraction": (1.0 - improving / with_cols) if with_cols else 0.0,
    }


def report(rows, betas, names, pricers) -> dict:
    summary = {"betas": betas, "uav": {}, "verdict": {}}

    for name in names:
        per = [r for r in rows if r["instance"] == name]
        if not per:
            continue
        print()
        print("=" * 116)
        print(f"=== {name}  n={per[0]['n']}  samples={per[0]['samples']}  "
              f"budget={per[0]['max_iters']} iterations ===")
        print("=" * 116)

        ref = per[0].get("reference_lp")
        hdr = (f"  {'pricer':>16} {'beta':>6} {'LP':>11} {'gap%':>7} {'cert':>5} "
               f"{'it':>4} {'cols':>6} {'neg/call':>9} {'dead%':>7} {'waste':>6} "
               f"{'wall':>9}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for kind in pricers:
            for b in betas + [None]:
                r = next((x for x in per
                          if x["pricer"] == kind and x["beta"] == b), None)
                if r is None:
                    continue
                a = analyse(r)
                gap = (100.0 * (r["lp_objective"] - ref) / abs(ref)
                       if ref else float("nan"))
                de = r["mean_dead_end_rate"]
                bt = "greedy" if b is None else f"{b:g}"
                print(f"  {kind:>16} {bt:>6} {r['lp_objective']:11.3f} {gap:7.3f} "
                      f"{str(r['lp_certified'])[:1]:>5} {r['iterations']:4d} "
                      f"{r['columns_in_pool']:6d} {r['mean_negative_per_call']:9.2f} "
                      f"{(100 * de if de is not None else float('nan')):7.2f} "
                      f"{a['wasted_fraction']:6.0%} {r['wall_clock']:8.2f}s")

        # ---- the actual question: can the model beat the dual at any temperature?
        dual = [r for r in per if r["pricer"] == "rollout-dual"]
        model = [r for r in per if r["pricer"] == "rollout-model"]
        if dual and model:
            tag = lambda r: "greedy" if r["beta"] is None else f"{r['beta']:g}"
            # Two readings, deliberately.  Each arm's own best temperature is
            # the fair one - it is what a user would actually run after tuning.
            # The matched-temperature reading is included because it is the
            # only one that isolates the score itself, with the exploration
            # rate held fixed; it can point the other way, and when it does,
            # the reason is that a temperature which flatters one score
            # degrades the other, not that one score is better.
            bd = max(dual, key=lambda r: r["mean_negative_per_call"])
            bm = max(model, key=lambda r: r["mean_negative_per_call"])
            print()
            print("  best configuration per arm, judged by negative columns per call")
            print(f"    dual  at beta={tag(bd)}: "
                  f"{bd['mean_negative_per_call']:.2f} neg/call, "
                  f"LP {bd['lp_objective']:.3f}, {bd['wall_clock']:.2f}s")
            print(f"    model at beta={tag(bm)}: "
                  f"{bm['mean_negative_per_call']:.2f} neg/call, "
                  f"LP {bm['lp_objective']:.3f}, {bm['wall_clock']:.2f}s")
            ratio = (bm["mean_negative_per_call"] / bd["mean_negative_per_call"]
                     if bd["mean_negative_per_call"] else float("nan"))
            cost = (bm["wall_clock"] / bd["wall_clock"]
                    if bd["wall_clock"] else float("nan"))
            print(f"    model / dual at their own best temperature: "
                  f"{ratio:.3f}x the columns for {cost:.2f}x the wall clock")
            matched = []
            for b in betas:
                d = next((r for r in dual if r["beta"] == b), None)
                m = next((r for r in model if r["beta"] == b), None)
                if d and m and d["mean_negative_per_call"]:
                    matched.append((b, m["mean_negative_per_call"]
                                    / d["mean_negative_per_call"]))
            best = max(matched, key=lambda t: t[1]) if matched else None
            if best:
                dd = next(r for r in dual if r["beta"] == best[0])
                mm = next(r for r in model if r["beta"] == best[0])
                print(f"    at a matched temperature the best ratio is "
                      f"{best[1]:.3f}x (beta={best[0]:g}); there the dual arm "
                      f"delivers {dd['mean_negative_per_call']:.2f} neg/call "
                      f"and the model {mm['mean_negative_per_call']:.2f}")
            summary["verdict"][name] = {
                "dual_best_beta": bd["beta"],
                "dual_best_neg_per_call": bd["mean_negative_per_call"],
                "model_best_beta": bm["beta"],
                "model_best_neg_per_call": bm["mean_negative_per_call"],
                "ratio_at_own_best": ratio,
                "cost_ratio_at_own_best": cost,
                "ratio_at_matched_best": best[1] if best else None,
                "ratio_at_matched_beta": best[0] if best else None,
                "dual_best_lp": bd["lp_objective"],
                "model_best_lp": bm["lp_objective"],
            }

        # ---- does temperature matter at all, and in which direction?
        for kind in pricers:
            sel = [r for r in per if r["pricer"] == kind]
            # `beta = None` is the deterministic greedy pass, not a temperature;
            # it is reported in the table but has to stay out of the sweep
            # statistics, which are about how the sampler's temperature behaves.
            swept = [r for r in sel if r["beta"] is not None]
            if len(swept) >= 3:
                negs = [r["mean_negative_per_call"] for r in swept]
                b = [r["beta"] for r in swept]
                lo, hi = int(np.argmin(negs)), int(np.argmax(negs))
                summary["uav"][f"{kind}|{name}"] = {
                    "betas": b, "neg_per_call": negs,
                    "best_beta": b[hi], "best_neg_per_call": negs[hi],
                    "spread": max(negs) - min(negs),
                    "walls": [r["wall_clock"] for r in swept],
                }
                if max(negs) - min(negs) < 1e-9:
                    print(f"\n  {kind}: neg/call is {negs[0]:.2f} at every beta "
                          f"- the score carries no information, so there is "
                          f"nothing for a temperature to sharpen")
                else:
                    print(f"\n  {kind}: neg/call ranges {min(negs):.2f} to "
                          f"{max(negs):.2f} over beta in "
                          f"[{min(b):g}, {max(b):g}], best at beta={b[hi]:g}")

    print()
    print("=" * 116)
    print("per-pricer means over all instances and temperatures")
    print("=" * 116)
    hdr = (f"{'pricer':>16} {'n':>3} {'neg/call':>10} {'dead%':>7} {'waste':>6} "
           f"{'iters':>6} {'cols':>7} {'wall':>9} {'certified':>10}")
    print(hdr)
    print("-" * len(hdr))
    for kind in pricers:
        sel = [r for r in rows if r["pricer"] == kind]
        if not sel:
            continue
        m = lambda k: sum(r[k] for r in sel) / len(sel)
        waste = sum(analyse(r)["wasted_fraction"] for r in sel) / len(sel)
        de = [r["mean_dead_end_rate"] for r in sel
              if r["mean_dead_end_rate"] is not None]
        print(f"{kind:>16} {len(sel):3d} {m('mean_negative_per_call'):10.2f} "
              f"{(100 * sum(de) / len(de) if de else float('nan')):7.2f} "
              f"{waste:6.0%} {m('iterations'):6.1f} {m('columns_in_pool'):7.1f} "
              f"{m('wall_clock'):8.2f}s "
              f"{sum(1 for r in sel if r['lp_certified']):3d}/{len(sel):<2d}")
    print()
    print("read neg/call against dead% - they are not independent.  A rollout")
    print("that dies early is truncated into a short route, and a short route is")
    print("cheap, so it is negative almost by construction.  A high neg/call with")
    print("a high dead% can therefore mean 'finds many short columns', not 'finds")
    print("good columns'.  `rollout-uniform` is the clean example: it tops")
    print("neg/call while the score it samples from carries no information at all.")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--betas", default=",".join(str(b) for b in DEFAULT_BETAS))
    ap.add_argument("--pricers", default="rollout-uniform,rollout-dual,rollout-model")
    ap.add_argument("--model", default=str(ROOT / "models" / "arc_scorer_n25.pkl"))
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--warmup-samples", type=int, default=0,
                    help="must match how the scorer was trained; the rank_freq "
                         "feature is 0 at training time when this was 0")
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--ng-k", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument("--max-cols", type=int, default=25)
    ap.add_argument("--stall", type=int, default=60)
    ap.add_argument("--target-gap", type=float, default=0.01)
    ap.add_argument("--exact-max-n", type=int, default=20,
                    help="default 20, matching run_experiment: at n=25 the exact "
                         "labeller costs ~200 s as a fallback and buys nothing "
                         "over the ng-route relaxation, which certifies just as "
                         "soundly")
    ap.add_argument("--fallback-max-n", type=int, default=120)
    ap.add_argument("--fleet", type=int, default=None)
    ap.add_argument("--uav", type=float, default=0.0)
    ap.add_argument("--load-factor", type=float, default=1.0)
    ap.add_argument("--no-tw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    ap.add_argument("--out", default=str(RESULT_DIR / "temperature.json"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--from-json", default=None)
    args = ap.parse_args()

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    pricers = [s.strip() for s in args.pricers.split(",") if s.strip()]

    scorer = None
    if "rollout-model" in pricers:
        scorer = ArcScorer.load(args.model)
        print(f"loaded scorer {scorer!r}")

    rows: list = []
    if args.from_json:
        with open(args.from_json, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        have = {r["instance"] for r in rows}
        names = [n for n in names if n in have] or sorted(have)
        print(f"re-reporting {args.from_json} ({len(rows)} runs)")
    else:
        if args.append and Path(args.out).exists():
            with open(args.out, encoding="utf-8") as fh:
                rows = json.load(fh)["rows"]
            print(f"appending to {args.out} (already {len(rows)} runs)")
        print("=" * 116)
        print(f"temperature sweep  (n={args.n}, samples={args.samples}, "
              f"betas={betas})")
        print("=" * 116)

        for name in names:
            # a certified reference: ng-route is cheap and its relaxation gives
            # a valid certificate, so it is the right yardstick for gap%
            inst = load_instance(name, args.n, args.uav, args.load_factor,
                                 args.no_tw, Path(args.data))
            ref_args = make_args(args, 1.0, name)
            ref_row = rx.run_one("ngroute", name, ref_args, None)
            reference_lp = ref_row["lp_objective"]
            ref_cert = ref_row["lp_certified"]
            print(f"  {name}: reference LP = {reference_lp:.3f} "
                  f"(ng-route, certified={ref_cert})", flush=True)

            for kind in pricers:
                for b in betas + [None]:
                    a = make_args(args, b, name)
                    try:
                        r = rx.run_one(kind, name, a, scorer)
                    except Exception as exc:
                        print(f"    {name} {kind} beta={b}: failed: {exc}",
                              flush=True)
                        continue
                    r["beta"] = b
                    r["samples"] = args.samples
                    r["max_iters"] = args.max_iters
                    r["reference_lp"] = reference_lp
                    r.update(analyse(r))
                    rows = [x for x in rows
                            if not (x["instance"] == name
                                    and x["pricer"] == kind
                                    and x["beta"] == b)] + [r]
                    print(f"    {name} {kind} beta={b}: LP={r['lp_objective']:.3f}"
                          f" cert={r['lp_certified']} it={r['iterations']}"
                          f" neg/call={r['mean_negative_per_call']:.2f}"
                          f" wall={r['wall_clock']:.2f}s", flush=True)

    summary = report(rows, betas, names, pricers)
    if not args.from_json:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "summary": summary, "rows": rows},
                      fh, indent=2)
        print(f"\nwrote {out}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
