"""Rank candidate instances by time-window width -- a cheap tractability proxy.

Why this exists
---------------
A single pricing call is the unit of cost, and measuring it is expensive: the
intractable cases do not return at all (R203/R207/RC203 at n=60 were still
running when `timeout` killed them, at each of x0.8/x0.6/x0.5).  But the
measurements that do exist line up with one number that costs nothing to
compute:

    C201  n=60  mean window width  160  ->   3.0 s, 191 k labels
    RC201 n=60  mean window width    ?  ->  14.3 s, 347 k labels
    C208  n=60  mean window width  640  -> >55 s untouched, 29.5 s at x0.6

Wide windows are the mechanism: they admit more feasible partial routes, which
is exactly what the labelling algorithm pays for.  Narrowing helps (C208
dropped from >55 s to 29.5 s) but only proportionally, so an instance whose
width starts an order of magnitude higher stays out of reach even at x0.5.

So: rank every candidate by width first, and only spend pricing calls on the
plausible ones.  This is a *proxy*, not a substitute -- the ranking says which
instances to probe, and the probe still decides.

    python scripts/probe_window_width.py --n 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.bench import load_instance

ROOT = Path(__file__).resolve().parents[1]


def widths(inst) -> dict:
    idx = np.arange(1, inst.n + 1)
    w = np.asarray(inst.due[idx] - inst.ready[idx], dtype=float)
    return {
        "n": inst.n,
        "width_mean": float(w.mean()),
        "width_median": float(np.median(w)),
        "width_max": float(w.max()),
        "width_min": float(w.min()),
        "depot_horizon": float(inst.due[0]),
        "frac_w_ge_300": float((w >= 300).mean()),
        "frac_w_ge_600": float((w >= 600).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "solomon100"))
    ap.add_argument("--names", default="",
                    help="comma list; default = every instance in --data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tw-scale", type=float, default=1.0,
                    help="apply the spec §3.3 narrowing before measuring, so the "
                         "width reported is the one the solver will actually see")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = Path(args.data)
    if args.names:
        names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        names = sorted(p.stem for p in data.glob("*.txt"))

    rows = []
    for name in names:
        try:
            inst = load_instance(name, n=args.n, data_dir=data, tw_scale=args.tw_scale)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"{name:>7}  SKIP  {exc}", flush=True)
            continue
        rec = {"instance": name, "tw_scale": args.tw_scale, **widths(inst)}
        rows.append(rec)

    rows.sort(key=lambda r: r["width_mean"])
    print(f"\nn={args.n}  tw_scale={args.tw_scale:g}   sorted by mean width "
          f"(lower = cheaper)\n")
    print(f"{'inst':>7} {'n':>3} {'mean':>8} {'median':>8} {'max':>8} "
          f"{'depot':>8} {'>=300':>6} {'>=600':>6}")
    for r in rows:
        print(f"{r['instance']:>7} {r['n']:>3} {r['width_mean']:>8.1f} "
              f"{r['width_median']:>8.1f} {r['width_max']:>8.1f} "
              f"{r['depot_horizon']:>8.1f} {r['frac_w_ge_300']:>6.0%} "
              f"{r['frac_w_ge_600']:>6.0%}")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
