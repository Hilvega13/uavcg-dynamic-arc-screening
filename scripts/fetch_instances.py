"""Fetch benchmark instances into ./data.

Sources (all public, no licence friction, verified reachable):
  * ML4VRP/ML4VRP2023      Solomon 100-customer and Homberger 200/400 sets,
                           published as the GECCO 2023 ML4VRP competition data
  * STOM-Group/TSP-D-Instances   TSP-with-drone instances with exact solutions

Usage:
    python scripts/fetch_instances.py --set solomon100
    python scripts/fetch_instances.py --set all --limit 6
    python scripts/fetch_instances.py --set list
    python scripts/fetch_instances.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.instance import parse_solomon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

REPO = "ML4VRP/ML4VRP2023"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
API = f"https://api.github.com/repos/{REPO}/contents"

SETS = {
    "solomon100": "Instances/text/Customer100",
    "homberger200": "Instances/text/Customer200",
    "homberger400": "Instances/text/Customer400",
}


def _opener():
    ctx = ssl.create_default_context()
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [("User-Agent", "uavcg-fetch")]
    return op


def list_remote(op, repo_path: str) -> list[str]:
    with op.open(f"{API}/{repo_path}", timeout=40) as r:
        return [d["name"] for d in json.load(r) if d["name"].endswith(".txt")]


def download(op, repo_path: str, name: str, dest: Path) -> dict:
    url = f"{RAW}/{repo_path}/{name}"
    with op.open(url, timeout=60) as r:
        body = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    info = {
        "name": dest.stem,
        "file": str(dest.relative_to(ROOT)),
        "bytes": len(body),
        "sha1": hashlib.sha1(body).hexdigest()[:12],
        "url": url,
    }
    try:
        inst, meta = parse_solomon(dest)
        info.update(
            customers=inst.n,
            capacity=meta["capacity"],
            max_vehicles=meta["num_vehicles"],
            horizon=meta["depot_horizon"],
        )
        info["parsed"] = True
    except Exception as exc:  # pragma: no cover - defensive
        info["parsed"] = False
        info["parse_error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="solomon100", help="solomon100 | homberger200 | homberger400 | all | list")
    ap.add_argument("--limit", type=int, default=0, help="download at most N per set (0 = all)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    op = _opener()

    if args.set == "list":
        for key, path in SETS.items():
            try:
                names = list_remote(op, path)
                print(f"{key:14s} {len(names):3d} instances: {', '.join(sorted(names))}")
            except Exception as exc:
                print(f"{key:14s} FAILED {type(exc).__name__}: {exc}")
        return 0

    keys = list(SETS) if args.set == "all" else [args.set]
    for key in keys:
        if key not in SETS:
            print(f"unknown set: {key}")
            return 2

    grand = 0
    for key in keys:
        repo_path = SETS[key]
        outdir = DATA / key
        print(f"\n=== {key}  <-  {REPO}/{repo_path}")
        try:
            names = sorted(list_remote(op, repo_path))
        except Exception as exc:
            print(f"  listing failed: {type(exc).__name__}: {exc}")
            continue
        if args.limit:
            names = names[: args.limit]

        manifest = []
        for name in names:
            dest = outdir / name
            if dest.exists() and not args.force:
                try:
                    inst, meta = parse_solomon(dest)
                    info = {
                        "name": dest.stem,
                        "file": str(dest.relative_to(ROOT)),
                        "bytes": dest.stat().st_size,
                        "cached": True,
                        "customers": inst.n,
                        "capacity": meta["capacity"],
                        "max_vehicles": meta["num_vehicles"],
                        "parsed": True,
                    }
                except Exception as exc:
                    info = {"name": dest.stem, "parsed": False, "parse_error": str(exc)}
                manifest.append(info)
                print(f"  cached  {info['name']:14s} n={info.get('customers')} cap={info.get('capacity')}")
                continue
            try:
                info = download(op, repo_path, name, dest)
                manifest.append(info)
                flag = "ok" if info.get("parsed") else "PARSE-FAIL"
                print(
                    f"  {flag:10s} {info['name']:14s} {info['bytes']:>7d} B  "
                    f"n={info.get('customers')} cap={info.get('capacity')} veh={info.get('max_vehicles')}"
                )
            except Exception as exc:
                print(f"  FAILED     {name:14s} {type(exc).__name__}: {exc}")

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        grand += len(manifest)

    print(f"\ntotal entries: {grand}")
    print(f"data root: {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
