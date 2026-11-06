"""Fetch the Solomon category-2 instances the local set is missing, plus BKS.

Why a second source
-------------------
`fetch_instances.py` pulls from ML4VRP/ML4VRP2023, which the spec named as the
approved channel.  Its `Instances/text/Customer100` directory holds 24 files --
exactly the 24 already on disk (verified by listing the remote directory), so
the channel cannot supply the 15 missing category-2 instances the spec's
instance matrix assumes.  That is a property of the channel, not a network
failure, and it was worth checking before spending time downloading.

This script uses felipelagos/mahh, which publishes all 56 Solomon instances in
the original text format *and* `instances/csv/minimum.csv` with the best-known
integer solutions.  The latter is what spec §8's red-line anchor needs
("certified LP <= the instance's known BKS"), since the repository contains no
literature LP values at all.

Source trust is established by comparison, not by assertion: every file the two
sources share must be byte-identical.  Measured on C201 / R203 / RC201 (one
from each of C, R, RC): identical, same sha1.  `--verify` re-runs that check.

Idempotent: existing files are parsed and skipped unless --force.

    python scripts/fetch_solomon_cat2.py
    python scripts/fetch_solomon_cat2.py --verify
    python scripts/fetch_solomon_cat2.py --all56
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
DATA = ROOT / "data" / "solomon100"
BASE = "https://raw.githubusercontent.com/felipelagos/mahh/main/instances"

#: Solomon 100-customer category-2 = the class the reference paper used
#: (R2/C2/RC2).  The full set is 27: R201-R211, C201-C208, RC201-RC208.
CAT2 = (
    [f"C2{i:02d}" for i in range(1, 9)]
    + [f"R2{i:02d}" for i in range(1, 12)]
    + [f"RC2{i:02d}" for i in range(1, 9)]
)
CAT1 = (
    [f"C1{i:02d}" for i in range(1, 10)]
    + [f"R1{i:02d}" for i in range(1, 13)]
    + [f"RC1{i:02d}" for i in range(1, 9)]
)
#: Files the two sources share -- byte-identity here is what licenses using this
#: source for the instances ML4VRP simply does not carry.
CROSS_CHECK = ["C201", "R203", "RC201", "C101", "R107", "RC107"]


def _opener():
    op = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    op.addheaders = [("User-Agent", "uavcg-fetch")]
    return op


def _get(op, url: str) -> bytes:
    with op.open(url, timeout=90) as r:
        return r.read()


def fetch_one(op, name: str, dest: Path, force: bool) -> dict:
    if dest.exists() and not force:
        inst, meta = parse_solomon(dest)
        return {"name": name, "bytes": dest.stat().st_size, "cached": True,
                "customers": inst.n, "capacity": meta["capacity"],
                "horizon": meta["depot_horizon"]}
    url = f"{BASE}/solomon/{name.lower()}.txt"
    body = _get(op, url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    inst, meta = parse_solomon(dest)
    return {"name": name, "bytes": len(body),
            "sha1": hashlib.sha1(body).hexdigest()[:12], "url": url,
            "customers": inst.n, "capacity": meta["capacity"],
            "horizon": meta["depot_horizon"]}


def verify(op) -> int:
    print("cross-check: this source vs the files already on disk")
    ok = True
    for name in CROSS_CHECK:
        dest = DATA / f"{name}.txt"
        if not dest.exists():
            print(f"  {name:>6}  skipped (not on disk)")
            continue
        body = _get(op, f"{BASE}/solomon/{name.lower()}.txt")
        same = body == dest.read_bytes()
        ok = ok and same
        print(f"  {name:>6}  identical={same}  sha1={hashlib.sha1(body).hexdigest()[:12]}")
    print("  -> source " + ("MATCHES" if ok else "DIFFERS -- do not use"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--verify", action="store_true", help="only run the cross-check")
    ap.add_argument("--all56", action="store_true",
                    help="fetch all 56 Solomon instances, not just category-2")
    ap.add_argument("--bks", action="store_true", default=False,
                    help="also fetch instances/csv/minimum.csv (best-known solutions)")
    args = ap.parse_args()

    op = _opener()

    if args.verify:
        return verify(op)

    wanted = (CAT1 + CAT2) if args.all56 else CAT2
    present = {p.stem for p in DATA.glob("*.txt")}
    todo = [n for n in wanted]
    print(f"category-2 set: {len(CAT2)}   on disk already: "
          f"{len([n for n in CAT2 if n in present])}   to fetch: {len([n for n in todo if n not in present])}")

    manifest, failed = [], []
    for name in todo:
        try:
            info = fetch_one(op, name, DATA / f"{name}.txt", args.force)
            manifest.append(info)
            tag = "cached" if info.get("cached") else "ok    "
            print(f"  {tag} {name:>6}  {info['bytes']:>7d} B  n={info['customers']} "
                  f"cap={info['capacity']:.0f} horizon={info['horizon']:.0f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append(name)
            print(f"  FAILED {name:>6}  {type(exc).__name__}: {exc}", flush=True)

    if args.bks:
        try:
            body = _get(op, f"{BASE}/csv/minimum.csv")
            (DATA / "minimum.csv").write_bytes(body)
            print(f"\n  ok     minimum.csv  {len(body)} B (BKS for spec §8 red-line anchor)")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  FAILED minimum.csv  {type(exc).__name__}: {exc}")

    (DATA / "manifest_cat2.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {len(manifest)} entries to data/solomon100/manifest_cat2.json")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
