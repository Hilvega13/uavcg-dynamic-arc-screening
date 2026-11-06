"""Frozen instance split for the main experimental tier.

The split lives in a data file (`data/splits/n60_cat2_v1.json`) rather than in
this module, because it is a *decision* with an audit trail -- which instance,
which time-window factor, measured how -- and a decision that only exists as
Python literals is a decision nobody can diff.  This module is the reader: it
turns the file into typed entries and, for convenience, straight into loaded
`Instance` objects with the per-instance narrowing already applied.

Why the narrowing factor is stored per instance
-----------------------------------------------
`tw_scale` is not a global "make it easier" knob, it is a property of the
instance: C201 is affordable untouched at x0.8 while R209 needed the x0.35
floor.  Re-deriving the factor at run time would mean re-running the probe, so
the factor measured once (spec §3.3) is frozen with the instance.  A run that
uses a different factor is a run on a different instance and must not be
compared against a table built from this file.

Why loading goes through `bench.load_instance`
---------------------------------------------
Truncation-then-narrowing order, the UAV budget derivation and the distance
convention all live in one place.  If this module re-implemented them, the
"same instance" guarantee would hold only until the next time one of the two
copies changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .instance import Instance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "n60_cat2_v1.json"

TRAIN = "train"
TEST = "test"
BUCKETS = (TRAIN, TEST)


@dataclass(frozen=True)
class SplitEntry:
    """One frozen instance: what to load, and what was measured about it."""

    name: str
    family: str
    bucket: str
    n: int
    tw_scale: float
    probe_seconds: float
    probe_labels: int
    probe_cap_frac: float
    clean: bool

    def __post_init__(self) -> None:
        if self.bucket not in BUCKETS:
            raise ValueError(f"unknown bucket {self.bucket!r}; expected {BUCKETS}")
        if not 0.0 < self.tw_scale <= 1.0:
            raise ValueError(
                f"{self.name}: tw_scale={self.tw_scale} outside (0, 1]; a factor "
                "above 1 would widen the windows, which no protocol authorises"
            )

    @property
    def label(self) -> str:
        """`C201(n60,x0.8)` — how this instance is named in a report table."""
        return f"{self.name}(n{self.n},x{self.tw_scale:g})"

    def with_tw_scale(self, tw_scale: float) -> "SplitEntry":
        """A copy at a different narrowing factor. Calibration use only.

        Deliberately not a mutation and deliberately not silently allowed
        through `Split.load`: an entry carrying a factor nobody has recorded is
        exactly the "table describes an instance that was never solved" failure
        `Split.load` guards against.  The caller has to ask for a variant by
        name, and the result is only meaningful once it has been written back
        into the split file.
        """
        import dataclasses
        return dataclasses.replace(self, tw_scale=float(tw_scale))


@dataclass(frozen=True)
class Split:
    version: str
    frozen: str
    entries: Tuple[SplitEntry, ...]
    families: Dict[str, dict]
    landing_instances: Tuple[str, ...]
    split_rule: Dict[str, str]

    # -- selection -----------------------------------------------------
    def bucket(self, bucket: str) -> Tuple[SplitEntry, ...]:
        self._check_bucket(bucket)
        return tuple(e for e in self.entries if e.bucket == bucket)

    def train(self) -> Tuple[SplitEntry, ...]:
        return self.bucket(TRAIN)

    def test(self) -> Tuple[SplitEntry, ...]:
        return self.bucket(TEST)

    def family(self, family: str, bucket: Optional[str] = None) -> Tuple[SplitEntry, ...]:
        if bucket is not None:
            self._check_bucket(bucket)
        return tuple(
            e for e in self.entries
            if e.family == family and (bucket is None or e.bucket == bucket)
        )

    def entry(self, name: str) -> SplitEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(f"{name!r} is not in split {self.version}")

    def names(self, bucket: Optional[str] = None) -> Tuple[str, ...]:
        pool = self.entries if bucket is None else self.bucket(bucket)
        return tuple(e.name for e in pool)

    def labels(self, bucket: Optional[str] = None) -> Tuple[str, ...]:
        pool = self.entries if bucket is None else self.bucket(bucket)
        return tuple(e.label for e in pool)

    # -- loading -------------------------------------------------------
    def load(self, entry: SplitEntry, **kwargs) -> Instance:
        """Materialise the frozen instance, narrowing applied.

        Extra keyword arguments go to `bench.load_instance` and must not
        contradict the frozen record: `n` and `tw_scale` are set from the entry
        and passing either again is an error, because silently overriding them
        is exactly how a table ends up describing an instance that was never
        solved.
        """
        for clash in ("n", "tw_scale", "name"):
            if clash in kwargs:
                raise TypeError(
                    f"load() takes n/tw_scale/name from the split entry; "
                    f"got an extra {clash!r}={kwargs[clash]!r}"
                )
        from .bench import load_instance  # local: bench imports this package

        return load_instance(
            entry.name, n=entry.n, tw_scale=entry.tw_scale, **kwargs
        )

    def load_bucket(self, bucket: str, **kwargs) -> List[Tuple[SplitEntry, Instance]]:
        return [(e, self.load(e, **kwargs)) for e in self.bucket(bucket)]

    # -- reporting -----------------------------------------------------
    def summary(self) -> str:
        """One table: bucket counts, mean probe cost, narrowing factors."""
        lines = [
            f"split {self.version}  (frozen {self.frozen})",
            f"{'family':6s} {'bucket':6s} {'k':>2s} {'mean_s':>8s} "
            f"{'x8':>4s} {'x6':>4s} {'x5':>4s} {'x35':>4s}  instances",
        ]
        for fam in sorted({e.family for e in self.entries}):
            for bucket in BUCKETS:
                pool = self.family(fam, bucket)
                if not pool:
                    continue
                mean = sum(e.probe_seconds for e in pool) / len(pool)
                x8 = sum(1 for e in pool if e.tw_scale == 0.8)
                x6 = sum(1 for e in pool if e.tw_scale == 0.6)
                x5 = sum(1 for e in pool if e.tw_scale == 0.5)
                x35 = sum(1 for e in pool if e.tw_scale == 0.35)
                names = ", ".join(e.label for e in pool)
                lines.append(
                    f"{fam:6s} {bucket:6s} {len(pool):>2d} {mean:>8.2f} "
                    f"{x8:>4d} {x6:>4d} {x5:>4d} {x35:>4d}  {names}"
                )
        lines.append(
            f"landing instances (spec §7.5 step 5): {', '.join(self.landing_instances)}"
        )
        return "\n".join(lines)

    def _check_bucket(self, bucket: str) -> None:
        if bucket not in BUCKETS:
            raise ValueError(f"unknown bucket {bucket!r}; expected {BUCKETS}")


def load_split(path: Optional[Path] = None) -> Split:
    """Read and validate the frozen split file."""
    path = Path(path) if path else DEFAULT_SPLIT
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(
        SplitEntry(
            name=r["name"],
            family=r["family"],
            bucket=r["bucket"],
            n=int(r["n"]),
            tw_scale=float(r["tw_scale"]),
            probe_seconds=float(r["probe_seconds"]),
            probe_labels=int(r["probe_labels"]),
            probe_cap_frac=float(r["probe_cap_frac"]),
            clean=bool(r["clean"]),
        )
        for r in raw["instances"]
    )

    # The file states its own bucket sizes; a count that disagrees with the
    # instances is a corrupt split, and a corrupt split is worse than a missing
    # one because it fails silently downstream.
    for fam, want in raw["families"].items():
        for bucket in BUCKETS:
            got = sum(1 for e in entries if e.family == fam and e.bucket == bucket)
            if got != int(want[bucket]):
                raise ValueError(
                    f"split {raw['version']}: family {fam} declares "
                    f"{want[bucket]} {bucket} instances but lists {got}"
                )
    seen = set()
    for e in entries:
        if e.name in seen:
            raise ValueError(f"split {raw['version']}: duplicate instance {e.name}")
        seen.add(e.name)

    return Split(
        version=raw["version"],
        frozen=raw["frozen"],
        entries=entries,
        families=raw["families"],
        landing_instances=tuple(raw["landing_instances"]),
        split_rule=dict(raw["split_rule"]),
    )


def landing_entries(split: Optional[Split] = None) -> Tuple[SplitEntry, ...]:
    """The three per-family representatives §7.5 step 5 lands a baseline on."""
    split = split or load_split()
    order = split.landing_instances
    return tuple(split.entry(n) for n in order)


def iter_families(entries: Iterable[SplitEntry]) -> Sequence[str]:
    return sorted({e.family for e in entries})


if __name__ == "__main__":  # python -m uavcg.splits
    print(load_split().summary())
