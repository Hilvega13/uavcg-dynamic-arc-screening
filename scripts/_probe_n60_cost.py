"""Bounded cost curve for one (instance, arm) on the frozen n=60 main tier.

Why this exists
---------------
The first attempt to land the C201 baseline ran for 12 minutes with no output,
because the shell pipeline buffered everything to process end.  That is a
measurement failure, not a performance failure: nothing was learned about
*where* the time went.  This probe answers the question that should have been
asked first -- how does the cost of a pricing call grow across a solve, and does
the run certify -- while writing one line per iteration so that a kill still
leaves the part of the curve that was measured.

The curve matters because it decides the stopping policy
--------------------------------------------------------
The single-call probe measured C201 at 0.69 s on the *seed* dual.  A CG run is
not a sequence of seed-dual calls: as columns accumulate the duals move, the
ng-route neighbourhood retains more non-dominated labels, and the per-call cost
rises.  A stopping rule chosen against the 0.69 s figure is chosen against the
cheapest call of the run.

Output goes to stdout with `flush=True` on every line, so it is safe to run as
    python scripts/_probe_n60_cost.py --name C201 --max-iters 40 --trace > log
and read the log while it is still being written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavcg.cg import CGConfig, ColumnGeneration
from uavcg.pricers import (
    ExactLabelingPricer,
    NgRoutePricer,
    RedCostArcPricer,
)
from uavcg.splits import load_split


#: v3.6 hard budget (user ruling 2026-10-05).  The descent phase and the
#: certification phase get SEPARATE reserves, and their SUM is the hard bound.
#: Written as two constants on purpose: a single "3600" would let one phase eat
#: the other's reserve, and the certification reserve is the only thing that
#: makes a certificate reachable at all.
DESCENT_SECONDS = 120.0     # per-call TIME CEILING while descending (see below)
DESCENT_BUDGET = 2400.0      # total seconds the descent phase may spend
#: NOMINAL per-call cap for a certification attempt.  Kept as a named constant
#: for the record, but it is NOT what binds: a certification attempt is granted
#: the whole remaining HARD TOTAL (see `TimedPricer._cap_for`).  Measured reason,
#: and it is the P2 defect: a COMPLETE full-network call needs >=1250 s on these
#: instances, so pinning the attempt at 1200 s meant the attempt was cut off
#: mid-search every single time -- it returned 200 columns instead of proving
#: emptiness, `untruncated_empty_calls` stayed 0, and the certification half of
#: the v3.6 verdict ("an un-truncated empty call") was UNREACHABLE.  The ruling's
#: hard bound is untouched: no call is ever handed more than the total reserve
#: left, so the run still cannot exceed DESCENT_BUDGET + CERT_BUDGET.
CERT_SECONDS = 1200.0
CERT_BUDGET = 1200.0         # total seconds certification attempts may spend
TOTAL_BUDGET = DESCENT_BUDGET + CERT_BUDGET     # 3600 s, and it is a BOUND
#: v3.6 P3: a "lap" is one full-network certification attempt.  The work gate
#: closes the DESCENT phase, not the run: a certification attempt is not extra
#: search, it IS the measurement, so the gate must not be allowed to veto it --
#: measured defect: the 2e7 gate always closed during descent, no certification
#: attempt was ever made, `certified` was False for EVERY attempt, and no row of
#: the n=40 tier could ever be admissible.  Laps are hard-capped, and a later lap
#: is only started when the previous one PROVED CONTENT (it returned columns);
#: an attempt that returned nothing -- empty-and-complete (a certificate) or
#: empty-and-truncated (blind) -- closes the question, because another lap of the
#: same search cannot change that verdict.
MAX_LAPS = 3
#: Label allowance granted to a full-network certification attempt (v3.6 P4).
#: NOT a schedule -- it is set so high that it cannot bind inside the hard total
#: (3600 s at the measured ~10k-100k labels/s is 3.6e7-3.6e8 labels).  It exists
#: because leaving the pricer's OWN ceiling in place made certification
#: unreachable for the same reason the 1200 s cap did: `NgRoutePricer` defaults to
#: `max_labels=3_000_000`, while a complete full-network call on n=40 builds
#: ~1e7-2.5e7 labels.  So every certification attempt was cut by the pricer's own
#: cap, came back truncated (proving nothing) and -- because an intrinsic-cap hit
#: is by definition NOT scheduled -- also marked the run degraded.  P2 and P4 are
#: the same defect at two different knobs: a certification attempt bounded by
#: anything smaller than the work it takes to finish cannot certify.
CERT_LABELS = 1_000_000_000
#: Per-call LABEL cap for descent rounds: **this, not the clock, is what
#: truncates a call.**  A measured fact forced the change.  Capping by time does
#: not merely add noise, it destroys determinism, because the cap decides how
#: much work happens.  Two back-to-back repeats of RC201(n40,x0.8) at a 1 s cap
#: gave 10 iterations and the same seed LP 3320.8965 but DIFFERENT work:
#: 581,265 labels / LP_min 1123.7996 vs 478,816 / 1132.4247.  Truncating at a
#: label count is exactly reproducible.
#:
#: Why the ceiling is 120 s and not 30 s.  With a 30 s *time* ceiling the first
#: call still hit the clock, because the initial label rate is the slowest of the
#: run: measured 7,538 labels/s on call #1 vs 20,000-88,000 labels/s later.  30 s
#: buys only ~226k labels there, so the timing -- and therefore the work -- stayed
#: machine-dependent (248,032 labels in one repeat, 220,391 in the other).  120 s
#: gives the 300,000-label cap a 3x margin even at that worst-case rate, which is
#: what makes the label cap the binding constraint and the run reproducible.
#: The ceiling has NOT been deleted: it is the safety net for a pathological
#: call, and if it ever fires the call is flagged `unscheduled` so the run is
#: marked degraded rather than quietly mixed in.
DESCENT_LABELS = 300_000
#: One deadline-sampling window, in seconds.  A time cap is SAMPLED, not
#: enforced per label: the pricer compares against the deadline every 256
#: labels, so a call can overrun its cap by up to one window.  At the slowest
#: label rate measured on these instances (~3,000 labels/s) a window costs
#: ~0.09 s, so 0.25 s covers it with room to spare -- and reserving it inside
#: the grant is what keeps the phase budget a bound in practice and not merely
#: in arithmetic.  This is an implementation detail of the budget, NOT a
#: measurement tolerance: nothing in the experiment is compared against it.
LABEL_WINDOW_SLACK = 0.25
#: Anti-hang fuse only (v3.6 ④).  This is NOT a performance criterion:
#: admissibility is decided on WORK (total labels), because the wall clock on
#: this machine was measured drifting by 1.68x on bit-identical output.  The
#: fuse exists so a pathological instance cannot hold a core for ever.
WALL_FUSE_SECONDS = 2 * 3600.0


def _positive(value):
    """`None`/0 means "no cap"; anything else is a cap in seconds."""
    if value is None:
        return None
    value = float(value)
    return None if value <= 0 else value


class TimedPricer:
    """v3.5 time-limited truncation protocol: cap a call, KEEP what it found.

    The whole mechanism rests on one fact about labelling: **a sink label that
    has been built is a real column with a real reduced cost, whatever happens
    next**.  Hitting a cap means the search did not finish; it does not
    invalidate what it already found.  So a truncated call may hand its columns
    to the master, and the master LP stays valid -- it is determined by the real
    columns in the pool and by nothing else.

    Budget v3.6: two phase reserves, and the total is a BOUND
    --------------------------------------------------------
    * **descent rounds** are cut at ``DESCENT_LABELS`` (300,000 labels) per call
      and may consume ``DESCENT_BUDGET`` (2400 s) of clock in total.  A truncated
      call returns its columns and the run simply continues;
    * **certification attempts** are full-network calls, bounded only by the
      remaining HARD TOTAL (nominal ``CERT_SECONDS`` = 1200 s -- see `CERT_SECONDS`
      for why that nominal figure cannot be what binds), with a label ceiling so
      high it cannot bind (``CERT_LABELS``; the pricer's own 3M default is an
      order of magnitude too small, which is defect P4), and there are at most
      ``MAX_LAPS`` (3) of them per run.  A truncated EMPTY descent call is
      *blind*, not converged (v3.5 ③), so it is retried once at the larger cap
      before the loop may conclude anything.

    ``2400 + 1200 = 3600`` is enforced, not hoped for.  Two mechanisms, and the
    second is the one that was missing:

    1. every call is checked BEFORE it starts -- once a phase's reserve (or the
       hard total) is gone, the pricer winds the run down instead of asking for
       more time;
    2. the per-call cap is ``min(phase call cap, phase reserve left, total
       reserve left)``, so the LAST call can only be handed what remains.
       Without this term the run could overshoot by a whole call: measured
       gate=1200 s -> PP=1840.8 s, i.e. 53% over, because the final 1203 s
       certification call had already been launched when the budget ran out.
       The certification attempt is the one exception to the *phase* term --
       not to the *total* term -- because a complete full-network call needs
       more than the phase's nominal 1200 s (see `_cap_for`).

    Cap v3.6: 300,000 labels, not a time cap.  The 300 s figure was chosen
    against the "call cost decreases as the dual improves" hypothesis, which is
    now VOID -- the wall clock on this machine drifts by ~1.7x on bit-identical
    output, so that hypothesis is not measurable here at all.  What the caps are
    chosen against instead is ``LP progress per unit of work``, and on that
    measure 300 s was the worst of three settings tested on C201 (810 vs 1929
    uncapped vs 2507 at 30 s per 1000 s) *and* it cost C201 its content proof
    (LP_min 480.71 > insertion 443.94, while 30 s and uncapped both dipped
    below).  A 30 s TIME cap then turned out not to work either: the first call
    of a run is the slowest (measured 7,538 labels/s), so 30 s bought only
    ~226k labels and the work stayed machine-dependent.  Truncating at 300,000
    LABELS is exactly reproducible, and the 120 s ceiling is loose enough never
    to bind (it is a safety net for a pathological call, and if it ever fires the
    call is flagged `unscheduled`, marking the run degraded rather than quietly
    mixing it in).

    Note what is NOT here: any check that stops the run because time has passed
    *for its own sake*.  ``gate_seconds`` is the 2 h anti-hang fuse and
    ``label_budget`` is the work gate; bounding the run is otherwise the
    caller's job.  This class bounds one call, its phase, and the total.
    """

    def __init__(self, inner, descent_seconds=DESCENT_SECONDS,
                 cert_seconds=CERT_SECONDS, label: str = "",
                 gate_seconds: float = 0.0, stop_after: float = 0.0,
                 descent_budget: float = DESCENT_BUDGET,
                 cert_budget: float = CERT_BUDGET,
                 label_budget: float = 0.0,
                 descent_labels: float = DESCENT_LABELS,
                 max_laps: int = MAX_LAPS,
                 cert_labels: float = CERT_LABELS) -> None:
        self.inner = inner
        #: The pricer's own label cap, remembered so a descent call's smaller
        #: cap can be undone for the certification attempt.
        self._intrinsic_max_labels = getattr(inner, "max_labels", None)
        #: `None` disables the cap entirely (the legacy probes need that to
        #: reproduce their recorded behaviour bit for bit).
        self.descent_seconds = _positive(descent_seconds)
        self.cert_seconds = _positive(cert_seconds)
        #: The cap that actually truncates a descent call (see `DESCENT_LABELS`).
        #: `None`/0 means "use the pricer's own intrinsic label cap".
        self.descent_labels = (
            None if (descent_labels is None or float(descent_labels) <= 0)
            else int(descent_labels)
        )
        #: Anti-hang fuse.  Once cumulative pricing time passes it the run
        #: winds down gracefully; it is NOT the admissibility criterion.
        self.gate_seconds = _positive(gate_seconds)
        #: Work gate: once cumulative labels created pass it the attempt cannot
        #: be admissible, so there is nothing left to learn from continuing.
        #: This is the v3.6 admissibility criterion (spec v3.6 ④): total labels,
        #: not seconds, because labels are environment-independent and the wall
        #: clock is not.
        self.label_budget = _positive(label_budget)
        #: v3.6 P3: hard cap on full-network certification attempts per run.
        self.max_laps = max(0, int(max_laps))
        self.label = label
        #: LEGACY, default off.  Reproduces the old "budget spent => raise
        #: TimeoutError" behaviour so the three historical probes
        #: (`_probe_main_tier`, `_probe_seed_composition`, `_probe_n60_cost`)
        #: still run exactly as they did when their numbers were recorded.
        #: Do NOT use this in new code: the raise discards `cg.run()`'s local
        #: snapshots, which is the defect that cost a 16 h sweep its verdicts.
        self.stop_after = float(stop_after)
        #: The budget machinery is armed only when there IS a descent cap.
        #: `descent_seconds=None` therefore means "no phase reserves, no total
        #: bound" -- which is what keeps the legacy probes byte-identical.
        bounded = self.descent_seconds is not None
        self.descent_budget = _positive(descent_budget) if bounded else None
        self.cert_budget = _positive(cert_budget) if bounded else None
        self.total_budget = (
            None if not bounded
            else (self.descent_budget or 0.0) + (self.cert_budget or 0.0)
        )
        if not bounded:
            # Legacy uncapped mode must stay byte-identical: no time cap, no
            # phase reserves, no label schedule, and no certification allowance
            # (the pricer's own ceiling is left exactly as the caller set it).
            self.descent_labels = None
        #: v3.6 P4: label allowance for a certification attempt (see CERT_LABELS).
        #: `None` means "leave the pricer's own ceiling in place", which is what
        #: legacy uncapped mode needs to stay byte-identical.
        self.cert_labels = (
            None if (not bounded or not cert_labels or float(cert_labels) <= 0)
            else int(cert_labels)
        )
        self.calls = 0
        self.spent = 0.0            # pricing seconds, both phases
        self.descent_spent = 0.0
        self.cert_spent = 0.0
        self.labels = 0             # labels created, both phases
        self.trace: list = []
        #: How often the certification cap had to decide an iteration.
        self.cert_escalations = 0
        #: v3.6 P3: full-network certification attempts actually launched, and
        #: whether the work gate has closed the DESCENT phase.  Both are needed
        #: together: the gate ends the descent, the laps end the run.
        #: `laps` is a COUNT of attempts launched -- never a latch, or a report
        #: could not tell "three attempts made" from "no more attempts allowed".
        self.laps = 0
        self.descent_closed = False
        #: Set when a lap came back with nothing: the question is closed and the
        #: next call winds down.  Separate from `laps` for exactly that reason.
        self.laps_exhausted = False
        #: Calls that returned an empty column set WITHOUT being truncated.
        #: This is the v3.6 certification evidence (spec v3.6 ③): an empty
        #: truncated call is blind, only an empty *complete* call proves that no
        #: negative column exists.  Read off the trace, so it cannot disagree
        #: with the per-call record a report prints.
        self.untruncated_empty_calls = 0
        #: Why the run was wound down: "budget" | "work_gate" | "wall_fuse".
        self.wind_down_reason = None

    def __getattr__(self, item):
        return getattr(self.inner, item)

    @property
    def name(self):
        return self.inner.name

    @property
    def guarantees_optimality(self):
        return self.inner.guarantees_optimality

    @property
    def arcs_total(self):
        return self.inner.arcs_total

    @property
    def masked_arcs(self):
        return self.inner.masked_arcs

    def _remaining(self, phase: str) -> float:
        """Seconds this phase may still spend, capped by the hard total.

        The `min` is what makes the total a bound: a phase can never borrow the
        other phase's reserve, and neither can spend past 3600 s in total.
        """
        left_total = self.total_budget - self.spent
        if left_total <= 0.0:
            return 0.0
        if phase == "desc":
            left_phase = (self.descent_budget or 0.0) - self.descent_spent
        else:
            left_phase = (self.cert_budget or 0.0) - self.cert_spent
        if left_phase <= 0.0:
            return 0.0
        return min(left_phase, left_total)

    def _cap_for(self, phase: str, *, allow_total: bool = False):
        """Per-call cap = min(phase cap, phase reserve left, total reserve left).

        Returns None only when there is no cap at all (legacy uncapped mode).
        A return of 0.0 means "no affordable call left"; the caller must wind
        down rather than launch a zero-length call.

        ``allow_total`` (v3.6 P2) drops the PHASE term and keeps only the TOTAL
        one, and it is for certification attempts alone.  The reason is measured:
        a COMPLETE full-network call needs >=1250 s on these instances, so an
        attempt bounded by the phase's nominal 1200 s was cut off every time --
        it returned 200 columns instead of proving emptiness, so
        `untruncated_empty_calls` stayed 0 and the certification half of the
        v3.6 verdict was unreachable.  What the ruling calls the hard bound is
        untouched: the attempt still cannot be handed more than
        ``total_budget - spent``, so the run stays inside 2400 + 1200 s.
        """
        base = self.descent_seconds if phase == "desc" else self.cert_seconds
        if self.total_budget is None:
            return base
        if allow_total:
            left = self.total_budget - self.spent
        else:
            left = self._remaining(phase)
        if left <= 0.0:
            return 0.0
        # Reserve one sampling window (see `LABEL_WINDOW_SLACK`): a call cannot
        # observe a deadline finer than the interval at which it looks, so the
        # grant has to be shrunk by that interval for the reserve to hold.
        grant = left - LABEL_WINDOW_SLACK
        if grant <= 0.0:
            return 0.0
        return grant if allow_total else min(base, grant)

    def _call(self, args, kwargs, cap, phase: str, label_cap=None):
        """One capped call.  Never raises and never discards columns."""
        if hasattr(self.inner, "max_seconds"):
            self.inner.max_seconds = cap
        # The label cap is the deterministic truncation knob (see
        # `DESCENT_LABELS`).  Resolution order for a call:
        #   * an int -- the protocol SCHEDULED that cap (a descent round);
        #   * None with a certification allowance configured -- a full-network
        #     certification attempt, which gets `CERT_LABELS` (P4: the pricer's
        #     own 3M ceiling is an order of magnitude too small to finish);
        #   * None in legacy uncapped mode -- restore the pricer's intrinsic cap,
        #     which is what keeps the old probes byte-identical.
        if hasattr(self.inner, "max_labels"):
            if label_cap is not None:
                self.inner.max_labels = int(label_cap)
            elif self.cert_labels is not None:
                self.inner.max_labels = int(self.cert_labels)
            else:
                self.inner.max_labels = self._intrinsic_max_labels
        t = time.perf_counter()
        res = self.inner.price(*args, **kwargs)
        dt = time.perf_counter() - t
        self.calls += 1
        self.spent += dt
        if phase == "desc":
            self.descent_spent += dt
        else:
            self.cert_spent += dt
        self.labels += res.labels_created
        #: v3.6 ③ certification evidence: empty AND complete.  An empty
        #: *truncated* call is blind; counting this here is what stops a capped
        #: run from ever being read as "no negative column exists".  Read off
        #: the same trace the report prints, so the two cannot disagree.
        if not res.truncated and not res.columns:
            self.untruncated_empty_calls += 1
        #: §3.5's `degraded` exists to catch a pricer quietly running at a cap.
        #: Under v3.6 a descent round is capped ON PURPOSE, so a scheduled hit
        #: must not count as degradation -- otherwise `degraded` is true for
        #: every run and a certificate is unreachable.  A TIME truncation is
        #: deliberately left unmarked: it means the cap that bound was the clock,
        #: which makes the run's work machine-dependent, and `degraded` flagging
        #: exactly those runs is the behaviour we want.
        #: The protocol's OWN choice of ceiling, marked on every call the protocol
        #: picked one for -- not only on the ones that actually hit it.  `cg.py`
        #: reads this key to keep `used_label_cap` (and therefore `degraded`'s
        #: proximity test) on the pricer's INTRINSIC cap.  Without the mark the
        #: 300k descent slice is mistaken for an intrinsic cap, the proximity
        #: threshold becomes CLEAN_LABEL_FRACTION * 300k = 240k, and -- since
        #: every run exceeds that -- `degraded` is true for every run, `status`
        #: can never be CERTIFIED and the §6.2 main table is unreachable even for
        #: a run holding a genuine certificate.  Measured: RC201(n40,x0.6)
        #: certified (empty complete call, LP_min 634.23 vs insertion 1005.41)
        #: and still reported INCOMPLETE / degraded.
        protocol_chose = bool(label_cap is not None
                              or (phase == "cert" and self.cert_labels is not None))
        if protocol_chose:
            res.stats["protocol_label_cap"] = 1.0
        # `scheduled` is keyed on the DESCENT schedule only.  A certification
        # attempt is never "scheduled truncation": its whole point is to finish,
        # so if it comes back truncated that is real evidence of failure (the
        # clock, or a ceiling so large it cannot have been the protocol's plan)
        # and it must mark the run, not be filed as policy.
        scheduled = bool(label_cap is not None
                         and int(res.stats.get("label_cap", 0) or 0) == int(label_cap)
                         and not res.stats.get("time_truncated"))
        if res.truncated and scheduled:
            res.stats["scheduled_cap"] = 1.0
        rec = dict(
            call=self.calls, phase=phase, cap=cap, seconds=dt,
            cumulative=self.spent, labels=res.labels_created,
            #: The deterministic truncation knob, recorded per call so a report
            #: can show that the label cap -- not the clock -- did the cutting.
            label_cap=(None if label_cap is None else int(label_cap)),
            #: The ceiling the pricer was actually handed, whatever the reason
            #: (`label_cap`, else `CERT_LABELS`, else the pricer's intrinsic one).
            effective_label_cap=getattr(self.inner, "max_labels", None),
            #: True when the PROTOCOL chose that ceiling for this call (as opposed
            #: to the pricer's own intrinsic cap).  The two must be told apart by
            #: every consumer of `label_cap`.
            protocol_cap=protocol_chose,
            scheduled=bool(res.stats.get("scheduled_cap", 0)),
            #: Reserve context, so a report can show that the bound is being
            #: respected rather than merely asserted.
            reserve_left=(None if self.total_budget is None
                          else self._remaining(phase)),
            phase_spent=(self.descent_spent if phase == "desc"
                         else self.cert_spent),
            columns=len(res.columns), masked=bool(res.masked),
            truncated=bool(res.truncated),
            #: Distinguishes "ran out of wall clock" from "ran out of labels".
            #: Both truncate; only the story differs.
            time_truncated=bool(res.stats.get("time_truncated", 0)),
            arcs_removed=int(res.stats.get("masked_arcs", 0) or 0),
            #: v3.6 P3: which full-network lap this call belongs to (0 = a
            #: descent round that is not part of any lap).
            lap=self.laps,
        )
        self.trace.append(rec)
        left = rec["reserve_left"]
        print(
            f"    [pp {self.label} #{self.calls:4d} {phase:4s}] {dt:9.3f}s "
            f"labels={res.labels_created:9d} "
            f"cols={len(res.columns):3d}"
            + ("" if left is None else f" left={left:8.1f}s")
            + ("  [SCHED-CUT]" if rec["scheduled"] else "")
            + ("  [TIME-CUT!]" if rec["time_truncated"] else "")
            + (f"  [LAP {self.laps}]" if phase == "cert" else ""),
            flush=True,
        )
        return res

    def _wind_down_result(self):
        """Empty and truncated -- "the attempt is decided, stop asking".

        Returned for any of three reasons, recorded on ``wind_down_reason`` and
        all of them meaning the same thing to the loop: no further iteration can
        change the verdict.  ``budget`` = this phase's reserve (or the hard
        total) is spent; ``work_gate`` = the label gate is out of reach;
        ``wall_fuse`` = the 2 h anti-hang fuse fired.

        Safe to return here precisely BECAUSE it cannot lose anything: the
        loop has already banked every column of the iterations behind it, and
        the verdict is decided.  Truncated, so it can never certify.
        """
        from uavcg.pricers import PricingResult

        return PricingResult(
            columns=[], min_reduced_cost=0.0, exact=False,
            relaxed_min_reduced_cost=None, truncated=True, masked=False,
            # `scheduled_cap`: the protocol declined to price -- no call was
            # truncated, so this must not read as an unscheduled truncation and
            # mark the run degraded.  Truncated is still True, so it can never
            # certify, which is the property that matters.
            stats={"gate_exceeded": 1.0, "scheduled_cap": 1.0},
        )

    def price(self, *a, **kw):
        # ---- wind-down checks, ALL of them before any call is launched ------
        # v3.6 requirement 2: the budget has to be checked before the call, or a
        # single in-flight call overruns the bound by its own entire length.
        # That is exactly what was measured: gate=1200 s -> PP=1840.8 s (+53%),
        # because the final certification call had already been launched.
        if self.gate_seconds and self.spent > self.gate_seconds:
            self.wind_down_reason = "wall_fuse"
            return self._wind_down_result()
        # v3.6 P3: the work gate closes the DESCENT phase, not the run.
        # Measured defect: with the gate enforced as a wind-down, the n=40 gate
        # (2e7 labels) closed during descent every single time -- roughly 67
        # capped calls -- so the certification attempt that the v3.6 verdict
        # depends on was never reached, `certified` was False for every attempt,
        # and no row of the tier could ever be admissible.  A certification
        # attempt is not extra search; it is the measurement, and it must be
        # allowed to happen even though the search work is already spent.
        if self.label_budget and self.labels > self.label_budget:
            self.descent_closed = True
        if self.descent_closed:
            if self.laps_exhausted or self.laps >= self.max_laps:
                # The laps are spent and the last one found nothing: no further
                # call can change the verdict.  This is the normal way a capped
                # run ends, so it is `work_gate`, not an error.
                self.wind_down_reason = "work_gate"
                return self._wind_down_result()
            ccap = self._cap_for("cert", allow_total=True)
            if ccap is not None and ccap <= 0.0:
                self.wind_down_reason = "budget"
                return self._wind_down_result()
            self.laps += 1
            res = self._call(a, kw, ccap, "cert", None)
            # A lap that returned NOTHING closes the question: empty-and-complete
            # is the certificate, empty-and-truncated is blind, and neither can
            # be improved by running the same full-network search again.  Only a
            # lap that came back WITH columns -- which is proof of content, and
            # the only state in which more search is justified -- earns another
            # lap.
            if not res.columns:
                self.laps_exhausted = True
            return res

        cap = self._cap_for("desc")
        if cap is not None and cap <= 0.0:
            self.wind_down_reason = "budget"
            return self._wind_down_result()

        res = self._call(a, kw, cap, "desc", self.descent_labels)

        # Blind, not converged (v3.5 ③) -- but ONLY when a cap is what stopped
        # the call.  v3.6: the binding cap is now the LABEL cap (`scheduled_cap`),
        # not the clock, so the test covers both.  Without the `scheduled_cap`
        # half the escalation would never fire under the label-capped protocol and
        # nothing could ever certify.  An INTRINSIC cap hit keeps its old meaning:
        # a bigger time budget would not help it, so it does not escalate.
        if (res.truncated and not res.columns
                and (res.stats.get("time_truncated")
                     or res.stats.get("scheduled_cap"))
                and self.cert_seconds is not None
                and self.cert_seconds != self.descent_seconds
                and not self.laps_exhausted
                and self.laps < self.max_laps):
            # `allow_total` (P2): a certification attempt is bounded by the hard
            # total, not by the certification phase's nominal 1200 s -- a
            # complete full-network call does not fit in 1200 s here.
            ccap = self._cap_for("cert", allow_total=True)
            # A certification attempt that cannot be afforded is skipped rather
            # than launched at zero length: a zero-length call is a truncation
            # dressed up as evidence.
            if ccap is None or ccap > 0.0:
                self.cert_escalations += 1
                self.laps += 1
                res = self._call(a, kw, ccap, "cert", None)

        if self.stop_after and self.spent > self.stop_after:
            # Legacy path, opt-in -- see `stop_after`.
            raise TimeoutError(
                f"PP budget {self.stop_after:.0f}s exceeded after "
                f"{self.calls} calls"
            )
        return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C201")
    ap.add_argument("--arm", default="baseline", choices=["baseline", "redcost"])
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--stall", type=int, default=0)
    ap.add_argument("--pp-budget", type=float, default=0.0,
                    help="stop once cumulative pricing time passes this (0 = off)")
    ap.add_argument("--max-cols", type=int, default=25)
    args = ap.parse_args()

    split = load_split()
    entry = split.entry(args.name)
    inst = split.load(entry)
    print(f"{entry.label}  family={entry.family} bucket={entry.bucket} "
          f"probe={entry.probe_seconds:.2f}s/{entry.probe_labels} labels", flush=True)

    if args.arm == "baseline":
        inner, fallback = NgRoutePricer(inst, k=10), ExactLabelingPricer(inst)
    else:
        inner, fallback = RedCostArcPricer(inst, k=10), ExactLabelingPricer(inst)

    pricer = TimedPricer(inner, descent_seconds=None, cert_seconds=None,
                         stop_after=args.pp_budget, label=args.arm)
    cfg = CGConfig(
        max_iterations=args.max_iters,
        max_columns_per_iter=args.max_cols,
        stall_limit=args.stall,
        fallback_max_n=40,
        log_every=1,
    )
    cg = ColumnGeneration(inst, pricer, config=cfg, fallback_pricer=fallback)

    t0 = time.perf_counter()
    try:
        res = cg.run()
    except TimeoutError as exc:
        print(f"\nBUDGET STOP: {exc}", flush=True)
        print(f"wall {time.perf_counter() - t0:.2f}s  pp {pricer.spent:.2f}s  "
              f"calls {pricer.calls}", flush=True)
        return 2

    wall = time.perf_counter() - t0
    print(f"\n=== {args.name} {args.arm} ===", flush=True)
    print(f"wall={wall:.2f}s  pp={res.pp_seconds:.2f}s  rmp={res.rmp_seconds:.2f}s "
          f"other={res.other_seconds:.2f}s", flush=True)
    print(f"iters={res.iterations}  calls={res.pricing_calls}  "
          f"cols={res.columns_in_pool}  LP={res.lp_objective:.6f}  "
          f"certified={res.optimal_lp}  degraded={res.degraded}  "
          f"truncated={res.truncated_pricing_calls}", flush=True)
    if pricer.trace:
        first = pricer.trace[0]["seconds"]
        mid = pricer.trace[len(pricer.trace) // 2]["seconds"]
        last = pricer.trace[-1]["seconds"]
        print(f"per-call seconds: first={first:.3f} mid={mid:.3f} last={last:.3f} "
              f"max={max(t['seconds'] for t in pricer.trace):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
