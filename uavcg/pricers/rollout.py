"""Feasibility-preserving serialised rollout pricing.

What this is
------------
The reference paper (Shen et al., AAAI 2022) turns a learned vertex score into
columns like this: sort vertices by predicted probability, walk down the list
adding a vertex whenever it keeps the set independent, repeat K times with
randomised sampling, and hand every accepted set to the master.  "Adding a
vertex to an independent set" can never fail, so that sampler always produces
a feasible column.  That property is what makes batch column production work
there, and it does not hold here.

Why it has to be rewritten
--------------------------
A drone route has resource constraints - capacity, time windows and (in UAV
mode) energy.  Appending the next customer can leave a partial route that no
completion can repair: the load no longer fits, the depot closes before the
drone can get back, the battery cannot cover the return leg.  Sampling points
in a fixed order and hoping is exactly how a rollout dies.  This is the single
place in the whole design where the migration can fail outright, which is why
it is built and measured before any learning is attached to it.

The three mechanisms that replace "always feasible"
---------------------------------------------------
1. `FeatureContext.mask` removes a candidate as soon as completing the route
   through it is provably impossible.  Every test is a necessary condition,
   so the mask can only discard routes that were already impossible - it is a
   weak prune, never an incorrect one.
2. Closing the route is attempted *after every extension*, not only at the
   end.  A dead-ended sample therefore still contributes the good prefixes it
   did manage to build instead of being thrown away, which is what keeps the
   per-iteration column count high.
3. One deterministic pass (argmax of the same score) runs alongside the
   stochastic ones, so a pricing call is never empty for a silly reason.

The pricer is heuristic and says so: `guarantees_optimality = False` and
`relaxed_min_reduced_cost` stays None, so the column generation loop still
calls an exact pricer before it certifies LP optimality.  A learned rollout
that performs badly degrades the search, never its correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import NUM_FEATURES, FeatureContext, RolloutState, arc_features
from ..instance import DEPOT, EPS, Instance
from .base import Pricer, PricingResult, RouteColumn


@dataclass
class RolloutDiagnostics:
    """Bookkeeping used by the Step-3 viability probe."""

    rollouts: int = 0
    stuck: int = 0            # ran out of candidates before closing
    no_close: int = 0         # produced no closable negative route at all
    closed_ok: int = 0
    rejected_tw: int = 0
    rejected_energy: int = 0
    rejected_rc: int = 0

    @property
    def dead_end_rate(self) -> float:
        return self.no_close / self.rollouts if self.rollouts else 0.0

    def as_stats(self) -> Dict[str, float]:
        return {
            "rollouts": self.rollouts,
            "stuck": self.stuck,
            "no_close": self.no_close,
            "closed_ok": self.closed_ok,
            "rejected_tw": self.rejected_tw,
            "rejected_energy": self.rejected_energy,
            "rejected_rc": self.rejected_rc,
            "dead_end_rate": self.dead_end_rate,
        }


class RolloutPricer(Pricer):
    """Probabilistic serialised construction of routes.

    `scoring` selects only the *score* used to bias the sampling:

        "uniform"  flat distribution            - lower bound on usefulness
        "dual"     softmax of the modified arc cost `dist_ij - pi_j`
                   (the classical nearest-neighbour-on-reduced-cost rule)
        "model"    a learned arc scorer         - the proposed method

    Nothing else changes between the three, which is what makes the comparison
    in the experiments a like-for-like one.

    `beta` is the sampling temperature and it is *not* a free knob to be left
    at 1.0 out of habit.  A sweep over beta in {0.1, 0.25, 0.5, 1, 2, 4}
    (`scripts/run_temperature.py`, results/temperature_n25.json) found the
    optimum at 2.0 on both scoring modes measured, and the loss from running
    below it is not small: on R105 the dual arm needs 16 iterations at beta=2
    against 22 at beta=1, with the dead-end rate falling from 65% to 38%.  The
    default is therefore 2.0.  Because the score is standardised per decision
    step, one unit of beta means "one standard deviation of this step's
    scores" in every mode, so the same number is comparable across the three.
    """

    name = "rollout"
    guarantees_optimality = False

    def __init__(
        self,
        inst: Instance,
        scoring: str = "dual",
        scorer=None,
        n_samples: int = 300,
        beta: float = 2.0,
        deterministic_passes: int = 1,
        seed: int = 0,
        record_prefixes: bool = True,
        warmup_samples: int = 0,
        max_rollout_len: Optional[int] = None,
        candidate_pool: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inst)
        if scoring not in ("uniform", "dual", "model"):
            raise ValueError(f"unknown scoring mode {scoring!r}")
        if scoring == "model" and scorer is None:
            raise ValueError("scoring='model' needs a scorer")
        self.scoring = scoring
        self.scorer = scorer
        self.n_samples = int(n_samples)
        self.beta = beta
        self.deterministic_passes = int(deterministic_passes)
        self.seed = int(seed)
        self.record_prefixes = bool(record_prefixes)
        self.warmup_samples = int(warmup_samples)
        self.max_rollout_len = int(max_rollout_len) if max_rollout_len else inst.n
        #: Keep only the `candidate_pool` most promising candidates (by
        #: modified arc cost) before computing features or sampling.
        #:
        #: Two reasons. Cost: the feature block is the dominant expense of a
        #: learned rollout, and it is paid per candidate per decision step, so
        #: dropping 30 candidates to 6 makes the call roughly five times
        #: cheaper. Statistics: the score is learned from arcs that a good
        #: heuristic would consider, so scoring arcs it would never look at is
        #: extrapolation. Whatever the pool drops is still reachable, because
        #: the pool is re-formed at every step from the current state.
        self.candidate_pool = int(candidate_pool) if candidate_pool else None
        self.ctx = FeatureContext(inst)
        #: Per-call feature cache, rebuilt at the top of `price` /
        #: `collect_arc_samples`.  A pricer instance is used single-threaded by
        #: the column generation loop, so one slot is enough; `arc_features`
        #: still works without it, which is what keeps the labelling path and
        #: the profiling script independent of this optimisation.
        self._dual = None
        self._calls = 0
        if name:
            self.name = name

    # ------------------------------------------------------------------
    def _pool(self, st: RolloutState, cand: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """The `candidate_pool` candidates with the most negative modified cost."""
        if self.candidate_pool is None or cand.size <= self.candidate_pool:
            return cand
        m = self.inst.dist[st.node, cand] - np.asarray(pi, dtype=float)[cand - 1]
        k = self.candidate_pool
        idx = np.argpartition(m, k - 1)[:k]
        return cand[np.sort(idx)]

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _logits(
        self,
        st: RolloutState,
        cand: np.ndarray,
        pi: np.ndarray,
        rank_freq: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.scoring == "uniform":
            return np.zeros(cand.size)
        if self.scoring == "dual":
            return -(self.inst.dist[st.node, cand] - np.asarray(pi, dtype=float)[cand - 1])
        X = arc_features(self.ctx, st, cand, pi, rank_freq, dual=self._dual)
        return np.asarray(self.scorer.logit(X), dtype=float)

    @staticmethod
    def _weights(logits: np.ndarray, beta: float) -> np.ndarray:
        """Sampling probabilities for one decision step.

        The score is standardised within the step before the temperature is
        applied, and that one design decision has a consequence worth stating
        plainly, because it explains a whole line of investigation that turned
        out to be a dead end.

        The reference paper (Shen et al., AAAI 2022) turns its linear-SVM
        distance `d` into a probability in a second step,

            psi(d) = 1 / (1 + exp(-(theta0 * d + theta1))),

        with theta0 = 9.7750 and theta1 = 12.5564 tuned by Bayesian
        optimisation over 300 evaluations.  Read as a sampler, that is an
        *affine* transform of the score followed by a softmax.  Standardising
        first therefore cancels it completely: a positive scale washes out in
        the division by the standard deviation, and the shift washes out
        because softmax is invariant to a constant offset.  So theta0 cannot
        be the reason a linear SVM scored differently from logistic
        regression here - and it was worth checking, because a scale
        mismatch is exactly what it looks like.  The paper's theta0 is, in
        effect, a temperature, and `beta` is this implementation's temperature,
        which is why the temperature sweep is the experiment that reproduces
        the paper's second step rather than a rerun of its parameter fit.

        The standardisation is also what makes `beta` comparable across
        scoring modes: one unit is always "one standard deviation of this
        step's scores", whether those scores are duals or model logits.
        Without it, equal `beta` would mean wildly different exploration in
        the two arms and the comparison would be an artefact.
        """
        if logits.size <= 1:
            return np.ones(max(logits.size, 1))
        z = logits - float(logits.mean())
        s = float(logits.std())
        if s > 1e-12:
            z = z / s
        w = np.exp(np.clip(beta * z, -60.0, 60.0))
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(logits.size, 1.0 / logits.size)
        return w / total

    def _pick(
        self,
        st: RolloutState,
        cand: np.ndarray,
        pi: np.ndarray,
        rng: np.random.Generator,
        beta: Optional[float],
        rank_freq: Optional[np.ndarray],
    ) -> int:
        logits = self._logits(st, cand, pi, rank_freq)
        if beta is None or cand.size == 1:
            return int(cand[int(np.argmax(logits))])
        return int(rng.choice(cand, p=self._weights(logits, beta)))

    # ------------------------------------------------------------------
    # one rollout
    # ------------------------------------------------------------------
    def _route_energy(self, path: Sequence[int]) -> float:
        inst = self.inst
        carried = float(sum(inst.demand[j] for j in path))
        e, prev = 0.0, DEPOT
        for j in path:
            e += inst.arc_energy(prev, j, carried)
            carried -= float(inst.demand[j])
            prev = j
        e += inst.arc_energy(prev, DEPOT, carried)
        return e

    def _try_close(
        self,
        path: List[int],
        node: int,
        time: float,
        dist_so_far: float,
        pi_sum: float,
        sigma: float,
        fixed_cost: float,
    ) -> Tuple[Optional[RouteColumn], str]:
        if not path:
            return None, "empty"
        inst = self.inst
        back = time + inst.service[node] + inst.travel_time[node, DEPOT]
        if back > inst.due[DEPOT] + EPS:
            return None, "tw"
        rc = dist_so_far + float(inst.dist[node, DEPOT]) - pi_sum + fixed_cost - sigma
        if rc >= -EPS:
            return None, "rc"
        if inst.params.energy_enabled:
            if self._route_energy(path) > float(inst.params.energy_capacity) + EPS:
                return None, "energy"
        return (tuple(path), rc), "ok"

    def _rollout(
        self,
        pi: np.ndarray,
        sigma: float,
        fixed_cost: float,
        rng: np.random.Generator,
        beta: Optional[float],
        diag: RolloutDiagnostics,
        trace: Optional[list] = None,
        rank_freq: Optional[np.ndarray] = None,
    ) -> Tuple[List[RouteColumn], List[Tuple[int, int]]]:
        inst = self.inst
        ctx = self.ctx
        n = ctx.n

        node, time, load = DEPOT, 0.0, 0.0
        dist_so_far, pi_sum, energy_lb = 0.0, 0.0, 0.0
        visited = np.zeros(n + 1, dtype=bool)
        visited[DEPOT] = True
        path: List[int] = []
        found: List[RouteColumn] = []
        arcs: List[Tuple[int, int]] = []
        stuck = False
        # True while a built state has not been close-tested yet.  A sample
        # that gets stuck still hands in whatever it managed to build instead
        # of being thrown away, which is what keeps the column batch large.
        pending = False

        def close_now() -> None:
            nonlocal pending
            col, why = self._try_close(
                path, node, time, dist_so_far, pi_sum, sigma, fixed_cost
            )
            if col is not None:
                found.append(col)
                diag.closed_ok += 1
            elif why == "tw":
                diag.rejected_tw += 1
            elif why == "energy":
                diag.rejected_energy += 1
            elif why == "rc":
                diag.rejected_rc += 1
            pending = False

        while len(path) < self.max_rollout_len:
            st = RolloutState(
                node=node, time=time, load=load, dist_so_far=dist_so_far,
                pi_sum=pi_sum, energy_lb=energy_lb, step=len(path),
                path=tuple(path), visited=visited.copy(),
            )
            cand = ctx.mask(st)
            if cand.size == 0:
                stuck = True
                break
            cand = self._pool(st, cand, pi)
            if trace is not None:
                trace.append((st, cand.copy()))

            j = self._pick(st, cand, pi, rng, beta, rank_freq)

            # --- extend -------------------------------------------------
            arr = time + inst.service[node] + inst.travel_time[node, j]
            start = arr if arr > inst.ready[j] else float(inst.ready[j])
            # energy lower bound on this arc: the load carried is at least the
            # demand of the node being flown to, since that demand must be on
            # board in order to be delivered at all
            rate_lb = ctx.base_rate * (1.0 + ctx.load_factor * float(inst.demand[j]) / ctx.cap)
            energy_lb += rate_lb * float(inst.dist[node, j])
            dist_so_far += float(inst.dist[node, j])
            pi_sum += float(pi[j - 1])
            load += float(inst.demand[j])
            arcs.append((node, j))
            node = j
            time = start
            visited[j] = True
            path.append(j)
            pending = True

            # --- closing is attempted at every step, not only at the end --
            if self.record_prefixes or len(path) == self.max_rollout_len:
                close_now()

        if pending:
            close_now()

        if not found:
            diag.no_close += 1
        if stuck:
            diag.stuck += 1
        return found, arcs

    # ------------------------------------------------------------------
    def _warmup_rank(
        self, pi: np.ndarray, sigma: float, fixed_cost: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Arc usage frequency over a cheap sample batch.

        The reference paper gives every vertex a "rank" feature built from how
        it scores inside sampled solutions.  The arc counterpart is how often
        an arc shows up in sampled routes: an arc that the sampler keeps
        choosing is one the search believes in.
        """
        n = self.ctx.n
        counts = np.zeros((n + 1, n + 1), dtype=float)
        scratch = RolloutDiagnostics()
        for _ in range(self.warmup_samples):
            _, arcs = self._rollout(pi, sigma, fixed_cost, rng, self.beta, scratch)
            for i, j in arcs:
                counts[i, j] += 1.0
        return counts / max(self.warmup_samples, 1)

    # ------------------------------------------------------------------
    def price(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        max_columns: int = 50,
    ) -> PricingResult:
        pi = np.asarray(pi, dtype=float)
        self._calls += 1
        # One dual solution per call, so the dual-dependent feature tables are
        # built once here and gathered several hundred times below.
        self._dual = self.ctx.dual_block(pi)
        rng = np.random.default_rng(self.seed * 1_000_003 + self._calls)

        rank_freq = None
        if self.scoring == "model" and self.warmup_samples > 0:
            rank_freq = self._warmup_rank(pi, sigma, fixed_cost, rng)

        diag = RolloutDiagnostics()
        best: Dict[Tuple[int, ...], float] = {}

        # deterministic backstop first: guarantees a non-empty answer
        for _ in range(self.deterministic_passes):
            diag.rollouts += 1
            found, _ = self._rollout(pi, sigma, fixed_cost, rng, None, diag, None, rank_freq)
            for path, rc in found:
                if path not in best or rc < best[path]:
                    best[path] = rc

        for _ in range(self.n_samples):
            diag.rollouts += 1
            found, _ = self._rollout(
                pi, sigma, fixed_cost, rng, self.beta, diag, None, rank_freq
            )
            for path, rc in found:
                if path not in best or rc < best[path]:
                    best[path] = rc

        cols: List[RouteColumn] = sorted(best.items(), key=lambda kv: kv[1])
        stats = diag.as_stats()
        stats.update({
            "negative_found": len(cols),
            "scoring": self.scoring,
            "n_samples": self.n_samples,
            "beta": self.beta,
            "mean_neg_per_rollout": len(cols) / max(diag.rollouts, 1),
        })
        return PricingResult(
            columns=cols[:max_columns],
            min_reduced_cost=cols[0][1] if cols else 0.0,
            exact=False,
            relaxed_min_reduced_cost=None,
            stats=stats,
        )

    # ------------------------------------------------------------------
    def collect_arc_samples(
        self,
        pi: np.ndarray,
        sigma: float = 0.0,
        fixed_cost: float = 0.0,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Replay the sampler and dump every arc decision it considered.

        Returns `(X, arcs)` with `X` of shape (M, NUM_FEATURES) and `arcs` of
        shape (M, 2) holding `(tail, head)`.  This is the training input: the
        label of each row comes from whether that arc appears in an optimal
        route for the same dual solution, which is what `labeling.py` builds.
        """
        pi = np.asarray(pi, dtype=float)
        self._dual = self.ctx.dual_block(pi)
        n_samples = self.n_samples if n_samples is None else int(n_samples)
        rng = np.random.default_rng(self.seed * 7919 + (seed if seed is not None else 0))

        rank_freq = None
        if self.scoring == "model" and self.warmup_samples > 0:
            rank_freq = self._warmup_rank(pi, sigma, fixed_cost, rng)

        Xs: List[np.ndarray] = []
        tails: List[np.ndarray] = []
        heads: List[np.ndarray] = []
        diag = RolloutDiagnostics()
        total = 0

        for _ in range(n_samples):
            trace: list = []
            diag.rollouts += 1
            self._rollout(pi, sigma, fixed_cost, rng, self.beta, diag, trace, rank_freq)
            for st, cand in trace:
                X = arc_features(self.ctx, st, cand, pi, rank_freq,
                                 dual=self._dual)
                Xs.append(X)
                tails.append(np.full(cand.size, st.node, dtype=int))
                heads.append(cand.astype(int))
                total += cand.size
                if max_samples is not None and total >= max_samples:
                    break
            if max_samples is not None and total >= max_samples:
                break

        if not Xs:
            return np.zeros((0, NUM_FEATURES)), np.zeros((0, 2), dtype=int)
        X_all = np.vstack(Xs)
        arcs = np.column_stack([np.concatenate(tails), np.concatenate(heads)])
        return X_all, arcs
