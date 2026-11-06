"""The learned arc scorer.

Model choice
------------
Three levels, deliberately, so the ablation is a switch and not a rewrite:

    "logistic"  L2 logistic regression on standardised features
    "svm"       linear SVM - the model family the reference paper uses, kept
                so its result can be reproduced rather than assumed
    "gbdt"      histogram gradient boosting - the default

The reference paper's linear-SVM-trained scores are turned into a probability
by a logistic function and used to sample.  That is kept, but it is not
assumed to be the right choice here.  Two pieces of evidence argue for a tree
model on this problem: the paper by Basso & Salani (2026) that measured
feature-target correlation on exactly this kind of labelling task reports
that *no single feature correlates strongly with the label* (mean correlation
near zero), which is the signature of a feature interaction that a linear
model cannot represent; and route decisions involve thresholds (does the
residual capacity still fit, does the window still close), which trees
represent natively and linear models do not.

Training uses balanced class weights rather than resampling.  Positive arcs
are a tiny minority - an arc is on an optimal route or it is not, and most
candidate arcs are not - and Basso & Salani's 1:1 resampling result is
reproduced here by weighting instead, which keeps every row.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES, NUM_FEATURES


@dataclass(frozen=True)
class PsiCalibration:
    """The reference paper's second step: signed distance -> probability.

        psi(d) = 1 / (1 + exp(-(theta0 * d + theta1)))

    The published constants are theta0 = 9.7750 and theta1 = 12.5564, tuned by
    Bayesian optimisation over 300 evaluations.  They exist because a linear
    SVM does not output a probability: `decision_function` returns an
    uncalibrated signed distance to the separating hyperplane, and the paper
    needs a number it can normalise into sampling weights.

    Kept here so the pipeline is a faithful copy of the paper rather than a
    paraphrase of it - and kept *separate* from `ArcScorer.proba`, because
    applying it by default would silently change every classification metric
    this project has already reported.  The published theta are large enough
    that psi saturates: any d above roughly -1.3 maps to a probability above
    0.5, and d above 0 maps to above 0.99999.  That is harmless when the
    consumer only needs relative weights, and destructive when the consumer
    applies a 0.5 threshold.

    Note for this project: `RolloutPricer` standardises scores within each
    decision step before sampling, which cancels any *affine* rescaling of the
    input.  psi is affine in d, so in the rollout it is exactly a no-op - the
    self-test asserts that rather than leaving it as a comment.  Its role in
    the paper is taken here by the sampling temperature `beta`.
    """

    theta0: float = 9.7750
    theta1: float = 12.5564

    def proba(self, d: np.ndarray) -> np.ndarray:
        z = self.theta0 * np.asarray(d, dtype=float) + self.theta1
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))

    def logit(self, d: np.ndarray) -> np.ndarray:
        return self.theta0 * np.asarray(d, dtype=float) + self.theta1


def label_confidence(y: np.ndarray) -> np.ndarray:
    """How much evidence a graded target carries, in [0.5, 1].

    A row sitting exactly on the decision boundary (grade 0.5) is half evidence,
    not none: it still counts towards the class balance, it just cannot pull
    the decision surface on its own.  Symmetric by construction, so a row graded
    0.6 as a positive and one graded 0.4 as a negative carry the same weight -
    otherwise the soft fit would systematically favour one class, and with a
    4%-positive label set that bias would be invisible in the metrics and loud
    in the search.
    """
    y = np.asarray(y, dtype=float)
    return np.maximum(y, 1.0 - y)


@dataclass
class Scores:
    """Diagnostics from a fit."""

    kind: str
    n_samples: int
    positive_rate: float
    train_seconds: float
    importance: Optional[List[Tuple[str, float]]] = None
    #: True when the fit consumed graded targets rather than binary ones.
    soft: bool = False
    #: Mean of the targets as they arrived, before thresholding at 0.5.
    mean_target: float = 0.0

    def top(self, k: int = 8) -> List[Tuple[str, float]]:
        if not self.importance:
            return []
        return sorted(self.importance, key=lambda kv: -abs(kv[1]))[:k]


class ArcScorer:
    """Wraps a scikit-learn classifier behind a stable `logit` interface.

    `logit` must be monotone in P(arc is optimal) and on a scale where a
    softmax makes sense; the rollout standardises it per decision step anyway,
    so only the ordering and relative gaps matter.
    """

    def __init__(
        self,
        kind: str = "gbdt",
        balance: bool = True,
        seed: int = 0,
        fast_path: bool = True,
        **kwargs,
    ) -> None:
        if kind not in ("logistic", "svm", "gbdt"):
            raise ValueError(f"unknown scorer kind {kind!r}")
        self.kind = kind
        self.balance = balance
        self.seed = seed
        #: Collapse a linear pipeline into a single affine map before use.
        #:
        #: This is not a micro-optimisation, it is the difference between the
        #: learned pricer being affordable and not.  `scripts/profile_rollout.py`
        #: measured the learned rollout at n=25 and found that **72% of a
        #: decision step is `logit` on a 15-row matrix** - the cost is
        #: scikit-learn's per-call validation and pipeline indirection, not the
        #: arithmetic, which on 15x17 floats is nothing.  Extracting
        #: `coef_`/`intercept_` (with the scaler folded in) removes sklearn from
        #: the hot path entirely and makes a pricing call roughly four times
        #: cheaper.  `fast_path=False` keeps the slow route reachable so the
        #: self-test can assert the two agree.
        self.fast_path = fast_path
        self.kwargs = kwargs
        self.model = None
        self.scores: Optional[Scores] = None
        self._affine: Optional[Tuple[np.ndarray, float]] = None

    # ------------------------------------------------------------------
    def _build(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        if self.kind == "logistic":
            return Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, C=self.kwargs.get("C", 1.0),
                    random_state=self.seed,
                )),
            ])
        if self.kind == "svm":
            return Pipeline([
                ("scale", StandardScaler()),
                ("clf", LinearSVC(
                    C=self.kwargs.get("C", 1.0), random_state=self.seed,
                    max_iter=self.kwargs.get("max_iter", 20000),
                )),
            ])
        return HistGradientBoostingClassifier(
            max_iter=self.kwargs.get("max_iter", 300),
            learning_rate=self.kwargs.get("learning_rate", 0.08),
            max_leaf_nodes=self.kwargs.get("max_leaf_nodes", 31),
            min_samples_leaf=self.kwargs.get("min_samples_leaf", 20),
            l2_regularization=self.kwargs.get("l2_regularization", 1.0),
            random_state=self.seed,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: Sequence[str] = FEATURE_NAMES,
        soft: bool = False,
    ) -> "ArcScorer":
        """Fit on either hard binary labels or graded targets in [0, 1].

        With `soft=True` the targets come from `labeling.near_optimal_arcs` and
        say *how* near-optimal the best route through an arc was, not merely
        whether it was within tolerance.  The classifier is still handed hard
        classes at a 0.5 threshold - that is the only thing a logistic
        regression or a GBDT can consume - and the grading enters through the
        weights: a row sitting at the edge of the tie set is down-weighted
        towards 0.5, because it is weaker evidence than a row on the best
        route.  Equal weights for the whole tie set was precisely the defect of
        the binary label.
        """
        import time

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
            raise ValueError(f"expected (m, {NUM_FEATURES}) features, got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("feature matrix contains non-finite values")
        if len(y) != X.shape[0]:
            raise ValueError("feature/label length mismatch")
        if soft and not np.all((y >= 0.0) & (y <= 1.0)):
            raise ValueError("graded targets must lie in [0, 1]")

        hard = (y > 0.5).astype(float) if soft else y

        t0 = time.perf_counter()
        self.model = self._build()
        if len(np.unique(hard)) < 2:
            # a single class cannot be learned; fall back to a constant score
            self.model = _ConstantScorer(float(hard[0]) if len(hard) else 0.0)
        else:
            w = None
            if self.balance:
                pos = float(hard.sum())
                neg = float(len(hard) - pos)
                if pos > 0 and neg > 0:
                    w = np.where(hard > 0.5,
                                 len(hard) / (2.0 * pos),
                                 len(hard) / (2.0 * neg))
            if soft:
                conf = label_confidence(y)
                w = conf if w is None else w * conf
            if hasattr(self.model, "named_steps"):
                # sklearn pipelines need the step name prefixed
                if w is not None:
                    self.model.fit(X, hard, **{"clf__sample_weight": w})
                else:
                    self.model.fit(X, hard)
            elif w is not None:
                self.model.fit(X, hard, sample_weight=w)
            else:
                self.model.fit(X, hard)
        dt = time.perf_counter() - t0

        self.scores = Scores(
            kind=self.kind, n_samples=int(X.shape[0]),
            positive_rate=float(hard.mean()) if len(hard) else 0.0,
            train_seconds=dt, importance=self._importance(X, hard, names),
            soft=bool(soft),
            mean_target=float(y.mean()) if len(y) else 0.0,
        )
        self._build_affine()
        return self

    # ------------------------------------------------------------------
    def _build_affine(self) -> None:
        """Fold `StandardScaler` + a linear classifier into `x @ w + b`.

        Exact, not approximate: for a pipeline whose first step scales each
        column by a per-column affine map, the composition is itself affine, so
        `clf.decision_function(scale(X))` equals `X @ (coef/sd) + (b - coef.mu/sd)`.
        The only guard needed is against a zero scale, which `StandardScaler`
        itself sets to 1 for constant columns.
        """
        self._affine = None
        if not self.fast_path or self.model is None:
            return
        m = self.model
        if not hasattr(m, "named_steps"):
            return
        clf = m.named_steps.get("clf")
        if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
            return
        w = np.asarray(clf.coef_, dtype=float).ravel()
        if w.size != NUM_FEATURES:
            return
        b = float(np.asarray(clf.intercept_, dtype=float).ravel()[0])
        sc = m.named_steps.get("scale")
        if sc is not None and getattr(sc, "mean_", None) is not None:
            mu = np.asarray(sc.mean_, dtype=float).ravel()
            sd = np.asarray(sc.scale_, dtype=float).ravel()
            sd = np.where(np.abs(sd) > 0.0, sd, 1.0)
            b -= float(np.dot(w / sd, mu))
            w = w / sd
        self._affine = (np.ascontiguousarray(w), b)

    # ------------------------------------------------------------------
    def _importance(self, X, y, names) -> Optional[List[Tuple[str, float]]]:
        try:
            if hasattr(self.model, "named_steps"):
                pipe = self.model
                clf = pipe.named_steps.get("clf")
                if hasattr(clf, "coef_"):
                    coef = np.asarray(clf.coef_).ravel()
                    if coef.size == NUM_FEATURES:
                        return [(str(n), float(c)) for n, c in zip(names, coef)]
                return None
            if X.shape[0] > 4000:
                idx = np.random.default_rng(0).choice(X.shape[0], 4000, replace=False)
                X, y = X[idx], y[idx]
            from sklearn.inspection import permutation_importance

            r = permutation_importance(
                self.model, X, y, n_repeats=3, random_state=0,
                scoring="roc_auc" if len(np.unique(y)) > 1 else None,
            )
            return [(str(n), float(v)) for n, v in zip(names, r.importances_mean)]
        except Exception:
            return None

    # ------------------------------------------------------------------
    def logit(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("scorer has not been fitted")
        X = np.asarray(X, dtype=float)
        if isinstance(self.model, _ConstantScorer):
            return self.model.logit(X)
        if self._affine is not None:
            # The hot path.  Hit once per candidate per decision step, so it is
            # deliberately the first branch: `scripts/profile_rollout.py`
            # measured 130us here against 1.3us for the bare matrix product,
            # all of it scikit-learn's per-call overhead on a 15-row matrix.
            w, b = self._affine
            return X @ w + b
        if hasattr(self.model, "named_steps"):
            return np.asarray(self.model.decision_function(X), dtype=float).ravel()
        p = np.clip(np.asarray(self.model.predict_proba(X), dtype=float)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1.0 - p))

    def proba(self, X: np.ndarray) -> np.ndarray:
        lg = self.logit(X)
        return 1.0 / (1.0 + np.exp(-lg))

    def psi_proba(self, X: np.ndarray, psi: Optional[PsiCalibration] = None) -> np.ndarray:
        """The reference paper's calibrated probability, on demand.

        Separate from `proba` on purpose: `proba` is the sigmoid of whatever
        `logit` returns, which for the linear SVM is the raw distance - i.e.
        *uncalibrated*.  Everything reported before this method existed used
        that definition, so changing the default here would silently
        invalidate those numbers rather than improve them.
        """
        return (psi or PsiCalibration()).proba(self.logit(X))

    # ------------------------------------------------------------------
    @property
    def affine(self) -> Optional[Tuple[np.ndarray, float]]:
        """The folded `(weights, bias)` on the raw feature scale, or None.

        Exposed because it is the thing worth reporting in a paper appendix: a
        linear pricing score whose coefficients can be read off the feature
        names is an interpretable result, and it is the same object the hot
        path uses, so the reported coefficients cannot drift from the deployed
        model.
        """
        return self._affine

    def coefficients(self, names: Sequence[str] = FEATURE_NAMES) -> List[Tuple[str, float]]:
        if self._affine is None:
            return []
        w, b = self._affine
        return [(str(n), float(c)) for n, c in zip(names, w)]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"kind": self.kind, "model": self.model,
                         "scores": self.scores, "affine": self._affine,
                         "fast_path": self.fast_path}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ArcScorer":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(kind=blob["kind"], fast_path=blob.get("fast_path", True))
        obj.model = blob["model"]
        obj.scores = blob.get("scores")
        obj._affine = blob.get("affine")
        if obj._affine is None:
            # models saved before the fast path existed (or saved by a version
            # that could not fold this model family) rebuild it on load
            obj._build_affine()
        return obj

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.scores is None:
            return f"<ArcScorer {self.kind} (unfitted)>"
        tag = " graded" if self.scores.soft else ""
        return (f"<ArcScorer {self.kind}{tag} n={self.scores.n_samples} "
                f"pos={self.scores.positive_rate:.3%} "
                f"fit={self.scores.train_seconds:.2f}s>")


class _ConstantScorer:
    """Placeholder used when a training batch contains only one class."""

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, X, y, **kwargs):  # pragma: no cover - trivial
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.value)

    def predict_proba(self, X):
        p = np.clip(self.value, 1e-6, 1 - 1e-6)
        m = np.asarray(X).shape[0]
        return np.column_stack([np.full(m, 1 - p), np.full(m, p)])

    def logit(self, X):
        p = np.clip(self.value, 1e-6, 1 - 1e-6)
        return np.full(np.asarray(X).shape[0], float(np.log(p / (1 - p))))


# --------------------------------------------------------------------------
def evaluate_scorer(
    scorer: ArcScorer,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Classification metrics that stay meaningful under heavy imbalance.

    Accepts graded targets.  The threshold metrics are computed against the
    hard reading (`y > 0.5`), because that is what a column generator consumes
    when it decides an arc is worth taking.  On top of that, `spearman` scores
    the same predictions against the *graded* target, which is the metric that
    can actually see the information grading was supposed to add: a model whose
    precision/recall is unchanged but whose ranking of the tie set improved
    will show it there and nowhere else.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=float)
    hard = (y > 0.5).astype(float)
    p = scorer.proba(X)
    pred = (p >= threshold).astype(float)
    out = {
        "n": float(len(y)),
        "positive_rate": float(hard.mean()) if len(hard) else 0.0,
        "mean_target": float(y.mean()) if len(y) else 0.0,
        "accuracy": float((pred == hard).mean()) if len(y) else 0.0,
        "precision": float(precision_score(hard, pred, zero_division=0)),
        "recall": float(recall_score(hard, pred, zero_division=0)),
        "f1": float(f1_score(hard, pred, zero_division=0)),
    }
    if len(np.unique(hard)) > 1:
        out["roc_auc"] = float(roc_auc_score(hard, p))
        out["avg_precision"] = float(average_precision_score(hard, p))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
    if len(y) > 2 and len(np.unique(y)) > 2:
        from scipy.stats import spearmanr

        rho = spearmanr(y, scorer.logit(X)).statistic
        out["spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    else:
        out["spearman"] = float("nan")
    return out
