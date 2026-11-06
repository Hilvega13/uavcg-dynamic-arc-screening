"""Pluggable pricing subproblem solvers.

    exact.py    ExactLabelingPricer  exact ESPPRC labelling; reference and
                                     final certifier, usable to about n=20
    ngroute.py  NgRoutePricer        ng-route relaxation; the practical
                                     workhorse for larger instances
    redcost.py  RedCostArcPricer     per-iteration reduced-cost arc reduction;
                                     the train-free baseline that has to be
                                     beaten (Morabit's RedCost-S)
    greedy.py   GreedyPricer         classical nearest-neighbour baseline
    rollout.py  RolloutPricer        feasibility-preserving serialised rollout;
                                     uniform / dual-guided / learned scoring

All of them implement the same Pricer interface and drop straight into
ColumnGeneration, so an experiment changes one constructor argument and
nothing else.
"""

from .base import Pricer, PricingResult, RouteColumn
from .exact import ExactLabelingPricer
from .greedy import GreedyPricer
from .ngroute import NgRoutePricer
from .redcost import RedCostArcPricer
from .rollout import RolloutDiagnostics, RolloutPricer

__all__ = [
    "Pricer",
    "PricingResult",
    "RouteColumn",
    "ExactLabelingPricer",
    "GreedyPricer",
    "NgRoutePricer",
    "RedCostArcPricer",
    "RolloutPricer",
    "RolloutDiagnostics",
]
