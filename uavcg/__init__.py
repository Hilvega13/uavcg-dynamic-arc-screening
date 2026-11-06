"""uavcg - column generation for multi-drone / VRPTW routing.

Layout
------
instance.py    problem data, Solomon reader, route evaluation
rmp.py         restricted master problem (set covering) + dual handling
pricers/       pluggable pricing subproblem solvers
heuristics.py  construction heuristic used to seed the master
cg.py          the column generation loop

Solver backend
--------------
Gurobi is NOT used: free academic licences were withdrawn for mainland
China on 2026-08-31, and the local node licence expired 2026-07-20.
The LP/MILP work is done by HiGHS through scipy. See rmp.py for the
(single) place that touches a solver, so swapping in COPT / Gurobi /
SCIP later is a local change.
"""

__version__ = "0.1.0"
