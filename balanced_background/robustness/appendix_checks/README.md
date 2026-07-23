# appendix_checks (balanced copy)

Only `tier3_lib.py` lives here — `run_robustness.py` and
`robustness_permutation.py` import it, and it resolves the cones and CSVs
relative to its own location, so each package needs its own copy one level
below `robustness/`. The actual check scripts are in the root package's
`robustness/appendix_checks/`; they were not duplicated for this background.
