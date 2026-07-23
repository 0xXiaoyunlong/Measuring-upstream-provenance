# Appendix checks

Small scripts for the paper's one-off numbers, the ones neither the main
pipeline nor the variant battery produces. Each writes its output into this
directory, next to the script.

`ledger_checks.py` covers four unrelated numbers in one pass. It backs the AUPRC row 0.538
[0.366, 0.714] and its ~0.15 chance level (Table 2a, ledger T8); the sign
test, 12 up / 5 down, p = 0.143 against the 0.53 empirical null (T6); the
analytic PU correction, 0.714 / 0.725 / 0.754 at 5 / 10 / 20 % (Box 1, T7);
and the cone-cap and coverage-stop counts, 6/167 and 5/169 (Appendix E).
Everything goes into a single `ledger_checks.json`. Because the script only
re-reads archived artifacts in `data/`, none of these numbers depend on the
sklearn build.

`paired_baselines.py` computes the paired variant-vs-baseline deltas for the
six reduced-fold extraction variants (Table F2) and writes
`paired_baselines.json`. It imports the local `tier3_lib.py`, which resolves
everything it needs relative to the package root.

`fold_reassignment.py` redoes the background fold reassignment: 100 seeds at
address-block level, range [0.595, 0.735], median 0.658, with the
prespecified assignment landing at the 94th percentile (T10). The same run
reports a tie-aware AUROC of 0.687. Output is `fold_reassignment_output.txt`.

Unlike `ledger_checks.py`, the last two actually refit models, so their exact
decimals depend on the sklearn/BLAS build, as the variant battery does. The
archived outputs here are from the runs cited in the paper.
