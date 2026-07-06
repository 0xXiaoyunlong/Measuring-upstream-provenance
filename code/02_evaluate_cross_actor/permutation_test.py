"""
Significance of the cross-crew signal by a crew-preserving permutation test.

A held-out AUROC of 0.68 only means something if a random labelling would not
score that high as easily. This test shuffles which crews are illicit (a whole
crew's deposits move together, never one deposit at a time), re-runs the entire
leave-one-crew-out evaluation, and records the shuffled AUROC; 1000 shuffles give
a null distribution. The p-value is the share of shuffles reaching the observed
score: p = (1 + #{shuffled >= observed}) / (1 + N).

Reproduces the paper's significance (observed 0.687, p = 0.044 for 17 crews):

    python permutation_test.py

The 1000 shuffled scores are shipped in ../../data/permutation_null.csv, because
each one re-runs the whole evaluation and they are slow to regenerate.
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")


def load_null(crew_set, model):
    """The shuffled AUROCs from the archived permutation run."""
    with open(os.path.join(DATA, "permutation_null.csv"), encoding="utf-8-sig") as fh:
        return [float(r["shuffled_auroc"]) for r in csv.DictReader(fh)
                if r["crew_set"] == crew_set and r["model"] == model
                and r["shuffled_auroc"] not in ("", None)]


def load_observed(crew_set, model):
    """The real (unshuffled) AUROC and the p-value recorded at analysis time."""
    with open(os.path.join(DATA, "permutation_summary.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["crew_set"] == crew_set and r["model"] == model:
                return r
    return None


def empirical_p(observed, null_scores):
    """Share of shuffles reaching the observed score (with the +1 correction)."""
    at_least_as_high = sum(1 for v in null_scores if v >= observed)
    return (1 + at_least_as_high) / (1 + len(null_scores))


def report(crew_set, model, label):
    summary = load_observed(crew_set, model)
    null_scores = load_null(crew_set, model)
    observed = float(summary["observed_auroc"])

    p = empirical_p(observed, null_scores)
    null_centre = float(np.mean(null_scores))

    print(f"{label}")
    print(f"  observed AUROC        {observed:.4f}")
    print(f"  shuffles run          {len(null_scores)}")
    print(f"  null distribution     mean {null_centre:.3f}, "
          f"middle 95% [{np.percentile(null_scores, 2.5):.3f}, {np.percentile(null_scores, 97.5):.3f}]")
    print(f"  recomputed p-value    {p:.5f}   (archived {summary['p_value']})")
    verdict = "significant at 0.05" if p < 0.05 else "not significant at 0.05"
    print(f"  verdict               {verdict}\n")


def main():
    print("Group-preserving permutation test (Ojala & Garriga, 2010)\n")
    report("17_crews", "logistic",
           "Primary: 17 crews, logistic regression")
    report("18_crews", "logistic",
           "Sensitivity: add one cross-chain crew (18 crews)")
    # The 17-crew signal clears 0.05; one added crew pushes it back over. The null
    # centres near 0.53, not 0.5, so compare against it rather than against 0.5.


if __name__ == "__main__":
    main()
