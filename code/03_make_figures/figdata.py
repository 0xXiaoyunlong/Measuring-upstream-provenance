# -*- coding: utf-8 -*-
"""
Data layer shared by the two-background figures (fig5/fig6/fig7).

Every number those plots show is read out of two frozen packages: the canonical
run at the repo root and the temporally balanced background under
balanced_background/. The plot scripts are not allowed to carry numbers of
their own -- if a figure needs a value it has to come through here, so a wrong
number is wrong in exactly one place.

Running the file directly executes verify(), which checks the paper's headline
numbers against both packages. Do that before trusting a redraw:

    python figdata.py

Both packages are read-only as far as this module is concerned; it writes
nothing.
"""
import csv
import io
import json
import os
import statistics
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CAN = os.path.abspath(os.path.join(HERE, "..", ".."))       # package root
SWAP = os.path.join(CAN, "balanced_background")

# The two OFAC dates are quoted from the Treasury press releases cited in the
# paper.
OFAC_DESIGNATION = datetime(2022, 8, 8, tzinfo=timezone.utc)
OFAC_DELISTING = datetime(2025, 3, 21, tzinfo=timezone.utc)

# frozen extraction protocol of Algorithm 2
PROTOCOL = {"window_days": 7, "depth": 2, "value_floor_eth": 10,
            "share_floor_V": 0.1, "cone_cap": 25, "coverage_stop": 0.3}


def csv_rows(which, name):
    """Read any frozen CSV that has no dedicated wrapper below."""
    return _rows(which, name)


def _pkg(which):
    return {"can": CAN, "swap": SWAP}[which]


def _rows(which, name):
    with io.open(os.path.join(_pkg(which), "data", name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def funnel():
    """[(stage, n, description)] in file order (census -> admitted); the funnel fig1 shows."""
    return [(r["stage"], int(r["n_candidates"]), r["description"])
            for r in _rows("can", "dataset_funnel.csv")]


# entrance events, parsed once and reused by timeline() and the count checks
def events(which="can"):
    rows = _rows(which, "entrance_events.csv")
    for r in rows:
        r["dt"] = datetime.strptime(r["deposit_time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        r["is_illicit"] = int(r["is_illicit"])
    return rows


def timeline():
    """Three date tracks plus the pre/post designation counts, all derived from events()."""
    can, swap = events("can"), events("swap")
    old_bg = [r["dt"] for r in can if r["is_illicit"] == 0]
    new_bg = [r["dt"] for r in swap if r["is_illicit"] == 0]
    crews = {}
    for r in can:
        if r["is_illicit"]:
            crews.setdefault(r["crew"], []).append(r["dt"])
    ill = [d for ds in crews.values() for d in ds]
    med = {c: sorted(ds)[len(ds) // 2] for c, ds in crews.items()}
    return {
        "old_bg": old_bg, "new_bg": new_bg, "crews": crews, "crew_median": med,
        "old_bg_pre": sum(d < OFAC_DESIGNATION for d in old_bg),
        "old_bg_post": sum(d >= OFAC_DESIGNATION for d in old_bg),
        "ill_pre": sum(d < OFAC_DESIGNATION for d in ill),
        "ill_post": sum(d >= OFAC_DESIGNATION for d in ill),
        "crew_pre": sum(m < OFAC_DESIGNATION for m in med.values()),
        "crew_post": sum(m >= OFAC_DESIGNATION for m in med.values()),
    }


# headline AUROCs and the permutation nulls
def main_results(which="can"):
    """Headline rows keyed '17'/'18': auroc, the CI bounds, and the legacy p,
    all as raw floats."""
    out = {}
    for r in _rows(which, "main_results.csv"):
        key = "17" if r["crew_set"].startswith("17") else "18"
        out[key] = {"auroc": float(r["auroc"]), "lo": float(r["ci_low"]),
                    "hi": float(r["ci_high"]),
                    "p": float(r["p_value"]) if r["p_value"] else None}
    return out


def per_crew(which="can"):
    return [{"crew": r["crew"], "type": r["crew_type"], "n": int(r["n_events"]),
             "n_bg": int(r["n_background_in_cohort"]), "auroc": float(r["auroc_logistic"])}
            for r in _rows(which, "per_crew_results.csv")]


def perm_null(which="can", crew_set="17_crews"):
    return [float(r["shuffled_auroc"]) for r in _rows(which, "permutation_null.csv")
            if r["crew_set"] == crew_set]


def corrected_null(which="can"):
    return [float(r["shuffled_auroc"]) for r in _rows(which, "permutation_sensitivity_null.csv")
            if r["null_variant"] == "corrected_154block"]


def perm_summary(which="can"):
    """p values and null means of the legacy (17/18) and corrected nulls."""
    out = {}
    for r in _rows(which, "permutation_legacy_summary.csv"):
        key = "legacy_17" if r["crew_set"].startswith("17") else "legacy_18"
        out[key] = {"p": float(r["p_value"]), "null_mean": float(r["null_mean"]),
                    "observed": float(r["observed_auroc"])}
    for r in _rows(which, "permutation_sensitivity_summary.csv"):
        if r["null_variant"] == "corrected_154block":
            out["corrected"] = {"p": float(r["p_value"]), "null_mean": float(r["null_mean"]),
                                "observed": float(r["observed_auroc"])}
        if r["null_variant"].startswith("singleton"):
            out["singleton"] = {"p": float(r["p_value"]), "null_mean": float(r["null_mean"])}
    return out


def audit(which="can"):
    """Missingness-audit results: before/after AUROC per feature set, and the
    increments as {comparison: (delta, lo, hi)}."""
    m = {"before": {}, "after": {}}
    for r in _rows(which, "missingness_audit_metrics.csv"):
        stage = "before" if r["stage"].startswith("before") else "after"
        m[stage][r["feature_set"]] = (float(r["auroc"]), float(r["ci_low"]), float(r["ci_high"]))
    inc = {}
    for r in _rows(which, "missingness_audit_increments.csv"):
        inc[r["comparison"]] = (float(r["auroc_change"]), float(r["ci_low"]), float(r["ci_high"]))
    m["increments"] = inc
    return m


def recovery(which="can"):
    return _rows(which, "outflow_recovery.csv")


# The forest data (fig5) ships as JSON, one file per package, unlike the CSVs above.
_FOREST = {
    "can": os.path.join(CAN, "robustness", "robustness_forest_data.json"),
    "swap": os.path.join(SWAP, "robustness", "robustness_forest_data.json"),
}


def forest(which="can"):
    with io.open(_FOREST[which], encoding="utf-8") as fh:
        return json.load(fh)


def forest_rows(which="can"):
    """All 30 variants (anchor + 29 perturbations) flattened, each tagged with its
    section. which='can' is the original 139-event background (anchor 0.703);
    'swap' the temporally balanced 140 (anchor 0.802)."""
    d = forest(which)
    rows = [dict(d["none"], section="anchor")]
    rows += [dict(r, section="loo") for r in d["loo"]]
    rows += [dict(r, section="params") for r in d["params"] if not r["label"].startswith("Main")]
    rows += [dict(r, section="feats") for r in d["feats"] if not r["label"].startswith("Full")]
    return rows


def forest_pair():
    """[(can_row, swap_row)] variants aligned one-to-one (same label order)."""
    c, s = forest_rows("can"), forest_rows("swap")
    assert [r["label"] for r in c] == [r["label"] for r in s], "forest labels misaligned"
    return list(zip(c, s))


def missingness_matrix():
    rows = _rows("can", "missingness_by_crew.csv")
    feats = [k for k in rows[0] if k not in ("crew", "n_events")]
    return {"crews": [r["crew"] for r in rows], "features": feats,
            "share": [[float(r[f]) for f in feats] for r in rows],
            "n_events": [int(r["n_events"]) for r in rows]}


# ---------------------------------------------------------------------------
# oracle checks: every headline number in the paper, re-read from the frozen
# packages. The check names below are printed verbatim, so leave those strings
# alone.
def verify():
    ok = []

    def chk(name, cond):
        ok.append((name, bool(cond)))
        print(("PASS  " if cond else "FAIL  ") + name)

    fu = dict((s, n) for s, n, _d in funnel())
    chk("funnel census=86 / admitted=12",
        fu.get("0_census") == 86 and 12 in fu.values())

    can = events("can")
    bg = [r for r in can if r["is_illicit"] == 0]
    chk("canonical 169 events = 30 illicit + 139 bg",
        len(can) == 169 and sum(r["is_illicit"] for r in can) == 30 and len(bg) == 139)
    chk("bg unique addresses = 137",
        len({r["deposit_address"].lower() for r in bg}) == 137)
    chk("17 crews", len({r["crew"] for r in can if r["is_illicit"]}) == 17)

    mr, pc = main_results("can"), per_crew("can")
    macro = statistics.mean(r["auroc"] for r in pc)
    chk("macro(per_crew)==main_results==0.702949",
        abs(macro - mr["17"]["auroc"]) < 1e-9 and abs(macro - 0.702949) < 1e-4)
    chk("canonical p legacy=.027 corrected=.031",
        abs(mr["17"]["p"] - 0.026973) < 1e-4
        and abs(perm_summary("can")["corrected"]["p"] - 0.030969) < 1e-4)
    chk("18-crew 0.631 / p=.146",
        abs(mr["18"]["auroc"] - 0.631) < 0.001 and abs(mr["18"]["p"] - 0.146) < 0.001)

    au = audit("can")
    inc = au["increments"]
    comb = [v for k, v in inc.items() if "+" in k]
    fact = inc.get("factory-pattern minus provenance")
    chk("audit before Phi=0.703, factory-alone-minus-Phi=-0.195",
        abs(au["before"]["provenance (Phi)"][0] - 0.702949) < 1e-6
        and fact is not None and abs(fact[0] + 0.194562) < 1e-4)
    chk("audit combined increment -0.006 spans zero",
        comb and abs(comb[0][0] + 0.0061) < 0.001 and comb[0][1] < 0 < comb[0][2])

    fr = forest_rows()
    aurocs = [r["auroc"] for r in fr]
    chk("forest 30 variants, range 0.576-0.779, 6 significant",
        len(fr) == 30 and abs(min(aurocs) - 0.576) < 0.001
        and abs(max(aurocs) - 0.779) < 0.001
        and sum(r["p"] < 0.05 for r in fr) == 6)

    sf = forest_rows("swap")
    chk("swap forest 30 variants, anchor 0.802 / p=.003, labels aligned",
        len(sf) == 30 and abs(sf[0]["auroc"] - 0.802) < 0.001
        and abs(sf[0]["p"] - 0.002997) < 1e-4
        and [r["label"] for r in sf] == [r["label"] for r in fr])

    sw = main_results("swap")
    ps = perm_summary("swap")
    chk("swap 0.802177 / legacy p=.003 / corrected p=.002",
        abs(sw["17"]["auroc"] - 0.802177) < 1e-4
        and abs(ps["legacy_17"]["p"] - 0.002997) < 1e-4
        and abs(ps["corrected"]["p"] - 0.001998) < 1e-4)
    chk("swap 18-crew 0.772 / p=.006",
        abs(sw["18"]["auroc"] - 0.772106) < 1e-4 and abs(sw["18"]["p"] - 0.005994) < 1e-4)
    swbg = [r for r in events("swap") if r["is_illicit"] == 0]
    chk("swap bg 140 events / 140 unique",
        len(swbg) == 140 and len({r["deposit_address"].lower() for r in swbg}) == 140)

    tl = timeline()
    chk("timeline old-bg 135 pre / 4 post; illicit 18/12; crews 5/12",
        tl["old_bg_pre"] == 135 and tl["old_bg_post"] == 4
        and tl["ill_pre"] == 18 and tl["ill_post"] == 12
        and tl["crew_pre"] == 5 and tl["crew_post"] == 12)

    mm = missingness_matrix()
    chk("missingness matrix 18 rows x 15 features",
        len(mm["crews"]) == 18 and len(mm["features"]) == 15)

    n_fail = sum(1 for _n, c in ok if not c)
    print(f"\n{len(ok) - n_fail}/{len(ok)} oracle checks passed")
    return n_fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if verify() else 1)
