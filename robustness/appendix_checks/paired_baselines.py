# -*- coding: utf-8 -*-
"""Paired baselines for the reduced-fold extraction variants (Table F2).

Six of the tightened settings drop events, so putting them straight against
the full 17-fold baseline would mix two effects: the setting itself and the
missing events. Each variant therefore gets its own baseline -- the default
settings re-run with the same events taken out -- and delta compares like
with like. tier3_lib resolves the frozen cones and archived CSVs relative
to the package root.

    python paired_baselines.py   ->  paired_baselines.json
"""
import io
import json
import os

import tier3_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))

# args = (window days, hop depth, value floor in ETH, share floor)
VARIANTS = [("Window 3 d", (3, 2, 10, 0.1)), ("Window 5 d", (5, 2, 10, 0.1)),
            ("Depth 1 hop", (7, 1, 10, 0.1)), ("Value floor 15 ETH", (7, 2, 15, 0.1)),
            ("Value floor 20 ETH", (7, 2, 20, 0.1)), ("Share floor 0.2 V", (7, 2, 10, 0.2))]

paired = {}
for name, args in VARIANTS:
    rows, eids, dropped = T.build_rows(*args)
    v_m, v_k = T.macro(rows)
    b_rows, b_eids, _ = T.build_rows(7, 2, 10, 0.1, exclude=set(dropped))  # same events out
    b_m, b_k = T.macro(b_rows)
    paired[name] = {"variant": v_m, "folds": v_k, "paired_baseline": b_m,
                    "baseline_folds": b_k, "delta": v_m - b_m, "n_dropped": len(dropped)}
    print(f"{name:<20} variant {v_m:.3f} (k{v_k})  paired baseline {b_m:.3f} (k{b_k})  "
          f"delta {v_m-b_m:+.3f}  dropped {len(dropped)}")

json.dump(paired, io.open(os.path.join(HERE, "paired_baselines.json"), "w",
                          encoding="utf-8"), indent=2)
print("saved -> paired_baselines.json")
