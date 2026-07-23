# 1. Collecting the data and building the features

This is how the dataset was gathered. You don't need it to reproduce the paper —
the feature table is already in `../../data/upstream_features.csv`. It's here so the
collection is auditable, and in case anyone wants to re-collect or extend it.

| file | what it does | API key? |
|---|---|---|
| `fetch_onchain_transactions.py` | finds Tornado deposits, pulls the funding transactions off a public block explorer, builds the backward "cone" | yes |
| `build_upstream_features.py` | turns one deposit's cone into the 15 features (Φ in the paper) | no |

## The idea

Take one deposit: an address `a0` that put `V` ETH into a pool at time `t0`. We
walk backward through whoever funded it. `fetch_onchain_transactions.py` builds the
"cone" (who funded `a0`, who funded them — capped at 10 ETH transfers, 7 days, 2
hops, 25 addresses), and `build_upstream_features.py` follows the biggest funder at
each hop and reads off 15 numbers describing that path.

No identities, no labels — just the shape of the money. The one thing worth knowing:
it separates *censored* (trail cut off at a contract) from *missing* (couldn't fetch
the record), and never zero-fills a missing value. The `build_upstream_features.py`
docstring explains why that mattered so much.

## Re-collecting from the chain

```bash
export ETHERSCAN_KEY=your_key_here     # or `set` on Windows
python fetch_onchain_transactions.py
```

With a key set, this loops over the illicit deposits in
`../../data/entrance_events.csv`, rebuilds their features from the chain, and
writes `recollected_upstream_features.csv` next to the script — diff that against
the shipped `../../data/upstream_features.csv` to check the extraction matches.

Two things to expect. It's slow: it rate-limits itself and backs off, because a
real re-collection walks thousands of addresses — which is also why the driver does
the 30 illicit deposits and not the 139 backgrounds by default (their (address,
time) is in the same file; re-fetching them just multiplies the runtime by five).
For the newly gated crews this reproduces the features fine; for four of them the
funder's outflow records just weren't fetchable (Blockscout kept rate-limiting us),
which is why those show up missing rather than zero — see `outflow_recovery.csv`.

To collect a completely fresh incident instead, call
`features_for_case(pool_address, start_block, end_block)`.

## Parameters

All fixed parameters live at the top of `build_upstream_features.py` and were set
once, before any result was seen: 7-day window, 10 ETH threshold, 2 hops, 25-node
cap, largest-funder ≥ 10% of the deposit, stop below 30% coverage. Figure 2 imports
these same constants from the code.
