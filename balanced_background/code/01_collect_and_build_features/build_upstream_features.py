"""
Build the upstream-provenance features (the paper's "Phi") for one deposit.

Given a mixer deposit -- an address a0 that put amount V into a Tornado Cash pool
at time t0 -- this walks *backwards* through the money that funded it and turns
the shape of that money trail into 15 plain numbers: how many hops, how long it
took, how long it rested, how much of the deposit we could trace, and so on.

The walk is deliberately narrow and label-free (it never looks at who anyone is):

  * only transfers in the 7 days before the deposit count,
  * only transfers of at least 10 ETH enter the trail at all,
  * we follow at most 2 hops back,
  * at each hop we follow the single largest funder (>= 10% of the deposit),
  * we stop at a contract, an exchange/swap router, or when the trail runs dry.

Two kinds of "we can't see it" are kept apart, which turns out to drive the paper's
main cautionary result:
  * censored  -- the trail hit a contract or the depth limit; the real source
                 exists but is cut off. The feature is flagged, with a reason.
  * missing   -- a record could not be fetched at all. The feature is marked
                 missing and is not filled with a zero: not seeing a transfer is
                 different from there being none.

This module is the reference implementation used to produce the frozen features
in ../../data/upstream_features.csv. It expects the backward "cone" of transfers
to be assembled first by fetch_onchain_transactions.py.
"""

# ---------------------------------------------------------------------------
# Fixed parameters. These were chosen once, before any result was looked at,
# and never changed -- so every feature is reproducible from public records.
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 7 * 24 * 3600     # only look 7 days before the deposit
MAX_HOPS_BACK = 2                  # follow the money at most two hops back
MIN_TRANSFER_ETH = 10              # a transfer must be at least this large to enter the cone
MIN_TRACED_SHARE = 0.30            # stop once we can trace < 30% of the deposit
MIN_FUNDER_SHARE = 0.10            # a funder must supply >= 10% of the deposit V

# Well-known DEX / swap-router contracts. Hitting one means the asset was
# converted, so the money trail changes form and we stop following it.
SWAP_ROUTERS = {
    "0x1111111254eeb25477b68fb85ed929f73a960582", "0x1111111254fb6c44bac0bed2854e76f90643097d",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "0xe592427a0aece92de3edee1f18e0157c05861564", "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45", "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",
}

# The Tornado Cash pools themselves (used only to ignore deposits *into* the mixer
# when measuring a node's local activity, so the mixer does not count as an outflow).
TORNADO_POOLS = {
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b", "0x722122df12d4e14e13ac3b6895a86e84145b6967",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291", "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936", "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc",
}

# The 15 features, in the order the paper lists them (Appendix D).
FEATURE_NAMES = [
    "observed_hops", "path_elapsed_time", "dwell_mean_h", "dwell_max_h",
    "coverage", "attribution_ambiguous", "node_context_out_degree",
    "node_context_in_degree", "wallet_age_min_h", "node_context_resid_Rrange_max",
    "node_context_resid_nin_mean", "node_context_resid_nout_mean",
    "boundary_type", "censoring_reason", "observation_confidence",
]


def feature(value, observed=True, censored=False, reason=None):
    """One feature value, tagged with whether we actually saw it."""
    return {"value": value, "observed": observed, "censored": censored, "reason": reason}


def within_window(timestamp, deposit_time):
    """True if a transfer happened inside the 7-day window before the deposit."""
    return (deposit_time - WINDOW_SECONDS) <= timestamp < deposit_time


def trace_main_funding_path(depositor, cone, deposit_time, deposit_value):
    """Walk backwards along the single largest funder at each hop.

    `cone` maps each address to its in/out transfers, already restricted to the
    window and threshold by the fetch step. We start at the depositor and, hop by
    hop, follow the largest qualifying inflow until we reach an origin, a contract
    boundary, a swap router, run low on traceable value, or hit the depth limit.

    Returns the path of addresses, the boundary node and how the walk ended, the
    still-traceable value, the edges we followed, and whether any hop was
    ambiguous (a smaller funder could have been the "real" one).
    """
    path = [depositor]
    edges = []
    traceable_value = float(deposit_value)
    current = depositor
    ambiguous = False

    for _ in range(MAX_HOPS_BACK):
        node = cone.get(current)
        if node is None:
            return path, current, "unresolved", "unresolved", traceable_value, edges, ambiguous

        # candidate funders: inflows within the window that are large enough
        funders = [(sender, amount, ts) for sender, amount, ts in node.get("in", [])
                   if within_window(ts, deposit_time) and amount >= MIN_FUNDER_SHARE * deposit_value]
        if not funders:
            # nothing more to follow: either a plain wallet (origin) or a contract
            ending = "contract" if node.get("boundary") else "eoa-source"
            reason = "contract-opacity" if ending == "contract" else "origin-reached"
            return path, current, ending, reason, traceable_value, edges, ambiguous

        # follow the largest funder; break ties by earliest, then address order
        sender, amount, ts = max(funders, key=lambda f: (f[1], -f[2], f[0]))
        if amount < traceable_value:
            ambiguous = True                     # a competing funder existed
        traceable_value = min(traceable_value, amount)
        edges.append((current, sender, amount, ts))

        if sender in SWAP_ROUTERS:
            return path + [sender], sender, "asset-conversion", "cross-asset", min(traceable_value, deposit_value), edges, ambiguous
        sender_node = cone.get(sender)
        if sender_node is None:
            return path + [sender], sender, "unresolved", "unresolved", min(traceable_value, deposit_value), edges, ambiguous
        if sender_node.get("boundary"):
            return path + [sender], sender, "contract", "contract-opacity", min(traceable_value, deposit_value), edges, ambiguous
        if min(amount, deposit_value) / deposit_value < MIN_TRACED_SHARE:
            return path + [sender], sender, "coverage-stop", "coverage-stop", min(traceable_value, deposit_value), edges, ambiguous

        path.append(sender)
        current = sender

    return path, current, "depth-limit", "depth-limit", min(traceable_value, deposit_value), edges, ambiguous


def local_balance_dynamics(node, deposit_time):
    """Small summary of a node's own activity inside the window.

    We look at how the node's balance moved (its inflows minus outflows, ignoring
    deposits straight into a Tornado pool) and how many distinct counterparties it
    had. This is *context*, not proof the money passed through here, so the paper
    files these under "node context" and treats their attribution as uncertain.
    """
    movements = (
        [(ts, amount) for sender, amount, ts in node.get("in", []) if within_window(ts, deposit_time)]
        + [(ts, -amount) for receiver, amount, ts in node.get("out", [])
           if within_window(ts, deposit_time) and receiver not in TORNADO_POOLS]
    )
    movements.sort()
    if not movements:
        return None

    running = 0
    balances = []
    for ts, delta in movements:
        running += delta
        balances.append(running)
    inflow_count = sum(1 for _, _, ts in node.get("in", []) if within_window(ts, deposit_time))
    outflow_count = sum(1 for receiver, _, ts in node.get("out", [])
                        if within_window(ts, deposit_time) and receiver not in TORNADO_POOLS)
    return {"balance_range": max(balances) - min(balances),
            "inflow_count": inflow_count, "outflow_count": outflow_count}


def compute_upstream_features(depositor, deposit_time, deposit_value, cone):
    """Return the 15 upstream-provenance features for one deposit."""
    depositor = depositor.lower()
    path, boundary_node, boundary_type, censoring_reason, traceable_value, edges, ambiguous = \
        trace_main_funding_path(depositor, cone, deposit_time, deposit_value)
    intermediaries = path[1:-1]     # nodes strictly between depositor and origin

    features = {}
    features["observed_hops"] = feature(len(path) - 1)

    # how long the traced money was in motion, from the earliest edge to deposit
    if edges:
        first_edge_time = min(e[3] for e in edges)
        was_cut_off = boundary_type in ("contract", "depth-limit", "coverage-stop", "unresolved")
        features["path_elapsed_time"] = feature((deposit_time - first_edge_time) / 3600.0,
                                                censored=was_cut_off,
                                                reason=(censoring_reason if was_cut_off else None))
    else:
        features["path_elapsed_time"] = feature(None, observed=False, reason="no_edge")

    # dwell = how long each intermediary held the money before passing it on
    dwell_hours = []
    negative_dwell = False
    for i in range(1, len(path) - 1):
        arrived = edges[i][3]        # money arrived at path[i]
        departed = edges[i - 1][3]   # money left path[i]
        gap = (departed - arrived) / 3600.0
        if gap < 0:
            negative_dwell = True    # out-of-order timestamps -> flag as suspect
        dwell_hours.append(gap)
    if dwell_hours:
        features["dwell_mean_h"] = feature(sum(dwell_hours) / len(dwell_hours),
                                           censored=negative_dwell,
                                           reason=("negative_dwell_anomaly" if negative_dwell else None))
        features["dwell_max_h"] = feature(max(dwell_hours),
                                          censored=negative_dwell,
                                          reason=("negative_dwell_anomaly" if negative_dwell else None))
    else:
        features["dwell_mean_h"] = feature(None, observed=False, reason="no_intermediary")
        features["dwell_max_h"] = feature(None, observed=False, reason="no_intermediary")

    # coverage = share of the deposit we could actually trace back to a source
    coverage = min(max(traceable_value, 0.0), deposit_value) / deposit_value
    features["coverage"] = feature(coverage, censored=ambiguous,
                                   reason=("attribution_ambiguous" if ambiguous else None))
    features["attribution_ambiguous"] = feature(bool(ambiguous))

    # local graph context of the intermediaries (uncertain attribution -> context)
    if intermediaries:
        out_degrees = [len({r for r, _, ts in cone.get(m, {}).get("out", [])
                            if within_window(ts, deposit_time) and r not in TORNADO_POOLS})
                       for m in intermediaries]
        in_degrees = [len({s for s, _, ts in cone.get(m, {}).get("in", [])
                           if within_window(ts, deposit_time)})
                      for m in intermediaries]
        features["node_context_out_degree"] = feature(sum(out_degrees) / len(out_degrees),
                                                       reason="node_context_not_path_isolated")
        features["node_context_in_degree"] = feature(sum(in_degrees) / len(in_degrees),
                                                      reason="node_context_not_path_isolated")
    else:
        features["node_context_out_degree"] = feature(None, observed=False, reason="no_intermediary")
        features["node_context_in_degree"] = feature(None, observed=False, reason="no_intermediary")

    # wallet age: we only see window activity, so this is a lower bound (left-censored)
    ages = []
    for m in intermediaries:
        stamps = ([ts for _, _, ts in cone.get(m, {}).get("in", []) if within_window(ts, deposit_time)]
                  + [ts for _, _, ts in cone.get(m, {}).get("out", []) if within_window(ts, deposit_time)])
        if stamps:
            ages.append((deposit_time - min(stamps)) / 3600.0)
    features["wallet_age_min_h"] = (feature(min(ages), censored=True, reason="left_censored_window_only")
                                    if ages else feature(None, observed=False, reason="no_intermediary"))

    # residual balance dynamics of the path nodes (context, uncertain attribution)
    dynamics = [local_balance_dynamics(cone.get(m, {}), deposit_time)
                for m in (intermediaries if intermediaries else path[1:])]
    dynamics = [d for d in dynamics if d]
    if dynamics:
        features["node_context_resid_Rrange_max"] = feature(max(d["balance_range"] for d in dynamics),
                                                            reason="node_context_not_path_isolated")
        features["node_context_resid_nin_mean"] = feature(sum(d["inflow_count"] for d in dynamics) / len(dynamics))
        features["node_context_resid_nout_mean"] = feature(sum(d["outflow_count"] for d in dynamics) / len(dynamics))
    else:
        for name in ("node_context_resid_Rrange_max", "node_context_resid_nin_mean", "node_context_resid_nout_mean"):
            features[name] = feature(None, observed=False, reason="no_path_node")

    # how the walk ended, and how confident we are in what we traced
    features["boundary_type"] = feature(boundary_type)
    features["censoring_reason"] = feature(censoring_reason)
    features["observation_confidence"] = feature(coverage)
    return features
