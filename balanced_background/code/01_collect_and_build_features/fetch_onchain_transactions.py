"""
Fetch the on-chain transactions that fund a deposit, and assemble the backward
"cone" that build_upstream_features.py turns into features.

This is the data-collection reference. Unlike the evaluation scripts, it talks to
a public Ethereum block explorer (Etherscan's v2 API), so it needs an API key and
internet access:

    set ETHERSCAN_KEY=your_key_here      # Windows
    export ETHERSCAN_KEY=your_key_here   # macOS / Linux
    python fetch_onchain_transactions.py

It is intentionally slow and polite (it backs off and rate-limits) because a full
re-collection walks thousands of addresses. Every record it reads is public, so
anyone can reproduce the feature table from scratch given the deposit list in
../../data/entrance_events.csv. The frozen feature table is shipped with the paper
so that reproduction does not require re-crawling the chain.
"""

import csv
import datetime
import json
import os
import time
import urllib.request

from build_upstream_features import (
    WINDOW_SECONDS, SWAP_ROUTERS, MAX_HOPS_BACK, MIN_TRANSFER_ETH, compute_upstream_features,
)

API_KEY = os.environ.get("ETHERSCAN_KEY", "")
# The event signature Tornado Cash emits on every deposit -- how we find deposits.
DEPOSIT_EVENT_TOPIC = "0xa945e51eec50ab98c161376f0db4cf2aeba3ec92755fe2fcd388bdbbb80ff196"
CONE_NODE_LIMIT = 25                    # never expand a deposit's cone past 25 addresses
CAMPAIGN_GAP_SECONDS = 48 * 3600        # deposits < 48h apart by the same address = one campaign


def call_api(query, max_tries=5):
    """One call to the explorer, with a few polite retries on failure."""
    url = f"https://api.etherscan.io/v2/api?chainid=1&{query}&apikey={API_KEY}"
    for attempt in range(max_tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception:
            time.sleep(0.7)             # back off and try again
    return {}


_contract_cache = {}


def is_contract_at(address, block):
    """Was this address a contract at the deposit's block? (state as of t0 only)."""
    key = (address, block)
    if key not in _contract_cache:
        code = call_api(f"module=proxy&action=eth_getCode&address={address}&tag=0x{block:x}").get("result", "0x")
        _contract_cache[key] = bool(code) and code != "0x" and len(code) > 3
    return _contract_cache[key]


def window_transfers(address, start_block, end_block, deposit_time):
    """The address's inflows and outflows inside the 7-day pre-deposit window.

    We pull both normal and internal transfers, keep only those of at least
    10 ETH that fall strictly inside [t0 - 7 days, t0), and split them into
    money coming in and money going out.
    """
    inflows, outflows = [], []
    for kind in ("txlist", "txlistinternal"):
        result = call_api(f"module=account&action={kind}&address={address}"
                          f"&startblock={start_block}&endblock={end_block}&sort=asc").get("result") or []
        for tx in result:
            if not isinstance(tx, dict):
                continue
            value_eth = int(tx.get("value", "0")) / 1e18
            timestamp = int(tx["timeStamp"])
            if value_eth < MIN_TRANSFER_ETH:
                continue
            if not (deposit_time - WINDOW_SECONDS <= timestamp < deposit_time):
                continue
            if (tx.get("to") or "").lower() == address:
                inflows.append(((tx.get("from") or "").lower(), value_eth, timestamp))
            if (tx.get("from") or "").lower() == address:
                outflows.append(((tx.get("to") or "").lower(), value_eth, timestamp))
    return inflows, outflows


def build_backward_cone(depositor, deposit_time, deposit_block):
    """Grow the backward cone from the depositor, one funding hop at a time.

    We do a breadth-first walk: start at the depositor, pull its window transfers,
    and queue up the addresses that funded it -- but stop expanding at contracts
    and swap routers (boundaries), and never let the cone grow past 25 nodes.
    """
    lookback_start = max(0, deposit_block - 50400)   # ~7 days of blocks
    cone = {}
    queue = [(depositor, 0)]
    while queue and len(cone) < CONE_NODE_LIMIT:
        address, depth = queue.pop(0)
        if address in cone:
            continue
        # boundaries: contracts and swap routers are recorded but not expanded
        if depth > 0 and (address in SWAP_ROUTERS or is_contract_at(address, deposit_block)):
            cone[address] = {"boundary": True, "in": [], "out": []}
            continue
        inflows, outflows = window_transfers(address, lookback_start, deposit_block, deposit_time)
        cone[address] = {"boundary": False, "in": inflows, "out": outflows}
        if depth < MAX_HOPS_BACK:
            for funder, _amount, _ts in inflows:
                if funder and funder not in cone:
                    queue.append((funder, depth + 1))
    return cone


def group_deposits_into_campaigns(deposits):
    """Merge an address's deposits made within 48 hours into one campaign (t0 = first)."""
    by_address = {}
    for d in deposits:
        by_address.setdefault(d["from"], []).append(d)
    campaigns = []
    for _address, deposit_list in by_address.items():
        deposit_list.sort(key=lambda d: d["ts"])
        current = [deposit_list[0]]
        for d in deposit_list[1:]:
            if d["ts"] - current[-1]["ts"] < CAMPAIGN_GAP_SECONDS:
                current.append(d)
            else:
                campaigns.append(current)
                current = [d]
        campaigns.append(current)
    return [{"address": c[0]["from"], "t0": c[0]["ts"], "block": c[0]["block"], "n_deposits": len(c)}
            for c in campaigns]


def find_pool_deposits(pool_address, start_block, end_block):
    """List every deposit into a Tornado pool in a block range (address + time)."""
    logs = call_api(f"module=logs&action=getLogs&address={pool_address}"
                    f"&topic0={DEPOSIT_EVENT_TOPIC}&fromBlock={start_block}&toBlock={end_block}").get("result") or []
    deposits, seen = [], set()
    for log in logs:
        tx_hash = log["transactionHash"]
        if tx_hash in seen:
            continue
        seen.add(tx_hash)
        tx = call_api(f"module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}").get("result")
        time.sleep(0.12)                # stay under the rate limit
        if tx and tx.get("from"):
            deposits.append({"from": tx["from"].lower(),
                             "ts": int(log["timeStamp"], 16),
                             "block": int(tx["blockNumber"], 16)})
    return deposits


def features_for_case(pool_address, start_block, end_block, deposit_value_eth=100.0):
    """End-to-end for one incident: find deposits, build cones, extract features."""
    deposits = find_pool_deposits(pool_address, start_block, end_block)
    results = []
    for campaign in group_deposits_into_campaigns(deposits):
        cone = build_backward_cone(campaign["address"], campaign["t0"], campaign["block"])
        try:
            features = compute_upstream_features(campaign["address"], campaign["t0"], deposit_value_eth, cone)
            results.append({"address": campaign["address"], "features": features})
        except Exception as error:
            results.append({"address": campaign["address"], "features": None, "error": str(error)})
    return results


def block_at_time(unix_time):
    """The last block mined at or before a timestamp (needed to bound the cone)."""
    result = call_api(f"module=block&action=getblocknobytime&timestamp={unix_time}&closest=before").get("result")
    return int(result) if result and str(result).isdigit() else None


def features_for_deposit(address, deposit_time_utc, deposit_value_eth=100.0):
    """Rebuild the 15 features for one deposit, given its address and time.

    Looks up the deposit's block, grows the backward cone, and extracts the
    features -- the same path used for the newly gated crews in the paper.
    """
    # TODO: deposit_value defaults to 100 ETH. That's fine for the ranking (coverage
    # is a ratio), but pull the actual deposit amount if you need it exact.
    deposit_time = int(datetime.datetime.strptime(deposit_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                       .replace(tzinfo=datetime.timezone.utc).timestamp())
    deposit_block = block_at_time(deposit_time)
    if deposit_block is None:
        return None
    cone = build_backward_cone(address.lower(), deposit_time, deposit_block)
    return compute_upstream_features(address.lower(), deposit_time, deposit_value_eth, cone)


def rebuild_features_from_event_list():
    """Re-collect the features for the illicit deposits, straight off the chain.

    Reads ../../data/entrance_events.csv and writes a fresh feature file next to
    it. Every row there has an (address, time); this driver re-fetches the 30
    illicit ones. The 139 background rows work the same way, they just make the
    run five times longer -- widen the filter below if you want them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    events_path = os.path.join(here, "..", "..", "data", "entrance_events.csv")
    out_path = os.path.join(here, "recollected_upstream_features.csv")

    with open(events_path, encoding="utf-8-sig") as fh:
        events = list(csv.DictReader(fh))
    refetchable = [e for e in events if e["is_illicit"] == "1" and e["deposit_time_utc"]]
    print(f"re-fetching {len(refetchable)} illicit deposits "
          f"({len(events) - len(refetchable)} background rows skipped by default).")

    feature_names = None
    rows = []
    for event in refetchable:
        features = features_for_deposit(event["deposit_address"], event["deposit_time_utc"])
        if features is None:
            print(f"  {event['event_id']} ({event['crew']}): could not resolve block, skipped")
            continue
        if feature_names is None:
            feature_names = list(features.keys())
        row = {"event_id": event["event_id"], "crew": event["crew"]}
        for name in feature_names:
            row[name] = features[name]["value"] if features[name]["observed"] else ""
        rows.append(row)
        print(f"  {event['event_id']} ({event['crew']}): {len([1 for n in feature_names if features[n]['observed']])}/15 features observed")

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["event_id", "crew"] + feature_names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")
    print("Compare it against ../../data/upstream_features.csv to confirm the extraction matches.")


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set ETHERSCAN_KEY to re-collect from the chain. "
                         "The frozen feature table in ../../data/ reproduces the results without a key.")
    rebuild_features_from_event_list()
