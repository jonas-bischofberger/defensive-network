"""
Per-ZONE network topology metrics.

Mirrors the full-match topology engine in 2026-05-28_team_level_analysis.py
(`process()`), but computes every graph metric *separately within each pitch
zone*, using the zone-filtered edge files produced by
2026-06-18_zone_network_edges.py.

For each edge-weight method we read
    scripts/2026-06-18_zone_network_edge(<method>).csv
and, for every (match_team_id, zone, weight-metric), build the co-defending
graph and compute:

    <metric>                          total network strength (Σ edge weights)
    <metric>_density                  realized / possible squad pairs (count ≥ thr)
    <metric>_gini                     inequality of node strength
    <metric>_cc_unweighted            average clustering (topology only)
    <metric>_cc_weighted              weighted average clustering
    <metric>_centralization           Freeman degree centralization (topology only)
    <metric>_centralization_weighted  strength-based centralization
    <metric>_assortativity            degree assortativity
    <metric>_kcore_max                max k-core
    <metric>_lcc_ratio                largest connected component / n nodes

Output: scripts/2026-06-19_zone_topology(<method>).csv — one row per
(match_team_id, zone).

Notes
-----
* Density denominator = the team's *full-match* squad size (from the node-level
  metrics CSV), so density is directly comparable across zones ("what fraction
  of all possible squad pairs co-defend in this zone").
* Edge-count threshold is fixed at 1 (every co-defending pair counts); at thr=1
  the node-strength Gini and total strength are identical whether computed from
  the graph or from the full edge list, so we read them straight off the graph.
* Zones follow the *thirds* scheme only (own / mid / high_press, ±17.5) — the
  same scheme the zone edge files were generated with.
"""
import os

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
METHODS = ["average", "min", "product", "sum"]
THR = 1

WEIGHT_COLS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]
ZONE_ORDER = ["own", "mid", "high_press"]

# ── Full-match squad size per match-team (density denominator) ──────────────────
_gs = os.path.join(HERE, "2026-06-07_node_level_metrics_with_gs.csv")
_nodes_path = _gs if os.path.exists(_gs) else os.path.join(
    HERE, "2026-05-06_node_level_metrics_with_mins.csv")
nodes = pd.read_csv(_nodes_path)
nodes["match_team_id"] = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
squad_size = nodes.groupby("match_team_id")["defender_id"].count()
print(f"Loaded squad sizes for {len(squad_size)} match-teams from {os.path.basename(_nodes_path)}")


def gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def zone_topology(edge_df):
    """One row per (match_team_id, zone); topology of each weight metric's graph."""
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = (
        edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str))

    rows = []
    for (mtid, zone), grp in edge_df.groupby(["match_team_id", "zone"]):
        rec = {
            "match_team_id": mtid,
            "match_id": int(grp["match_id"].iloc[0]),
            "defending_team": int(grp["defending_team"].iloc[0]),
            "zone": zone,
        }
        ssize = squad_size.get(mtid, np.nan)
        mp = ssize * (ssize - 1) / 2 if pd.notna(ssize) else np.nan
        rec["squad_size"] = int(ssize) if pd.notna(ssize) else np.nan

        for c in WEIGHT_COLS:
            if c not in grp.columns:
                continue
            ec = c + "_edge_count"
            # total network strength: Σ edge weights over this metric's edges
            rec[c] = float(grp[c].fillna(0).sum())

            e = (grp[grp[ec].fillna(0) >= THR][["player_1", "player_2", c]]
                 if ec in grp.columns else grp[["player_1", "player_2", c]])
            rec[c + "_density"] = (len(e) / mp) if (pd.notna(mp) and mp > 0) else np.nan
            if len(e) < 2:
                continue

            G = nx.Graph()
            for _, r in e.iterrows():
                G.add_edge(r["player_1"], r["player_2"], weight=r[c])
            n = G.number_of_nodes()

            strengths = np.array([s for _, s in G.degree(weight="weight")], dtype=float)
            rec[c + "_gini"]          = gini(strengths)
            rec[c + "_cc_unweighted"] = nx.average_clustering(G)
            rec[c + "_cc_weighted"]   = nx.average_clustering(G, weight="weight")
            if n > 2:
                dc = nx.degree_centrality(G)
                max_dc = max(dc.values())
                rec[c + "_centralization"] = sum(max_dc - v for v in dc.values()) / (n - 2)
                s_max = strengths.max()
                if s_max > 0:
                    s_norm = strengths / s_max
                    rec[c + "_centralization_weighted"] = (1 - s_norm).sum() / (n - 2)
            if G.number_of_edges() >= 2:
                try:
                    rec[c + "_assortativity"] = nx.degree_assortativity_coefficient(G)
                except Exception:
                    pass
            rec[c + "_kcore_max"] = max(nx.core_number(G).values())
            lcc = max(len(comp) for comp in nx.connected_components(G))
            rec[c + "_lcc_ratio"] = lcc / n

        rows.append(rec)

    return pd.DataFrame(rows)


for method in METHODS:
    in_path = os.path.join(HERE, f"2026-06-18_zone_network_edge({method}).csv")
    edge_df = pd.read_csv(in_path)
    out = (zone_topology(edge_df)
           .sort_values(["match_id", "defending_team", "zone"])
           .reset_index(drop=True))
    out_path = os.path.join(HERE, f"2026-06-19_zone_topology({method}).csv")
    out.to_csv(out_path, index=False)
    print(f"[{method}] {len(out):,} rows  ->  {os.path.basename(out_path)}")

print("\nDone.")
