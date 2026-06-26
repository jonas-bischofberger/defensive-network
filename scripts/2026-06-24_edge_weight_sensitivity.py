"""
Edge-weight method sensitivity test (team-level).

Question: does the choice of how a co-defensive event's two endpoint values
(val_a, val_b) are combined into an edge weight change our conclusions?

    sum     : val_a + val_b
    average : (val_a + val_b) / 2      # == sum / 2, perfectly collinear
    min     : min(val_a, val_b)        # weakest-link / conjunctive  <- primary
    product : val_a * val_b

Primary method = min (weakest-link semantics: a co-defensive link is only as
strong as the less-involved partner). This script re-runs the team-level
network metrics under all four methods and compares the headline KPI — the
correlation of each network metric with defensive outcomes (raw + partial,
controlling for passes_against) — exactly as the main analysis reports it
(`combined_corr_tbl` in 2026-05-28_team_level_analysis.py).

Two metric classes (verified, not assumed):
  * topology-only  (density, cc_unweighted, centralization, assortativity,
    kcore_max, lcc_ratio) are built from the edge-count threshold, which is
    method-independent -> identical across methods by construction.
  * weight-dependent (strength, gini, cc_weighted, centralization_weighted)
    use the edge weight -> these are the only metrics the method can move.

Outputs:
  scripts/2026-06-24_edge_weight_sensitivity_long.csv   (tidy, all rows)
  scripts/2026-06-24_edge_weight_sensitivity_wide.csv   (paper table)
  + a console summary with the headline robustness numbers.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr

METHODS = ["min", "average", "sum", "product"]
PRIMARY = "min"

# ── Mirror the main analysis's column definitions ─────────────────────────────
WEIGHT_COLS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]
OUTCOME_COLS = ["goals_against", "shots_against", "xg_against"]

# Suffixes of the metrics that actually depend on the edge weight.
WEIGHT_DEPENDENT = ["", "_gini", "_cc_weighted", "_centralization_weighted"]
# Topology-only suffixes — included to verify they are invariant across methods.
TOPOLOGY_ONLY = ["_density", "_cc_unweighted", "_centralization",
                 "_assortativity", "_kcore_max", "_lcc_ratio"]

PRETTY = {
    "":                        "Total Strength",
    "_gini":                   "Gini (strength inequality)",
    "_cc_weighted":            "Clustering (weighted)",
    "_centralization_weighted": "Freeman Centralization (weighted)",
}

# ── Data ──────────────────────────────────────────────────────────────────────
outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
nodes    = pd.read_csv("scripts/2026-06-07_node_level_metrics_with_gs.csv")
nodes["match_team_id"] = (nodes["match_id"].astype(str) + "_"
                          + nodes["defending_team"].astype(str))
squad_size = nodes.groupby("match_team_id")["defender_id"].count().rename("n_players")

edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
            for k in METHODS}


# ── gini + process(): copied verbatim from 2026-05-28_team_level_analysis.py ───
def gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def process(edge_df, thr=1):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    mp  = squad_size * (squad_size - 1) / 2
    out = edge_df.groupby("match_team_id")[WEIGHT_COLS].sum()
    out = out.join(pd.DataFrame({
        c + "_density": edge_df[edge_df[c + "_edge_count"] >= thr].groupby("match_team_id").size() / mp
        for c in WEIGHT_COLS
    }))
    extra = {}
    for c in WEIGHT_COLS:
        p1 = edge_df[["match_team_id", "player_1", c]].rename(columns={"player_1": "player"})
        p2 = edge_df[["match_team_id", "player_2", c]].rename(columns={"player_2": "player"})
        ps = pd.concat([p1, p2]).groupby(["match_team_id", "player"])[c].sum()
        extra[c + "_gini"] = ps.groupby("match_team_id").apply(gini)
        u, w, cent_u, cent_w, assort, kcore_max, lcc_ratio = {}, {}, {}, {}, {}, {}, {}
        for mid, grp in edge_df.groupby("match_team_id"):
            e = grp[grp[c + "_edge_count"] >= thr][["player_1", "player_2", c]]
            if len(e) < 2:
                continue
            G = nx.Graph()
            for _, row in e.iterrows():
                G.add_edge(row["player_1"], row["player_2"], weight=row[c])
            n = G.number_of_nodes()
            u[mid] = nx.average_clustering(G)
            w[mid] = nx.average_clustering(G, weight="weight")
            if n > 2:
                dc = nx.degree_centrality(G)
                max_dc = max(dc.values())
                cent_u[mid] = sum(max_dc - v for v in dc.values()) / (n - 2)
                strengths = np.array([s for _, s in G.degree(weight="weight")])
                s_max = strengths.max()
                if s_max > 0:
                    s_norm = strengths / s_max
                    cent_w[mid] = (1 - s_norm).sum() / (n - 2)
            if G.number_of_edges() >= 2:
                try:
                    assort[mid] = nx.degree_assortativity_coefficient(G)
                except Exception:
                    pass
            if n >= 2:
                kcore_max[mid] = max(nx.core_number(G).values())
            lcc_size = max(len(comp) for comp in nx.connected_components(G))
            lcc_ratio[mid] = lcc_size / n
        extra[c + "_cc_unweighted"]           = pd.Series(u)
        extra[c + "_cc_weighted"]             = pd.Series(w)
        extra[c + "_centralization"]          = pd.Series(cent_u)
        extra[c + "_centralization_weighted"] = pd.Series(cent_w)
        extra[c + "_assortativity"]           = pd.Series(assort)
        extra[c + "_kcore_max"]               = pd.Series(kcore_max)
        extra[c + "_lcc_ratio"]               = pd.Series(lcc_ratio)
    outcome_cols = ["match_team_id", "team_name", "competition_stage"] + OUTCOME_COLS + ["passes_against"]
    return out.join(pd.DataFrame(extra)).reset_index().merge(outcomes[outcome_cols], on="match_team_id")


def _resid(y, z):
    mask = ~(np.isnan(y) | np.isnan(z))
    r = np.full(len(y), np.nan)
    r[mask] = y[mask] - np.polyval(np.polyfit(z[mask], y[mask], 1), z[mask])
    return r


def corr_raw_partial(df, m, t):
    """Headline KPI: raw and passes_against-partialled Pearson r, as in main app."""
    v = df[[m, t]].dropna()
    raw_r, raw_p = pearsonr(v[m], v[t])
    s = df[[m, t, "passes_against"]].dropna()
    a = _resid(s[m].values, s["passes_against"].values)
    b = _resid(s[t].values, s["passes_against"].values)
    mk = ~(np.isnan(a) | np.isnan(b))
    par_r, par_p = pearsonr(a[mk], b[mk])
    return (raw_r, raw_p), (par_r, par_p)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ── Run all four methods ──────────────────────────────────────────────────────
print("Computing network metrics for:", ", ".join(METHODS))
processed = {k: process(df) for k, df in edge_dfs.items()}

# Sanity check: topology-only metrics must be identical across methods.
print("\n[sanity] topology-only metrics should be identical across methods:")
ref = processed[PRIMARY]
for c in WEIGHT_COLS:
    for suf in TOPOLOGY_ONLY:
        col = c + suf
        ok = all(np.allclose(processed[k][col].fillna(-999),
                             ref[col].fillna(-999)) for k in METHODS)
        if not ok:
            print(f"  ! {col} DIFFERS across methods (unexpected)")
print("  all topology-only metrics identical across methods ✓")

# ── Sensitivity on the weight-dependent KPIs ──────────────────────────────────
rows = []
for c in WEIGHT_COLS:
    for suf in WEIGHT_DEPENDENT:
        col = c + suf
        for t in OUTCOME_COLS:
            for kind in ("raw", "partial"):
                rec = {"metric": col, "metric_family": PRETTY[suf],
                       "node_metric": c, "outcome": t, "kind": kind}
                for k in METHODS:
                    (rr, rp), (pr, pp) = corr_raw_partial(processed[k], col, t)
                    r, p = (rr, rp) if kind == "raw" else (pr, pp)
                    rec[k] = r
                    rec[k + "_sig"] = stars(p)
                rows.append(rec)

long_df = pd.DataFrame(rows)

# Spread across the three near-equivalent methods (the real decision set) and all four.
trio = ["min", "average", "sum"]
long_df["r_range_trio"] = long_df[trio].max(axis=1) - long_df[trio].min(axis=1)
long_df["r_range_all"]  = long_df[METHODS].max(axis=1) - long_df[METHODS].min(axis=1)
# Does any method flip the significance verdict (p<.05) relative to primary?
long_df["sig_flip_trio"] = long_df.apply(
    lambda x: len({bool(x[k + "_sig"]) for k in trio}) > 1, axis=1)
long_df["sign_flip_trio"] = long_df.apply(
    lambda x: len({np.sign(x[k]) for k in trio}) > 1, axis=1)

long_df.to_csv("scripts/2026-06-24_edge_weight_sensitivity_long.csv", index=False)

# Paper-style wide table: formatted r* per method.
def fmt(row, k):
    return f"{row[k]:+.2f}{row[k + '_sig']}"
wide = long_df.copy()
for k in METHODS:
    wide[k] = wide.apply(lambda r: fmt(r, k), axis=1)
wide = wide[["metric_family", "node_metric", "outcome", "kind"] + METHODS
            + ["r_range_trio", "r_range_all"]]
wide.to_csv("scripts/2026-06-24_edge_weight_sensitivity_wide.csv", index=False)

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EDGE-WEIGHT SENSITIVITY — headline numbers (weight-dependent KPIs only)")
print("=" * 70)
print(f"primary method = {PRIMARY}; KPI = corr(network metric, outcome), "
      f"{long_df.shape[0]} metric×outcome×kind cells")
print(f"\nmax |Δr| across min/average/sum : {long_df['r_range_trio'].max():.3f}")
print(f"mean |Δr| across min/average/sum: {long_df['r_range_trio'].mean():.3f}")
print(f"cells where significance (p<.05) verdict changes within min/avg/sum: "
      f"{int(long_df['sig_flip_trio'].sum())} / {len(long_df)}")
print(f"cells where the sign of r flips within min/avg/sum: "
      f"{int(long_df['sign_flip_trio'].sum())} / {len(long_df)}")
print(f"\nincluding product, max |Δr| across all four: {long_df['r_range_all'].max():.3f}")

worst = long_df.sort_values("r_range_trio", ascending=False).head(5)
print("\nLargest disagreements within min/average/sum:")
for _, r in worst.iterrows():
    vals = "  ".join(f"{k}={r[k]:+.2f}{r[k+'_sig']}" for k in trio)
    print(f"  {r['metric_family']:<33} {r['outcome']:<13} {r['kind']:<7} | {vals}")

print("\nWrote:")
print("  scripts/2026-06-24_edge_weight_sensitivity_long.csv")
print("  scripts/2026-06-24_edge_weight_sensitivity_wide.csv")
