"""
Per-zone *spatial / spatially-embedded-network* predictors of defensive success.

For each (match_team_id, zone, thirds scheme) we combine the zone-specific average
player positions (2026-06-18_zone_network_positions.csv) with the zone co-defending
edges (2026-06-18_zone_network_edge(average).csv) to build metrics that each map to a
visual feature of the co-defending network drawn on the pitch:

  POSITION-ONLY (one value per match-team-zone; method-independent):
    block_spread   size of the node cloud         sqrt(var_x+var_y)
    spread_x       vertical extent (depth)
    spread_y       horizontal extent (width)
    x_range        deepest-to-highest span        <- strongest predictor (depth)
    hull_area      area of enclosing polygon
    nn_dist        mean gap to closest team-mate
    aspect_xy      tall vs wide (spread_x/spread_y)
    lateral_off    cloud shifted to a flank (|mean y|)

  EDGE UNEVENNESS, per weight metric (CV of edge lengths over THAT metric's edge set,
  i.e. pairs with <metric>_edge_count >= 1 — a different sub-network per metric):
    <metric>_edge_cv      uneven vs even web of links
    <metric>_edge_mean    average line length (for reference / redundant w/ spread)

Edge length is reflection-invariant, so the extra-time mirroring caveat does not bite.
Output: scripts/2026-06-24_zone_spatial_metrics.csv (one row per match_team_id × zone).
Join to 2026-06-08_team_zone_metrics.csv (scheme==thirds) for the success-rate target.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

HERE = os.path.dirname(__file__)
WEIGHT_COLS = ["raw_involvement", "raw_fault", "raw_contribution",
               "valued_involvement", "valued_contribution", "valued_fault"]
ZONE_ORDER = ["own", "mid", "high_press"]

pos = pd.read_csv(os.path.join(HERE, "2026-06-18_zone_network_positions.csv")).dropna(
    subset=["overall_avg_x", "overall_avg_y"])
pos["match_team_id"] = pos["match_id"].astype(str) + "_" + pos["defending_team"].astype(str)
edges = pd.read_csv(os.path.join(HERE, "2026-06-18_zone_network_edge(average).csv"))
edges["match_team_id"] = edges["match_id"].astype(str) + "_" + edges["defending_team"].astype(str)
pk = pos.set_index(["match_id", "defending_team", "zone", "defender_name"])
PX = pk["overall_avg_x"].to_dict(); PY = pk["overall_avg_y"].to_dict()


def cv(a):
    a = np.asarray(a, float)
    return float(a.std() / a.mean()) if len(a) >= 3 and a.mean() > 0 else np.nan


def spatial_record(X, Y, pos_lookup, g):
    """Build the spatial / edge-unevenness metric dict (no id fields) from player
    positions X, Y, a name->(x, y) lookup, and that network's edge rows `g`
    (player_1/player_2 + <metric>_edge_count cols). Identical formulas to the per-zone
    block below, so the whole-network "all" row is directly comparable to the zones."""
    rec = {"block_spread": float(np.sqrt(np.var(X) + np.var(Y))),
           "spread_x": float(X.std()), "spread_y": float(Y.std()),
           "x_range": float(X.max() - X.min()), "lateral_off": abs(float(Y.mean()))}
    rec["aspect_xy"] = rec["spread_x"] / rec["spread_y"] if rec["spread_y"] > 0 else np.nan
    try:
        rec["hull_area"] = float(ConvexHull(np.c_[X, Y]).volume)
    except Exception:
        rec["hull_area"] = np.nan
    P = np.c_[X, Y]; D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    rec["nn_dist"] = float(D.min(1).mean())

    def _len(r):
        a = pos_lookup.get(r["player_1"]); b = pos_lookup.get(r["player_2"])
        return float(np.hypot(a[0] - b[0], a[1] - b[1])) if a is not None and b is not None else np.nan
    L = g.apply(_len, axis=1)
    rec["edge_cv"] = cv(L.dropna().values)
    rec["edge_mean"] = float(L.dropna().mean()) if L.notna().any() else np.nan
    for c in WEIGHT_COLS:
        ec = c + "_edge_count"
        mask = ((g[ec].fillna(0) >= 1) & L.notna()) if ec in g.columns else L.notna()
        sub = L[mask]
        rec[c + "_edge_cv"] = cv(sub.values)
        rec[c + "_edge_mean"] = float(sub.mean()) if len(sub) >= 3 else np.nan
    return rec


rows = []
for (mtid, zone), g in edges.groupby(["match_team_id", "zone"]):
    mid = g["match_id"].iloc[0]; team = g["defending_team"].iloc[0]
    bp = pos[(pos.match_team_id == mtid) & (pos.zone == zone)]
    if len(bp) < 4:
        continue
    X = bp.overall_avg_x.values; Y = bp.overall_avg_y.values
    rec = {"match_team_id": mtid, "match_id": int(mid), "defending_team": int(team), "zone": zone,
           "block_spread": float(np.sqrt(np.var(X) + np.var(Y))),
           "spread_x": float(X.std()), "spread_y": float(Y.std()),
           "x_range": float(X.max() - X.min()), "lateral_off": abs(float(Y.mean()))}
    rec["aspect_xy"] = rec["spread_x"] / rec["spread_y"] if rec["spread_y"] > 0 else np.nan
    try:
        rec["hull_area"] = float(ConvexHull(np.c_[X, Y]).volume)
    except Exception:
        rec["hull_area"] = np.nan
    P = np.c_[X, Y]; D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    rec["nn_dist"] = float(D.min(1).mean())

    # precompute each edge's length once
    g = g.copy()
    def _len(r):
        k1 = (mid, team, zone, r["player_1"]); k2 = (mid, team, zone, r["player_2"])
        if k1 in PX and k2 in PX:
            return float(np.hypot(PX[k1] - PX[k2], PY[k1] - PY[k2]))
        return np.nan
    g["_len"] = g.apply(_len, axis=1)
    rec["edge_cv"] = cv(g["_len"].dropna().values)          # any co-defending pair
    rec["edge_mean"] = float(g["_len"].dropna().mean()) if g["_len"].notna().any() else np.nan
    for c in WEIGHT_COLS:
        ec = c + "_edge_count"
        sub = g[(g[ec].fillna(0) >= 1) & g["_len"].notna()]["_len"] if ec in g.columns else g["_len"].dropna()
        rec[c + "_edge_cv"] = cv(sub.values)
        rec[c + "_edge_mean"] = float(sub.mean()) if len(sub) >= 3 else np.nan
    rows.append(rec)

# ── whole-network "all" rows: same metrics on the non-zoned (full-match) network ──
# Positions: 2026-06-07_node_level_metrics_with_gs.csv (overall_avg_x/y, all 128 teams).
# Edges:     2026-04-28_defensive_network_edge(average).csv (full-match co-defending).
nd = pd.read_csv(os.path.join(HERE, "2026-06-07_node_level_metrics_with_gs.csv")).dropna(
    subset=["overall_avg_x", "overall_avg_y"])
nd["match_team_id"] = nd["match_id"].astype(str) + "_" + nd["defending_team"].astype(str)
eavg = pd.read_csv(os.path.join(HERE, "2026-04-28_defensive_network_edge(average).csv"))
eavg["match_team_id"] = eavg["match_id"].astype(str) + "_" + eavg["defending_team"].astype(str)
for mtid, g in eavg.groupby("match_team_id"):
    bp = nd[nd.match_team_id == mtid]
    if len(bp) < 4:
        continue
    mid = int(g["match_id"].iloc[0]); team = int(g["defending_team"].iloc[0])
    look = {r.defender_name: (r.overall_avg_x, r.overall_avg_y) for r in bp.itertuples()}
    rec = {"match_team_id": mtid, "match_id": mid, "defending_team": team, "zone": "all"}
    rec.update(spatial_record(bp.overall_avg_x.values, bp.overall_avg_y.values, look, g))
    rows.append(rec)

out = pd.DataFrame(rows).sort_values(["match_id", "defending_team", "zone"]).reset_index(drop=True)
out_path = os.path.join(HERE, "2026-06-24_zone_spatial_metrics.csv")
out.to_csv(out_path, index=False)
print(f"{len(out):,} rows -> {os.path.basename(out_path)}")

# ── diagnostic: correlate each metric with zone success rate, per zone, BH-FDR ──
zr = pd.read_csv(os.path.join(HERE, "2026-06-08_team_zone_metrics.csv"))
zr["match_team_id"] = zr["match_team_id"].astype(str)
zr = zr[zr.scheme == "thirds"].copy(); zr["succ_rate"] = zr["n_stop_def"] / zr["n_actions"]
m = out.merge(zr[["match_team_id", "zone", "succ_rate"]], on=["match_team_id", "zone"], how="left")
from scipy.stats import pearsonr
metrics = [c for c in out.columns if c not in ("match_team_id", "match_id", "defending_team", "zone")]
res = []
for z in ZONE_ORDER:
    dz = m[m.zone == z]
    for v in metrics:
        s = dz[[v, "succ_rate"]].dropna()
        if len(s) >= 10 and s[v].std() > 0 and s["succ_rate"].std() > 0:
            r, p = pearsonr(s[v], s["succ_rate"]); res.append({"metric": v, "zone": z, "r": r, "p": p})
res = pd.DataFrame(res)
pv = res["p"].values; order = np.argsort(pv); q = np.full(len(pv), np.nan)
adj = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
q[order] = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0); res["q"] = q
print(f"\nvs zone success rate: {len(res)} tests, raw p<.05={int((res.p<.05).sum())}, q<.05={int((res.q<.05).sum())}")
print("\nEDGE UNEVENNESS (edge_cv) per weight metric × zone:")
ecv = res[res.metric.str.endswith("edge_cv")].copy()
ecv["sig"] = np.where(ecv.q < .05, "†", np.where(ecv.p < .05, "*", ""))
print(ecv.pivot_table(index="metric", columns="zone", values="r").reindex(columns=ZONE_ORDER).round(2).to_string())
print("\nedge_cv significant cells:")
print(ecv[ecv.p < .05].sort_values("p")[["metric","zone","r","p","q"]].to_string(index=False,
      formatters={"r":lambda v:f"{v:+.2f}","p":lambda v:f"{v:.4f}","q":lambda v:f"{v:.3f}"}) or "  none")
