"""
Team-level defensive network analysis (combined).

Tabs:
  1. Concentrated vs Balanced
  2. Self vs Shared
  3. Defensive Style
  4. Co-Defenders
  5. Zones
  6. Zone Topology
  7. Zone Contrasts (ICC) — within-team zone differences/slopes, partial-pooled
     (mixed-model) ICC + shrunk per-team fingerprints
  8. Pressing Style — role-projected co-defending pattern as a team trait
     (permutation-tested); the one geometry-free network signal, strongest in
     the high press
  9. Correlation
 10. Robustness (ICC)
 11. Regression
 12. Sensitivity
 13. Data
"""
import warnings
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu, t as t_dist, f as f_dist

# statsmodels powers the partial-pooling (mixed-model) view in the Zone Contrasts tab.
# Optional: the tab degrades to ANOVA-only ICC if it's missing.
try:
    import statsmodels.formula.api as _smf
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ── Data loading ──────────────────────────────────────────────────────────────
import os as _os
outcomes  = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
_gs_csv   = "scripts/2026-06-07_node_level_metrics_with_gs.csv"
nodes     = pd.read_csv(_gs_csv if _os.path.exists(_gs_csv) else "scripts/2026-05-06_node_level_metrics_with_mins.csv")
m2_edges  = pd.read_csv("scripts/2026-05-05_player_net_m2_edges.csv")
edge_dfs  = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
             for k in ("average", "min", "product", "sum")}

nodes["match_team_id"]    = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
m2_edges["match_team_id"] = m2_edges["match_id"].astype(str) + "_" + m2_edges["defending_team"].astype(str)

# Reached the knockout stage = team played any non-group-stage match (16 of 32 teams)
_reached_ko = outcomes.groupby("team_name")["competition_stage"].apply(
    lambda s: bool((s != "Group Stage").any()))
outcomes["reached_knockout"] = outcomes["team_name"].map(_reached_ko)

# Exogenous team-strength proxy: mean FIFA overall rating of the squad (player-level
# ratings, deduped per team). Used as a control variable. Optional.
try:
    _fifa = pd.read_csv("scripts/2026-05-25_player_level_enriched_with_ratings.csv",
                        usecols=["defending_team_name", "defender_id", "overall_rating"])
    _fifa["overall_rating"] = pd.to_numeric(_fifa["overall_rating"], errors="coerce")
    fifa_team_rating = (_fifa.dropna(subset=["overall_rating"])
                        .drop_duplicates(["defending_team_name", "defender_id"])
                        .groupby("defending_team_name")["overall_rating"].mean())
except (FileNotFoundError, ValueError):
    fifa_team_rating = None

# Pitch-zone defensive metrics (produced by 2026-06-08_team_zone_metrics.py).
# Optional: app still runs without it (the Zones tab shows a hint instead).
try:
    zone_raw = pd.read_csv("scripts/2026-06-08_team_zone_metrics.csv")
    zone_raw["match_team_id"] = zone_raw["match_team_id"].astype(str)
except FileNotFoundError:
    zone_raw = None

# Per-zone network topology (produced by 2026-06-19_zone_topology.py). One file
# per edge-weight method, one row per (match_team_id, zone). Optional.
try:
    zone_topo_dfs = {k: pd.read_csv(f"scripts/2026-06-19_zone_topology({k}).csv")
                     for k in ("average", "min", "product", "sum")}
    for _d in zone_topo_dfs.values():
        _d["match_team_id"] = _d["match_team_id"].astype(str)
except FileNotFoundError:
    zone_topo_dfs = None

# Per-zone spatial / spatially-embedded-network predictors (produced by
# 2026-06-24_zone_spatial_metrics.py). Method-independent, one row per
# (match_team_id, zone). Optional.
try:
    zone_spatial_df = pd.read_csv("scripts/2026-06-24_zone_spatial_metrics.csv")
    zone_spatial_df["match_team_id"] = zone_spatial_df["match_team_id"].astype(str)
except FileNotFoundError:
    zone_spatial_df = None

# Pressing-style fingerprint inputs (zone co-defending edges + per-zone avg
# positions). Powers the role-projected co-defending pattern — the one network
# signal that is geometry-free AND a (faint) team trait, strongest in the high
# press (validated by team-label permutation). Optional.
try:
    zone_edge_avg = pd.read_csv("scripts/2026-06-18_zone_network_edge(average).csv")
    zone_pos_df   = pd.read_csv("scripts/2026-06-18_zone_network_positions.csv")
    zone_edge_avg["mtid"] = (zone_edge_avg["match_id"].astype(str) + "_"
                             + zone_edge_avg["defending_team"].astype(str))
except FileNotFoundError:
    zone_edge_avg = zone_pos_df = None

match_mins   = nodes.groupby("match_team_id")["mins_played"].max().rename("match_mins")
squad_size   = nodes.groupby("match_team_id")["defender_id"].count().rename("n_players")
team_names   = nodes.groupby("match_team_id")["team_name"].first()
_match_teams = outcomes.groupby("match_id")["team_name"].apply(list)
match_labels = (
    outcomes.set_index("match_team_id")["match_id"]
    .map(_match_teams.apply(lambda t: " vs ".join(sorted(t))))
    .rename("match")
)
_self_inv_cols = [c for c in nodes.columns if c.endswith("_self_inv")]
self_inv_match = nodes.groupby("match_team_id")[_self_inv_cols].sum().reset_index()
GS_AVAILABLE   = any(c.endswith("_self_inv_gs") for c in nodes.columns)

# ── Column groups ─────────────────────────────────────────────────────────────
WEIGHT_COLS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]
OUTCOME_COLS      = ["goals_against", "shots_against", "xg_against"]
CONTRIBUTION_COLS = ["raw_contribution", "valued_contribution"]
FAULT_COLS        = ["raw_fault", "valued_fault"]
INV_COLS          = ["raw_involvement", "valued_involvement"]

# ── Pressing-style fingerprint (role-projected co-defending pattern) ───────────
PRESS_ZONES = ["own", "mid", "high_press"]
FULL_NETWORK = "full"                       # sentinel: all zones combined
PRESS_ZONE_OPTIONS = [FULL_NETWORK] + PRESS_ZONES
# defending_team id -> team name (fingerprint row labels)
_TEAM_BY_ID = (nodes.drop_duplicates("defending_team")
               .set_index("defending_team")["team_name"].to_dict())

# Goalkeeper lookup for the optional GK-exclusion toggle. Keyed by
# (match_id, defending_team, defender_name) from the enriched player file's
# most_common_position. Dropping these from the role map both reassigns the
# role tertiles among outfield players AND removes any edge touching the GK
# (the role lookup misses -> the edge is filtered out downstream).
try:
    _gk_pos = pd.read_csv("scripts/2026-05-25_player_level_enriched_with_ratings.csv",
                          usecols=["match_id", "defending_team", "defender_name",
                                   "most_common_position"])
    GK_KEYS = set(map(tuple,
                      _gk_pos.loc[_gk_pos["most_common_position"] == "Goalkeeper",
                                  ["match_id", "defending_team", "defender_name"]]
                      .itertuples(index=False, name=None)))
except (FileNotFoundError, ValueError):
    GK_KEYS = set()


def _drop_gk_rows(p):
    """Filter a positions frame to outfield players (drop goalkeepers)."""
    if not GK_KEYS:
        return p
    keep = [(r.match_id, r.defending_team, r.defender_name) not in GK_KEYS
            for r in p.itertuples()]
    return p[keep]


def _build_press_role_map(pos_df, drop_gk=False):
    """Assign each player a pitch role per (match, team, zone): line B/M/F ×
    channel L/C/R, by within-block tertiles of avg x and y. Position-*relative*,
    so the pattern is comparable across matches with different personnel and
    press heights."""
    p = pos_df.dropna(subset=["overall_avg_x", "overall_avg_y"]).copy()
    if drop_gk:
        p = _drop_gk_rows(p)
    gk = ["match_id", "defending_team", "zone"]
    p = p[p.groupby(gk)["defender_name"].transform("count") >= 4].copy()

    def _tert(s):
        try:
            return pd.qcut(s.rank(method="first"), 3, labels=False).astype(int)
        except Exception:                       # <3 distinct -> all middle
            return pd.Series(1, index=s.index)

    g = p.groupby(gk)
    p["_xl"] = g["overall_avg_x"].transform(_tert)
    p["_yc"] = g["overall_avg_y"].transform(_tert)
    p["role"] = (p["_xl"].map({0: "B", 1: "M", 2: "F"})
                 + p["_yc"].map({0: "L", 1: "C", 2: "R"}))
    return p.set_index(gk + ["defender_name"])["role"].to_dict()


def _build_press_role_map_full(pos_df, drop_gk=False):
    """Like _build_press_role_map but for the *full* network: collapse the three
    pitch zones into one avg position per (match, team, player), so roles reflect
    the whole-match shape rather than a single press height. Keyed by
    (match_id, defending_team, defender_name)."""
    p = pos_df.dropna(subset=["overall_avg_x", "overall_avg_y"]).copy()
    if drop_gk:
        p = _drop_gk_rows(p)
    p = (p.groupby(["match_id", "defending_team", "defender_name"], as_index=False)
           [["overall_avg_x", "overall_avg_y"]].mean())
    gk = ["match_id", "defending_team"]
    p = p[p.groupby(gk)["defender_name"].transform("count") >= 4].copy()

    def _tert(s):
        try:
            return pd.qcut(s.rank(method="first"), 3, labels=False).astype(int)
        except Exception:                       # <3 distinct -> all middle
            return pd.Series(1, index=s.index)

    g = p.groupby(gk)
    p["_xl"] = g["overall_avg_x"].transform(_tert)
    p["_yc"] = g["overall_avg_y"].transform(_tert)
    p["role"] = (p["_xl"].map({0: "B", 1: "M", 2: "F"})
                 + p["_yc"].map({0: "L", 1: "C", 2: "R"}))
    return p.set_index(gk + ["defender_name"])["role"].to_dict()


PRESS_ROLE_MAP = _build_press_role_map(zone_pos_df) if zone_pos_df is not None else None
PRESS_ROLE_MAP_FULL = (_build_press_role_map_full(zone_pos_df)
                       if zone_pos_df is not None else None)
# Goalkeeper-excluded role maps (optional toggle). Tertiles are recomputed among
# outfield players only; GK edges drop out for free (role lookup misses).
PRESS_ROLE_MAP_NOGK = (_build_press_role_map(zone_pos_df, drop_gk=True)
                       if zone_pos_df is not None else None)
PRESS_ROLE_MAP_FULL_NOGK = (_build_press_role_map_full(zone_pos_df, drop_gk=True)
                            if zone_pos_df is not None else None)


@st.cache_data(show_spinner="Building pressing-style patterns…")
def pressing_role_patterns(weight, zone, drop_gk=False):
    """Per match-team: row-normalised role-pair co-defending weight vector (a
    *pattern*, not a volume). Returns (matrix, mtids, team_names, role_pairs).
    drop_gk=True uses the goalkeeper-excluded role map, so GK edges are removed
    and roles are re-tertiled among outfield players."""
    if zone == FULL_NETWORK:
        # all zones combined: keep every edge, use the whole-match role map. The
        # pivot below sums across the (now multiple) zone rows per role pair.
        ed = zone_edge_avg[zone_edge_avg[weight] > 0].copy()
        rm = PRESS_ROLE_MAP_FULL_NOGK if drop_gk else PRESS_ROLE_MAP_FULL
        r1 = [rm.get((r.match_id, r.defending_team, r.player_1)) for r in ed.itertuples()]
        r2 = [rm.get((r.match_id, r.defending_team, r.player_2)) for r in ed.itertuples()]
    else:
        ed = zone_edge_avg[(zone_edge_avg[weight] > 0)
                           & (zone_edge_avg["zone"] == zone)].copy()
        rm = PRESS_ROLE_MAP_NOGK if drop_gk else PRESS_ROLE_MAP
        r1 = [rm.get((r.match_id, r.defending_team, zone, r.player_1)) for r in ed.itertuples()]
        r2 = [rm.get((r.match_id, r.defending_team, zone, r.player_2)) for r in ed.itertuples()]
    ed["_rp"] = ["-".join(sorted([a, b])) if (a and b) else None
                 for a, b in zip(r1, r2)]
    ed = ed.dropna(subset=["_rp"])
    piv = ed.groupby(["mtid", "_rp"])[weight].sum().unstack(fill_value=0.0)
    piv = piv.div(piv.sum(axis=1), axis=0)              # row-normalise -> pattern
    teams = [_TEAM_BY_ID.get(int(m.split("_")[1]), m.split("_")[1]) for m in piv.index]
    return piv.values.astype(float), list(piv.index), teams, list(piv.columns)


def _bh_fdr(pvals):
    """Benjamini–Hochberg q-values. NaNs are excluded from the family and stay
    NaN. BH is valid under positive dependence (PRDS), which the correlated edge
    weights satisfy, so it is the right (not over-conservative) family correction
    here. Returns an array aligned to the input."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    if m == 0:
        return q
    order = ok[np.argsort(p[ok])]
    qv = p[order] * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]      # enforce monotonicity
    q[order] = np.clip(qv, 0.0, 1.0)
    return q


def _meff_li_ji(corr):
    """Li & Ji (2005) effective number of independent tests from a correlation
    matrix: Σ [1(λ≥1) + (λ − ⌊λ⌋)] over its (clipped) eigenvalues."""
    ev = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0)), 0.0, None)
    return float(np.sum((ev >= 1).astype(float) + (ev - np.floor(ev))))


@st.cache_data(show_spinner="Effective # tests…")
def pressing_weight_meff(zone, drop_gk=False):
    """Effective number of independent tests among the 6 edge weights for a zone.
    The weights (raw/valued × involvement/fault/contribution) are highly
    correlated, so the 6 columns carry far fewer than 6 independent comparisons;
    computed from the correlation of their (match-team × role-pair) pattern
    matrices, aligned on the common rows/columns. Used to make the family-wise
    penalty gracious instead of treating all 24 weight×zone cells as independent."""
    mats = {}
    for w in WEIGHT_COLS:
        V, mtids, teams, cols = pressing_role_patterns(w, zone, drop_gk)
        mats[w] = pd.DataFrame(V, index=mtids, columns=cols)
    idx = cols = None
    for w in WEIGHT_COLS:
        idx = mats[w].index if idx is None else idx.intersection(mats[w].index)
        cols = mats[w].columns if cols is None else cols.intersection(mats[w].columns)
    if len(idx) < 3 or len(cols) < 2:
        return float(len(WEIGHT_COLS))
    F = pd.DataFrame({w: mats[w].loc[idx, cols].values.flatten() for w in WEIGHT_COLS})
    return _meff_li_ji(F.corr().values)


def _cos_sim_matrix(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = V / n
    return U @ U.T


def _within_between(C, teams):
    wi, bw = [], []
    for i, j in combinations(range(len(teams)), 2):
        (wi if teams[i] == teams[j] else bw).append(C[i, j])
    return float(np.mean(wi)), float(np.mean(bw)), len(wi), len(bw)


def _press_icc11(vals, teams):
    """ICC(1,1): team = subject, match = replicate."""
    df = pd.DataFrame({"v": vals, "t": teams})
    g = df.groupby("t")["v"]; k = g.count(); nt = df["t"].nunique()
    if nt < 2 or len(df) == nt:
        return np.nan
    grand = df["v"].mean()
    msb = ((g.mean() - grand) ** 2 * k).sum() / (nt - 1)
    msw = ((df["v"] - df["t"].map(g.mean())) ** 2).sum() / (len(df) - nt)
    den = msb + (k.mean() - 1) * msw
    return float((msb - msw) / den) if den else np.nan


def _perm_delta(C, teams, nperm, rng):
    """Permutation p-value for within−between cosine gap."""
    wi, bw, _, _ = _within_between(C, teams)
    obs = wi - bw
    ge = 0
    for _ in range(nperm):
        p = rng.permutation(teams)
        w2, b2, _, _ = _within_between(C, p)
        if (w2 - b2) >= obs:
            ge += 1
    return obs, wi, bw, (ge + 1) / (nperm + 1)


@st.cache_data(show_spinner="Permuting team labels…")
def pressing_style_stats(weight, zone, nperm=2000, seed=20260625, drop_gk=False):
    """Within- vs between-team cosine similarity of the role-pair pattern, with a
    team-label permutation null (correct for non-independent match-team pairs).
    Also runs a formation-confound check: binary presence/absence vs continuous
    weight, and a common-pairs-only (>=80% presence) re-test."""
    V, mtids, teams, cols = pressing_role_patterns(weight, zone, drop_gk)
    teams = np.array(teams)
    rng = np.random.default_rng(seed)

    C = _cos_sim_matrix(V)
    obs, wi, bw, perm_p = _perm_delta(C, teams, nperm, rng)

    # --- formation-confound checks ---
    V_bin = (V > 0).astype(float)
    C_bin = _cos_sim_matrix(V_bin)
    obs_bin, _, _, p_bin = _perm_delta(C_bin, teams, nperm, rng)

    presence = (V > 0).sum(axis=0)
    common_mask = presence >= len(V) * 0.8
    n_common = int(common_mask.sum())
    obs_com, p_com = np.nan, np.nan
    if n_common >= 3:
        V_com = V[:, common_mask]
        V_com = V_com / V_com.sum(axis=1, keepdims=True)
        C_com = _cos_sim_matrix(V_com)
        obs_com, _, _, p_com = _perm_delta(C_com, teams, nperm, rng)

    icc = (pd.DataFrame([(c, _press_icc11(V[:, i], teams)) for i, c in enumerate(cols)],
                        columns=["role_pair", "icc"])
           .dropna().sort_values("icc", ascending=False).reset_index(drop=True))
    fingerprint = pd.DataFrame(V, columns=cols, index=teams).groupby(level=0).mean()
    return {"within": wi, "between": bw, "delta": obs, "perm_p": perm_p,
            "n_mt": len(teams), "n_teams": int(pd.Series(teams).nunique()),
            "icc": icc, "fingerprint": fingerprint,
            "bin_delta": obs_bin, "bin_p": p_bin,
            "com_delta": obs_com, "com_p": p_com, "n_common": n_common,
            "n_total_pairs": len(cols)}


def _unit_rows(V):
    """Row-normalise to unit L2 norm (cosine space)."""
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(nrm == 0, 1.0, nrm)


def _loo_topk_acc(U, codes, K):
    """Leave-one-match-out nearest-centroid team identification, fully vectorised.

    U: (n,d) unit-norm role-pair patterns. codes: (n,) integer team labels in
    0..K-1. For each row i, assign it to the team whose mean fingerprint (cosine)
    is closest; the row's OWN team centroid excludes row i (leave-one-out), every
    other team uses its full mean. A row is *eligible* only if its team has ≥2
    matches (else there is no LOO centroid for the true team). Returns
    (top1, top3, n_eligible, correct1_mask, eligible_mask)."""
    n = U.shape[0]
    Y = np.zeros((n, K))
    Y[np.arange(n), codes] = 1.0
    S = Y.T @ U                                   # (K,d) per-team sums
    c = Y.sum(0)                                  # (K,) per-team counts
    cb_norm = np.linalg.norm(np.divide(S, np.where(c[:, None] == 0, 1.0, c[:, None])), axis=1)
    D = U @ S.T                                   # (n,K) Uᵢ·S_t
    denom = c[None, :] * cb_norm[None, :]
    M = np.divide(D, denom, out=np.full((n, K), -1.0), where=denom > 0)  # cosine to base centroid
    # replace the own-team column with the leave-one-out cosine (Uᵢ unit norm):
    #   cos(Uᵢ, S_own − Uᵢ) = (Uᵢ·S_own − 1) / ‖S_own − Uᵢ‖
    own = codes
    cc = c[own]
    UdotSown = D[np.arange(n), own]
    Snorm_own = cb_norm[own] * c[own]             # ‖S_own‖
    loo_norm = np.sqrt(np.maximum(Snorm_own ** 2 - 2 * UdotSown + 1.0, 0.0))
    loo_cos = np.where(loo_norm > 0, (UdotSown - 1.0) / np.where(loo_norm == 0, 1.0, loo_norm), -1.0)
    M[np.arange(n), own] = loo_cos
    own_cos = M[np.arange(n), own]
    greater = (M > own_cos[:, None]).sum(1)       # 0 = own team ranked first
    elig = cc >= 2
    correct1 = (greater == 0) & elig
    correct3 = (greater < 3) & elig
    return int(correct1.sum()), int(correct3.sum()), int(elig.sum()), correct1, elig


@st.cache_data(show_spinner="Leave-one-match-out team ID…")
def pressing_loo_identification(weight, zone, nperm=2000, seed=20260627, drop_gk=False):
    """#1 — robustness as identifiability. Can the role-pair pattern alone *name
    the team*? Each held-out match is classified to the nearest team by its LOO
    mean fingerprint; accuracy is tested against a team-label permutation null
    (chance = mean permuted accuracy). A multivariate, intuitive alternative to
    per-pair ICC. Returns accuracies, perm-p, empirical chance, per-team table."""
    V, mtids, teams, cols = pressing_role_patterns(weight, zone, drop_gk)
    teams = np.asarray(teams)
    U = _unit_rows(V)
    codes, uniq = pd.factorize(teams)
    K = len(uniq)
    t1, t3, total, correct1, elig = _loo_topk_acc(U, codes, K)
    if total == 0:
        return {"total": 0}
    rng = np.random.default_rng(seed)
    perm_t1 = np.empty(nperm)
    for p in range(nperm):
        perm_t1[p] = _loo_topk_acc(U, rng.permutation(codes), K)[0]
    perm_p = (int((perm_t1 >= t1).sum()) + 1) / (nperm + 1)
    chance = float(perm_t1.mean() / total)
    per_team = (pd.DataFrame({"team": teams, "correct": correct1.astype(float), "elig": elig})
                .query("elig").groupby("team")
                .agg(acc=("correct", "mean"), n=("correct", "size"))
                .reset_index())
    return {"top1": t1 / total, "top3": t3 / total, "total": total,
            "perm_p": perm_p, "chance": chance,
            "n_teams": int(pd.Series(teams)[elig].nunique()), "per_team": per_team}


@st.cache_data(show_spinner="Within-team deviation vs outcome…")
def pressing_deviation_outcome(weight, zone, scheme="thirds", nperm=2000, seed=20260627,
                               drop_gk=False):
    """#2 — trait → outcome, dominance-controlled. Per match-team: cosine DISTANCE
    between its role-pair pattern and its team's LOO mean fingerprint. Does
    deviating from a team's own pressing identity cost zone defensive success
    (stop_rate) *that match*? Tested WITHIN team (each team is its own baseline,
    so squad-quality / dominance confounds cancel) via team-demeaned correlation
    with a within-team permutation null. Raw (uncontrolled) r returned for
    contrast. Outcome = stop_rate in the matching pitch zone; for the full network
    it is pooled across zones (Σ stops / Σ actions)."""
    if zone_raw is None:
        return {"n": 0}
    V, mtids, teams, cols = pressing_role_patterns(weight, zone, drop_gk)
    teams = np.asarray(teams)
    U = _unit_rows(V)
    codes, uniq = pd.factorize(teams)
    K = len(uniq)
    Y = np.zeros((len(U), K)); Y[np.arange(len(U)), codes] = 1.0
    S = Y.T @ U; c = Y.sum(0)
    own = codes; cc = c[own]
    D_own = (U * S[own]).sum(1)                    # Uᵢ·S_own
    loo_norm = np.sqrt(np.maximum(np.linalg.norm(S[own], axis=1) ** 2 - 2 * D_own + 1.0, 0.0))
    cos = np.where((loo_norm > 0) & (cc >= 2),
                   (D_own - 1.0) / np.where(loo_norm == 0, 1.0, loo_norm), np.nan)
    deviation = 1.0 - cos                          # cosine distance to own identity

    zr = zone_raw[zone_raw["scheme"] == scheme]
    if zone == FULL_NETWORK:
        g = zr.groupby("match_team_id").agg(ns=("n_stop_def", "sum"), na=("n_actions", "sum"))
        success = g["ns"] / g["na"].replace(0, np.nan)
    else:
        success = (zr[zr["zone"] == zone].drop_duplicates("match_team_id")
                   .set_index("match_team_id")["stop_rate"])
    succ = np.array([success.get(m, np.nan) for m in mtids])

    df = pd.DataFrame({"mtid": mtids, "team": teams, "deviation": deviation, "success": succ})
    df = df.dropna(subset=["deviation", "success"])
    df = df[df.groupby("team")["team"].transform("size") >= 2]
    if len(df) < 8:
        return {"n": len(df)}
    df["dev_demean"] = df["deviation"] - df.groupby("team")["deviation"].transform("mean")
    df["suc_demean"] = df["success"] - df.groupby("team")["success"].transform("mean")
    r_within, _ = pearsonr(df["dev_demean"], df["suc_demean"])
    n, nt = len(df), df["team"].nunique()
    dof = max(n - nt - 1, 1)
    tstat = r_within * np.sqrt(dof / max(1 - r_within ** 2, 1e-12))
    p_within_param = float(2 * t_dist.sf(abs(tstat), dof))

    # within-team permutation null: shuffle stop_rate within each team (preserves
    # each team's baseline → team means, hence the demeaning, are unchanged).
    rng = np.random.default_rng(seed)
    dev_d = df["dev_demean"].values
    team_mean = df.groupby("team")["success"].transform("mean").values
    suc = df["success"].values
    grp_pos = [np.array([df.index.get_loc(i) for i in g.index])
               for _, g in df.groupby("team")]
    obs, ge = abs(r_within), 0
    for _ in range(nperm):
        sd = suc.copy()
        for gp in grp_pos:
            sd[gp] = rng.permutation(sd[gp])
        rp = np.corrcoef(dev_d, sd - team_mean)[0, 1]
        if abs(rp) >= obs:
            ge += 1
    p_within_perm = (ge + 1) / (nperm + 1)

    r_raw, p_raw = pearsonr(df["deviation"], df["success"])
    rho_raw, p_rho = spearmanr(df["deviation"], df["success"])
    return {"n": n, "n_teams": nt, "r_within": r_within,
            "p_within_perm": p_within_perm, "p_within_param": p_within_param,
            "r_raw": r_raw, "p_raw": p_raw, "rho_raw": rho_raw, "p_rho_raw": p_rho,
            "df": df, "zone": zone, "scheme": scheme}


GROUPS = {
    "Total Network Strength":              WEIGHT_COLS,
    "Network Density":                     [c + "_density"                  for c in WEIGHT_COLS],
    "Gini (player strength inequality)":   [c + "_gini"                    for c in WEIGHT_COLS],
    "Clustering Coefficient (unweighted)": [c + "_cc_unweighted"           for c in WEIGHT_COLS],
    "Clustering Coefficient (weighted)":   [c + "_cc_weighted"             for c in WEIGHT_COLS],
    "Freeman Centralization (unweighted)": [c + "_centralization"          for c in WEIGHT_COLS],
    "Freeman Centralization (weighted)":   [c + "_centralization_weighted" for c in WEIGHT_COLS],
    "Degree Assortativity":                [c + "_assortativity"           for c in WEIGHT_COLS],
    "Max K-core":                          [c + "_kcore_max"               for c in WEIGHT_COLS],
    "LCC Ratio":                           [c + "_lcc_ratio"               for c in WEIGHT_COLS],
}

GROUP_DESC = {
    "Total Network Strength":
        "Sum of all edge weights — total volume of co-defensive activity across all player pairs. "
        "Scales with match exposure; higher values indicate greater overall defensive engagement.",
    "Network Density":
        "Proportion of possible player pairs that co-defended at least once (edge count ≥ threshold). "
        "Measures how broadly defensive collaboration is spread across the squad. "
        "Topology-only: unaffected by edge-weight method.",
    "Gini (player strength inequality)":
        "Inequality in individual players' defensive load (node strength). "
        "High Gini = a few players carry most of the defensive burden; "
        "low Gini = workload is evenly distributed across the squad.",
    "Clustering Coefficient (unweighted)":
        "Average probability that two co-defenders of a given player also co-defend with each other "
        "(triangle closure). Reflects the tightness of local defensive groups. Topology-only.",
    "Clustering Coefficient (weighted)":
        "Weighted extension of clustering — accounts for the intensity of co-defensive links, "
        "not just their presence. Higher values indicate denser and stronger local defensive triangles.",
    "Freeman Centralization (unweighted)":
        "Degree to which the network's connectivity is concentrated around a single hub player "
        "(Freeman 1979). High = one organiser dominates the defensive structure; "
        "low = no single hub, responsibility is spread. Topology-only.",
    "Freeman Centralization (weighted)":
        "Weighted extension of Freeman centralization using node strength (sum of edge weights). "
        "Captures structural dominance weighted by defensive intensity — "
        "how much one player's co-defensive load towers above teammates.",
    "Degree Assortativity":
        "Pearson correlation between the degrees of adjacent nodes. "
        "Positive = highly connected defenders tend to co-defend with other highly connected defenders "
        "('stars pair with stars'); negative = key defenders paired with peripheral role players. Topology-only.",
    "Max K-core":
        "The highest k such that a subgraph exists where every node has ≥ k co-defenders within it — "
        "the densest mutually connected defensive nucleus. "
        "Higher = tighter collective defensive core; lower = hierarchical or sparse co-defending. Topology-only.",
    "LCC Ratio":
        "Proportion of squad players belonging to the largest connected component at the given threshold. "
        "High = defense operates as one unified, interconnected group; "
        "low = fragmented into isolated sub-units with little cross-group collaboration. Topology-only.",
}

# ── Stage constants ───────────────────────────────────────────────────────────
STAGE_ORDER    = {"Group Stage": 1, "Round of 16": 2, "Quarter-finals": 3,
                  "Semi-finals": 4, "3rd Place Final": 4, "Final": 5}
STAGE_LABEL    = {"Group Stage": "Group Stage", "Round of 16": "Round of 16",
                  "Quarter-finals": "Quarter-finals", "Semi-finals": "Semi-finals",
                  "3rd Place Final": "Semi-finals", "Final": "Final"}
STAGE_PALETTE  = {"Group Stage": "#a8d8ea", "Round of 16": "#78c1e0",
                  "Quarter-finals": "#f4a261", "Semi-finals": "#e76f51", "Final": "#b5179e"}
STAGE_CATEGORY_ORDER = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
STYLE_STAGE_ORDER    = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "3rd Place Final", "Final"]
STYLE_STAGE_PALETTE  = {"Group Stage": "#adb5bd", "Round of 16": "#4cc9f0",
                        "Quarter-finals": "#4361ee", "Semi-finals": "#f77f00",
                        "3rd Place Final": "#9b2226", "Final": "#2dc653"}


# ── Shared helpers ────────────────────────────────────────────────────────────
def gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def _join_outcomes(df):
    return df.join(team_names).join(
        outcomes.set_index("match_team_id")[OUTCOME_COLS + ["competition_stage", "passes_against"]]
    )


def _furthest_stage(df):
    return (df.groupby("team_name")["competition_stage"]
              .apply(lambda s: max(s, key=lambda x: STAGE_ORDER.get(x, 0)))
              .map(STAGE_LABEL).rename("furthest_stage").reset_index())


def _add_quadrant_lines(fig, df, x_col, y_col, opacity=1.0):
    fig.add_vline(x=df[x_col].median(), line_dash="dash", line_color="grey", opacity=opacity)
    fig.add_hline(y=df[y_col].median(), line_dash="dash", line_color="grey", opacity=opacity)


# ── Concentrated vs Balanced ──────────────────────────────────────────────────
def _cent_w(x):
    n = len(x)
    if n <= 2 or x.max() == 0:
        return np.nan
    return (1 - x / x.max()).sum() / (n - 2)


def build_conc_match(edge_df, metric, thr=1):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    ec_col = metric + "_edge_count"
    strength = edge_df.groupby("match_team_id")[metric].sum().rename("strength")
    p1 = edge_df[["match_team_id", "player_1", metric]].rename(columns={"player_1": "player"})
    p2 = edge_df[["match_team_id", "player_2", metric]].rename(columns={"player_2": "player"})
    player_str = pd.concat([p1, p2]).groupby(["match_team_id", "player"])[metric].sum()
    cent_w = player_str.groupby("match_team_id").apply(_cent_w).rename("centralization_w")
    kcore_dict   = {}
    density_dict = {}
    for mid, grp in edge_df.groupby("match_team_id"):
        e = grp[grp[ec_col] >= thr] if ec_col in edge_df.columns else grp
        G = nx.Graph()
        G.add_edges_from(zip(e["player_1"], e["player_2"]))
        if G.number_of_nodes() >= 2:
            kcore_dict[mid]   = max(nx.core_number(G).values())
            density_dict[mid] = nx.density(G)
    kcore_s   = pd.Series(kcore_dict,   name="kcore_max")
    density_s = pd.Series(density_dict, name="network_density")
    df = (_join_outcomes(pd.concat([strength, cent_w, kcore_s, density_s], axis=1))
          .join(match_labels)
          .join(match_mins)
          .dropna(subset=["strength"]))
    df["strength_per90"]    = df["strength"] / (df["match_mins"] / 90)
    df["strength_per_pass"] = df["strength"] / df["passes_against"]
    for _oc in OUTCOME_COLS:
        df[_oc + "_per_pass"] = df[_oc] / df["passes_against"]
    return df.reset_index()


def build_conc_team(df):
    _pp_oc = [oc + "_per_pass" for oc in OUTCOME_COLS if oc + "_per_pass" in df.columns]
    agg = df.groupby("team_name")[
        ["strength", "strength_per90", "strength_per_pass",
         "centralization_w", "kcore_max", "network_density"] + OUTCOME_COLS + _pp_oc
    ].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


_Y_HIGH = {"centralization_w": "Centralized", "kcore_max": "Dense core",    "network_density": "Dense"}
_Y_LOW  = {"centralization_w": "Distributed", "kcore_max": "Sparse core", "network_density": "Sparse"}


def plot_conc_match(df, outcome_col, strength_col="strength", x_label="Total Strength",
                    y_col="centralization_w", y_label="Centralization (weighted)"):
    fig = px.scatter(df, x=strength_col, y=y_col, color=outcome_col, size=outcome_col,
                     hover_name="team_name", hover_data=["match", "competition_stage"] + OUTCOME_COLS,
                     color_continuous_scale="RdYlGn_r",
                     title="Total Strength vs " + y_label + " — match level",
                     labels={strength_col: x_label, y_col: y_label})
    _add_quadrant_lines(fig, df, strength_col, y_col)
    fig.add_annotation(x=df[strength_col].max(), y=df[y_col].max(),
                       text=f"High strength<br>{_Y_HIGH.get(y_col, 'High Y')}",
                       showarrow=False, font_size=10)
    fig.add_annotation(x=df[strength_col].max(), y=df[y_col].min(),
                       text=f"High strength<br>{_Y_LOW.get(y_col, 'Low Y')}",
                       showarrow=False, font_size=10)
    return fig


def plot_conc_team(df, outcome_col, strength_col="strength", x_label="Total Strength",
                   y_col="centralization_w", y_label="Centralization (weighted)"):
    fig = px.scatter(df, x=strength_col, y=y_col, color="furthest_stage", size=outcome_col,
                     color_discrete_map=STAGE_PALETTE,
                     category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
                     hover_name="team_name", hover_data=OUTCOME_COLS + ["furthest_stage"],
                     title="Total Strength vs " + y_label + " — team level",
                     labels={strength_col: x_label, y_col: y_label,
                             "furthest_stage": "Furthest Stage"})
    _add_quadrant_lines(fig, df, strength_col, y_col)
    fig.add_annotation(x=df[strength_col].max(), y=df[y_col].max(),
                       text=f"High strength<br>{_Y_HIGH.get(y_col, 'High Y')}",
                       showarrow=False, font_size=10)
    fig.add_annotation(x=df[strength_col].max(), y=df[y_col].min(),
                       text=f"High strength<br>{_Y_LOW.get(y_col, 'Low Y')}",
                       showarrow=False, font_size=10)
    return fig


# ── Self vs Shared ────────────────────────────────────────────────────────────
def build_selfshared_match(edge_df, metric, use_gs=False):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    if use_gs and GS_AVAILABLE:
        self_inv = nodes.groupby("match_team_id")[metric + "_self_inv_gs"].sum().rename("self_inv")
        shared   = nodes.groupby("match_team_id")[metric + "_shared_inv_gs"].sum().rename("shared_inv")
        ratio    = (shared / (self_inv + shared)).rename("self_ratio")  # sharedness: high = shared
    else:
        shared   = edge_df.groupby("match_team_id")[metric].sum().rename("shared_inv")
        self_inv = nodes.groupby("match_team_id")[metric + "_self_inv"].sum().rename("self_inv")
        ratio    = (self_inv / (self_inv + shared)).rename("self_ratio")
    df = _join_outcomes(pd.concat([self_inv, shared, ratio], axis=1)).dropna(subset=["self_inv", "shared_inv"])
    return df.reset_index()


def build_selfshared_team(df):
    _cols = ["self_inv", "shared_inv", "self_ratio"] + OUTCOME_COLS
    if "passes_against" in df.columns:
        _cols = _cols + ["passes_against"]
    agg = df.groupby("team_name")[_cols].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


def plot_selfshared(df_team, outcome_col, use_gs=False, correct_possession=False):
    df_sorted = df_team.sort_values("self_ratio", ascending=False).reset_index(drop=True)
    fig_bar = px.bar(df_sorted, x="team_name", y=["self_inv", "shared_inv"],
                     title="Self vs Shared Involvement per Team",
                     labels={"value": "Total Involvement", "team_name": "Team"}, barmode="stack")
    fig_bar.update_xaxes(tickangle=45)

    ratio_label = "Sharedness (shared / total) [Gini-Simpson]" if use_gs else "Self ratio (self / total)"
    def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    # build plot df — residualize y if possession correction requested
    if correct_possession and "passes_against" in df_sorted.columns:
        _s = df_sorted[[outcome_col, "passes_against"]].dropna()
        _ry = np.full(len(df_sorted), np.nan)
        _ry[_s.index] = _resid(_s[outcome_col].values, _s["passes_against"].values)
        df_plot = df_sorted.copy()
        df_plot["_y"] = _ry
        y_col_plot, y_label = "_y", f"{outcome_col} (residual after passes against)"
    else:
        df_plot, y_col_plot, y_label = df_sorted, outcome_col, outcome_col

    _valid = df_plot[["self_ratio", y_col_plot]].dropna()
    r_val, p_val = pearsonr(_valid["self_ratio"], _valid[y_col_plot])
    rho,   p_rho = spearmanr(_valid["self_ratio"], _valid[y_col_plot])

    if "passes_against" in df_sorted.columns and not correct_possession:
        _s = df_sorted[["self_ratio", outcome_col, "passes_against"]].dropna()
        _a = _resid(_s["self_ratio"].values, _s["passes_against"].values)
        _b = _resid(_s[outcome_col].values,  _s["passes_against"].values)
        _mk = ~(np.isnan(_a) | np.isnan(_b))
        r_p,   p_p  = pearsonr(_a[_mk],  _b[_mk])
        rho_p, p_rp = spearmanr(_a[_mk], _b[_mk])
        _title_r = (
            f"r = {r_val:.3f}, p = {p_val:.3f} ({_sig(p_val)})  |  partial r = {r_p:.3f}, p = {p_p:.3f} ({_sig(p_p)}, ctrl passes_against)  |  "
            f"ρ = {rho:.3f}, p = {p_rho:.3f} ({_sig(p_rho)})  |  partial ρ = {rho_p:.3f}, p = {p_rp:.3f} ({_sig(p_rp)}, ctrl passes_against)"
        )
    else:
        _note = " corrected" if correct_possession else ""
        _title_r = (f"r = {r_val:.3f}, p = {p_val:.3f} ({_sig(p_val)}){_note}  |  "
                    f"ρ = {rho:.3f}, p = {p_rho:.3f} ({_sig(p_rho)}){_note}")

    fig_scatter = px.scatter(df_plot, x="self_ratio", y=y_col_plot,
                             text="team_name", color="furthest_stage",
                             color_discrete_map=STAGE_PALETTE,
                             category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
                             trendline="ols", trendline_scope="overall",
                             trendline_color_override="black",
                             title=f"Self-involvement ratio vs Defensive Performance  |  {_title_r}",
                             labels={"self_ratio": ratio_label, y_col_plot: y_label,
                                     "furthest_stage": "Furthest Stage"})
    fig_scatter.update_traces(textposition="top center", selector={"mode": "markers+text"})
    fig_scatter.update_layout(title_font_size=11)
    return fig_bar, fig_scatter, df_sorted


# ── Defensive Style ───────────────────────────────────────────────────────────
def _best_stage(stages):
    ranked = [s for s in STYLE_STAGE_ORDER if s in stages.values]
    return ranked[-1] if ranked else stages.iloc[0]


def build_style_team(edge_df, x_col, y_col, size_col, x_normalize="raw", y_normalize="raw"):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    match_level = (
        edge_df.groupby("match_team_id")[[x_col, y_col]].sum()
        .merge(outcomes[["match_team_id", "team_name", "competition_stage", "passes_against"] + OUTCOME_COLS], on="match_team_id")
        .merge(self_inv_match[["match_team_id", x_col + "_self_inv", y_col + "_self_inv"]], on="match_team_id", how="left")
    )
    for col, norm_key in [(x_col, x_normalize), (y_col, y_normalize)]:
        if norm_key == "per_90":
            norm = match_level["match_team_id"].map(match_mins) / 90
        elif norm_key == "per_pass_against":
            norm = match_level["passes_against"]
        else:
            continue
        match_level[col] = match_level[col] / norm
        si = col + "_self_inv"
        if si in match_level.columns:
            match_level[si] = match_level[si] / norm
    match_level["x_total"] = match_level[x_col] + match_level[x_col + "_self_inv"]
    match_level["y_total"] = match_level[y_col] + match_level[y_col + "_self_inv"]
    team_df = (match_level.groupby("team_name")
               .agg(x=(x_col, "mean"), y=(y_col, "mean"),
                    x_total=("x_total", "mean"), y_total=("y_total", "mean"),
                    size=(size_col, "mean"), n_matches=("match_team_id", "count"),
                    best_stage=("competition_stage", _best_stage),
                    **{oc: (oc, "mean") for oc in OUTCOME_COLS})
               .reset_index())
    team_df["best_stage"] = pd.Categorical(team_df["best_stage"], categories=STYLE_STAGE_ORDER, ordered=True)
    return match_level, team_df


def plot_style(team_df, x_col, y_col, size_col, size_scale):
    shared = dict(size="size", size_max=size_scale, color="best_stage",
                  color_discrete_map=STYLE_STAGE_PALETTE,
                  category_orders={"best_stage": STYLE_STAGE_ORDER}, hover_name="team_name")
    fig1 = px.scatter(team_df, x="x", y="y",
                      title="Network contribution vs fault  |  colour = best stage",
                      labels={"x": x_col, "y": y_col, "size": size_col, "best_stage": "Best stage"},
                      hover_data={"x": ":.3f", "y": ":.3f", "size": ":.3f",
                                  "n_matches": True, "best_stage": True}, **shared)
    _add_quadrant_lines(fig1, team_df, "x", "y", opacity=0.5)
    fig1.update_layout(height=600)
    fig2 = px.scatter(team_df, x="x_total", y="y_total",
                      title="Total contribution vs fault (network + self-inv)  |  colour = best stage",
                      labels={"x_total": f"{x_col} (network+self)", "y_total": f"{y_col} (network+self)",
                              "size": size_col, "best_stage": "Best stage"},
                      hover_data={"x_total": ":.3f", "y_total": ":.3f", "size": ":.3f",
                                  "n_matches": True, "best_stage": True}, **shared)
    _add_quadrant_lines(fig2, team_df, "x_total", "y_total", opacity=0.5)
    fig2.update_layout(height=600)
    return fig1, fig2


# ── Co-Defenders ──────────────────────────────────────────────────────────────
@st.cache_data
def build_co_defender_data():
    df = m2_edges.copy()
    # Aggregate to pass level first: count unique defenders per (match_team_id, passer, receiver),
    # then subtract 1 to get co-defenders (excluding the defender themselves).
    # This avoids inflating the average by counting each co-defended pass once per defender.
    n_co_per_pass = (
        df.groupby(["match_team_id", "passer_id", "receiver_id"])["defender_id"]
          .nunique()
          .reset_index(name="n_co")
    )
    avg_co = n_co_per_pass.groupby("match_team_id")["n_co"].mean().reset_index(name="avg_co_defenders")
    avg_co = avg_co.merge(outcomes[["match_team_id", "team_name", "competition_stage"] + OUTCOME_COLS],
                          on="match_team_id", how="left")
    avg_co_team = (avg_co.groupby("team_name")
                         .agg(avg_co_defenders=("avg_co_defenders", "mean"),
                              **{c: (c, "mean") for c in OUTCOME_COLS})
                         .reset_index()
                         .merge(_furthest_stage(avg_co), on="team_name"))
    return avg_co, avg_co_team


@st.cache_data
def build_partnerships():
    df = m2_edges.copy()
    groups = (df.groupby(["match_id", "defending_team", "passer_id", "receiver_id"])
               ["defender_name"].apply(list).reset_index())
    rows = []
    for _, row in groups.iterrows():
        mtid = str(row["match_id"]) + "_" + str(row["defending_team"])
        for a, b in combinations(sorted(set(row["defender_name"])), 2):
            rows.append({"match_team_id": mtid, "player_a": a, "player_b": b})
    if not rows:
        return pd.DataFrame(columns=["match_team_id", "player_a", "player_b", "co_defenses"])
    pairs = (pd.DataFrame(rows)
               .groupby(["match_team_id", "player_a", "player_b"]).size()
               .reset_index(name="co_defenses"))
    pairs = pairs.merge(outcomes[["match_team_id", "team_name"]], on="match_team_id", how="left")
    return pairs.sort_values("co_defenses", ascending=False)


def plot_avg_co_defenders(avg_co, avg_co_team, outcome_col):
    fig_match = px.scatter(
        avg_co, x="avg_co_defenders", y=outcome_col,
        color="competition_stage", hover_name="team_name",
        title="Avg co-defenders per pass vs outcome — match level",
        labels={"avg_co_defenders": "Avg co-defenders per pass"},
    )
    _sorted_teams = avg_co_team.sort_values("avg_co_defenders", ascending=False)
    fig_team = px.bar(
        _sorted_teams,
        x="team_name", y="avg_co_defenders",
        color="furthest_stage", color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER,
                         "team_name": _sorted_teams["team_name"].tolist()},
        title="Avg co-defenders per pass — by team",
        labels={"avg_co_defenders": "Avg co-defenders", "team_name": "Team"},
    )
    fig_team.update_xaxes(tickangle=45)
    return fig_match, fig_team


def plot_partnership_heatmap(pairs, team_name):
    team_pairs = pairs[pairs["team_name"] == team_name]
    if team_pairs.empty:
        return None
    all_p = sorted(set(team_pairs["player_a"]) | set(team_pairs["player_b"]))
    mat   = pd.DataFrame(0, index=all_p, columns=all_p, dtype=float)
    for _, r in team_pairs.iterrows():
        mat.loc[r["player_a"], r["player_b"]] += r["co_defenses"]
        mat.loc[r["player_b"], r["player_a"]] += r["co_defenses"]
    n = len(all_p)
    cell_px = 36
    fig_size = max(400, n * cell_px)
    fig = px.imshow(mat, color_continuous_scale="Blues", aspect="equal",
                    title=f"{team_name} — co-defending heatmap (passes co-defended together)")
    fig.update_layout(width=fig_size + 200, height=fig_size + 150)
    fig.update_xaxes(tickmode="array", tickvals=list(range(n)), ticktext=all_p,
                     tickangle=45, tickfont_size=11)
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)), ticktext=all_p,
                     tickfont_size=11)
    return fig


# ── Correlation & ICC ─────────────────────────────────────────────────────────
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


def combined_corr_tbl(df, cols):
    """Single table: outcomes as top-level headers, raw / partial as sub-columns."""
    raw_r, raw_p, par_r, par_p = {}, {}, {}, {}
    for m in cols:
        for t in OUTCOME_COLS:
            v = df[[m, t]].dropna()
            raw_r[(m, t)], raw_p[(m, t)] = pearsonr(v[m], v[t])
            s = df[[m, t, "passes_against"]].dropna()
            a = _resid(s[m].values, s["passes_against"].values)
            b = _resid(s[t].values, s["passes_against"].values)
            mk = ~(np.isnan(a) | np.isnan(b))
            par_r[(m, t)], par_p[(m, t)] = pearsonr(a[mk], b[mk])

    def _fmt(r, p):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"{r:.2f}{sig}"

    mi_cols = pd.MultiIndex.from_tuples(
        [(t, kind) for t in OUTCOME_COLS for kind in ("raw", "partial")]
    )
    rows = [
        [_fmt(raw_r[(m, t)], raw_p[(m, t)]) if kind == "raw"
         else _fmt(par_r[(m, t)], par_p[(m, t)])
         for t in OUTCOME_COLS for kind in ("raw", "partial")]
        for m in cols
    ]
    disp = pd.DataFrame(rows, index=cols, columns=mi_cols)
    gmap = np.array([
        [raw_r[(m, t)] if kind == "raw" else par_r[(m, t)]
         for t in OUTCOME_COLS for kind in ("raw", "partial")]
        for m in cols
    ])
    st.dataframe(
        disp.style.background_gradient(cmap="RdYlGn", gmap=gmap, axis=None, vmin=-1, vmax=1),
        use_container_width=True,
    )


def corr_tbl(df, cols, partial=False):
    def _r(m, t):
        if not partial:
            return pearsonr(df[m].dropna(), df.loc[df[m].notna(), t])
        s = df[[m, t, "passes_against"]].dropna()
        a = _resid(s[m].values, s["passes_against"].values)
        b = _resid(s[t].values, s["passes_against"].values)
        mask = ~(np.isnan(a) | np.isnan(b))
        return pearsonr(a[mask], b[mask])
    r = {(m, t): _r(m, t) for m in cols for t in OUTCOME_COLS}
    to_df  = lambda i: pd.Series({k: v[i] for k, v in r.items()}).rename_axis(["metric", "outcome"]).unstack("outcome")
    rdf, pdf = to_df(0), to_df(1)
    disp = rdf.map(lambda v: f"{v: .2f}") + " (" + pdf.map(lambda v: f"{v:.3f}") + ")"
    st.dataframe(disp.style.background_gradient(cmap="RdYlGn", gmap=rdf.values, axis=None, vmin=-1, vmax=1))


def _bh_qvalues(pvals):
    """Benjamini-Hochberg FDR-adjusted q-values for a sequence of p-values (NaN-safe).
    NaNs stay NaN and don't count toward the test family m."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]                 # ascending p
    adj = p[order] * m / np.arange(1, m + 1)        # BH step-up
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity
    q[order] = np.minimum(adj, 1.0)
    return q


def compute_icc_rows(df, cols):
    """ICC(1,1) per column — teams = subjects, match-team rows = replicate measurements.
    Returns a list of dicts (one per usable column). Guards against the sparse,
    NaN-heavy columns produced by zone contrasts (teams with no within-team
    replication, zero within-team variance, degenerate denominators).
    `q` = Benjamini-Hochberg FDR-adjusted p across the supplied `cols` (the family is
    whatever the caller passes / displays); `sig` is based on `q`, not raw `p`."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[["team_name", c]].dropna()
        if s["team_name"].nunique() < 2:
            continue
        g   = s.groupby("team_name")[c]
        nt, ng, sz, mn = len(s), g.ngroups, g.count(), g.mean()
        if nt - ng < 1:                       # need ≥1 within-team replicate overall
            continue
        msb = (sz * (mn - s[c].mean()) ** 2).sum() / (ng - 1)
        msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0  = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        denom = msb + (k0 - 1) * msw
        if denom == 0:
            continue
        icc = (msb - msw) / denom
        # F-test for H0: ICC = 0 (MSB/MSW ~ F(ng-1, nt-ng))
        f_stat = msb / msw if msw > 0 else float("nan")
        p_val  = f_dist.sf(f_stat, ng - 1, nt - ng) if not np.isnan(f_stat) else float("nan")
        rows.append({
            "metric": c,
            "ICC": round(icc, 3),
            "F": round(f_stat, 2),
            "p": round(p_val, 4),
            "q": np.nan,                      # filled below (BH over the family)
            "sig": "",                        # set from q below
            "n_teams": ng,
            "n_obs": nt,
            "interpretation": "stable trait" if icc > 0.5 else "match-driven",
        })
    if rows:
        qs = _bh_qvalues([r["p"] for r in rows])
        for r, q in zip(rows, qs):
            r["q"] = round(float(q), 4) if not np.isnan(q) else np.nan
            r["sig"] = ("***" if q < 0.001 else "**" if q < 0.01
                        else "*" if q < 0.05 else "")
    return rows


def icc_tbl(df, cols):
    rows = compute_icc_rows(df, cols)
    if not rows:
        st.caption("Not enough replicated data for ICC on this group.")
        return
    st.dataframe(
        pd.DataFrame(rows).style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
        use_container_width=True,
    )


# ── Axis Selection & Quadrant Analysis ───────────────────────────────────────

def eta_sq_tbl(df, only_from=None, nperm=0, seed=20260627):
    """η² (between-team variance share) per metric.

    With nperm>0, add a **team-label permutation** test: η² is a *biased*
    variance-share estimator whose null is not 0 but ≈(k−1)/(n−1) (≈0.25 here),
    so a raw η² is uninterpretable on its own. The permutation rebuilds the null
    η² distribution by shuffling team labels and reports:
        floor = mean null η² (the chance level for this metric's n,k)
        p     = P(null η² ≥ observed)   [team identity ↑ separates teams]
    ss_tot is invariant under label permutation, so only ss_bet is recomputed —
    vectorised as (Yᵀ·Xperm)² / nₜ via an indicator matrix.
    """
    skip = {"match_team_id", "team_name", "competition_stage", "passes_against"} | set(OUTCOME_COLS)
    if only_from is not None:
        cols = [c for c in df.columns if c not in skip
                and any(c == p or c.startswith(p + "_") for p in only_from)]
    else:
        cols = [c for c in df.columns if c not in skip]
    rng = np.random.default_rng(seed)
    rows = []
    for m in cols:
        s = df[["team_name", m]].dropna()
        if s["team_name"].nunique() < 2:
            continue
        grand = s[m].mean()
        ss_tot = ((s[m] - grand) ** 2).sum()
        if ss_tot == 0:
            continue
        x = s[m].to_numpy(dtype=float)
        codes = pd.factorize(s["team_name"])[0]
        n, k = len(x), codes.max() + 1
        n_t = np.bincount(codes, minlength=k).astype(float)        # (k,)
        const = x.sum() ** 2 / n                                    # n·grand²
        ss_bet = (np.bincount(codes, weights=x, minlength=k) ** 2 / n_t).sum() - const
        rec = {"metric": m, "η²": round(ss_bet / ss_tot, 3), "n": n, "teams": int(k)}
        if nperm > 0:
            Y = np.zeros((n, k)); Y[np.arange(n), codes] = 1.0      # indicator (n,k)
            Xp = rng.permuted(np.broadcast_to(x, (nperm, n)), axis=1).T   # (n, nperm)
            GS = Y.T @ Xp                                           # (k, nperm) group sums
            eta_perm = ((GS ** 2 / n_t[:, None]).sum(0) - const) / ss_tot
            obs = ss_bet / ss_tot
            rec["floor"] = round(float(eta_perm.mean()), 3)
            rec["p"] = round((int((eta_perm >= obs).sum()) + 1) / (nperm + 1), 4)
            rec["sig"] = ("***" if rec["p"] < 0.001 else "**" if rec["p"] < 0.01
                          else "*" if rec["p"] < 0.05 else "ns")
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("η²", ascending=False).reset_index(drop=True)
    if nperm > 0 and len(out):
        # Benjamini-Hochberg FDR across the metric family (q), so the table is
        # honest about multiple comparisons — only small q is a defensible trait.
        order = out["p"].to_numpy().argsort()
        ranked = out["p"].to_numpy()[order]
        m = len(ranked)
        q = ranked * m / (np.arange(1, m + 1))
        q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
        qcol = np.empty(m); qcol[order] = q
        out["q"] = qcol.round(4)
        out["sig"] = np.where(out["q"] < 0.001, "***", np.where(out["q"] < 0.01, "**",
                              np.where(out["q"] < 0.05, "*", "ns")))
    return out


def quadrant_analysis(df, x_col, y_col, outcome_cols=None):
    if outcome_cols is None:
        outcome_cols = OUTCOME_COLS
    _has_stage = "competition_stage" in df.columns
    sel = [x_col, y_col, "team_name"] + outcome_cols + (["competition_stage"] if _has_stage else [])
    d = df[[c for c in sel if c in df.columns]].dropna(subset=[x_col, y_col]).copy()
    xm, ym = d[x_col].median(), d[y_col].median()
    d["quadrant"] = d.apply(
        lambda r: ("H" if r[x_col] >= xm else "L") + "X / " +
                  ("H" if r[y_col] >= ym else "L") + "Y",
        axis=1,
    )
    quads = sorted(d["quadrant"].unique())
    n_pairs = len(list(combinations(quads, 2)))

    rows = []
    for q in quads:
        dq = d[d["quadrant"] == q]
        row = {"quadrant": q, "n": len(dq)}
        for oc in outcome_cols:
            if oc in dq.columns:
                row[f"{oc} mean"] = round(dq[oc].mean(), 3)
                row[f"{oc} sd"]   = round(dq[oc].std(),  3)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("quadrant")

    kw_rows = []
    for oc in outcome_cols:
        if oc not in d.columns:
            continue
        groups = [g for g in
                  [d[d["quadrant"] == q][oc].dropna().values for q in quads]
                  if len(g) >= 2]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            kw_rows.append({"outcome": oc, "H": round(stat, 3), "p": round(p, 4),
                            "sig": "***" if p < 0.001 else "**" if p < 0.01
                                   else "*" if p < 0.05 else "ns"})

    mw_rows = []
    for oc in outcome_cols:
        if oc not in d.columns:
            continue
        for qa, qb in combinations(quads, 2):
            a = d[d["quadrant"] == qa][oc].dropna().values
            b = d[d["quadrant"] == qb][oc].dropna().values
            if len(a) >= 3 and len(b) >= 3:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                p_bonf = min(p * n_pairs, 1.0)
                mw_rows.append({
                    "outcome": oc, "Q1": qa, "Q2": qb,
                    "U": round(stat, 1), "p": round(p, 4), "p_bonf": round(p_bonf, 4),
                    "sig": "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01
                           else "*" if p_bonf < 0.05 else "ns",
                })

    if _has_stage:
        stage_cols = [s for s in STYLE_STAGE_ORDER if s in d["competition_stage"].values]
        stage_dist = (
            d.groupby(["quadrant", "competition_stage"]).size()
             .reset_index(name="n")
             .pivot(index="quadrant", columns="competition_stage", values="n")
             .reindex(columns=stage_cols, fill_value=0).fillna(0).astype(int)
        )
    else:
        stage_dist = pd.DataFrame()
    return d, summary, pd.DataFrame(kw_rows), (pd.DataFrame(mw_rows) if mw_rows else pd.DataFrame()), stage_dist


def _style_mw(mw: pd.DataFrame):
    def _row(r):
        if "p_bonf" in r.index and r["p_bonf"] < 0.05:
            return ["background-color: #c6efce"] * len(r)   # green — Bonferroni significant
        if "p" in r.index and r["p"] < 0.05:
            return ["background-color: #ffeb9c"] * len(r)   # yellow — raw significant only
        return [""] * len(r)
    return mw.style.apply(_row, axis=1)


def _style_kw(kw: pd.DataFrame):
    def _row(r):
        if "sig" in r.index and r["sig"] != "ns":
            return ["background-color: #c6efce"] * len(r)
        return [""] * len(r)
    return kw.style.apply(_row, axis=1)


def marginal_analysis(df, x_col, y_col, outcome_cols=None):
    """2-group MW tests: High vs Low on X alone, and High vs Low on Y alone."""
    if outcome_cols is None:
        outcome_cols = OUTCOME_COLS
    cols = [x_col, y_col, "team_name"] + [c for c in outcome_cols if c in df.columns]
    d = df[[c for c in cols if c in df.columns]].dropna(subset=[x_col, y_col]).copy()
    xm, ym = d[x_col].median(), d[y_col].median()
    d["grp_x"] = d[x_col].apply(lambda v: "High X" if v >= xm else "Low X")
    d["grp_y"] = d[y_col].apply(lambda v: "High Y" if v >= ym else "Low Y")

    rows = []
    for axis, grp_col, desc in [("X", "grp_x", x_col), ("Y", "grp_y", y_col)]:
        a_label, b_label = (f"High {axis}", f"Low {axis}")
        for oc in outcome_cols:
            if oc not in d.columns:
                continue
            a = d[d[grp_col] == a_label][oc].dropna().values
            b = d[d[grp_col] == b_label][oc].dropna().values
            if len(a) >= 3 and len(b) >= 3:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                rows.append({
                    "axis": f"{axis} = {desc}",
                    "comparison": f"{a_label} vs {b_label}",
                    "outcome": oc,
                    f"mean {a_label}": round(a.mean(), 3),
                    f"mean {b_label}": round(b.mean(), 3),
                    "U": round(stat, 1), "p": round(p, 4),
                    "sig": "***" if p < 0.001 else "**" if p < 0.01
                           else "*" if p < 0.05 else "ns",
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── OLS Regression ───────────────────────────────────────────────────────────
def run_ols(df, x_cols, y_col):
    """OLS with β, standardised β, SE, t, p, 95% CI. Returns (coef_df, r2, r2_adj, n)."""
    d = df[x_cols + [y_col]].dropna()
    n, k = len(d), len(x_cols)
    if n <= k + 1:
        return None
    X = np.column_stack([np.ones(n), d[x_cols].values])
    y = d[y_col].values
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta   = XtX_inv @ X.T @ y
    resid  = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    sigma2 = ss_res / (n - k - 1)
    se     = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stat = beta / se
    p_val  = 2 * t_dist.sf(np.abs(t_stat), df=n - k - 1)
    ci_lo  = beta - t_dist.ppf(0.975, df=n - k - 1) * se
    ci_hi  = beta + t_dist.ppf(0.975, df=n - k - 1) * se
    y_sd   = d[y_col].std()
    std_b  = np.concatenate([
        [np.nan],
        beta[1:] * d[x_cols].std().values / y_sd if y_sd > 0 else np.full(k, np.nan)
    ])
    coef = pd.DataFrame({
        "β":        np.round(beta,   4),
        "std β":    np.round(std_b,  4),
        "SE":       np.round(se,     4),
        "t":        np.round(t_stat, 3),
        "p":        np.round(p_val,  4),
        "CI 2.5%":  np.round(ci_lo,  4),
        "CI 97.5%": np.round(ci_hi,  4),
        "sig": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                for p in p_val],
    }, index=["(Intercept)"] + x_cols)
    return coef, r2, r2_adj, n


# ── Zones (pitch-area defensive metrics) ───────────────────────────────────────
ZONE_ORDER   = ["own", "mid", "high_press"]              # near own goal -> high press
ZONE_LABEL   = {"own": "Own-third (forced)", "mid": "Midfield", "high_press": "High press",
                "all": "Whole network (all zones)"}
ZONE_SHORT   = {"own": "Own", "mid": "Mid", "high_press": "High press", "all": "Whole network"}
SCHEME_LABEL = {"thirds": "Pitch thirds (±17.5)",
                "scheme_4060": "40 / 60 split (±10.5)",
                "half": "Half — own vs opponent (0)"}
# zone weight -> (label, value col, contribution col, fault col).  con + fault == total.
WEIGHT_SPLIT = {
    "n_actions":          ("Defended passes",  "n_actions",
                           "raw_contribution_npass",  "raw_fault_npass"),
    "raw_involvement":    ("Raw involvement",  "raw_involvement_sum",
                           "raw_contribution_sum",    "raw_fault_sum"),
    "valued_involvement": ("Valued involvement", "valued_involvement_sum",
                           "valued_contribution_sum", "valued_fault_sum"),
}
# medium hue per zone (con = solid, fault = hatched, same colour)
ZONE_COLOR = {
    "own":        "#5e9eca",   # blue
    "mid":        "#71bc63",   # green
    "high_press": "#ec5c5d",   # red
}
# metrics whose per-zone SUMS we correlate with outcomes (raw + valued)
ZONE_CORR_METRICS = ["raw_involvement", "valued_involvement",
                     "raw_contribution", "valued_contribution",
                     "raw_fault", "valued_fault"]
METRIC_LABEL = {"raw_involvement": "raw inv",  "valued_involvement": "val inv",
                "raw_contribution": "raw con", "valued_contribution": "val con",
                "raw_fault": "raw fault",      "valued_fault": "val fault"}
def build_zone_corr(zone_df, scheme, outcomes_df, outcome_cols,
                    partial=False, control="total"):
    """Correlation between each (zone, metric) SUM and each outcome, match-team level.

    Each match-team is one observation; a zone with no actions counts as 0.
    partial: control='total' residualises on the match's total passes_against;
    control='zone' residualises on that zone's own faced-pass count (n_passes).
    Returns long df: metric, zone, outcome, r, p, n.
    """
    d = zone_df[zone_df["scheme"] == scheme].copy()
    o = outcomes_df.set_index("match_team_id")
    npass = (d.pivot_table(index="match_team_id", columns="zone", values="n_passes",
                           aggfunc="sum", fill_value=0)
             if partial and control == "zone" else None)
    rows = []
    for m in ZONE_CORR_METRICS:
        wide = d.pivot_table(index="match_team_id", columns="zone",
                             values=f"{m}_sum", aggfunc="sum", fill_value=0)
        for z in [zz for zz in ZONE_ORDER if zz in wide.columns]:
            base = pd.concat([wide[z].rename("x"), o[outcome_cols]], axis=1)
            if partial:
                ctrl = o["passes_against"] if control == "total" else npass[z]
                base = base.join(ctrl.rename("ctrl"))
            for oc in outcome_cols:
                if partial:
                    s = base[["x", oc, "ctrl"]].dropna()
                    if len(s) <= 3 or s["x"].std() == 0:
                        continue
                    a = _resid(s["x"].values, s["ctrl"].values)
                    b = _resid(s[oc].values, s["ctrl"].values)
                    mk = ~(np.isnan(a) | np.isnan(b))
                    if a[mk].std() == 0 or b[mk].std() == 0:
                        continue
                    r, p = pearsonr(a[mk], b[mk]); n = int(mk.sum())
                else:
                    s = base[["x", oc]].dropna()
                    if len(s) <= 3 or s["x"].std() == 0:
                        continue
                    r, p = pearsonr(s["x"], s[oc]); n = len(s)
                rows.append(dict(metric=m, zone=z, outcome=oc, r=r, p=p, n=n))
    return pd.DataFrame(rows)


def build_zone_ratio_corr(zone_df, scheme, outcomes_df, outcome_cols,
                          partial=False, control="passes_against"):
    """Correlation between per-zone contribution/fault RATIO and outcomes, match-team level.

    Ratio = contribution_sum / fault_sum per match-team-zone (raw and valued). Match-teams
    with zero fault in a zone are dropped (undefined ratio). partial=True controls for
    `control`. Returns long df: ratio, zone, outcome, r, p, n.
    """
    d = zone_df[zone_df["scheme"] == scheme]
    cols = outcome_cols + ([control] if partial else [])
    o = outcomes_df.set_index("match_team_id")[cols]
    rows = []
    for kind, ccol, fcol in [("raw", "raw_contribution_sum", "raw_fault_sum"),
                             ("valued", "valued_contribution_sum", "valued_fault_sum")]:
        con = d.pivot_table(index="match_team_id", columns="zone", values=ccol,
                            aggfunc="sum", fill_value=0)
        fau = d.pivot_table(index="match_team_id", columns="zone", values=fcol,
                            aggfunc="sum", fill_value=0)
        for z in [zz for zz in ZONE_ORDER if zz in con.columns]:
            j = pd.concat([(con[z] / fau[z].replace(0, np.nan)).rename("x"), o], axis=1)
            if partial:
                s = j[["x"] + cols].dropna()
                if len(s) <= 3 or s["x"].std() == 0:
                    continue
                for oc in outcome_cols:
                    a = _resid(s["x"].values, s[control].values)
                    b = _resid(s[oc].values, s[control].values)
                    mk = ~(np.isnan(a) | np.isnan(b))
                    if a[mk].std() == 0 or b[mk].std() == 0:
                        continue
                    r, p = pearsonr(a[mk], b[mk])
                    rows.append(dict(ratio=kind, zone=z, outcome=oc, r=r, p=p, n=int(mk.sum())))
            else:
                for oc in outcome_cols:
                    s = j[["x", oc]].dropna()
                    if len(s) <= 3 or s["x"].std() == 0:
                        continue
                    r, p = pearsonr(s["x"], s[oc])
                    rows.append(dict(ratio=kind, zone=z, outcome=oc, r=r, p=p, n=len(s)))
    return pd.DataFrame(rows)


def build_zone_volume_diag(zone_df, scheme, outcomes_df, outcome, kind="raw"):
    """Per-zone diagnostic (match-team level): how much each zone's metric SUM is just
    volume (n_passes), whether that volume itself relates to the outcome, and the
    per-faced-pass intensity vs outcome (volume removed)."""
    d = zone_df[zone_df["scheme"] == scheme].merge(
        outcomes_df[["match_team_id", outcome]], on="match_team_id")

    def _r(a, b):
        s = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        return (pearsonr(s["a"], s["b"])[0]
                if len(s) > 2 and s["a"].std() > 0 and s["b"].std() > 0 else np.nan)

    oc = outcome.replace("_against", " ag.")
    rows = []
    for z in [zz for zz in ZONE_ORDER if zz in set(d["zone"])]:
        g = d[(d["zone"] == z) & (d["n_passes"] > 0)]
        npass = g["n_passes"]
        rows.append({"zone": ZONE_LABEL[z],
                     "npass↔inv":  _r(npass, g[f"{kind}_involvement_sum"]),
                     "npass↔con":  _r(npass, g[f"{kind}_contribution_sum"]),
                     "npass↔fault": _r(npass, g[f"{kind}_fault_sum"]),
                     f"npass↔{oc}": _r(npass, g[outcome]),
                     f"invPerPass↔{oc}": _r(g[f"{kind}_involvement_sum"] / npass, g[outcome])})
    return pd.DataFrame(rows).set_index("zone").round(2)


def build_team_style(zone_df, scheme, outcomes_df, kind="raw"):
    """Team-level defensive-style profile + performance (pooled across a team's matches).

    Zone proportions are computed for involvement, contribution AND fault (each = that
    zone's sum ÷ the team's total of that metric across zones). Joined with mean outcomes
    & knockout. Proportion columns: '{zone}_share' (inv), 'con_{zone}_share', 'fault_{zone}_share'.
    """
    d = zone_df[zone_df["scheme"] == scheme].merge(
        outcomes_df[["match_team_id", "team_name"]], on="match_team_id")
    t = None
    for metric, pre in [("involvement", ""), ("contribution", "con_"), ("fault", "fault_")]:
        col = f"{kind}_{metric}_sum"
        m = (d.groupby(["team_name", "zone"], observed=True)[col].sum()
               .unstack().fillna(0))
        sh = m.div(m.sum(axis=1).replace(0, np.nan), axis=0)
        if t is None:
            t = pd.DataFrame(index=sh.index)
        for z in ZONE_ORDER:
            if z in sh.columns:
                t[f"{pre}{z}_share"] = sh[z]
    # team-level faced-pass totals per zone (control variable for zone-pass partial)
    npz = (d.groupby(["team_name", "zone"], observed=True)["n_passes"].sum()
             .unstack().fillna(0))
    for z in ZONE_ORDER:
        if z in npz.columns:
            t[f"npass_{z}"] = npz[z]
    perf = outcomes_df.groupby("team_name").agg(
        shots_against=("shots_against", "mean"), xg_against=("xg_against", "mean"),
        goals_against=("goals_against", "mean"),
        passes_against=("passes_against", "mean"),
        reached_knockout=("reached_knockout", "first"))
    return t.join(perf)


def build_team_zone_ratio(zone_df, scheme, kind, mode):
    """Team × zone contribution/fault ratio matrix (unweighted mean of per-match ratios).

    kind: 'raw' or 'valued'.
    mode: 'plain'     -> contribution_sum / fault_sum  per match, then mean over matches.
          'per_event' -> (contribution_sum/contribution_npass) / (fault_sum/fault_npass),
                         i.e. mean contribution intensity ÷ mean fault intensity per match,
                         then mean over matches.
    Matches with an undefined ratio in a zone (zero denominator) are dropped.
    """
    ccol, fcol = f"{kind}_contribution_sum", f"{kind}_fault_sum"
    d = zone_df[zone_df["scheme"] == scheme].copy()
    if mode == "per_event":
        cnp, fnp = f"{kind}_contribution_npass", f"{kind}_fault_npass"
        con = d[ccol] / d[cnp].replace(0, np.nan)
        fau = d[fcol] / d[fnp].replace(0, np.nan)
        d["ratio"] = con / fau.replace(0, np.nan)
    else:
        d["ratio"] = d[ccol] / d[fcol].replace(0, np.nan)
    agg = (d.groupby(["defending_team_name", "zone"], observed=True)["ratio"]
             .mean().reset_index())
    zones = [z for z in ZONE_ORDER if z in set(agg["zone"])]
    mat = (agg.pivot(index="defending_team_name", columns="zone", values="ratio")
              .reindex(columns=zones))
    return mat, zones


def build_team_zone_volume_ratio(zone_df, scheme, outcomes_df, kind):
    """Per team × zone: involvement volume (total & per-match), pooled con/fault ratio,
    and the team's mean-per-match outcomes (for colour). Aligned via match_team_id to
    avoid team-name mismatches. Returns long df: team, zone, inv, inv_pm, ratio, outcomes.
    """
    ic, cc, fc = f"{kind}_involvement_sum", f"{kind}_contribution_sum", f"{kind}_fault_sum"
    d = zone_df[zone_df["scheme"] == scheme]
    g = (d.groupby(["defending_team_name", "zone"], observed=True)
           .agg(inv=(ic, "sum"), con=(cc, "sum"), fault=(fc, "sum")).reset_index())
    g["ratio"] = g["con"] / g["fault"].replace(0, np.nan)
    # matches played + mean-per-match outcomes per team (aligned by match_team_id)
    om = (d[["match_team_id", "defending_team_name"]].drop_duplicates()
          .merge(outcomes_df.set_index("match_team_id")[OUTCOME_COLS],
                 left_on="match_team_id", right_index=True, how="left"))
    nm = om.groupby("defending_team_name")["match_team_id"].nunique().rename("n_matches")
    perf = om.groupby("defending_team_name")[OUTCOME_COLS].mean()
    g = g.merge(nm, on="defending_team_name").merge(perf, on="defending_team_name", how="left")
    g["inv_pm"] = g["inv"] / g["n_matches"]
    return g


def build_zone_split(zone_df, scheme, weight):
    """Team × zone × {contribution, fault} long table, pooled across a team's matches.

    `weight` selects the measure (defended passes / raw inv / valued inv); it is split
    into its contribution and fault parts (which sum to the whole). `share` is each
    part's fraction of the team's total across all zones (one bar per team sums to 1).
    """
    _, _, con_col, fault_col = WEIGHT_SPLIT[weight]
    d = zone_df[zone_df["scheme"] == scheme]
    agg = (d.groupby(["defending_team_name", "zone"], observed=True)
             .agg(contribution=(con_col, "sum"), fault=(fault_col, "sum"))
             .reset_index())
    long = agg.melt(id_vars=["defending_team_name", "zone"],
                    value_vars=["contribution", "fault"],
                    var_name="type", value_name="value")
    team_total = long.groupby("defending_team_name")["value"].transform("sum")
    long["share"] = long["value"] / team_total.replace(0, np.nan)
    zones = [z for z in ZONE_ORDER if z in set(long["zone"])]
    return long, zones


# ── Zone topology (per-zone network structure) ─────────────────────────────────
# label -> column suffix on a weight col (empty suffix = the bare strength sum).
TOPO_METRICS = {
    "Total strength":               "",
    "Density":                      "_density",
    "Gini (strength inequality)":   "_gini",
    "Clustering (unweighted)":      "_cc_unweighted",
    "Clustering (weighted)":        "_cc_weighted",
    "Centralization (unweighted)":  "_centralization",
    "Centralization (weighted)":    "_centralization_weighted",
    "Degree assortativity":         "_assortativity",
    "Max k-core":                   "_kcore_max",
    "LCC ratio":                    "_lcc_ratio",
}
# higher value means... (for reading the descriptive plot)
TOPO_HINT = {
    "_density": "more squad pairs co-defend together",
    "_gini": "load concentrated on fewer players",
    "_cc_unweighted": "tighter local triangles",
    "_cc_weighted": "tighter, stronger local triangles",
    "_centralization": "one hub dominates connectivity",
    "_centralization_weighted": "one hub dominates the load",
    "_assortativity": "hubs pair with hubs",
    "_kcore_max": "a denser mutually-connected core",
    "_lcc_ratio": "defense acts as one connected group",
    "": "more total co-defensive activity",
}
# compact row labels for the all-metrics overview grid
TOPO_SHORT = {
    "Total strength": "strength", "Density": "density",
    "Gini (strength inequality)": "gini", "Clustering (unweighted)": "cc",
    "Clustering (weighted)": "cc(w)", "Centralization (unweighted)": "centr",
    "Centralization (weighted)": "centr(w)", "Degree assortativity": "assort",
    "Max k-core": "kcore", "LCC ratio": "lcc",
}


def build_zone_topo_corr(topo_df, outcomes_df, outcome_cols, suffix, partial=False,
                         control="total", zone_npass=None):
    """Correlate a per-zone topology metric (one column per weight metric) with each
    outcome, at match-team level. One observation per match-team that has a value in
    the zone. partial=True residualises both sides on a control variable (mirrors the
    Zones tab): control='total' uses the match's total passes_against; control='zone'
    uses that zone's own faced-pass count (zone_npass: match_team_id × zone pivot).
    Returns long df: metric (weight col), zone, outcome, r, p, n.
    """
    o = outcomes_df.set_index("match_team_id")
    rows = []
    for z in ZONE_ORDER:
        dz = topo_df[topo_df["zone"] == z].set_index("match_team_id")
        if dz.empty:
            continue
        for c in WEIGHT_COLS:
            col = c + suffix if suffix else c
            if col not in dz.columns:
                continue
            base = pd.concat([dz[col].rename("x"), o[outcome_cols]], axis=1)
            if partial:
                if control == "zone":
                    if zone_npass is None or z not in zone_npass.columns:
                        continue
                    ctrl = zone_npass[z]
                else:
                    ctrl = o["passes_against"]
                base = base.join(ctrl.rename("ctrl"))
            for oc in outcome_cols:
                if partial:
                    s = base[["x", oc, "ctrl"]].dropna()
                    if len(s) <= 3 or s["x"].std() == 0:
                        continue
                    a = _resid(s["x"].values, s["ctrl"].values)
                    b = _resid(s[oc].values, s["ctrl"].values)
                    mk = ~(np.isnan(a) | np.isnan(b))
                    if a[mk].std() == 0 or b[mk].std() == 0:
                        continue
                    r, p = pearsonr(a[mk], b[mk]); n = int(mk.sum())
                else:
                    s = base[["x", oc]].dropna()
                    if len(s) <= 3 or s["x"].std() == 0:
                        continue
                    r, p = pearsonr(s["x"], s[oc]); n = len(s)
                rows.append(dict(metric=c, zone=z, outcome=oc, r=r, p=p, n=n))
    return pd.DataFrame(rows)


def build_zone_topo_corr_all(topo_df, outcomes_df, outcome_cols, partial=False,
                             control="total", zone_npass=None):
    """Every (topology metric × weight metric × zone × outcome) correlation in one
    long df. Adds a `topo` column (topology-metric label) on top of build_zone_topo_corr."""
    frames = []
    for label, suffix in TOPO_METRICS.items():
        d = build_zone_topo_corr(topo_df, outcomes_df, outcome_cols, suffix, partial=partial,
                                 control=control, zone_npass=zone_npass)
        if not d.empty:
            d.insert(0, "topo", label)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _icc_pval(df, col):
    """ICC(1,1) + F-test p for one column (teams = subjects, matches = replicates).
    Returns (icc, p, n_obs) or None if not estimable."""
    s = df[["team_name", col]].dropna()
    if s["team_name"].nunique() < 3:
        return None
    g = s.groupby("team_name")[col]
    nt, ng, sz, mn = len(s), g.ngroups, g.count(), g.mean()
    if nt - ng < 1:
        return None
    msb = (sz * (mn - s[col].mean()) ** 2).sum() / (ng - 1)
    msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
    k0  = (nt - (sz ** 2).sum() / nt) / (ng - 1)
    den = msb + (k0 - 1) * msw
    if den == 0 or msw <= 0:
        return None
    icc = (msb - msw) / den
    return icc, float(f_dist.sf(msb / msw, ng - 1, nt - ng)), nt


def build_zone_topo_icc(topo_df):
    """ICC(1,1) of every per-zone topology metric **level** across teams — the test of
    whether a team's zonal co-defending *structure* is a repeatable trait or match-driven
    noise. Long df: topo (label), metric (weight), zone, ICC, p, n. A Benjamini-Hochberg
    `q` is added across the whole returned family (all metrics × all zones)."""
    df = topo_df.copy()
    df["team_name"] = team_names.reindex(df["match_team_id"]).values
    rows = []
    for z in ZONE_ORDER:
        dz = df[df["zone"] == z]
        if dz.empty:
            continue
        for label, suffix in TOPO_METRICS.items():
            for w in WEIGHT_COLS:
                col = w + suffix if suffix else w
                if col not in dz.columns:
                    continue
                r = _icc_pval(dz, col)
                if r is not None:
                    rows.append({"topo": label, "metric": w, "zone": z,
                                 "ICC": r[0], "p": r[1], "n": r[2]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = _bh_qvalues(out["p"].values)
    return out


# ── Spatial / spatially-embedded-network predictors of zone defensive success ───
# Each metric maps to a visual feature of the co-defending network drawn on the pitch.
SPATIAL_VIS = {
    "x_range":      "depth span (deepest→highest)",
    "spread_x":     "vertical spread (depth)",
    "spread_y":     "horizontal spread (width)",
    "block_spread": "node-cloud size",
    "hull_area":    "enclosing-polygon area",
    "nn_dist":      "nearest-neighbour gap",
    "aspect_xy":    "tall vs wide shape",
    "lateral_off":  "shift to a flank",
    "edge_cv":                     "edge unevenness — all co-defending",
    "raw_involvement_edge_cv":     "edge unevenness — raw involvement",
    "raw_fault_edge_cv":           "edge unevenness — raw fault",
    "raw_contribution_edge_cv":    "edge unevenness — raw contribution",
    "valued_involvement_edge_cv":  "edge unevenness — valued involvement",
    "valued_contribution_edge_cv": "edge unevenness — valued contribution",
    "valued_fault_edge_cv":        "edge unevenness — valued fault",
}


def build_zone_spatial_corr(sp_df, zone_raw_df):
    """Pearson r of each spatial / edge predictor with per-zone defensive **stop
    rate** (n_stop_def/n_actions), at match-team level, per zone. Stop rate is a
    consistent-denominator defensive-success target: of the passes the defence
    engaged (raw_involvement>0), the share it kept from completing — higher = better
    defending. (Supersedes the old n_success/n_actions, which mixed a completed-pass
    numerator over *all* passes with a defended-pass denominator and was unbounded.)
    BH-FDR q across all (metric × zone) cells. Long df: metric, zone, r, p, n, q."""
    zr = zone_raw_df[zone_raw_df["scheme"] == "thirds"].copy()
    zr["succ_rate"] = zr["n_stop_def"] / zr["n_actions"]
    tgt = zr[["match_team_id", "zone", "succ_rate"]]
    # whole-match ("all") stop rate = pooled Σstop / Σactions across the thirds
    allt = (zr.groupby("match_team_id")[["n_stop_def", "n_actions"]].sum()
              .assign(succ_rate=lambda d: d["n_stop_def"] / d["n_actions"], zone="all")
              .reset_index()[["match_team_id", "zone", "succ_rate"]])
    tgt = pd.concat([tgt, allt], ignore_index=True)
    m = sp_df.merge(tgt, on=["match_team_id", "zone"], how="left")
    skip = {"match_team_id", "match_id", "defending_team", "zone"}
    metrics = [c for c in sp_df.columns if c not in skip]
    rows = []
    for z in ZONE_ORDER + ["all"]:
        dz = m[m["zone"] == z]
        for v in metrics:
            s = dz[[v, "succ_rate"]].dropna()
            if len(s) >= 10 and s[v].std() > 0 and s["succ_rate"].std() > 0:
                r, p = pearsonr(s[v], s["succ_rate"])
                rows.append({"metric": v, "zone": z, "r": r, "p": p, "n": len(s)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = _bh_qvalues(out["p"].values)
    return out


# ── Zone contrasts (within match-team differences across zones) ─────────────────
# Press-height coding for the slope contrast: own third (deep) → high press.
ZONE_DEPTH = {"own": -1.0, "mid": 0.0, "high_press": 1.0}
# Pairwise simple deltas (value = hi_zone − lo_zone).
ZONE_DELTAS = {
    "Δ High press − Own": ("high_press", "own"),
    "Δ Mid − Own":        ("mid", "own"),
    "Δ High press − Mid": ("high_press", "mid"),
}


def _resid_series(y, x):
    """Residualise Series y on Series x (linear fit), aligned on y's index; NaN-safe.
    Returns NaN where either is missing or x has no spread."""
    y = y.astype(float)
    x = x.astype(float).reindex(y.index)
    out = pd.Series(np.nan, index=y.index)
    m = y.notna() & x.notna()
    if m.sum() >= 4 and x[m].std() > 0:
        out[m] = y[m].values - np.polyval(np.polyfit(x[m].values, y[m].values, 1), x[m].values)
    return out


def _zone_pivot(topo_df, col, correct, zone_npass):
    """match_team_id × zone table of one metric, optionally residualised within each
    zone on that zone's faced-pass count (so the contrast reflects structure, not how
    much the ball sat in that zone)."""
    piv = topo_df.pivot_table(index="match_team_id", columns="zone", values=col, aggfunc="first")
    if correct and zone_npass is not None:
        for z in list(piv.columns):
            if z in zone_npass.columns:
                piv[z] = _resid_series(piv[z], zone_npass[z])
    return piv


def _row_slope(row):
    """OLS slope of metric over press depth, using whatever zones are present (≥2)."""
    xs, ys = [], []
    for z, d in ZONE_DEPTH.items():
        if z in row.index and pd.notna(row[z]):
            xs.append(d); ys.append(float(row[z]))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else np.nan


def build_zone_contrasts(topo_df, contrast, correct=False, zone_npass=None):
    """Wide df: one row per match-team, one column per (weight × topology) named exactly
    like GROUPS (e.g. 'raw_involvement_centralization_weighted'), holding the requested
    within-team zone contrast. Differencing within a match-team cancels squad size and
    overall style, so the contrast isolates how structure *changes* across the pitch.
    `contrast` is a ZONE_DELTAS key or 'slope'."""
    data = {}
    for suffix in TOPO_METRICS.values():
        for w in WEIGHT_COLS:
            col = w + suffix if suffix else w
            if col not in topo_df.columns:
                continue
            piv = _zone_pivot(topo_df, col, correct, zone_npass)
            if contrast == "slope":
                data[col] = piv.apply(_row_slope, axis=1)
            else:
                hi, lo = ZONE_DELTAS[contrast]
                if hi in piv.columns and lo in piv.columns:
                    data[col] = piv[hi] - piv[lo]
    cdf = pd.DataFrame(data)
    cdf.insert(0, "team_name", team_names.reindex(cdf.index).values)
    return cdf.reset_index()


def _gate_sparse_zones(topo_df, zone_npass, min_pass):
    """Drop (match_team_id, zone) rows whose zone faced fewer than `min_pass` passes,
    so degenerate near-empty zone graphs don't enter the contrasts. No-op at min_pass=0
    or without a faced-pass table."""
    if min_pass <= 0 or zone_npass is None:
        return topo_df
    long = zone_npass.stack().rename("n_passes").reset_index()   # match_team_id, zone, n_passes
    m = topo_df.merge(long, on=["match_team_id", "zone"], how="left")
    return m[m["n_passes"].fillna(0) >= min_pass].drop(columns="n_passes")


def lme_partial_pool(df, col):
    """Partial-pool a per-match-team contrast across teams with a random-intercept mixed
    model (`col ~ 1 + (1|team_name)`, REML). This is the (B) answer to "teams don't play
    the same formation each match": match-to-match variation (formation, personnel,
    opponent) is absorbed into the within-team residual, and each team's estimate is
    shrunk toward the league mean in proportion to how noisy/few its matches are — no
    node correspondence across matches required. Returns a dict or None.
      icc_lme : σ²_team / (σ²_team + σ²_resid) — REML; handles the unbalanced 3–7
                matches/team design more precisely than the ANOVA ICC(1,1)
      fe_mean : fixed intercept = league-average contrast (the typical gradient)
      blups   : team_name -> partial-pooled (shrunk) contrast = fe_mean + random effect
    A singular RE covariance means no detectable between-team variance: icc_lme≈0 and
    every team shrinks fully to fe_mean. Teams need ≥2 contrast observations to inform
    the within-team variance."""
    if not _HAS_SM:
        return None
    d = df[["team_name", col]].dropna().rename(columns={col: "y"})
    cnt = d.groupby("team_name")["y"].count()
    d = d[d["team_name"].isin(cnt[cnt >= 2].index)]
    if d["team_name"].nunique() < 3 or len(d) < 6 or d["y"].std() == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _smf.mixedlm("y ~ 1", d, groups=d["team_name"]).fit(reml=True, method="lbfgs")
    except Exception:
        return None
    s_b, s_e = float(res.cov_re.iloc[0, 0]), float(res.scale)
    if s_b + s_e == 0:
        return None
    fe = float(res.fe_params["Intercept"])
    try:
        blups = {t: fe + float(v.iloc[0]) for t, v in res.random_effects.items()}
    except Exception:                       # singular RE cov => full shrinkage to mean
        blups = {t: fe for t in d["team_name"].unique()}
    return {"icc_lme": s_b / (s_b + s_e), "fe_mean": fe, "blups": blups,
            "n_obs": len(d), "n_teams": d["team_name"].nunique()}


@st.cache_data(show_spinner="Fitting partial-pooling models…")
def partial_pool_all(cdf, cols, key):
    """All metrics' partial-pool fits, cached on (cdf, cols, key) so changing only the
    inspected metric in the UI doesn't refit the whole set."""
    return {c: lme_partial_pool(cdf, c) for c in cols}


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Defensive Network Analysis — Team Level")

with st.sidebar:
    method  = st.selectbox("Edge weight method", list(edge_dfs))
    st.subheader("Concentrated vs Balanced")
    metric_conc_inv   = st.selectbox("Involvement", INV_COLS, key="metric_conc_inv")
    metric_conc_fault = st.selectbox("Fault", FAULT_COLS, key="metric_conc_fault")
    metric_conc_cont  = st.selectbox("Contribution", CONTRIBUTION_COLS, key="metric_conc_cont")
    _y_opts = {"centralization_w": "Centralization (weighted)", "kcore_max": "Max K-core",
               "network_density": "Network Density"}
    y_conc = st.selectbox("Y axis", list(_y_opts), format_func=_y_opts.__getitem__, key="y_conc")
    st.subheader("Self vs Shared")
    metric_self  = st.selectbox("Metric", WEIGHT_COLS, key="metric_self")
    outcome_col  = st.selectbox("Outcome metric", OUTCOME_COLS)
    if GS_AVAILABLE:
        _gs_opts = {"Binary (solo = self, multi = shared)": False, "Gini-Simpson (continuous)": True}
        use_gs = _gs_opts[st.selectbox("Self/Shared method", list(_gs_opts))]
    else:
        use_gs = False
        st.caption("Gini-Simpson not available — regenerate node CSV.")
    correct_possession = st.checkbox("Correct for passes against")
    st.subheader("Defensive Style")
    _norm_opts   = ["raw", "per_90", "per_pass_against"]
    _norm_labels = {"raw": "Raw", "per_90": "Per 90 min", "per_pass_against": "Per pass against"}
    x_col       = st.selectbox("X axis (contribution)", CONTRIBUTION_COLS)
    x_normalize = st.selectbox("X normalization", _norm_opts,
                               format_func=_norm_labels.__getitem__, key="x_norm")
    y_col       = st.selectbox("Y axis (fault)", FAULT_COLS)
    y_normalize = st.selectbox("Y normalization", _norm_opts,
                               format_func=_norm_labels.__getitem__, key="y_norm")
    size_col    = st.selectbox("Bubble size", OUTCOME_COLS, key="style_size")
    size_scale  = st.slider("Bubble size scale", 10, 80, 40)
    st.subheader("Correlation / ICC")
    thr = st.slider("Edge count threshold (≥)", 1, 20, 1)
    st.subheader("Zones (pitch area)")
    zone_scheme = st.selectbox("Zoning scheme", list(SCHEME_LABEL),
                               format_func=SCHEME_LABEL.__getitem__, key="zone_scheme")
    zone_weight = st.selectbox("Zone weight", list(WEIGHT_SPLIT),
                               format_func=lambda c: WEIGHT_SPLIT[c][0], key="zone_weight")

df_conc_inv_match   = build_conc_match(edge_dfs[method], metric_conc_inv,   thr)
df_conc_inv_team    = build_conc_team(df_conc_inv_match)
df_conc_fault_match = build_conc_match(edge_dfs[method], metric_conc_fault, thr)
df_conc_fault_team  = build_conc_team(df_conc_fault_match)
df_conc_cont_match  = build_conc_match(edge_dfs[method], metric_conc_cont,  thr)
df_conc_cont_team   = build_conc_team(df_conc_cont_match)
df_self_match       = build_selfshared_match(edge_dfs[method], metric_self, use_gs=use_gs)
df_self_team        = build_selfshared_team(df_self_match)
df_style_match, df_style_team = build_style_team(edge_dfs[method], x_col, y_col, size_col, x_normalize, y_normalize)
avg_co, avg_co_team = build_co_defender_data()
partnerships        = build_partnerships()
df_corr             = process(edge_dfs[method], thr)

(tab_conc, tab_self, tab_style, tab_codef, tab_zone, tab_ztopo, tab_zcon, tab_pstyle,
 tab_corr, tab_icc, tab_reg, tab_sens, tab_data) = st.tabs([
    "Concentrated vs Balanced", "Self vs Shared", "Defensive Style", "Co-Defenders", "Zones",
    "Zone Topology", "Zone Contrasts (ICC)", "Pressing Style",
    "Correlation", "Robustness (ICC)", "Regression", "Sensitivity", "Data",
])

_QUAD_EXPLAIN = {
    "centralization_w": {
        "HX / HY": ("High X", "Centralized around a few players",
                    "Dependent defense — high workload concentrated in key defenders"),
        "HX / LY": ("High X", "Distributed across squad",
                    "Collective defense — everyone participates heavily"),
        "LX / HY": ("Low X",  "Centralized around a few players",
                    "Passive + reliant — defense falls on a small number of players"),
        "LX / LY": ("Low X",  "Distributed across squad",
                    "Passive but organized — light workload spread evenly"),
    },
    "kcore_max": {
        "HX / HY": ("High X", "Dense defensive nucleus (high k-core)",
                    "High-volume defense with a tight interconnected core group"),
        "HX / LY": ("High X", "Sparse/no tight nucleus (low k-core)",
                    "High overall volume but fragmented — no coherent defensive core"),
        "LX / HY": ("Low X",  "Dense defensive nucleus (high k-core)",
                    "Tight but small core, limited overall defensive coverage"),
        "LX / LY": ("Low X",  "Sparse/no tight nucleus (low k-core)",
                    "Passive and fragmented — no coherent defensive structure"),
    },
}

def _render_conc_group(match_df, team_df, group_label, y_conc, y_label_conc, outcome_col):
    x_noun = group_label.lower()
    if y_conc in _QUAD_EXPLAIN:
        rows_expl = [
            {"Quadrant": q, f"X ({group_label} strength)": v[0],
             f"Y ({y_label_conc})": v[1], "Style interpretation": v[2]}
            for q, v in _QUAD_EXPLAIN[y_conc].items()
        ]
        st.dataframe(pd.DataFrame(rows_expl).set_index("Quadrant"), use_container_width=True)
    for _label, _scol, _xl in [
        ("Raw",              "strength",          f"Total {x_noun} strength"),
        ("Per 90 min",       "strength_per90",    f"Total {x_noun} strength per 90 min"),
        ("Per pass against", "strength_per_pass", f"Total {x_noun} strength per pass against"),
    ]:
        st.markdown(f"#### {_label}")
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            plot_conc_match(match_df, outcome_col, _scol, _xl, y_conc, y_label_conc),
            use_container_width=True,
        )
        c2.plotly_chart(
            plot_conc_team(team_df, outcome_col, _scol, _xl, y_conc, y_label_conc),
            use_container_width=True,
        )
        with st.expander(f"Quadrant statistics — {group_label} / {_label}"):
            _pp_ocs = [oc + "_per_pass" for oc in OUTCOME_COLS]
            _is_pp  = (_scol == "strength_per_pass")

            def _show_quad_tables(df_q, oc_list, label):
                _uid = f"{group_label}__{_label}__{_scol}__{label}"
                _, sm, kw, mw, _ = quadrant_analysis(df_q, _scol, y_conc, outcome_cols=oc_list)
                st.markdown(f"**{label} — Outcome means per quadrant**")
                st.dataframe(sm, use_container_width=True)
                if not kw.empty:
                    st.markdown(f"**{label} — Kruskal-Wallis**")
                    st.dataframe(_style_kw(kw), use_container_width=True)
                if not mw.empty:
                    st.markdown(f"**{label} — Pairwise Mann-Whitney U** "
                                "(p = raw · p_bonf = Bonferroni corrected · sig based on p_bonf"
                                " · 🟢 p_bonf<0.05 · 🟡 p<0.05 only)")
                    st.dataframe(_style_mw(mw), use_container_width=True)
                mg = marginal_analysis(df_q, _scol, y_conc, outcome_cols=oc_list)
                if not mg.empty:
                    st.markdown(f"**{label} — Marginal analysis** "
                                "(High vs Low on X alone / Y alone — isolates each axis's independent effect)")
                    def _style_mg(r):
                        if "sig" in r.index and r["sig"] != "ns":
                            return ["background-color: #c6efce"] * len(r)
                        return [""] * len(r)
                    st.dataframe(mg.style.apply(_style_mg, axis=1), use_container_width=True)

                # Stage distribution + rank quantification per quadrant
                if _scol in df_q.columns and y_conc in df_q.columns:
                    _tmp = df_q[[_scol, y_conc]].dropna().copy()
                    _xm, _ym = _tmp[_scol].median(), _tmp[y_conc].median()
                    _tmp["quadrant"] = _tmp.apply(
                        lambda r: ("H" if r[_scol] >= _xm else "L") + "X / " +
                                  ("H" if r[y_conc] >= _ym else "L") + "Y", axis=1)
                    # stage analysis — team level only
                    if "furthest_stage" in df_q.columns:
                        _valid = _tmp.copy()
                        _valid["stage"] = df_q.loc[_tmp.index, "furthest_stage"]
                        _valid = _valid.dropna(subset=["stage"])
                        if not _valid.empty:
                            _s_order = STAGE_CATEGORY_ORDER
                            _s_pal   = STAGE_PALETTE

                            # 100% proportion bar
                            _piv = (_valid.groupby(["quadrant", "stage"])
                                          .size().unstack(fill_value=0))
                            _piv_pct = _piv.div(_piv.sum(axis=1), axis=0).mul(100).round(1)
                            _piv_long = (_piv_pct.reset_index()
                                                  .melt(id_vars="quadrant",
                                                        var_name="stage", value_name="pct"))
                            _fig_pct = px.bar(
                                _piv_long, x="quadrant", y="pct", color="stage",
                                color_discrete_map=_s_pal,
                                category_orders={"stage": _s_order},
                                title=f"{label} — Furthest stage proportion per quadrant (%)",
                                labels={"pct": "% of teams", "quadrant": "Quadrant",
                                        "stage": "Furthest stage"},
                                barmode="stack",
                            )
                            _fig_pct.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(_fig_pct, use_container_width=True,
                                            key=f"stage_pct_{_uid}")

                            # success rate table — all stages
                            _n_q = _valid.groupby("quadrant").size().rename("n")
                            _stage_cols = [s for s in _s_order if s in _valid["stage"].values]
                            _sr = pd.DataFrame(index=_n_q.index)
                            _sr["n"] = _n_q
                            for _s in _stage_cols:
                                _sr[f"% {_s}"] = (
                                    _valid[_valid["stage"] == _s]
                                    .groupby("quadrant").size()
                                    .reindex(_sr.index, fill_value=0) / _n_q * 100
                                ).round(1)
                            _pct_cols = [c for c in _sr.columns if c.startswith("%")]
                            st.markdown(f"**{label} — Stage distribution per quadrant (%)**")
                            st.dataframe(
                                _sr.style.background_gradient(
                                    cmap="YlGn", subset=_pct_cols, vmin=0, vmax=100),
                                use_container_width=True)

            st.markdown("##### Match level")
            _show_quad_tables(match_df, OUTCOME_COLS, "Raw outcomes")
            if _is_pp:
                st.markdown("*Per-pass outcomes (goals/shots/xg per attacking pass) — consistent with x-axis normalization*")
                _show_quad_tables(match_df, _pp_ocs, "Per-pass outcomes")

            st.markdown("##### Team level")
            st.caption("n = 32 teams — treat p-values as indicative only.")
            _show_quad_tables(team_df, OUTCOME_COLS, "Raw outcomes")
            if _is_pp:
                st.markdown("*Per-pass outcomes*")
                _show_quad_tables(team_df, _pp_ocs, "Per-pass outcomes")

with tab_conc:
    @st.fragment
    def _frag_tab_conc():
        _y_label_conc = _y_opts[y_conc]
        if y_conc == "centralization_w":
            st.caption("Centralization (weighted) is scale-invariant — y axis values are identical across all normalisation sections; only quadrant *boundaries* shift as x changes.")

        for _group, _mdf, _tdf in [
            ("Involvement",  df_conc_inv_match,   df_conc_inv_team),
            ("Fault",        df_conc_fault_match,  df_conc_fault_team),
            ("Contribution", df_conc_cont_match,   df_conc_cont_team),
        ]:
            st.subheader(_group)
            _render_conc_group(_mdf, _tdf, _group, y_conc, _y_label_conc, outcome_col)
            st.divider()

        st.divider()
        st.subheader("Axis Selection — η²")
        st.caption(
            "η² = proportion of total variance explained by team identity.  "
            "**Higher → this metric better differentiates teams.**  "
            "Pick two metrics with high η² *and* low mutual correlation as quadrant axes.  \n"
            "⚠️ η² is *biased upward*: under no team effect its null is **≈(k−1)/(n−1) ≈ 0.25** here, "
            "not 0. Enable the permutation test below to see each metric's chance **floor** and a "
            "**p**-value (H₀: team labels carry no information) and a BH-FDR **q** across the metric "
            "family. **sig** is based on q — only metrics with η² clearly above their floor *and* small q "
            "are defensible team traits."
        )
        _eta_nperm = st.select_slider(
            "η² permutation test (team-label shuffles; 0 = off)",
            options=[0, 500, 1000, 2000, 5000], value=2000, key="eta_nperm")
        eta_df = eta_sq_tbl(df_corr, only_from=INV_COLS, nperm=_eta_nperm)
        _eta_sty = eta_df.style.background_gradient(cmap="YlOrRd", subset=["η²"], vmin=0, vmax=1)
        if "floor" in eta_df.columns:
            _eta_sty = _eta_sty.background_gradient(cmap="YlOrRd", subset=["floor"], vmin=0, vmax=1)
        st.dataframe(_eta_sty, use_container_width=True, height=360)
        all_eta_metrics = eta_df["metric"].tolist()
        if len(all_eta_metrics) >= 2:
            with st.expander("Pairwise correlations — all metrics"):
                st.caption(
                    "Lower correlation = more independent dimensions — better for a 2-axis quadrant plot. "
                    "The table below gives each pair's Pearson **r**, two-sided **p** (H₀: ρ=0) and a "
                    "BH-FDR **q** across all unique pairs; **sig** is based on q."
                )
                corr_all = df_corr[all_eta_metrics].corr().round(2)
                st.dataframe(
                    corr_all.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
                    use_container_width=True,
                )
                _pair_rows = []
                for _a_m, _b_m in combinations(all_eta_metrics, 2):
                    _pv = df_corr[[_a_m, _b_m]].dropna()
                    if len(_pv) < 3:
                        continue
                    _r, _p = pearsonr(_pv[_a_m], _pv[_b_m])
                    _pair_rows.append({"metric A": _a_m, "metric B": _b_m,
                                       "r": round(_r, 3), "n": len(_pv), "p": _p})
                if _pair_rows:
                    pair_df = pd.DataFrame(_pair_rows)
                    _order = pair_df["p"].to_numpy().argsort()
                    _ranked = pair_df["p"].to_numpy()[_order]
                    _m = len(_ranked)
                    _q = _ranked * _m / np.arange(1, _m + 1)
                    _q = np.minimum.accumulate(_q[::-1])[::-1].clip(max=1.0)
                    _qcol = np.empty(_m); _qcol[_order] = _q
                    pair_df["q"] = _qcol
                    pair_df["sig"] = np.where(pair_df["q"] < 0.001, "***",
                                     np.where(pair_df["q"] < 0.01, "**",
                                     np.where(pair_df["q"] < 0.05, "*", "ns")))
                    pair_df["p"] = pair_df["p"].round(4)
                    pair_df["q"] = pair_df["q"].round(4)
                    pair_df = (pair_df.reindex(pair_df["r"].abs().sort_values(ascending=False).index)
                               .reset_index(drop=True))
                    st.dataframe(
                        pair_df.style.background_gradient(cmap="RdYlGn_r", subset=["r"], vmin=-1, vmax=1),
                        use_container_width=True, height=360,
                    )
    _frag_tab_conc()

with tab_self:
    @st.fragment
    def _frag_tab_self():
        global df_self_sorted  # consumed by the Data tab
        fig_bar, fig_scatter, df_self_sorted = plot_selfshared(df_self_team, outcome_col, use_gs=use_gs, correct_possession=correct_possession)
        st.plotly_chart(fig_bar, use_container_width=True)

        def _msig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        _vm = df_self_match[["self_ratio", outcome_col]].dropna()
        _rm,  _pm   = pearsonr(_vm["self_ratio"],  _vm[outcome_col])
        _rhom, _pms = spearmanr(_vm["self_ratio"], _vm[outcome_col])
        _sm = df_self_match[["self_ratio", outcome_col, "passes_against"]].dropna()
        _a = _resid(_sm["self_ratio"].values, _sm["passes_against"].values)
        _b = _resid(_sm[outcome_col].values,  _sm["passes_against"].values)
        _mk = ~(np.isnan(_a) | np.isnan(_b))
        _rmp,  _pmp  = pearsonr(_a[_mk],  _b[_mk])
        _rhomp, _pmps = spearmanr(_a[_mk], _b[_mk])

        fig_scatter_m = px.scatter(
            df_self_match, x="self_ratio", y=outcome_col,
            color="competition_stage",
            color_discrete_map=STYLE_STAGE_PALETTE,
            category_orders={"competition_stage": STYLE_STAGE_ORDER},
            hover_name="team_name",
            trendline="ols", trendline_scope="overall",
            trendline_color_override="black",
            title=(f"Match level (n={len(_vm)}) — Self ratio vs {outcome_col}  |  "
                   f"r = {_rm:.3f}, p = {_pm:.3f} ({_msig(_pm)})  |  partial r = {_rmp:.3f}, p = {_pmp:.3f} ({_msig(_pmp)})  |  "
                   f"ρ = {_rhom:.3f}, p = {_pms:.3f} ({_msig(_pms)})  |  partial ρ = {_rhomp:.3f}, p = {_pmps:.3f} ({_msig(_pmps)})"),
            labels={"self_ratio": "Sharedness (shared / total) [Gini-Simpson]" if use_gs else "Self ratio (self / total)",
                    "competition_stage": "Stage"},
        )
        fig_scatter_m.update_layout(title_font_size=11)

        _c1, _c2 = st.columns(2)
        _c1.plotly_chart(fig_scatter, use_container_width=True)
        _c2.plotly_chart(fig_scatter_m, use_container_width=True)
    _frag_tab_self()

with tab_style:
    @st.fragment
    def _frag_tab_style():
        cs1, cs2 = st.columns(2)

        fig_m = px.scatter(
            df_style_match, x=x_col, y=y_col, color="competition_stage",
            hover_name="team_name", hover_data=OUTCOME_COLS,
            color_discrete_map=STYLE_STAGE_PALETTE,
            category_orders={"competition_stage": STYLE_STAGE_ORDER},
            title=f"{x_col} vs {y_col} — match level",
            labels={x_col: x_col, y_col: y_col, "competition_stage": "Stage"},
        )
        _add_quadrant_lines(fig_m, df_style_match, x_col, y_col, opacity=0.5)
        fig_m.update_layout(height=550)
        cs1.plotly_chart(fig_m, use_container_width=True)

        fig1, _ = plot_style(df_style_team, x_col, y_col, size_col, size_scale)
        fig1.update_layout(height=550)
        cs2.plotly_chart(fig1, use_container_width=True)

        st.subheader("Team-level values")
        st.dataframe(
            df_style_team[["team_name", "x", "y", "size", "n_matches", "best_stage"]]
            .rename(columns={"x": x_col, "y": y_col, "size": size_col})
            .sort_values("best_stage", ascending=False).round(3),
            use_container_width=True,
        )

        with st.expander("Quadrant Analysis — split at median"):
            st.caption(
                f"X = `{x_col}`  |  Y = `{y_col}`  "
                f"|  HX = high contribution · LX = low  |  HY = high fault · LY = low"
            )
            _mw_caption = ("**Pairwise Mann-Whitney U** (p = raw · p_bonf = Bonferroni corrected"
                           " · sig based on p_bonf · 🟢 p_bonf<0.05 · 🟡 p<0.05 only)")
            st.markdown("##### Match level")
            _, sm_m, kw_m, mw_m, _ = quadrant_analysis(df_style_match, x_col, y_col)
            st.markdown("**Outcome means per quadrant**")
            st.dataframe(sm_m, use_container_width=True)
            if not kw_m.empty:
                st.markdown("**Kruskal-Wallis**")
                st.dataframe(_style_kw(kw_m), use_container_width=True)
            if not mw_m.empty:
                st.markdown(_mw_caption)
                st.dataframe(_style_mw(mw_m), use_container_width=True)
            mg_m = marginal_analysis(df_style_match, x_col, y_col)
            if not mg_m.empty:
                st.markdown("**Marginal analysis** (High vs Low on X alone / Y alone)")
                st.dataframe(mg_m.style.apply(
                    lambda r: ["background-color: #c6efce"] * len(r) if r.get("sig", "ns") != "ns" else [""] * len(r),
                    axis=1), use_container_width=True)

            st.markdown("##### Team level")
            st.caption("n = 32 teams — treat p-values as indicative only.")
            _, sm_t, kw_t, mw_t, _ = quadrant_analysis(df_style_team, "x", "y")
            st.markdown("**Outcome means per quadrant**")
            st.dataframe(sm_t, use_container_width=True)
            if not kw_t.empty:
                st.markdown("**Kruskal-Wallis**")
                st.dataframe(_style_kw(kw_t), use_container_width=True)
            if not mw_t.empty:
                st.markdown(_mw_caption)
                st.dataframe(_style_mw(mw_t), use_container_width=True)
            mg_t = marginal_analysis(df_style_team, "x", "y")
            if not mg_t.empty:
                st.markdown("**Marginal analysis**")
                st.dataframe(mg_t.style.apply(
                    lambda r: ["background-color: #c6efce"] * len(r) if r.get("sig", "ns") != "ns" else [""] * len(r),
                    axis=1), use_container_width=True)
    _frag_tab_style()

with tab_codef:
    @st.fragment
    def _frag_tab_codef():
        st.caption(
            "**Co-defender** = number of teammates also defending the same attacking pass. "
            "Computed from Method-2 player edges. "
            "High avg co-defenders → team defends in groups (pressing style). "
            "Low → more zonal / individual defending."
        )
        st.subheader("Avg co-defenders per pass")
        fig_match, fig_team = plot_avg_co_defenders(avg_co, avg_co_team, outcome_col)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_match, use_container_width=True)
        c2.plotly_chart(fig_team, use_container_width=True)

        st.subheader("Top co-defending pairs")
        n_top = st.slider("Top N pairs", 10, 50, 20, key="pair_n")
        top = (partnerships.groupby(["player_a", "player_b"])
               .agg(co_defenses=("co_defenses", "sum"), team_name=("team_name", "first"))
               .reset_index()
               .sort_values("co_defenses", ascending=False)
               .head(n_top).copy())
        top["pair"] = top["player_a"] + "  +  " + top["player_b"]
        fig_pairs = px.bar(top, x="co_defenses", y="pair", orientation="h",
                           hover_data={"team_name": True},
                           title=f"Top {n_top} co-defending pairs (all teams)")
        fig_pairs.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig_pairs, use_container_width=True)

        st.subheader("Team co-defending heatmap")
        team_sel = st.selectbox("Select team", sorted(partnerships["team_name"].dropna().unique()))
        fig_heat = plot_partnership_heatmap(partnerships, team_sel)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No partnership data for this team.")
    _frag_tab_codef()

with tab_zone:
    @st.fragment
    def _frag_tab_zone():
        st.caption(
            "Defensive actions split by **where the ball is** when the pass is made "
            "(`x_def = -x_norm`, defending-team perspective: larger = closer to the "
            "opponent goal = higher press). `x_norm` already folds in home/away "
            "orientation and the first/second-half flip, so both teams share one "
            "direction. Both successful (C) and unsuccessful (B/D) passes are included."
        )
        _need = ["n_actions"] + [WEIGHT_SPLIT[w][i] for w in WEIGHT_SPLIT for i in (1, 2, 3)]
        _missing = [] if zone_raw is None else [c for c in set(_need) if c not in zone_raw.columns]
        if zone_raw is None:
            st.warning(
                "Zone data not found. Run `scripts/2026-06-08_team_zone_metrics.py` "
                "first to generate `2026-06-08_team_zone_metrics.csv`."
            )
        elif _missing:
            st.warning(
                "Zone CSV is out of date (missing columns: "
                f"`{', '.join(sorted(_missing))}`). Re-run "
                "`scripts/2026-06-08_team_zone_metrics.py` to regenerate it."
            )
        else:
            _wlabel = WEIGHT_SPLIT[zone_weight][0]
            st.subheader("Zone composition per team")
            st.caption(
                f"Each team's defensive **{_wlabel}**, split by pitch zone (colour) and, "
                "within each zone, contribution (solid) vs fault (hatched). Bars sum to 1; "
                "pooled across all of a team's matches."
            )
            zsplit, zones = build_zone_split(zone_raw, zone_scheme, zone_weight)
            zsplit["Zone"] = zsplit["zone"].map(ZONE_LABEL)
            zone_label_order = [ZONE_LABEL[z] for z in zones]
            zone_color = {ZONE_LABEL[z]: ZONE_COLOR[z] for z in zones}
            team_order = (zsplit[zsplit["zone"] == zones[-1]]
                          .groupby("defending_team_name")["share"].sum()
                          .sort_values(ascending=False).index.tolist())
            fig_share = px.bar(
                zsplit, x="share", y="defending_team_name",
                color="Zone", pattern_shape="type",
                orientation="h", barmode="stack",
                color_discrete_map=zone_color,
                pattern_shape_map={"contribution": "", "fault": "/"},
                category_orders={"Zone": zone_label_order,
                                 "type": ["contribution", "fault"],
                                 "defending_team_name": team_order},
                labels={"share": f"Proportion of team's total {_wlabel.lower()}"},
                title=f"Zone composition of {_wlabel} ({SCHEME_LABEL[zone_scheme]})")
            # denser hatching for the fault segments
            fig_share.update_traces(marker_pattern_size=3, marker_pattern_solidity=0.35)
            fig_share.update_layout(height=max(420, 24 * len(team_order)),
                                    yaxis=dict(autorange="reversed", title=None),
                                    xaxis=dict(tickformat=".0%"),
                                    legend_title_text="Zone / con-fault")
            st.plotly_chart(fig_share, use_container_width=True)
            with st.expander("Composition table (proportion)"):
                tbl = (zsplit.assign(seg=zsplit["Zone"] + " · " + zsplit["type"])
                       .pivot_table(index="defending_team_name", columns="seg", values="share")
                       .loc[team_order])
                st.dataframe(tbl.round(3))

            # ── Correlation: zone metric vs outcomes ──────────────────────────────
            st.subheader("Zone metric × outcome correlation")
            _zc_modes = {"Raw": (False, "total"),
                         "Partial — control total passes against": (True, "total"),
                         "Partial — control this zone's passes faced": (True, "zone")}
            zc_mode = st.radio("Correlation", list(_zc_modes), horizontal=True,
                               key="zone_corr_partial")
            zc_partial, zc_control = _zc_modes[zc_mode]
            st.caption(
                "Correlation between each zone metric **sum** (rows: raw/val × inv/con/fault) "
                "and each outcome (cols), at **match-team level** (one point per team per "
                "match). One heatmap per zone. Red = positive (more ↔ conceding more, worse); "
                "blue = negative (↔ conceding less). `*` = p<0.05. "
                + {"total": "Partial = residualised on the match's **total** passes_against.",
                   "zone": "Partial = residualised on **this zone's** faced-pass count "
                           "(n_passes)."}[zc_control] if zc_partial else ""
            )
            corr_df = build_zone_corr(zone_raw, zone_scheme, outcomes, OUTCOME_COLS,
                                      partial=zc_partial, control=zc_control)
            zones_c = [z for z in ZONE_ORDER if z in set(corr_df["zone"])]
            ocl = [c.replace("_against", " ag.") for c in OUTCOME_COLS]
            cols = st.columns(len(zones_c))
            for col, z in zip(cols, zones_c):
                rmat, tmat = [], []
                for m in ZONE_CORR_METRICS:
                    rrow, trow = [], []
                    for oc in OUTCOME_COLS:
                        sel = corr_df[(corr_df["metric"] == m) & (corr_df["zone"] == z)
                                      & (corr_df["outcome"] == oc)]
                        if len(sel):
                            r, p = sel["r"].iloc[0], sel["p"].iloc[0]
                            rrow.append(r); trow.append(f"{r:+.2f}{'*' if p < 0.05 else ''}")
                        else:
                            rrow.append(np.nan); trow.append("")
                    rmat.append(rrow); tmat.append(trow)
                rmat = pd.DataFrame(rmat, index=[METRIC_LABEL[m] for m in ZONE_CORR_METRICS],
                                    columns=ocl)
                fig = px.imshow(rmat, color_continuous_scale="RdBu_r", zmin=-0.6, zmax=0.6,
                                aspect="equal", labels=dict(color="r"),
                                title=ZONE_LABEL[z])
                fig.update_traces(text=tmat, texttemplate="%{text}", textfont_size=12)
                fig.update_xaxes(side="top")
                fig.update_layout(width=300, height=520, margin=dict(l=10, r=10, t=60, b=10),
                                  coloraxis_showscale=False)
                col.plotly_chart(fig, use_container_width=False)
            with st.expander("Correlation table (r, p, n)"):
                st.dataframe(
                    corr_df.assign(metric=corr_df["metric"].map(METRIC_LABEL),
                                   zone=corr_df["zone"].map(ZONE_LABEL))
                           .round({"r": 3, "p": 4}))

            with st.expander("Volume diagnostic — why 'control passes' behaves differently per zone"):
                vd1, vd2 = st.columns(2)
                vdo = vd1.selectbox("Outcome", OUTCOME_COLS, index=1, key="vol_diag_out")
                vdk = vd2.selectbox("raw / valued", ["raw", "valued"], key="vol_diag_kind")
                st.caption(
                    "Match-team level (N=128). **npass↔metric**: how much the zone's metric "
                    "SUM is just volume (n_passes). **npass↔outcome**: whether that volume "
                    "itself relates to the outcome (large + in own third = being pinned back). "
                    "**invPerPass↔outcome**: per-faced-pass intensity vs outcome (volume "
                    "removed). Explains why controlling zone passes helps high-press but "
                    "erases own-third's real exposure signal."
                )
                st.table(build_zone_volume_diag(zone_raw, zone_scheme, outcomes, vdo, vdk))

            # ── Team × zone contribution/fault ratio ──────────────────────────────
            st.subheader("Team × zone contribution / fault ratio")
            st.caption(
                "Per team, mean of per-match contribution÷fault ratios in each zone (each "
                "match weighted equally). **Original** = Σcon/Σfault per match; **per-event** "
                "= (con/con-passes) ÷ (fault/fault-passes), i.e. mean contribution intensity "
                "÷ mean fault intensity (strips how *often* you act, keeps how *impactful*). "
                "Darker = higher. Teams sorted by raw-original high-press ratio."
            )
            # team order from raw / original / high-press
            _base, _zb = build_team_zone_ratio(zone_raw, zone_scheme, "raw", "plain")
            team_order = _base[_zb[-1]].sort_values(ascending=False).index.tolist()
            for mode, mlab in [("plain", "Original  con/fault"),
                               ("per_event", "Per-event  con/fault")]:
                st.markdown(f"**{mlab}**")
                for col, kind in zip(st.columns(2), ("raw", "valued")):
                    m, zns = build_team_zone_ratio(zone_raw, zone_scheme, kind, mode)
                    mm = m.reindex(index=team_order)
                    mm.columns = [ZONE_SHORT[z] for z in zns]
                    fig_tr = px.imshow(mm, color_continuous_scale="Reds", text_auto=".2f",
                                       aspect="auto", labels=dict(color="ratio"),
                                       title=f"{kind}")
                    fig_tr.update_xaxes(side="top", tickangle=0, tickfont_size=13)
                    fig_tr.update_yaxes(title=None, tickfont_size=12)
                    fig_tr.update_traces(textfont_size=13)
                    fig_tr.update_layout(height=max(900, 28 * len(team_order)),
                                         margin=dict(l=8, r=8, t=50, b=8))
                    col.plotly_chart(fig_tr, use_container_width=True)

            # ── Volume vs efficiency scatter (per zone) ───────────────────────────
            st.subheader("Volume vs efficiency, per zone")
            st.caption(
                "One point per team. X = involvement in the zone (how much they defend "
                "there); Y = con/fault ratio (efficiency, pooled Σcon/Σfault). Colour = "
                "mean-per-match outcome (darker = concedes more). Top-left = efficient but "
                "barely defends there; top-right = efficient AND high volume. Dashed line = "
                "ratio 1 (con=fault). OLS fit + r in each title."
            )
            ve1, ve2, ve3 = st.columns(3)
            ve_kind = ve1.selectbox("raw / valued", ["raw", "valued"], key="ve_kind")
            ve_xmode = ve2.radio("X axis", ["per-match", "total"], horizontal=True, key="ve_xmode")
            ve_color = ve3.selectbox("Colour by", OUTCOME_COLS, index=1, key="ve_color")
            xcol = "inv_pm" if ve_xmode == "per-match" else "inv"
            xlab = f"{'per-match' if ve_xmode == 'per-match' else 'total'} involvement"
            ve = build_team_zone_volume_ratio(zone_raw, zone_scheme, outcomes, ve_kind)
            ve_zones = [z for z in ZONE_ORDER if z in set(ve["zone"])]
            cmax = ve[ve_color].max()
            for col, z in zip(st.columns(len(ve_zones)), ve_zones):
                g = ve[ve["zone"] == z].dropna(subset=[xcol, "ratio"])
                r, p = pearsonr(g[xcol], g["ratio"]) if len(g) > 2 else (np.nan, np.nan)
                fig_ve = px.scatter(g, x=xcol, y="ratio", text="defending_team_name",
                                    color=ve_color, color_continuous_scale="Reds",
                                    range_color=(0, cmax), trendline="ols",
                                    trendline_scope="overall", trendline_color_override="black",
                                    labels={xcol: xlab, "ratio": "con/fault",
                                            ve_color: ve_color.replace("_", " ")},
                                    title=f"{ZONE_LABEL[z]}  (r={r:+.2f}, p={p:.3f})")
                fig_ve.update_traces(textposition="top center", textfont_size=9,
                                     marker=dict(size=10, line=dict(width=1, color="white")),
                                     selector=dict(mode="markers+text"))
                fig_ve.add_hline(y=1.0, line_dash="dash", line_color="grey", opacity=0.6)
                fig_ve.update_layout(height=500, margin=dict(l=8, r=8, t=50, b=8),
                                     coloraxis_showscale=(z == ve_zones[-1]))
                col.plotly_chart(fig_ve, use_container_width=True)

            # ── con/fault RATIO × outcome correlation (match level, table) ────────
            st.subheader("Contribution / Fault ratio × outcome correlation")
            st.caption(
                "Pearson r between each zone's **contribution÷fault** ratio (per match-team) "
                "and each outcome, at **match-team level**. Match-teams with zero fault in a "
                "zone are dropped. `*` p<0.05, `**` p<0.01. "
                + ("Partial: controlling passes_against." if zc_partial else "Raw correlation.")
            )
            rc = build_zone_ratio_corr(zone_raw, zone_scheme, outcomes, OUTCOME_COLS,
                                       partial=zc_partial)
            if rc.empty:
                st.info("Not enough data for ratio correlations in this scheme.")
            else:
                rc["cell"] = rc.apply(
                    lambda x: f"{x.r:+.2f}{'**' if x.p < 0.01 else '*' if x.p < 0.05 else ''}"
                              f" (n={x.n})", axis=1)
                rc["row"] = rc["zone"].map(ZONE_LABEL) + " · " + rc["ratio"]
                row_order = [f"{ZONE_LABEL[z]} · {k}"
                             for z in ZONE_ORDER if z in set(rc["zone"])
                             for k in ("raw", "valued")]
                tbl_rc = (rc.pivot(index="row", columns="outcome", values="cell")
                            .reindex(index=row_order,
                                     columns=[c for c in OUTCOME_COLS if c in set(rc["outcome"])]))
                tbl_rc.columns = [c.replace("_against", " ag.") for c in tbl_rc.columns]
                st.table(tbl_rc.fillna("—"))

            # ── Zone profile by stage (boxplots: reached knockout?) ───────────────
            st.subheader("Zone profile — group-stage-out vs knockout teams")
            bx1, bx2 = st.columns(2)
            bx_measure = bx1.selectbox(
                "Measure", ["involvement proportion", "contribution proportion",
                            "fault proportion"], key="bx_measure")
            bx_kind = bx2.selectbox("raw / valued", ["raw", "valued"], key="bx_kind")
            dd = zone_raw[zone_raw["scheme"] == zone_scheme].copy()
            metric = {"involvement proportion": "involvement",
                      "contribution proportion": "contribution",
                      "fault proportion": "fault"}[bx_measure]
            mcol = f"{bx_kind}_{metric}_sum"
            tot = dd.groupby("match_team_id")[mcol].transform("sum")
            dd["val"] = dd[mcol] / tot.replace(0, np.nan)
            ylab = f"{bx_kind} {metric} zone proportion"
            dd = dd.merge(outcomes[["match_team_id", "reached_knockout"]], on="match_team_id")
            dd["Zone"] = dd["zone"].map(ZONE_LABEL)
            dd["Stage"] = dd["reached_knockout"].map({True: "Reached knockout", False: "Group only"})
            zorder = [ZONE_LABEL[z] for z in ZONE_ORDER if z in set(dd["zone"])]
            dd = dd.dropna(subset=["val"])
            fig_bx = px.box(dd, x="Zone", y="val", color="Stage",
                            category_orders={"Zone": zorder,
                                             "Stage": ["Group only", "Reached knockout"]},
                            color_discrete_map={"Group only": "#bbbbbb",
                                                "Reached knockout": "#1f78b4"},
                            labels={"val": ylab},
                            title=f"{ylab} by zone (match-team level)")
            st.plotly_chart(fig_bx, use_container_width=True)
            mw = []
            for z in [zz for zz in ZONE_ORDER if zz in set(dd["zone"])]:
                g = dd[dd["zone"] == z]
                a = g[g["reached_knockout"]]["val"]; b = g[~g["reached_knockout"]]["val"]
                if len(a) > 2 and len(b) > 2:
                    _, p = mannwhitneyu(a, b)
                    mw.append(dict(Zone=ZONE_LABEL[z],
                                   knockout_median=round(a.median(), 3),
                                   group_median=round(b.median(), 3),
                                   p=round(p, 4), sig="*" if p < 0.05 else ""))
            st.caption("Median per group + Mann–Whitney U p (knockout vs group-only; "
                       "a knockout team's group matches count as 'knockout').")
            if mw:
                st.table(pd.DataFrame(mw).set_index("Zone"))

            _present_zones = [z for z in ZONE_ORDER
                              if z in set(zone_raw.loc[zone_raw["scheme"] == zone_scheme, "zone"])]

            # ── Defensive style map (team level) ──────────────────────────────────
            st.subheader("Defensive style map (team level)")
            st.caption(
                "One point per team (pooled across its matches). X = the chosen metric's "
                "proportion in the chosen zone (zone sum ÷ the team's total of that metric); "
                "Y = mean-per-match outcome (lower = better defence); colour = reached "
                "knockout. OLS fit + r in title; dashed lines = medians."
            )
            _METRIC6 = {"raw inv": ("raw", ""), "valued inv": ("valued", ""),
                        "raw con": ("raw", "con_"), "valued con": ("valued", "con_"),
                        "raw fault": ("raw", "fault_"), "valued fault": ("valued", "fault_")}
            sc1, sc2, sc3 = st.columns(3)
            sc_metric = sc1.selectbox("Metric", list(_METRIC6), key="style_metric")
            sc_zone = sc2.selectbox("Zone", _present_zones,
                                    format_func=ZONE_SHORT.__getitem__, key="style_zone")
            sc_y = sc3.selectbox("Y axis (outcome)", OUTCOME_COLS, index=1, key="style_y")
            _sk, _spre = _METRIC6[sc_metric]
            ts_sc = build_team_style(zone_raw, zone_scheme, outcomes, kind=_sk)
            if fifa_team_rating is not None:
                ts_sc["fifa_rating"] = ts_sc.index.map(fifa_team_rating)
            _xcol = f"{_spre}{sc_zone}_share"
            _xlab = f"{sc_metric} · {ZONE_SHORT[sc_zone]} proportion"
            if _xcol in ts_sc.columns:
                ts = ts_sc.reset_index().dropna(subset=[_xcol, sc_y])
                ts["Stage"] = ts["reached_knockout"].map({True: "Reached knockout", False: "Group only"})
                r_sm, p_sm = pearsonr(ts[_xcol], ts[sc_y])
                fig_sm = px.scatter(ts, x=_xcol, y=sc_y, color="Stage", text="team_name",
                                    color_discrete_map={"Group only": "#bbbbbb",
                                                        "Reached knockout": "#1f78b4"},
                                    trendline="ols", trendline_scope="overall",
                                    trendline_color_override="black",
                                    labels={_xcol: _xlab, sc_y: sc_y.replace("_", " ")},
                                    title=f"{_xlab} vs {sc_y.replace('_',' ')}  "
                                          f"(r={r_sm:+.2f}, p={p_sm:.3f}, N={len(ts)})")
                fig_sm.update_traces(textposition="top center",
                                     marker=dict(size=11, line=dict(width=1, color="white")),
                                     selector=dict(mode="markers+text"))
                fig_sm.add_vline(x=ts[_xcol].median(), line_dash="dash", line_color="grey", opacity=0.5)
                fig_sm.add_hline(y=ts[sc_y].median(), line_dash="dash", line_color="grey", opacity=0.5)
                fig_sm.update_layout(height=620)
                st.plotly_chart(fig_sm, use_container_width=True)

            # ── Merged correlation table: rows = metric × zone, cols = control × outcome ──
            st.subheader("Style proportion × outcome correlation (team level)")
            tbl_kind = st.selectbox("raw / valued", ["raw", "valued"], key="style_tbl_kind")
            ts_t = build_team_style(zone_raw, zone_scheme, outcomes, kind=tbl_kind)
            if fifa_team_rating is not None:
                ts_t["fifa_rating"] = ts_t.index.map(fifa_team_rating)
            _controls = [("raw", None), ("ctrl total-pass", "passes_against"),
                         ("ctrl zone-pass", "zone")]
            if fifa_team_rating is not None:
                _controls.append(("ctrl FIFA", "fifa_rating"))
            _rows_def = [(f"{pre}{z}_share", f"{ml} · {zl}", z)
                         for pre, ml in [("", "inv"), ("con_", "con"), ("fault_", "fault")]
                         for z, zl in [("high_press", "high"), ("mid", "mid"), ("own", "own")]]

            def _fmt(rr, pp):
                return "—" if np.isnan(rr) else \
                    f"{rr:+.2f}{'**' if pp < 0.01 else '*' if pp < 0.05 else ''}"

            data, data_num, rlabels = {}, {}, []
            for key, lab, z in _rows_def:
                if key not in ts_t.columns:
                    continue
                rlabels.append(lab)
                for clab, ctrl in _controls:
                    cc = f"npass_{z}" if ctrl == "zone" else ctrl
                    for oc in OUTCOME_COLS:
                        ocl = oc.replace("_against", " ag.")
                        if ctrl is None:
                            s = ts_t[[key, oc]].dropna()
                            rr, pp = pearsonr(s[key], s[oc]) if len(s) > 3 else (np.nan, np.nan)
                        else:
                            s = ts_t[[key, oc, cc]].dropna()
                            if len(s) > 3:
                                a = _resid(s[key].values, s[cc].values)
                                b = _resid(s[oc].values, s[cc].values)
                                mk = ~(np.isnan(a) | np.isnan(b))
                                rr, pp = pearsonr(a[mk], b[mk]) if mk.sum() > 3 else (np.nan, np.nan)
                            else:
                                rr, pp = np.nan, np.nan
                        data.setdefault((clab, ocl), []).append(_fmt(rr, pp))
                        data_num.setdefault((clab, ocl), []).append(rr)
            txt = pd.DataFrame(data, index=rlabels)
            num = pd.DataFrame(data_num, index=rlabels)
            txt.columns = pd.MultiIndex.from_tuples(txt.columns)
            num.columns = pd.MultiIndex.from_tuples(num.columns)

            def _bg(v):  # diverging: 0 = lightest (white), + -> red, - -> blue
                if pd.isna(v):
                    return ""
                tt = max(-1.0, min(1.0, v / 0.8))
                if tt >= 0:
                    r, g, b = 255, int(255 - 215 * tt), int(255 - 215 * tt)
                else:
                    r, g, b = int(255 + 215 * tt), int(255 + 165 * tt), 255
                return f"background-color: rgb({r},{g},{b}); color: #000"

            css = num.apply(lambda c: c.map(_bg))
            st.caption(
                f"N={len(ts_t)}, {tbl_kind}. Pearson r between each zone proportion (rows: "
                "inv/con/fault × high/mid/own) and each outcome, under no control (raw) and "
                "three partial controls (total opponent passes / that zone's passes / FIFA "
                "rating). Colour: 0 = white, red = positive (concedes more), blue = negative. "
                "`*` p<0.05, `**` p<0.01."
            )
            st.table(txt.style.apply(lambda _: css, axis=None))
            st.caption(
                "Robustness: controlling **FIFA rating** (exogenous strength) barely changes "
                "the raw values (own-third +0.72→+0.67, high-press −0.62→−0.54) — the "
                "style↔conceding link is not just 'stronger teams'. **Zone-pass** control "
                "also barely moves it. **Total-pass** control shrinks it sharply, but that "
                "over-corrects (passes_against is partly a consequence of a deep style)."
            )

            st.markdown("**Style proportion by stage** — median (knockout vs group-only) + "
                        "Mann–Whitney U p.")
            rowsB = []
            for key, lab, z in _rows_def:
                if key not in ts_t.columns:
                    continue
                a = ts_t[ts_t["reached_knockout"]][key].dropna()
                b = ts_t[~ts_t["reached_knockout"]][key].dropna()
                if len(a) > 2 and len(b) > 2:
                    _, pp = mannwhitneyu(a, b)
                    rowsB.append(dict(Index=lab, knockout_med=round(a.median(), 3),
                                      group_med=round(b.median(), 3), p=round(pp, 3),
                                      sig="*" if pp < 0.05 else ""))
            if rowsB:
                st.table(pd.DataFrame(rowsB).set_index("Index"))
    _frag_tab_zone()

with tab_ztopo:
    st.caption(
        "Network **structure** computed *within each pitch zone* (own / mid / "
        "high-press, thirds scheme). Same graph metrics as the Correlation tab, but "
        "the co-defending graph is rebuilt from only the passes whose ball position "
        "falls in that zone — so you can ask whether a team centralises in the press "
        "but fragments in the low block, and whether that structure relates to "
        "conceding. Built from `scripts/2026-06-19_zone_topology(<method>).csv`."
    )
    if zone_topo_dfs is None:
        st.warning(
            "Zone topology files not found. Run "
            "`scripts/2026-06-19_zone_topology.py` to generate "
            "`2026-06-19_zone_topology(<method>).csv` (one per edge-weight method)."
        )
    else:
        # Fragment: changing the metric/weight/correlation widgets below reruns only
        # this block, not the whole app. `method` is a sidebar control outside the
        # fragment, so switching edge-weight method still triggers a full rerun.
        @st.fragment
        def _render_zone_topology(topo_df):
            zt_zones = [z for z in ZONE_ORDER if z in set(topo_df["zone"])]
            topo_labels = list(TOPO_METRICS)

            # ── All topology × weight correlations, one outcome at a time ──────
            st.subheader("Zone topology × outcome correlation — all metrics")
            cc1, cc2 = st.columns([1, 2])
            zt_outcome = cc1.selectbox("Outcome", OUTCOME_COLS, index=2, key="zt_outcome")
            # control modes mirror the Zones tab; the per-zone option needs zone_raw
            _zt_modes = {"Raw": (False, "total"),
                         "Partial — control total passes against": (True, "total")}
            if zone_raw is not None:
                _zt_modes["Partial — control this zone's passes faced"] = (True, "zone")
            zt_mode = cc2.radio("Correlation", list(_zt_modes),
                                horizontal=True, key="zt_corr_mode")
            zt_partial, zt_control = _zt_modes[zt_mode]
            zt_npass = None
            if zt_control == "zone":
                _zr = zone_raw[zone_raw["scheme"] == "thirds"]
                zt_npass = _zr.pivot_table(index="match_team_id", columns="zone",
                                           values="n_passes", aggfunc="sum", fill_value=0)
            st.caption(
                f"Pearson r between every **topology metric** (rows) × **weight metric** "
                f"(cols) and **{zt_outcome.replace('_', ' ')}**, at match-team level — one "
                "heatmap per zone, all combinations shown at once. Red = positive (more ↔ "
                "conceding more, worse); blue = negative. Every cell shows its r; "
                "`*` marks p<0.05. "
                + ({"total": "Partial = residualised on the match's total passes_against.",
                    "zone": "Partial = residualised on this zone's own faced-pass count "
                            "(n_passes), so the structure is isolated from how much the "
                            "team was pinned back in that zone."}[zt_control]
                   if zt_partial else "Raw correlation.")
            )
            corr_all = build_zone_topo_corr_all(topo_df, outcomes, OUTCOME_COLS,
                                                partial=zt_partial, control=zt_control,
                                                zone_npass=zt_npass)
            corr_df = corr_all[corr_all["outcome"] == zt_outcome] if not corr_all.empty \
                else corr_all
            if corr_df.empty:
                st.info("Not enough data to correlate for this selection.")
            else:
                zones_c = [z for z in ZONE_ORDER if z in set(corr_df["zone"])]
                wlabels = [METRIC_LABEL[m] for m in WEIGHT_COLS]
                row_labels = [TOPO_SHORT[t] for t in topo_labels]
                cols_ui = st.columns(len(zones_c))
                for i, (col_ui, z) in enumerate(zip(cols_ui, zones_c)):
                    col_ui.markdown(f"<div style='text-align:center;font-weight:600'>"
                                    f"{ZONE_LABEL[z]}</div>", unsafe_allow_html=True)
                    rmat, tmat = [], []
                    for tl in topo_labels:
                        rrow, trow = [], []
                        for m in WEIGHT_COLS:
                            sel = corr_df[(corr_df["topo"] == tl) & (corr_df["metric"] == m)
                                          & (corr_df["zone"] == z)]
                            if len(sel):
                                r, p = sel["r"].iloc[0], sel["p"].iloc[0]
                                rrow.append(r)
                                # every cell labelled; * marks p<0.05
                                trow.append(f"{r:+.2f}{'*' if p < 0.05 else ''}")
                            else:
                                rrow.append(np.nan); trow.append("")
                        rmat.append(rrow); tmat.append(trow)
                    rmat = pd.DataFrame(rmat, index=row_labels, columns=wlabels)
                    last = (i == len(zones_c) - 1)
                    fig = px.imshow(rmat, color_continuous_scale="RdBu_r", zmin=-0.5, zmax=0.5,
                                    aspect="auto", labels=dict(color="r"))
                    # trace text renders reliably and Plotly auto-contrasts it per cell
                    fig.update_traces(text=tmat, texttemplate="%{text}", textfont_size=11)
                    fig.update_xaxes(side="top", tickangle=45, tickfont_size=10)
                    fig.update_yaxes(showticklabels=(i == 0), tickfont_size=10)
                    fig.update_layout(height=460, margin=dict(l=4, r=4, t=55, b=4),
                                      coloraxis_showscale=last)
                    col_ui.plotly_chart(fig, use_container_width=True)
                with st.expander("Full correlation table (all outcomes · r, p, n)"):
                    st.dataframe(
                        corr_all.assign(metric=corr_all["metric"].map(METRIC_LABEL),
                                        zone=corr_all["zone"].map(ZONE_LABEL))
                                .round({"r": 3, "p": 4}),
                        use_container_width=True, height=360)

            # ── Level robustness: is the zonal topology a stable team trait? ───
            st.subheader("Per-zone level robustness — ICC across a team's matches")
            st.caption(
                "ICC(1,1) of each topology metric's **level** within a zone — teams = "
                "subjects, their 3–7 matches = replicates. Asks whether a team's zonal "
                "co-defending *structure* is a repeatable trait or just match-to-match "
                "noise. Since defending depends heavily on the opponent, most of it is "
                "noise. Green = more team-stable (ICC>0.5 would be a usable trait). "
                "`*` = raw p<0.05 · `†` = survives BH-FDR (q<0.05) across **all** cells "
                "shown. One heatmap per zone.")
            icc_long = build_zone_topo_icc(topo_df)
            if icc_long.empty:
                st.info("Not enough replicated data for level ICC.")
            else:
                n_raw = int((icc_long["p"] < 0.05).sum())
                n_q   = int((icc_long["q"] < 0.05).sum())
                n_hi  = int((icc_long["ICC"] > 0.5).sum())
                tp = icc_long.sort_values("ICC", ascending=False).iloc[0]
                st.markdown(
                    f"**{n_raw}** / {len(icc_long)} cells sig at raw p<0.05 "
                    f"(~{0.05 * len(icc_long):.0f} expected by chance) → **{n_q}** survive "
                    f"BH-FDR · **{n_hi}** reach ICC>0.5 · strongest: "
                    f"`{TOPO_SHORT.get(tp['topo'], tp['topo'])}` × {METRIC_LABEL[tp['metric']]} "
                    f"in {ZONE_SHORT[tp['zone']]} (ICC={tp['ICC']:.3f}, p={tp['p']:.3f}"
                    f"{', q<0.05' if tp['q'] < 0.05 else ''}).")
                zones_i = [z for z in ZONE_ORDER if z in set(icc_long["zone"])]
                wlabels = [METRIC_LABEL[m] for m in WEIGHT_COLS]
                row_labels = [TOPO_SHORT[t] for t in TOPO_METRICS]
                cols_ui = st.columns(len(zones_i))
                for i, (col_ui, z) in enumerate(zip(cols_ui, zones_i)):
                    col_ui.markdown(f"<div style='text-align:center;font-weight:600'>"
                                    f"{ZONE_LABEL[z]}</div>", unsafe_allow_html=True)
                    rmat, tmat = [], []
                    for tl in TOPO_METRICS:
                        rrow, trow = [], []
                        for m in WEIGHT_COLS:
                            sel = icc_long[(icc_long["topo"] == tl) & (icc_long["metric"] == m)
                                           & (icc_long["zone"] == z)]
                            if len(sel):
                                v, p, q = sel["ICC"].iloc[0], sel["p"].iloc[0], sel["q"].iloc[0]
                                mark = "†" if q < 0.05 else ("*" if p < 0.05 else "")
                                rrow.append(v); trow.append(f"{v:.2f}{mark}")
                            else:
                                rrow.append(np.nan); trow.append("")
                        rmat.append(rrow); tmat.append(trow)
                    rmat = pd.DataFrame(rmat, index=row_labels, columns=wlabels)
                    last = (i == len(zones_i) - 1)
                    fig = px.imshow(rmat, color_continuous_scale="YlGn", zmin=0, zmax=0.5,
                                    aspect="auto", labels=dict(color="ICC"))
                    fig.update_traces(text=tmat, texttemplate="%{text}", textfont_size=11)
                    fig.update_xaxes(side="top", tickangle=45, tickfont_size=10)
                    fig.update_yaxes(showticklabels=(i == 0), tickfont_size=10)
                    fig.update_layout(height=460, margin=dict(l=4, r=4, t=55, b=4),
                                      coloraxis_showscale=last)
                    col_ui.plotly_chart(fig, use_container_width=True, key=f"zt_icc_{z}")
                with st.expander("Full level-ICC table (all metrics · ICC, p, q, n)"):
                    st.dataframe(
                        icc_long.assign(topo=icc_long["topo"].map(TOPO_SHORT),
                                        metric=icc_long["metric"].map(METRIC_LABEL),
                                        zone=icc_long["zone"].map(ZONE_LABEL))
                                .sort_values("ICC", ascending=False)
                                .round({"ICC": 3, "p": 4, "q": 3}),
                        use_container_width=True, height=360)

            # ── Inspect a single metric's distribution across zones ────────────
            with st.expander("Inspect a single metric across zones"):
                c1, c2 = st.columns(2)
                zt_metric_label = c1.selectbox("Topology metric", topo_labels, key="zt_metric")
                zt_weight = c2.selectbox("Weight metric", WEIGHT_COLS, key="zt_weight")
                suffix = TOPO_METRICS[zt_metric_label]
                col = zt_weight + suffix if suffix else zt_weight
                st.caption(
                    f"Distribution of **{zt_metric_label}** ({zt_weight}) per zone, one point "
                    f"per match-team. Higher = {TOPO_HINT.get(suffix, '')}."
                )
                dd = topo_df[["zone", col]].dropna().copy()
                if dd.empty:
                    st.info("No values for this metric/weight combination.")
                else:
                    dd["Zone"] = dd["zone"].map(ZONE_LABEL)
                    zorder = [ZONE_LABEL[z] for z in zt_zones]
                    fig_zt = px.box(
                        dd, x="Zone", y=col, color="Zone", points="all",
                        category_orders={"Zone": zorder},
                        color_discrete_map={ZONE_LABEL[z]: ZONE_COLOR[z] for z in zt_zones},
                        labels={col: zt_metric_label},
                        title=f"{zt_metric_label} ({zt_weight}) by zone — match-team level")
                    fig_zt.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_zt, use_container_width=True)

                    mean_rows = []
                    for c in WEIGHT_COLS:
                        ccol = c + suffix if suffix else c
                        if ccol in topo_df.columns:
                            mean_rows.append(
                                topo_df.groupby("zone")[ccol].mean().rename(METRIC_LABEL[c]))
                    if mean_rows:
                        mean_mat = pd.concat(mean_rows, axis=1).T.reindex(columns=zt_zones)
                        mean_mat.columns = [ZONE_SHORT[z] for z in zt_zones]
                        st.markdown(f"**Mean {zt_metric_label} per zone** (rows = weight metric)")
                        st.dataframe(
                            mean_mat.style.background_gradient(cmap="Blues", axis=None).format("{:.3f}"),
                            use_container_width=True)

            # ── Spatial / network-geometry predictors of zone success ──────────
            if zone_spatial_df is not None and zone_raw is not None:
                st.subheader("Spatial predictors of zone defensive success")
                st.caption(
                    "Pearson r between each **spatial / network-geometry** metric and that "
                    "zone's **defensive stop rate** (n_stop_def/n_actions) — of the passes "
                    "the defence engaged (raw_involvement>0), the share it kept from "
                    "completing; higher = better defending. This is the confound-cleaner "
                    "target (the zone already conditions on press height) and uses a "
                    "consistent denominator — it **replaces the earlier n_success/n_actions**, "
                    "which divided completed passes (over *all* passes) by defended passes "
                    "and was unbounded (>1). Each metric maps to a visual feature of the "
                    "co-defending network drawn on the pitch (row labels name the feature). "
                    "Red = higher metric ↔ more successful defending, blue = less. `*` p<0.05 "
                    "· `†` survives BH-FDR (q<0.05). Edge unevenness is shown for every weight "
                    "metric — each defines a different co-defending edge set. The **Whole "
                    "network** column applies the same metrics to the full-match (non-zoned) "
                    "co-defending network vs. the pooled match stop rate (Σstop/Σactions), "
                    "for comparison against the per-zone cells. Built from "
                    "`2026-06-24_zone_spatial_metrics.csv`.")
                scorr = build_zone_spatial_corr(zone_spatial_df, zone_raw)
                if scorr.empty:
                    st.info("Not enough data to correlate spatial predictors with zone success.")
                else:
                    n_q = int((scorr["q"] < 0.05).sum())
                    tp = scorr.loc[scorr["r"].abs().sort_values(ascending=False).index].iloc[0]
                    st.markdown(
                        f"**{n_q}** / {len(scorr)} predictor×zone cells survive BH-FDR · "
                        f"strongest: *{SPATIAL_VIS.get(tp['metric'], tp['metric'])}* in "
                        f"{ZONE_SHORT[tp['zone']]} (r={tp['r']:+.2f}, q={tp['q']:.3f})")
                    zones_s = [z for z in ZONE_ORDER + ["all"] if z in set(scorr["zone"])]
                    disp_rows = [m for m in SPATIAL_VIS if m in set(scorr["metric"])]
                    rmat, tmat, ylab = [], [], []
                    for mname in disp_rows:
                        rrow, trow = [], []
                        for z in zones_s:
                            sel = scorr[(scorr["metric"] == mname) & (scorr["zone"] == z)]
                            if len(sel):
                                r, p, q = sel["r"].iloc[0], sel["p"].iloc[0], sel["q"].iloc[0]
                                mark = "†" if q < 0.05 else ("*" if p < 0.05 else "")
                                rrow.append(r); trow.append(f"{r:+.2f}{mark}")
                            else:
                                rrow.append(np.nan); trow.append("")
                        rmat.append(rrow); tmat.append(trow); ylab.append(SPATIAL_VIS[mname])
                    rmat = pd.DataFrame(rmat, index=ylab, columns=[ZONE_SHORT[z] for z in zones_s])
                    fig_s = px.imshow(rmat, color_continuous_scale="RdBu_r", zmin=-0.6, zmax=0.6,
                                      aspect="auto", labels=dict(color="r"))
                    fig_s.update_traces(text=tmat, texttemplate="%{text}", textfont_size=11)
                    fig_s.update_xaxes(side="top", tickfont_size=11)
                    fig_s.update_yaxes(tickfont_size=10)
                    fig_s.update_layout(height=26 * len(disp_rows) + 90,
                                        margin=dict(l=4, r=4, t=40, b=4))
                    st.plotly_chart(fig_s, use_container_width=True, key="zspatial_heat")
                    with st.expander("Full spatial-predictor table (all metrics · r, p, q, n)"):
                        st.dataframe(
                            scorr.assign(visual=scorr["metric"].map(SPATIAL_VIS),
                                         zone=scorr["zone"].map(ZONE_LABEL))
                                 .sort_values(["zone", "p"])
                                 .round({"r": 3, "p": 4, "q": 3}),
                            use_container_width=True, height=360)

        _render_zone_topology(zone_topo_dfs[method])

with tab_zcon:
    st.caption(
        "Within-team **zone contrasts**: how a team's co-defending structure *changes* "
        "across the pitch (own third → midfield → high press) rather than its level in "
        "any one zone. Each match-team gives one difference per metric, so squad size and "
        "overall style **cancel out**. We then **partial-pool** across each team's matches "
        "with a random-intercept mixed model — match-to-match formation/personnel changes "
        "become within-team noise that gets shrunk away, so no node correspondence across "
        "matches is needed — and ask whether the *gradient* is a stable team fingerprint. "
        "**Δ** = simple zone difference; **slope** = OLS over press depth "
        "(own −1 · mid 0 · high +1). Built from `2026-06-19_zone_topology(<method>).csv`."
    )
    if zone_topo_dfs is None:
        st.warning(
            "Zone topology files not found. Run `scripts/2026-06-19_zone_topology.py` to "
            "generate `2026-06-19_zone_topology(<method>).csv` (one per edge-weight method)."
        )
    else:
        @st.fragment
        def _render_zone_contrasts(topo_df):
            # faced-pass table (thirds) — used for both gating and the optional correction
            znp = None
            if zone_raw is not None:
                _zr = zone_raw[zone_raw["scheme"] == "thirds"]
                znp = _zr.pivot_table(index="match_team_id", columns="zone",
                                      values="n_passes", aggfunc="sum")

            c1, c2, c3 = st.columns([3, 1.5, 1.6])
            contrast_opts = list(ZONE_DELTAS) + ["slope"]
            _fmt = lambda k: "Slope (own→high press)" if k == "slope" else k
            contrast = c1.radio("Contrast", contrast_opts, format_func=_fmt,
                                horizontal=True, key="zcon_contrast")
            zcon_correct = c2.checkbox(
                "Correct for zone passes against", value=False, key="zcon_correct",
                disabled=(zone_raw is None),
                help="Residualise each zone's metric on that zone's own faced-pass count "
                     "(n_passes) before differencing, so the contrast reflects defensive "
                     "structure rather than how much the ball was in that zone.")
            min_pass = c3.slider(
                "Min zone passes faced", 0, 100, 0, step=5, key="zcon_minpass",
                disabled=(zone_raw is None),
                help="Exclude a team's zone observation when that zone faced fewer than "
                     "this many passes, so degenerate near-empty zone graphs (where "
                     "clustering/centralization are ill-defined) don't enter the contrasts.")

            topo_g = _gate_sparse_zones(topo_df, znp, min_pass)
            cdf = build_zone_contrasts(topo_g, contrast,
                                       correct=zcon_correct,
                                       zone_npass=(znp if zcon_correct else None))
            metric_cols = [c for c in cdf.columns if c not in ("match_team_id", "team_name")]
            lme_all = partial_pool_all(cdf, metric_cols,
                                       key=(method, contrast, zcon_correct, min_pass))

            # ── Ranked summary: which contrasts are stable team traits? ────────
            st.subheader("Stable contrasts — ranked by partial-pooled ICC")
            st.caption(
                "**ICC_lme** = partial-pooling (REML mixed-model) reliability of the "
                "contrast across each team's 3–7 matches — the headline fingerprint test, "
                "robust to the unbalanced design. **ICC** = the simpler ANOVA ICC(1,1) for "
                "reference. High ICC + low q ⇒ a repeatable team trait, not match noise. "
                "**fe_mean** = league-average contrast (is there a *systematic* gradient "
                "at all?). **p** = raw F-test · **q** = Benjamini-Hochberg FDR across all "
                f"{len(metric_cols)} metrics in this view · **sig** from **q**. "
                "🟢 stable (ICC>0.5). `n_obs` is small (a contrast needs the metric in "
                "*both* zones; derived metrics drop out on sparse high-press graphs), so "
                "read q as indicative."
                + ("  \n*Correction on:* each zone residualised on its faced-pass count "
                   "before differencing." if zcon_correct else "")
                + (f"  \n*Gating on:* zone observations with <{min_pass} faced passes "
                   "excluded." if min_pass > 0 else ""))
            summary = pd.DataFrame(compute_icc_rows(cdf, metric_cols))
            if summary.empty:
                st.info("Not enough replicated data to compute contrast ICCs for this selection.")
            else:
                lme_rows = [{"metric": c, "ICC_lme": round(r["icc_lme"], 3),
                             "fe_mean": round(r["fe_mean"], 3)}
                            for c, r in lme_all.items() if r is not None]
                if lme_rows:
                    summary = summary.merge(pd.DataFrame(lme_rows), on="metric", how="left")
                else:
                    summary["ICC_lme"] = np.nan
                    summary["fe_mean"] = np.nan
                _sort = "ICC_lme" if summary["ICC_lme"].notna().any() else "ICC"
                summary = summary.sort_values(_sort, ascending=False).reset_index(drop=True)
                n_stable = int((summary["ICC_lme"] > 0.5).sum())
                n_raw    = int((summary["p"] < 0.05).sum())
                n_sig    = int((summary["q"] < 0.05).sum())
                top = summary.iloc[0]
                st.markdown(
                    f"**{n_stable}** / {len(summary)} contrasts stable (ICC_lme>0.5) · "
                    f"**{n_raw}** sig at raw p<0.05 → **{n_sig}** survive BH-FDR (q<0.05) · "
                    f"strongest: `{top['metric']}` — ICC_lme={top['ICC_lme']} · ICC={top['ICC']} {top['sig']}")
                _order = [c for c in ["metric", "ICC_lme", "ICC", "fe_mean", "p", "q", "sig",
                                      "n_obs", "n_teams", "interpretation"] if c in summary.columns]
                st.dataframe(
                    summary[_order].style.background_gradient(
                        cmap="RdYlGn",
                        subset=[c for c in ["ICC_lme", "ICC"] if c in _order], vmin=0, vmax=1),
                    use_container_width=True, height=380)

            # ── Per-team fingerprint (partial-pooled / shrunk) ─────────────────
            st.subheader("Per-team fingerprint (partial-pooled)")
            if not _HAS_SM:
                st.info("statsmodels not installed — partial-pooled per-team estimates unavailable.")
            else:
                fmetric = st.selectbox("Metric (weight × topology column)", metric_cols,
                                       key="zcon_fmetric")
                r = lme_all.get(fmetric)
                if r is None:
                    st.info("Not enough replicated data to partial-pool this metric.")
                else:
                    blups = pd.Series(r["blups"], name="contrast").sort_values()
                    bdf = blups.reset_index().rename(columns={"index": "team_name"})
                    # discrete above/below-mean colour (a continuous scale centred on the
                    # mean paints near-mean bars white → invisible when ICC_lme≈0 collapses
                    # every team onto the mean).
                    bdf["vs_mean"] = np.where(bdf["contrast"] >= r["fe_mean"],
                                              "above mean", "below mean")
                    fig_bl = px.bar(
                        bdf, x="contrast", y="team_name", orientation="h",
                        color="vs_mean",
                        color_discrete_map={"above mean": "#C62828", "below mean": "#1565C0"},
                        category_orders={"team_name": bdf["team_name"].tolist()},
                        labels={"contrast": f"{_fmt(contrast)}  ({fmetric})",
                                "team_name": "", "vs_mean": ""},
                        title=f"Shrunk per-team contrast — {fmetric}  "
                              f"(ICC_lme={r['icc_lme']:.3f}, n={r['n_obs']} obs / {r['n_teams']} teams)")
                    fig_bl.add_vline(x=r["fe_mean"], line_dash="dash", line_color="#555")
                    fig_bl.update_layout(height=max(360, 20 * len(bdf) + 120),
                                         legend_title_text="")
                    st.plotly_chart(fig_bl, use_container_width=True, key="zcon_blup")
                    st.caption(
                        "Bars = **partial-pooled** (shrunk) contrast per team; teams with "
                        "few/noisy matches are pulled toward the dashed league mean "
                        f"(fe_mean={r['fe_mean']:.3f}). When ICC_lme≈0 the model finds no "
                        "between-team signal and every bar collapses to the mean — i.e. no "
                        "fingerprint in this metric. Red = above mean, blue = below.")

            # ── Per topology-metric ICC tables (grouped like the Robustness tab) ──
            with st.expander("ANOVA ICC tables grouped by topology metric"):
                for name, cols in GROUPS.items():
                    st.subheader(name)
                    if name in GROUP_DESC:
                        st.caption(GROUP_DESC[name])
                    icc_tbl(cdf, cols)

        _render_zone_contrasts(zone_topo_dfs[method])

with tab_corr:
    @st.fragment
    def _frag_tab_corr():
        for _tab, _partial in zip(
            st.tabs(["Raw", "Partial (controlling passes_against)"]), [False, True]
        ):
            with _tab:
                for name, cols in GROUPS.items():
                    st.subheader(name)
                    if name in GROUP_DESC:
                        st.caption(GROUP_DESC[name])
                    corr_tbl(df_corr, cols, _partial)
    _frag_tab_corr()

with tab_icc:
    @st.fragment
    def _frag_tab_icc():
        st.markdown(
            "**ICC(1,1)**: >0.75 stable trait · 0.5–0.75 moderate · <0.5 match-driven  \n"
            "**p** = raw F-test (H₀: ICC = 0) · **q** = Benjamini-Hochberg FDR-adjusted "
            "across the 6 weight metrics in each table  \n"
            "**sig** (based on **q**): \\* q<0.05 · \\*\\* q<0.01 · \\*\\*\\* q<0.001"
        )
        stage = st.selectbox("Competition stage",
                             ["All"] + sorted(df_corr["competition_stage"].dropna().unique().tolist()))
        dff = df_corr if stage == "All" else df_corr[df_corr["competition_stage"] == stage]
        if stage != "All":
            st.caption(f"{len(dff)} obs · {dff['team_name'].nunique()} teams")
        for name, cols in GROUPS.items():
            st.subheader(name)
            if name in GROUP_DESC:
                st.caption(GROUP_DESC[name])
            icc_tbl(dff, cols)
    _frag_tab_icc()

with tab_reg:
    @st.fragment
    def _frag_tab_reg():
        st.caption(
            "OLS regression predicting defensive outcomes from network metrics. "
            "**std β** allows comparison of effect sizes across variables. "
            "Data: match-team level from df_corr."
        )

        _skip_cols = {"match_team_id", "team_name", "competition_stage", "passes_against"} | set(OUTCOME_COLS)
        _all_metrics = [c for c in df_corr.columns if c not in _skip_cols]

        rc1, rc2 = st.columns([2, 1])
        with rc1:
            reg_x = st.multiselect(
                "X variables (predictors)",
                _all_metrics,
                default=[m for m in ["raw_involvement", "raw_involvement_centralization_weighted"]
                         if m in _all_metrics],
                key="reg_x",
            )
        with rc2:
            reg_y      = st.selectbox("Y (outcome)", OUTCOME_COLS, key="reg_y")
            reg_ctrl   = st.checkbox("Control for passes_against", value=True, key="reg_ctrl")

        if reg_ctrl and "passes_against" in df_corr.columns:
            x_full = reg_x + ["passes_against"]
        else:
            x_full = reg_x

        if len(x_full) == 0:
            st.info("Select at least one X variable.")
        else:
            res = run_ols(df_corr, x_full, reg_y)
            if res is None:
                st.warning("Not enough observations or singular matrix — try fewer predictors.")
            else:
                coef, r2, r2_adj, n = res
                st.markdown(
                    f"**n = {n}** &nbsp;|&nbsp; **R² = {r2:.3f}** &nbsp;|&nbsp;"
                    f" **Adj. R² = {r2_adj:.3f}**"
                )

                def _style_coef(row):
                    if row.name == "(Intercept)":
                        return [""] * len(row)
                    p = row.get("p", 1.0)
                    if p < 0.05:
                        return ["background-color: #c6efce"] * len(row)
                    return [""] * len(row)

                st.subheader("Coefficients")
                st.dataframe(
                    coef.style.apply(_style_coef, axis=1)
                              .format({"β": "{:.4f}", "std β": "{:.4f}", "SE": "{:.4f}",
                                       "t": "{:.3f}", "p": "{:.4f}",
                                       "CI 2.5%": "{:.4f}", "CI 97.5%": "{:.4f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "β = raw coefficient · std β = standardised (comparable effect size) · "
                    "🟢 highlighted rows: p < 0.05"
                )

                # std β bar chart for quick visual comparison (exclude intercept)
                _coef_plot = coef.drop("(Intercept)").reset_index()
                _coef_plot.columns = ["variable"] + list(_coef_plot.columns[1:])
                if not _coef_plot["std β"].isna().all():
                    fig_coef = px.bar(
                        _coef_plot, x="std β", y="variable", orientation="h",
                        color="std β", color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        error_x=None,
                        title="Standardised coefficients (std β)",
                        labels={"variable": "", "std β": "std β"},
                    )
                    fig_coef.update_layout(yaxis={"autorange": "reversed"}, height=40 * len(_coef_plot) + 120)
                    st.plotly_chart(fig_coef, use_container_width=True, key="reg_std_beta")
    _frag_tab_reg()

# ── Edge-weight sensitivity ───────────────────────────────────────────────────
# Headline KPI = corr(network metric, outcome), raw + passes_against-partial,
# computed under all four edge-weight methods. Only the four weight-dependent
# metric families can move; topology-only metrics are method-invariant by
# construction (they use the method-independent edge-count threshold).
_SENS_METHODS    = ["min", "average", "sum", "product"]
_SENS_PRIMARY    = "min"
_SENS_DEPENDENT  = {
    "":                         "Total Strength",
    "_gini":                    "Gini (strength inequality)",
    "_cc_weighted":             "Clustering (weighted)",
    "_centralization_weighted": "Freeman Centralization (weighted)",
}


def _sens_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


@st.cache_data
def build_edge_weight_sensitivity(thr=1):
    proc = {k: process(edge_dfs[k], thr) for k in _SENS_METHODS}
    rows = []
    for c in WEIGHT_COLS:
        for suf, fam in _SENS_DEPENDENT.items():
            col = c + suf
            for t in OUTCOME_COLS:
                for kind in ("raw", "partial"):
                    rec = {"metric_family": fam, "node_metric": c,
                           "outcome": t, "kind": kind}
                    for k in _SENS_METHODS:
                        d = proc[k]
                        if kind == "raw":
                            v = d[[col, t]].dropna()
                            r, p = pearsonr(v[col], v[t])
                        else:
                            s = d[[col, t, "passes_against"]].dropna()
                            a = _resid(s[col].values, s["passes_against"].values)
                            b = _resid(s[t].values,   s["passes_against"].values)
                            mk = ~(np.isnan(a) | np.isnan(b))
                            r, p = pearsonr(a[mk], b[mk])
                        rec[k] = r
                        rec[k + "_sig"] = _sens_stars(p)
                    rows.append(rec)
    return pd.DataFrame(rows)


def _safe_corr(fn, a, b):
    """Pairwise correlation that returns NaN instead of raising on a constant
    input (some metrics are all-zero under a given method)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return fn(a, b)[0]
    except Exception:
        return np.nan


@st.cache_data
def build_edge_weight_agreement(thr=1, methods=tuple(_SENS_METHODS)):
    """Outcome-free sensitivity: do the combining operators produce the
    *same metric values* (up to scale/rank)? For each weight-dependent metric
    we align the `methods` on match_team_id and compute pairwise Pearson and
    Spearman over the match-teams — no reference to any defensive outcome. If
    the operators agree, every downstream result is robust to the choice by
    construction. Note `average = sum/2` exactly, so sum-vs-average is 1.0 by
    definition and acts as a built-in sanity check. `methods` lets the caller
    drop an operator (e.g. the multiplicative `product`) from the comparison."""
    methods = list(methods)
    proc = {k: process(edge_dfs[k], thr).set_index("match_team_id")
            for k in methods}
    pairs = list(combinations(methods, 2))
    rows = []
    for c in WEIGHT_COLS:
        for suf, fam in _SENS_DEPENDENT.items():
            col = c + suf
            M = pd.DataFrame({k: proc[k][col] for k in methods}).dropna()
            rec = {"metric_family": fam, "node_metric": c, "n": len(M)}
            for k in methods:                             # each method vs primary (min)
                rec["p_" + k] = _safe_corr(pearsonr,  M[_SENS_PRIMARY], M[k]) if len(M) else np.nan
                rec["s_" + k] = _safe_corr(spearmanr, M[_SENS_PRIMARY], M[k]) if len(M) else np.nan
            pp = [_safe_corr(pearsonr,  M[a], M[b]) for a, b in pairs]
            ss = [_safe_corr(spearmanr, M[a], M[b]) for a, b in pairs]
            rec["p_min_pair"] = np.nanmin(pp) if np.any(~np.isnan(pp)) else np.nan
            rec["s_min_pair"] = np.nanmin(ss) if np.any(~np.isnan(ss)) else np.nan
            rec["p_sumavg"] = _safe_corr(pearsonr,  M["sum"], M["average"]) if len(M) else np.nan
            rec["s_sumavg"] = _safe_corr(spearmanr, M["sum"], M["average"]) if len(M) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def _sens_intrinsic_view(thr):
    """Outcome-free view of the edge-weight sensitivity tab (option A)."""
    st.markdown(
        "Do the four operators produce the **same metric values**? For every "
        "weight-dependent metric we align all four methods on the 128 "
        "match-teams and correlate them pairwise — **no outcome involved**. "
        "If the operators agree, every downstream result is robust to the "
        "choice by construction. Pearson is scale/location-invariant and "
        "Spearman is rank-only, so the differing scales (sum is 2× average, "
        "etc.) don't distort the comparison.")
    drop_product = st.checkbox(
        "Exclude `product`",
        value=False,
        help="`product` is the only multiplicative operator — it squashes the "
             "value scale so a few high-value edges dominate, which can depress "
             "the linear (Pearson) agreement while leaving the rank order "
             "intact. Tick this to compare only the additive/min operators.")
    methods = [m for m in _SENS_METHODS if not (drop_product and m == "product")]
    agr = build_edge_weight_agreement(thr, tuple(methods))
    kind = st.radio("Correlation", ["Spearman (rank)", "Pearson"],
                    horizontal=True, index=0)
    pre = "s_" if kind.startswith("Spearman") else "p_"
    label = "ρ" if pre == "s_" else "r"
    others = [m for m in methods if m != _SENS_PRIMARY]
    minpair = agr[pre + "min_pair"]
    sumavg  = agr[pre + "sumavg"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("worst-case min pairwise", f"{np.nanmin(minpair):.3f}")
    c2.metric("mean min pairwise", f"{np.nanmean(minpair):.3f}")
    c3.metric("metrics with min pairwise < 0.95",
              f"{int((minpair < 0.95).sum())} / {minpair.notna().sum()}")
    # sum == average/2 exactly -> this must be 1.000; flags a pipeline bug if not
    sa_ok = np.allclose(sumavg.dropna(), 1.0, atol=1e-6)
    c4.metric("sum≡average sanity", "✓ 1.000" if sa_ok else "✗ BROKEN",
              delta=None if sa_ok else f"min {np.nanmin(sumavg):.3f}",
              delta_color="off" if sa_ok else "inverse")
    st.caption(
        f"High = robust: the operators rank/scale the match-teams the same way, "
        f"so the choice of **min** is immaterial *regardless of any outcome*. "
        f"`sum` and `average` are a monotone rescale of each other, so their "
        f"correlation is exactly 1 by construction — a built-in pipeline check. "
        f"Columns show {label}(min, method); **min pairwise** is the worst "
        f"agreement over all six method pairs.")

    disp = agr[["metric_family", "node_metric", "n"]].copy()
    for m in others:
        disp[f"{label}(min,{m})"] = agr[pre + m].round(3)
    disp["min pairwise"] = minpair.round(3)
    st.dataframe(
        disp.style.background_gradient(
            cmap="Greens", subset=["min pairwise"], vmin=0.8, vmax=1.0),
        use_container_width=True, hide_index=True)
    st.download_button(
        "Download agreement table (CSV)",
        agr.to_csv(index=False).encode(),
        "edge_weight_agreement.csv", "text/csv")


with tab_sens:
    @st.fragment
    def _frag_tab_sens():
        st.subheader("Edge-weight method sensitivity")
        st.markdown(
            "Each co-defensive event's two endpoint values are combined into an "
            "edge weight by one of four operators: **sum**, **average** (=sum/2), "
            "**min** (weakest-link / conjunctive — *primary*), **product**. "
            "Two ways to ask whether that arbitrary choice matters:")
        view = st.radio(
            "Robustness view",
            ["vs. outcome (headline KPI)", "intrinsic agreement (outcome-free)"],
            horizontal=True,
            help="'vs. outcome' checks whether each metric's correlation with "
                 "defensive outcomes moves across operators. 'intrinsic "
                 "agreement' checks whether the operators produce the same "
                 "metric values at all — independent of any outcome.")

        if view.startswith("intrinsic"):
            _sens_intrinsic_view(thr)
            return

        st.markdown(
            "This re-runs every network metric under all four and compares the "
            "headline KPI: each metric's correlation with defensive outcomes "
            "(raw + partial, controlling `passes_against`).\n\n"
            "Only the four **weight-dependent** families below can move — the "
            "topology-only metrics (density, unweighted clustering / "
            "centralization, assortativity, k-core, LCC) use the "
            "method-independent edge-count threshold and are **identical across "
            "methods by construction**.")

        sens = build_edge_weight_sensitivity(thr)
        other = st.selectbox("Compare primary (min) against",
                             ["average", "sum", "product"], index=0)
        dr_pair = (sens[_SENS_PRIMARY] - sens[other]).abs()
        dr_all  = sens[_SENS_METHODS].max(axis=1) - sens[_SENS_METHODS].min(axis=1)
        sig_flip = (sens[_SENS_PRIMARY + "_sig"].astype(bool)
                    != sens[other + "_sig"].astype(bool))
        n = len(sens)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"max |Δr|  min vs {other}", f"{dr_pair.max():.3f}")
        c2.metric(f"mean |Δr|  min vs {other}", f"{dr_pair.mean():.3f}")
        c3.metric("p<.05 verdict changes", f"{int(sig_flip.sum())} / {n}")
        c4.metric("max |Δr| across all 4", f"{dr_all.max():.3f}")
        st.caption(
            "Small = robust: the primary choice (min) barely changes any "
            "correlation. Sign flips at r≈0 are near-zero noise, not "
            "substantive disagreement — read the significance-change count, "
            "not raw sign flips.")

        disp = sens.copy()
        for k in _SENS_METHODS:
            disp[k] = [f"{r:+.2f}{s}" for r, s in zip(sens[k], sens[k + "_sig"])]
        disp[f"|Δr| min-{other}"] = dr_pair.round(3)
        disp = disp[["metric_family", "node_metric", "outcome", "kind",
                     *_SENS_METHODS, f"|Δr| min-{other}"]]
        st.dataframe(
            disp.style.background_gradient(
                cmap="Reds", subset=[f"|Δr| min-{other}"], vmin=0, vmax=0.3),
            use_container_width=True, hide_index=True)
        st.download_button(
            "Download sensitivity table (CSV)",
            sens.to_csv(index=False).encode(),
            "edge_weight_sensitivity.csv", "text/csv")
    _frag_tab_sens()


with tab_pstyle:
    @st.fragment
    def _frag_pstyle_overview():
        st.subheader("Results overview — all weights × zones")
        if zone_edge_avg is None or PRESS_ROLE_MAP is None:
            return
        st.caption(
            "One row per **edge-weight × zone** combination, summarising the three "
            "estimators below: **self-similarity** (is the pattern a team trait?), "
            "**LOO identification** (does the pattern name the team?), and "
            "**deviation → outcome** (does straying from your identity cost stop "
            "rate, within-team?). Generation runs the full permutation sweep, so it "
            "is **off by default** — flip the toggle to compute it.")
        if not st.toggle("Generate overview table", value=False,
                         key="pstyle_overview_on"):
            return
        _oc1, _oc2 = st.columns(2)
        _nperm = _oc1.select_slider("Permutations (overview)", [500, 1000, 2000],
                                    value=1000, key="pstyle_overview_nperm")
        _drop_gk = _oc2.toggle("Exclude goalkeeper", value=False,
                               key="pstyle_overview_drop_gk", disabled=not GK_KEYS,
                               help="Recompute the whole sweep with goalkeeper edges "
                                    "removed and roles re-tertiled among outfielders.")

        _combos = [(w, z) for w in WEIGHT_COLS for z in PRESS_ZONE_OPTIONS]
        _rows, _bar = [], st.progress(0.0, "Sweeping weight × zone…")
        for _i, (_w, _z) in enumerate(_combos):
            _zlabel = "full" if _z == FULL_NETWORK else _z
            _s = pressing_style_stats(_w, _z, _nperm, drop_gk=_drop_gk)
            _loo = pressing_loo_identification(_w, _z, _nperm, drop_gk=_drop_gk)
            _dev = pressing_deviation_outcome(_w, _z, nperm=_nperm, drop_gk=_drop_gk)
            _has_loo = _loo.get("total", 0) > 0
            _has_dev = _dev.get("n", 0) >= 8
            _mult = (_loo["top1"] / _loo["chance"]
                     if _has_loo and _loo["chance"] > 0 else np.nan)
            _rows.append({
                "weight": _w, "zone": _zlabel, "n_mt": _s["n_mt"],
                "Δcos": round(_s["delta"], 3), "self p": _s["perm_p"],
                "LOO top-1": _loo["top1"] if _has_loo else np.nan,
                "chance": _loo["chance"] if _has_loo else np.nan,
                "×chance": round(_mult, 2) if pd.notna(_mult) else np.nan,
                "LOO p": _loo["perm_p"] if _has_loo else np.nan,
                "within-r": round(_dev["r_within"], 3) if _has_dev else np.nan,
                "within p": _dev["p_within_perm"] if _has_dev else np.nan,
                "n dev": _dev.get("n", 0),
            })
            _bar.progress((_i + 1) / len(_combos), f"{_w} × {_zlabel}")
        _bar.empty()

        ov = pd.DataFrame(_rows)

        # --- multiple-comparisons correction --------------------------------
        # BH-FDR q-values per estimator across the 24 weight×zone cells. The 6
        # edge weights are highly correlated, so the *effective* family is far
        # smaller than 24 — reported below via Li & Ji M_eff so a Bonferroni-24
        # penalty is not applied blindly.
        ov["self q"] = _bh_fdr(ov["self p"].values)
        ov["LOO q"] = _bh_fdr(ov["LOO p"].values)
        ov["within q"] = _bh_fdr(ov["within p"].values)

        _meff_by_zone = {z: pressing_weight_meff(z, _drop_gk) for z in PRESS_ZONE_OPTIONS}
        _m_eff = float(sum(_meff_by_zone.values()))      # zones ~indep, weights collapse
        _nominal = int(ov["self p"].notna().sum())
        _alpha_eff = 1.0 - (1.0 - 0.05) ** (1.0 / _m_eff) if _m_eff > 0 else 0.05

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("nominal tests / estimator", _nominal)
        mc2.metric("effective tests (M_eff)", f"{_m_eff:.1f}",
                   help="Σ over zones of Li & Ji effective # among the 6 correlated "
                        "weights. The 6 weights collapse to ~2–3 independent tests "
                        "per zone, so the real family is far below 24.")
        mc3.metric("Šidák α (effective)", f"{_alpha_eff:.4f}",
                   help="Family-wise α threshold using M_eff effective tests: "
                        "1−(1−0.05)^(1/M_eff). More gracious than Bonferroni-24 "
                        f"(α={0.05/max(_nominal,1):.4f}).")

        def _stars(p):
            if pd.isna(p):
                return ""
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        # stars now reflect BH-FDR q (corrected significance)
        ov["self"] = ov["self q"].map(_stars)
        ov["LOO"] = ov["LOO q"].map(_stars)
        ov["dev"] = ov["within q"].map(_stars)
        for _pc in ("self p", "LOO p", "within p", "self q", "LOO q", "within q"):
            ov[_pc] = ov[_pc].round(4)
        ov = ov[["weight", "zone", "n_mt", "Δcos", "self p", "self q", "self",
                 "LOO top-1", "chance", "×chance", "LOO p", "LOO q", "LOO",
                 "within-r", "within p", "within q", "dev", "n dev"]]

        def _hi_q(col):
            return ["background-color:#c7e9c0" if (pd.notna(v) and v < 0.05) else ""
                    for v in col]
        sty = (ov.style
               .apply(_hi_q, subset=["self q", "LOO q", "within q"])
               .background_gradient(cmap="Greens", subset=["Δcos"])
               .format({"LOO top-1": "{:.1%}", "chance": "{:.1%}"}, na_rep="—"))
        st.dataframe(sty, use_container_width=True, height=min(40 + 28 * len(ov), 760))
        st.caption(
            ("**Goalkeeper excluded.** " if _drop_gk else "") +
            "Green / stars = **BH-FDR q<0.05** (multiple-comparisons-corrected); raw "
            "permutation p shown alongside. **Δcos** = within− between-team cosine "
            "similarity; **×chance** = LOO top-1 ÷ permuted chance; **within-r** "
            "negative = deviating from identity lowers stop rate. The 6 edge weights "
            "are strongly correlated (M_eff ≈ 2–3 per zone), so BH-FDR — valid under "
            "positive dependence — is used rather than a Bonferroni over 24 cells, "
            "and the effective family (M_eff above) is what a Šidák threshold should "
            "use. Blanks (—) = too few eligible match-teams.")
        _fname = ("pressing_style_overview_nogk.csv" if _drop_gk
                  else "pressing_style_overview.csv")
        st.download_button("Download overview (CSV)", ov.to_csv(index=False).encode(),
                           _fname, "text/csv", key="pstyle_overview_dl")
    _frag_pstyle_overview()

    @st.fragment
    def _frag_tab_pstyle():
        st.subheader("Pressing-style fingerprint — role-pair co-defending pattern")
        if zone_edge_avg is None or PRESS_ROLE_MAP is None:
            st.info("Needs `scripts/2026-06-18_zone_network_edge(average).csv` and "
                    "`scripts/2026-06-18_zone_network_positions.csv`.")
            return
        st.markdown(
            "Projects the co-defending network onto **pitch roles** (line "
            "B/M/F × channel L/C/R, by within-block tertiles) instead of player "
            "identities, so a team's *pattern* of who-presses-with-whom is "
            "comparable across matches with different line-ups. Each match-team's "
            "role-pair weight vector is row-normalised — a **pattern**, not a "
            "volume.\n\n"
            "**Question:** is this pattern a *team trait* — are a team's matches "
            "more similar to **each other** than to other teams? Measured by cosine "
            "similarity, tested against a **team-label permutation** null (the "
            "correct test for non-independent match-team pairs). This is the one "
            "network signal that is **geometry-free** *and* survives as a "
            "faint-but-real team trait — strongest in the **high press**. It is a "
            "**style descriptor** (scouting), not a predictor of defensive success.")

        c1, c2, c3 = st.columns(3)
        weight = c1.selectbox("Edge weight", WEIGHT_COLS,
                              index=WEIGHT_COLS.index("valued_involvement"))
        zone = c2.selectbox("Zone", PRESS_ZONE_OPTIONS,
                            index=PRESS_ZONE_OPTIONS.index("high_press"),
                            format_func=lambda z: "full network (all zones)"
                            if z == FULL_NETWORK else z)
        nperm = c3.select_slider("Permutations", [500, 1000, 2000, 5000], value=2000)
        drop_gk = st.toggle(
            "Exclude goalkeeper", value=False, key="pstyle_drop_gk",
            help="Drop the goalkeeper from the role map: GK edges are removed and "
                 "the B/M/F × L/C/R role tertiles are recomputed among outfield "
                 "players only. Tests whether the team-trait signal is carried by "
                 "the keeper or by the outfield pressing shape.",
            disabled=not GK_KEYS)
        if not GK_KEYS:
            st.caption("⚠️ Goalkeeper labels unavailable — exclusion toggle disabled.")

        s = pressing_style_stats(weight, zone, nperm, drop_gk=drop_gk)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("within-team cos", f"{s['within']:.3f}")
        m2.metric("between-team cos", f"{s['between']:.3f}")
        m3.metric("Δ (within − between)", f"{s['delta']:+.3f}")
        sig = s["perm_p"] < 0.05
        m4.metric("permutation p", f"{s['perm_p']:.4f}",
                  delta="team trait" if sig else "n.s.",
                  delta_color="normal" if sig else "off")
        st.caption(
            f"{s['n_mt']} match-teams across {s['n_teams']} teams. A team's matches "
            f"are {'**significantly** ' if sig else ''}more self-similar than under "
            "random team labels. Read the effect as **small by design** (Δcos≈0.03 "
            "in the high press): a *detectable* identity, not a strong fingerprint — "
            "it wants club-season data to aggregate. Own-third / midfield patterns "
            "are reactive (opponent-driven) and typically **not** significant.")

        with st.expander("Formation-confound check"):
            st.markdown(
                "Not every role pair exists in every match-team (e.g. BC-BC needs ≥2 "
                "central back-line players in the zone). Is the team-trait signal "
                "driven by **which pairs exist** (= formation structure) or by **how "
                "weight is distributed** across pairs (= pressing style)?")
            f1, f2, f3 = st.columns(3)
            sig_bin = s["bin_p"] < 0.05
            f1.metric("binary (presence/absence)", f"Δ={s['bin_delta']:+.4f}",
                      delta=f"p={s['bin_p']:.3f} — {'formation signal' if sig_bin else 'null'}",
                      delta_color="normal" if sig_bin else "off")
            f2.metric("full pattern (original)", f"Δ={s['delta']:+.4f}",
                      delta=f"p={s['perm_p']:.4f}",
                      delta_color="normal" if sig else "off")
            sig_com = s["com_p"] < 0.05 if pd.notna(s["com_p"]) else False
            f3.metric(f"common pairs only ({s['n_common']}/{s['n_total_pairs']})",
                      f"Δ={s['com_delta']:+.4f}" if pd.notna(s["com_delta"]) else "n/a",
                      delta=f"p={s['com_p']:.4f} — ≥80% presence" if pd.notna(s["com_p"]) else "too few pairs",
                      delta_color="normal" if sig_com else "off")
            st.caption(
                "**Binary** tests whether which role pairs *exist* is team-consistent "
                "(formation artifact). **Common pairs only** restricts to role pairs "
                "present in ≥80% of match-teams and re-normalises — removes the "
                "presence/absence layer entirely. If the signal survives in common "
                "pairs but binary is null, it is **pressing style, not formation**.")

        st.markdown("**Which role partnerships are most team-distinctive?** "
                    "ICC across a team's matches — higher = more of a signature.")
        icc = s["icc"].head(12)
        if not icc.empty:
            fig = px.bar(icc, x="icc", y="role_pair", orientation="h",
                         labels={"icc": "ICC (team trait)", "role_pair": "role pair"})
            fig.update_layout(yaxis=dict(autorange="reversed"), height=380,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Roles: line **B**ack/**M**id/**F**ront × channel **L**eft/"
                   "**C**entre/**R**ight, e.g. **BC-BC** = two central back-line "
                   "pressers; **FR-MR** = right-side forward + midfielder pressing "
                   "together.")

        st.markdown("**Team fingerprints** — mean role-pair share per team "
                    "(rows = teams, columns = role pairs, ordered by prevalence).")
        fp = s["fingerprint"]
        fp = fp[fp.mean().sort_values(ascending=False).index]
        fig2 = px.imshow(fp, aspect="auto", color_continuous_scale="Blues",
                         labels=dict(color="share", x="role pair", y="team"))
        fig2.update_layout(height=max(360, 18 * len(fp)),
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("Download fingerprints (CSV)", fp.to_csv().encode(),
                           f"pressing_fingerprint_{weight}_{zone}.csv", "text/csv")

        # ── #1 Leave-one-match-out team identification ─────────────────────────
        st.divider()
        st.markdown("### 1 · Leave-one-match-out team identification")
        st.markdown(
            "A stronger, more intuitive robustness test than per-pair ICC: can the "
            "pattern alone **name the team**? Each match is assigned to the team "
            "whose **leave-one-out** mean fingerprint — built *only* from that "
            "team's other matches — it most resembles (cosine, nearest-centroid). "
            "Accuracy well above chance means the co-defending pattern carries a "
            "real, **multivariate** team identity, even where any single role pair "
            "is only faintly stable.")
        loo = pressing_loo_identification(weight, zone, nperm, drop_gk=drop_gk)
        if loo.get("total", 0) == 0:
            st.info("No teams with ≥2 matches in this zone — cannot leave one out.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("top-1 accuracy", f"{loo['top1']:.1%}")
            k2.metric("top-3 accuracy", f"{loo['top3']:.1%}")
            k3.metric("chance (perm mean)", f"{loo['chance']:.1%}")
            sig_loo = loo["perm_p"] < 0.05
            k4.metric("permutation p", f"{loo['perm_p']:.4f}",
                      delta="identifiable" if sig_loo else "n.s.",
                      delta_color="normal" if sig_loo else "off")
            _mult = loo["top1"] / loo["chance"] if loo["chance"] > 0 else float("nan")
            st.caption(
                f"{loo['total']} held-out matches across {loo['n_teams']} teams with "
                f"≥2 matches. Chance = mean top-1 accuracy under {nperm} team-label "
                f"permutations ({loo['chance']:.1%}); observed top-1 is "
                f"**{_mult:.1f}× chance**. With only 3–7 matches per team this is a "
                "lower bound — the identity aggregates further on club-season data.")
            pt = loo["per_team"]
            if not pt.empty:
                figL = px.bar(pt.sort_values("acc"), x="acc", y="team", orientation="h",
                              labels={"acc": "top-1 accuracy", "team": ""},
                              title="How identifiable is each team by its pressing pattern?")
                figL.add_vline(x=loo["chance"], line_dash="dash", line_color="grey")
                figL.update_layout(height=max(360, 16 * len(pt)),
                                   margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(figL, use_container_width=True)

        # ── #2 Deviation from own identity vs match outcome ────────────────────
        st.divider()
        st.markdown("### 2 · Deviation from own identity vs match outcome")
        st.markdown(
            "Turns the descriptive trait into an **outcome** signal while "
            "controlling for team strength. For each match, the **cosine distance** "
            "between its pattern and the team's leave-one-out mean fingerprint — how "
            "far the team strayed from its own pressing identity. Does straying cost "
            "defensive success **that match**? The test is **within team** (each "
            "team is its own baseline), so squad-quality and dominance confounds — "
            "which sank every cross-sectional model — cancel. Outcome = **stop "
            "rate** in the matching pitch zone (scheme: pitch thirds).")
        if zone_raw is None:
            st.info("Needs `scripts/2026-06-08_team_zone_metrics.csv`.")
        else:
            dev = pressing_deviation_outcome(weight, zone, nperm=nperm, drop_gk=drop_gk)
            if dev.get("n", 0) < 8:
                st.info("Too few within-team observations for this zone "
                        "(need teams with ≥2 matches that also have zone stop-rate).")
            else:
                d1, d2, d3 = st.columns(3)
                sig_w = dev["p_within_perm"] < 0.05
                d1.metric("within-team r (deviation vs stop rate)",
                          f"{dev['r_within']:+.3f}",
                          delta=f"perm p={dev['p_within_perm']:.4f}",
                          delta_color="normal" if sig_w else "off")
                d2.metric("raw r (no team control)", f"{dev['r_raw']:+.3f}",
                          delta=f"p={dev['p_raw']:.3f} — confounded",
                          delta_color="off")
                d3.metric("obs / teams", f"{dev['n']} / {dev['n_teams']}")
                st.caption(
                    "**Negative** within-team r = deviating from the team's usual "
                    "pressing pattern goes with a **lower** stop rate (imposing your "
                    "identity helps defend). The within-team p permutes stop rate "
                    "**within each team**, preserving every team's baseline. The raw "
                    "r ignores team identity and is dominance-confounded — shown only "
                    "for contrast; the within-team estimate is the clean test.")
                figD = px.scatter(
                    dev["df"], x="dev_demean", y="suc_demean", hover_name="team",
                    hover_data=["deviation", "success"],
                    trendline="ols", trendline_color_override="black",
                    title="Within-team: deviation from identity vs zone stop rate",
                    labels={"dev_demean": "deviation from own fingerprint (team-demeaned)",
                            "suc_demean": "zone stop rate (team-demeaned)"})
                _add_quadrant_lines(figD, dev["df"], "dev_demean", "suc_demean", opacity=0.4)
                st.plotly_chart(figD, use_container_width=True)
    _frag_tab_pstyle()


with tab_data:
    @st.fragment
    def _frag_tab_data():
        st.subheader("Concentrated vs Balanced — Involvement (match level)")
        st.dataframe(df_conc_inv_match)
        st.subheader("Concentrated vs Balanced — Involvement (team level)")
        st.dataframe(df_conc_inv_team)
        st.subheader("Concentrated vs Balanced — Fault (match level)")
        st.dataframe(df_conc_fault_match)
        st.subheader("Concentrated vs Balanced — Fault (team level)")
        st.dataframe(df_conc_fault_team)
        st.subheader("Concentrated vs Balanced — Contribution (match level)")
        st.dataframe(df_conc_cont_match)
        st.subheader("Concentrated vs Balanced — Contribution (team level)")
        st.dataframe(df_conc_cont_team)
        st.subheader("Self vs Shared — match level")
        st.dataframe(df_self_match)
        st.subheader("Self vs Shared — team level")
        st.dataframe(df_self_sorted)
        st.subheader("Co-defenders — match level")
        st.dataframe(avg_co)
        st.subheader("Partnerships")
        st.dataframe(partnerships.head(200))
        st.subheader("Correlation / ICC — match level")
        st.dataframe(df_corr)
    _frag_tab_data()
