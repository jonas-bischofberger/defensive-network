"""
Team-level defensive network analysis (combined).

Tabs:
  1. Concentrated vs Balanced
  2. Self vs Shared
  3. Defensive Style
  4. Co-Defenders
  5. Correlation
  6. Robustness (ICC)
  7. Data
"""
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu, t as t_dist, f as f_dist

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

# Pitch-zone defensive metrics (produced by 2026-06-08_team_zone_metrics.py).
# Optional: app still runs without it (the Zones tab shows a hint instead).
try:
    zone_raw = pd.read_csv("scripts/2026-06-08_team_zone_metrics.csv")
    zone_raw["match_team_id"] = zone_raw["match_team_id"].astype(str)
except FileNotFoundError:
    zone_raw = None

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
    kcore_dict = {}
    for mid, grp in edge_df.groupby("match_team_id"):
        e = grp[grp[ec_col] >= thr] if ec_col in edge_df.columns else grp
        G = nx.Graph()
        G.add_edges_from(zip(e["player_1"], e["player_2"]))
        if G.number_of_nodes() >= 2:
            kcore_dict[mid] = max(nx.core_number(G).values())
    kcore_s = pd.Series(kcore_dict, name="kcore_max")
    df = (_join_outcomes(pd.concat([strength, cent_w, kcore_s], axis=1))
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
         "centralization_w", "kcore_max"] + OUTCOME_COLS + _pp_oc
    ].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


_Y_HIGH = {"centralization_w": "Centralized", "kcore_max": "Dense core"}
_Y_LOW  = {"centralization_w": "Distributed", "kcore_max": "Sparse core"}


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


def icc_tbl(df, cols):
    rows = []
    for c in cols:
        s = df[["team_name", c]].dropna()
        if s["team_name"].nunique() < 2:
            continue
        g   = s.groupby("team_name")[c]
        nt, ng, sz, mn = len(s), g.ngroups, g.count(), g.mean()
        msb = (sz * (mn - s[c].mean()) ** 2).sum() / (ng - 1)
        msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0  = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        # F-test for H0: ICC = 0 (MSB/MSW ~ F(ng-1, nt-ng))
        f_stat = msb / msw if msw > 0 else float("nan")
        p_val  = f_dist.sf(f_stat, ng - 1, nt - ng) if not np.isnan(f_stat) else float("nan")
        rows.append({
            "metric": c,
            "ICC": round(icc, 3),
            "F": round(f_stat, 2),
            "p": round(p_val, 4),
            "sig": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")),
            "n_teams": ng,
            "n_obs": nt,
            "interpretation": "stable trait" if icc > 0.5 else "match-driven",
        })
    st.dataframe(
        pd.DataFrame(rows).style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
        use_container_width=True,
    )


# ── Axis Selection & Quadrant Analysis ───────────────────────────────────────

def eta_sq_tbl(df, only_from=None):
    skip = {"match_team_id", "team_name", "competition_stage", "passes_against"} | set(OUTCOME_COLS)
    if only_from is not None:
        cols = [c for c in df.columns if c not in skip
                and any(c == p or c.startswith(p + "_") for p in only_from)]
    else:
        cols = [c for c in df.columns if c not in skip]
    rows = []
    for m in cols:
        s = df[["team_name", m]].dropna()
        if s["team_name"].nunique() < 2:
            continue
        grand = s[m].mean()
        ss_tot = ((s[m] - grand) ** 2).sum()
        if ss_tot == 0:
            continue
        ss_bet = s.groupby("team_name")[m].apply(
            lambda x: len(x) * (x.mean() - grand) ** 2
        ).sum()
        rows.append({"metric": m, "η²": round(ss_bet / ss_tot, 3)})
    return pd.DataFrame(rows).sort_values("η²", ascending=False).reset_index(drop=True)


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
ZONE_LABEL   = {"own": "Own-third (forced)", "mid": "Midfield", "high_press": "High press"}
ZONE_SHORT   = {"own": "Own", "mid": "Mid", "high_press": "High press"}
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

    Zone proportions are of involvement; press_index = high_press_share − own_share
    (mid is the neutral pivot). Joined with mean outcomes & knockout flag.
    """
    icol = f"{kind}_involvement_sum"
    d = zone_df[zone_df["scheme"] == scheme].merge(
        outcomes_df[["match_team_id", "team_name"]], on="match_team_id")
    g = (d.groupby(["team_name", "zone"], observed=True)
           .agg(inv=(icol, "sum")).reset_index())
    inv = g.pivot(index="team_name", columns="zone", values="inv").fillna(0)
    share = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0)
    t = pd.DataFrame(index=inv.index)
    for z in ZONE_ORDER:
        if z in share.columns:
            t[f"{z}_share"] = share[z]
    t["press_index"] = share.get("high_press", 0) - share.get("own", 0)
    perf = outcomes_df.groupby("team_name").agg(
        shots_against=("shots_against", "mean"), xg_against=("xg_against", "mean"),
        goals_against=("goals_against", "mean"),
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


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Defensive Network Analysis — Team Level")

with st.sidebar:
    method  = st.selectbox("Edge weight method", list(edge_dfs))
    st.subheader("Concentrated vs Balanced")
    metric_conc_inv   = st.selectbox("Involvement", INV_COLS, key="metric_conc_inv")
    metric_conc_fault = st.selectbox("Fault", FAULT_COLS, key="metric_conc_fault")
    metric_conc_cont  = st.selectbox("Contribution", CONTRIBUTION_COLS, key="metric_conc_cont")
    _y_opts = {"centralization_w": "Centralization (weighted)", "kcore_max": "Max K-core"}
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

(tab_conc, tab_self, tab_style, tab_codef, tab_zone,
 tab_corr, tab_icc, tab_reg, tab_data) = st.tabs([
    "Concentrated vs Balanced", "Self vs Shared", "Defensive Style", "Co-Defenders", "Zones",
    "Correlation", "Robustness (ICC)", "Regression", "Data",
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
    _y_label_conc = {"centralization_w": "Centralization (weighted)", "kcore_max": "Max K-core"}[y_conc]
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
        "Pick two metrics with high η² *and* low mutual correlation as quadrant axes."
    )
    eta_df = eta_sq_tbl(df_corr, only_from=INV_COLS)
    st.dataframe(
        eta_df.style.background_gradient(cmap="YlOrRd", subset=["η²"], vmin=0, vmax=1),
        use_container_width=True, height=320,
    )
    all_eta_metrics = eta_df["metric"].tolist()
    if len(all_eta_metrics) >= 2:
        with st.expander("Pairwise correlations — all metrics"):
            st.caption("Lower correlation = more independent dimensions — better for a 2-axis quadrant plot.")
            corr_all = df_corr[all_eta_metrics].corr().round(2)
            st.dataframe(
                corr_all.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
                use_container_width=True,
            )

with tab_self:
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

with tab_style:
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

with tab_codef:
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

with tab_zone:
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

        # ── Defensive style map (team level) ──────────────────────────────────
        st.subheader("Defensive style map (team level)")
        st.caption(
            "One point per team (pooled across its matches). **press index** = "
            "high-press proportion − own-third proportion (right = proactive/high line; "
            "left = forced/deep; midfield is the neutral pivot, not in this axis). Colour = "
            "reached knockout. Lower y (shots/xG against) = better defence."
        )
        sm1, sm2, sm3 = st.columns(3)
        _xopts = {"press_index": "Press index (high − own proportion)",
                  "high_press_share": "High-press proportion", "own_share": "Own-third proportion",
                  "mid_share": "Midfield proportion"}
        sm_kind = sm1.selectbox("raw / valued", ["raw", "valued"], key="style_kind")
        team_style = build_team_style(zone_raw, zone_scheme, outcomes, kind=sm_kind)
        _xavail = [k for k in _xopts if k in team_style.columns]
        sm_x = sm2.selectbox("X axis", _xavail, format_func=_xopts.__getitem__, key="style_x")
        sm_y = sm3.selectbox("Y axis (outcome)", OUTCOME_COLS, index=1, key="style_y")
        ts = team_style.reset_index().dropna(subset=[sm_x, sm_y])
        ts["Stage"] = ts["reached_knockout"].map({True: "Reached knockout", False: "Group only"})
        r_sm, p_sm = pearsonr(ts[sm_x], ts[sm_y])
        fig_sm = px.scatter(ts, x=sm_x, y=sm_y, color="Stage", text="team_name",
                            color_discrete_map={"Group only": "#bbbbbb",
                                                "Reached knockout": "#1f78b4"},
                            trendline="ols", trendline_scope="overall",
                            trendline_color_override="black",
                            labels={sm_x: _xopts[sm_x], sm_y: sm_y.replace("_", " ")},
                            title=f"{_xopts[sm_x]} vs {sm_y.replace('_',' ')}  "
                                  f"(r={r_sm:+.2f}, p={p_sm:.3f}, N={len(ts)})")
        fig_sm.update_traces(textposition="top center",
                             marker=dict(size=11, line=dict(width=1, color="white")),
                             selector=dict(mode="markers+text"))
        fig_sm.add_vline(x=ts[sm_x].median(), line_dash="dash", line_color="grey", opacity=0.5)
        fig_sm.add_hline(y=ts[sm_y].median(), line_dash="dash", line_color="grey", opacity=0.5)
        fig_sm.update_layout(height=620)
        st.plotly_chart(fig_sm, use_container_width=True)
        st.caption(
            "Note: press index strongly tracks **defensive** quality (fewer shots/xG "
            "against) but **not** knockout qualification — deep/forced teams can still "
            "advance via attack & results, so colour (knockout) scatters across the plot."
        )

        # index × performance summary tables (team level)
        _idx = [k for k in _xopts if k in team_style.columns]
        st.markdown(f"**Style index × outcome correlation** (team level, N={len(team_style)}, "
                    f"{sm_kind}). `*` p<0.05, `**` p<0.01.")
        rowsA = []
        for ix in _idx:
            cell = {}
            for oc in OUTCOME_COLS:
                s = team_style[[ix, oc]].dropna()
                r, p = pearsonr(s[ix], s[oc])
                cell[oc.replace("_against", " ag.")] = \
                    f"{r:+.2f}{'**' if p < 0.01 else '*' if p < 0.05 else ''}"
            rowsA.append(pd.Series(cell, name=_xopts[ix]))
        st.table(pd.DataFrame(rowsA))

        st.markdown("**Style index by stage** — median (knockout vs group-only) + "
                    "Mann–Whitney U p.")
        rowsB = []
        for ix in _idx:
            a = team_style[team_style["reached_knockout"]][ix].dropna()
            b = team_style[~team_style["reached_knockout"]][ix].dropna()
            if len(a) > 2 and len(b) > 2:
                _, p = mannwhitneyu(a, b)
                rowsB.append(dict(Index=_xopts[ix], knockout_med=round(a.median(), 3),
                                  group_med=round(b.median(), 3), p=round(p, 3),
                                  sig="*" if p < 0.05 else ""))
        if rowsB:
            st.table(pd.DataFrame(rowsB).set_index("Index"))

with tab_corr:
    for _tab, _partial in zip(
        st.tabs(["Raw", "Partial (controlling passes_against)"]), [False, True]
    ):
        with _tab:
            for name, cols in GROUPS.items():
                st.subheader(name)
                if name in GROUP_DESC:
                    st.caption(GROUP_DESC[name])
                corr_tbl(df_corr, cols, _partial)

with tab_icc:
    st.markdown(
        "**ICC(1,1)**: >0.75 stable trait · 0.5–0.75 moderate · <0.5 match-driven  \n"
        "**sig** (F-test H₀: ICC = 0): \\* p<0.05 · \\*\\* p<0.01 · \\*\\*\\* p<0.001"
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

with tab_reg:
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

with tab_data:
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
