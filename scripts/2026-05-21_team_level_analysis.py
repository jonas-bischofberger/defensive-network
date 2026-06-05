"""
Team-level defensive network analysis.
Extends 2026-05-05_team_level_analysis.py with a Co-Defenders tab.

Tabs:
  1. Concentrated vs Balanced
  2. Self vs Shared
  3. Defensive Style
  4. Co-Defenders  ← new: avg co-defenders, partnership heatmap, top pairs
  5. Data
"""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Data loading ──────────────────────────────────────────────────────────────
outcomes  = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
# nodes     = pd.read_csv("scripts/2026-04-29-player_level_metrics.csv")
nodes     = pd.read_csv("scripts/2026-05-06_node_level_metrics_with_mins.csv")
m2_edges  = pd.read_csv("scripts/2026-05-05_player_net_m2_edges.csv")
edge_dfs  = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
             for k in ("average", "min", "product", "sum")}

nodes["match_team_id"]    = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
m2_edges["match_team_id"] = m2_edges["match_id"].astype(str) + "_" + m2_edges["defending_team"].astype(str)

# match duration per match-team (for per-90 normalisation)
match_mins = nodes.groupby("match_team_id")["mins_played"].max().rename("match_mins")
team_names   = nodes.groupby("match_team_id")["team_name"].first()
_match_teams = outcomes.groupby("match_id")["team_name"].apply(list)
match_labels = (
    outcomes.set_index("match_team_id")["match_id"]
    .map(_match_teams.apply(lambda t: " vs ".join(sorted(t))))
    .rename("match")
)
_self_inv_cols  = [c for c in nodes.columns if c.endswith("_self_inv")]
self_inv_match  = nodes.groupby("match_team_id")[_self_inv_cols].sum().reset_index()

# ── Column groups ─────────────────────────────────────────────────────────────
WEIGHT_COLS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]
OUTCOME_COLS      = ["goals_against", "shots_against", "xg_against"]
CONTRIBUTION_COLS = ["raw_contribution", "valued_contribution"]
FAULT_COLS        = ["raw_fault", "valued_fault"]

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
def build_conc_match(edge_df, metric):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    strength = edge_df.groupby("match_team_id")[metric].sum().rename("strength")
    p1 = edge_df[["match_team_id", "player_1", metric]].rename(columns={"player_1": "player"})
    p2 = edge_df[["match_team_id", "player_2", metric]].rename(columns={"player_2": "player"})
    player_str = pd.concat([p1, p2]).groupby(["match_team_id", "player"])[metric].sum()
    gini_s = player_str.groupby("match_team_id").apply(gini).rename("gini")
    df = (_join_outcomes(pd.concat([strength, gini_s], axis=1))
          .join(match_labels)
          .join(match_mins)
          .dropna(subset=["strength", "gini"]))
    df["strength_per90"]    = df["strength"] / (df["match_mins"] / 90)
    df["strength_per_pass"] = df["strength"] / df["passes_against"]
    return df.reset_index()


def build_conc_team(df):
    agg = df.groupby("team_name")[["strength", "strength_per90", "strength_per_pass", "gini"] + OUTCOME_COLS].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


def plot_conc_match(df, outcome_col, strength_col="strength"):
    fig = px.scatter(df, x=strength_col, y="gini", color=outcome_col, size=outcome_col,
                     hover_name="team_name", hover_data=["match", "competition_stage"] + OUTCOME_COLS,
                     color_continuous_scale="RdYlGn_r",
                     title="Concentrated vs Balanced — team-match level",
                     labels={strength_col: "Total Strength", "gini": "Gini (concentration)"})
    _add_quadrant_lines(fig, df, strength_col, "gini")
    fig.add_annotation(x=df[strength_col].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df[strength_col].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


def plot_conc_team(df, outcome_col, strength_col="strength"):
    fig = px.scatter(df, x=strength_col, y="gini", color="furthest_stage", size=outcome_col,
                     color_discrete_map=STAGE_PALETTE, category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
                     hover_name="team_name", hover_data=OUTCOME_COLS + ["furthest_stage"],
                     title="Concentrated vs Balanced — team level (by stage)",
                     labels={strength_col: "Total Strength", "gini": "Gini (concentration)",
                             "furthest_stage": "Furthest Stage"})
    _add_quadrant_lines(fig, df, strength_col, "gini")
    fig.add_annotation(x=df[strength_col].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df[strength_col].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


# ── Self vs Shared ────────────────────────────────────────────────────────────
def build_selfshared_match(edge_df, metric):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    shared   = edge_df.groupby("match_team_id")[metric].sum().rename("shared_inv")
    self_inv = nodes.groupby("match_team_id")[metric + "_self_inv"].sum().rename("self_inv")
    ratio    = (self_inv / (self_inv + shared)).rename("self_ratio")
    df = _join_outcomes(pd.concat([self_inv, shared, ratio], axis=1)).dropna(subset=["self_inv", "shared_inv"])
    return df.reset_index()


def build_selfshared_team(df):
    agg = df.groupby("team_name")[["self_inv", "shared_inv", "self_ratio"] + OUTCOME_COLS].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


def plot_selfshared(df_team, outcome_col):
    df_sorted = df_team.sort_values("self_ratio", ascending=False).reset_index(drop=True)
    fig_bar = px.bar(df_sorted, x="team_name", y=["self_inv", "shared_inv"],
                     title="Self vs Shared Involvement per Team",
                     labels={"value": "Total Involvement", "team_name": "Team"}, barmode="stack")
    fig_bar.update_xaxes(tickangle=45)
    fig_scatter = px.scatter(df_sorted, x="self_ratio", y=outcome_col,
                             text="team_name", color="furthest_stage",
                             color_discrete_map=STAGE_PALETTE,
                             category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
                             title="Self-involvement ratio vs Defensive Performance",
                             labels={"self_ratio": "Self ratio (self / total)", "furthest_stage": "Furthest Stage"})
    fig_scatter.update_traces(textposition="top center")
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
                    best_stage=("competition_stage", _best_stage))
               .reset_index())
    team_df["best_stage"] = pd.Categorical(team_df["best_stage"], categories=STYLE_STAGE_ORDER, ordered=True)
    return team_df


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


# ── Co-Defenders (new) ────────────────────────────────────────────────────────
@st.cache_data
def build_co_defender_data():
    df = m2_edges.copy()

    # number of defenders involved per pass
    n_def = df.groupby(["match_id", "defending_team", "passer_id", "receiver_id"]
                        )["defender_id"].transform("nunique")
    df["n_co"] = (n_def - 1).clip(lower=0)

    # avg co-defenders per match-team
    avg_co = (df.groupby("match_team_id")["n_co"].mean()
                .reset_index(name="avg_co_defenders"))
    avg_co = avg_co.merge(outcomes[["match_team_id", "team_name", "competition_stage"] + OUTCOME_COLS],
                          on="match_team_id", how="left")

    # team-level average
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
    fig_team = px.bar(
        avg_co_team.sort_values("avg_co_defenders", ascending=False),
        x="team_name", y="avg_co_defenders",
        color="furthest_stage", color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
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
    return px.imshow(mat, color_continuous_scale="Blues",
                     title=f"{team_name} — co-defending heatmap (passes co-defended together)")


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Defensive Network Analysis — Team Level")

with st.sidebar:
    method  = st.selectbox("Edge weight method", list(edge_dfs))
    st.subheader("Concentrated / Self vs Shared")
    metric      = st.selectbox("Metric", WEIGHT_COLS)
    outcome_col = st.selectbox("Outcome metric", OUTCOME_COLS)
    st.subheader("Defensive Style")
    _norm_opts   = ["raw", "per_90", "per_pass_against"]
    _norm_labels = {"raw": "Raw", "per_90": "Per 90 min", "per_pass_against": "Per pass against"}
    x_col       = st.selectbox("X axis (contribution)", CONTRIBUTION_COLS)
    x_normalize = st.selectbox("X normalization", _norm_opts,
                               format_func=_norm_labels.__getitem__, key="x_norm")
    y_col       = st.selectbox("Y axis (fault)", FAULT_COLS)
    y_normalize = st.selectbox("Y normalization", _norm_opts,
                               format_func=_norm_labels.__getitem__, key="y_norm")
    size_col   = st.selectbox("Bubble size", OUTCOME_COLS, key="style_size")
    size_scale = st.slider("Bubble size scale", 10, 80, 40)

df_conc_match  = build_conc_match(edge_dfs[method], metric)
df_conc_team   = build_conc_team(df_conc_match)
df_self_match  = build_selfshared_match(edge_dfs[method], metric)
df_self_team   = build_selfshared_team(df_self_match)
df_style_team  = build_style_team(edge_dfs[method], x_col, y_col, size_col, x_normalize, y_normalize)
avg_co, avg_co_team = build_co_defender_data()
partnerships   = build_partnerships()

tab_conc, tab_self, tab_style, tab_codef, tab_data = st.tabs([
    "Concentrated vs Balanced", "Self vs Shared", "Defensive Style", "Co-Defenders", "Data"
])

with tab_conc:
    for _label, _scol in [
        ("Raw", "strength"),
        ("Per 90 min", "strength_per90"),
        ("Per pass against", "strength_per_pass"),
    ]:
        st.subheader(_label)
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_conc_match(df_conc_match, outcome_col, _scol), use_container_width=True)
        c2.plotly_chart(plot_conc_team(df_conc_team, outcome_col, _scol), use_container_width=True)

with tab_self:
    fig_bar, fig_scatter, df_self_sorted = plot_selfshared(df_self_team, outcome_col)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab_style:
    fig1, fig2 = plot_style(df_style_team, x_col, y_col, size_col, size_scale)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Team-level values")
    st.dataframe(
        df_style_team[["team_name","x","x_total","y","y_total","size","n_matches","best_stage"]]
        .rename(columns={"x": f"network_{x_col}", "x_total": f"total_{x_col}",
                         "y": f"network_{y_col}", "y_total": f"total_{y_col}", "size": size_col})
        .sort_values("best_stage", ascending=False).round(3),
        use_container_width=True,
    )

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
    n_top   = st.slider("Top N pairs", 10, 50, 20, key="pair_n")
    top     = partnerships.head(n_top).copy()
    top["pair"] = top["player_a"] + "  +  " + top["player_b"]
    fig_bar = px.bar(top, x="co_defenses", y="pair", orientation="h",
                     hover_data={"team_name": True},
                     title=f"Top {n_top} co-defending pairs (all teams)")
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Team co-defending heatmap")
    team_sel = st.selectbox("Select team", sorted(partnerships["team_name"].dropna().unique()))
    fig_heat = plot_partnership_heatmap(partnerships, team_sel)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No partnership data for this team.")

with tab_data:
    st.subheader("Concentrated vs Balanced — match level")
    st.dataframe(df_conc_match)
    st.subheader("Concentrated vs Balanced — team level")
    st.dataframe(df_conc_team)
    st.subheader("Self vs Shared — match level")
    st.dataframe(df_self_match)
    st.subheader("Self vs Shared — team level")
    st.dataframe(df_self_sorted)
    st.subheader("Co-defenders — match level")
    st.dataframe(avg_co)
    st.subheader("Partnerships")
    st.dataframe(partnerships.head(200))
