import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Data loading ──────────────────────────────────────────────────────────────
outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
nodes = pd.read_csv("scripts/2026-04-29-player_level_metrics.csv")
edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
            for k in ("average", "min", "product", "sum")}

nodes["match_team_id"] = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
team_names = nodes.groupby("match_team_id")["team_name"].first()

_match_teams = outcomes.groupby("match_id")["team_name"].apply(list)
match_labels = (
    outcomes.set_index("match_team_id")["match_id"]
    .map(_match_teams.apply(lambda t: " vs ".join(sorted(t))))
    .rename("match")
)

_self_inv_cols = [c for c in nodes.columns if c.endswith("_self_inv")]
self_inv_match = nodes.groupby("match_team_id")[_self_inv_cols].sum().reset_index()

# ── Column groups ─────────────────────────────────────────────────────────────
WEIGHT_COLS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r",
]
OUTCOME_COLS = ["goals_against", "shots_against", "xg_against"]
CONTRIBUTION_COLS = ["raw_contribution", "valued_contribution", "raw_contribution_r", "valued_contribution_r"]
FAULT_COLS = ["raw_fault", "valued_fault", "raw_fault_r", "valued_fault_r"]

# ── Stage constants ───────────────────────────────────────────────────────────
STAGE_ORDER = {
    "Group Stage": 1, "Round of 16": 2, "Quarter-finals": 3,
    "Semi-finals": 4, "3rd Place Final": 4, "Final": 5,
}
STAGE_LABEL = {
    "Group Stage": "Group Stage", "Round of 16": "Round of 16",
    "Quarter-finals": "Quarter-finals", "Semi-finals": "Semi-finals",
    "3rd Place Final": "Semi-finals", "Final": "Final",
}
STAGE_PALETTE = {
    "Group Stage": "#a8d8ea", "Round of 16": "#78c1e0",
    "Quarter-finals": "#f4a261", "Semi-finals": "#e76f51", "Final": "#b5179e",
}
STAGE_CATEGORY_ORDER = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]

# Style scatter keeps 3rd Place Final as its own category
STYLE_STAGE_ORDER = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "3rd Place Final", "Final"]
STYLE_STAGE_PALETTE = {
    "Group Stage": "#adb5bd", "Round of 16": "#4cc9f0",
    "Quarter-finals": "#4361ee", "Semi-finals": "#f77f00",
    "3rd Place Final": "#9b2226", "Final": "#2dc653",
}


# ── Shared helpers ────────────────────────────────────────────────────────────
def gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def _join_outcomes(df):
    return df.join(team_names).join(
        outcomes.set_index("match_team_id")[OUTCOME_COLS + ["competition_stage"]]
    )


def _furthest_stage(df):
    return (
        df.groupby("team_name")["competition_stage"]
        .apply(lambda s: max(s, key=lambda x: STAGE_ORDER.get(x, 0)))
        .map(STAGE_LABEL)
        .rename("furthest_stage")
        .reset_index()
    )


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

    df = _join_outcomes(pd.concat([strength, gini_s], axis=1)).join(match_labels).dropna(subset=["strength", "gini"])
    return df.reset_index()


def build_conc_team(df):
    agg = df.groupby("team_name")[["strength", "gini"] + OUTCOME_COLS].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


def plot_conc_match(df, outcome_col):
    fig = px.scatter(
        df, x="strength", y="gini",
        color=outcome_col, size=outcome_col,
        hover_name="team_name", hover_data=["match", "competition_stage"] + OUTCOME_COLS,
        color_continuous_scale="RdYlGn_r",
        title="Concentrated vs Balanced — team-match level",
        labels={"strength": "Total Strength", "gini": "Gini (concentration)"},
    )
    _add_quadrant_lines(fig, df, "strength", "gini")
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


def plot_conc_team(df, outcome_col):
    fig = px.scatter(
        df, x="strength", y="gini",
        color="furthest_stage", size=outcome_col,
        color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
        hover_name="team_name", hover_data=OUTCOME_COLS + ["furthest_stage"],
        title="Concentrated vs Balanced — team level (by stage)",
        labels={"strength": "Total Strength", "gini": "Gini (concentration)",
                "furthest_stage": "Furthest Stage"},
    )
    _add_quadrant_lines(fig, df, "strength", "gini")
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


# ── Self vs Shared ────────────────────────────────────────────────────────────
def build_selfshared_match(edge_df, metric):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)

    shared = edge_df.groupby("match_team_id")[metric].sum().rename("shared_inv")
    self_inv = nodes.groupby("match_team_id")[metric + "_self_inv"].sum().rename("self_inv")
    ratio = (self_inv / (self_inv + shared)).rename("self_ratio")

    df = _join_outcomes(pd.concat([self_inv, shared, ratio], axis=1)).dropna(subset=["self_inv", "shared_inv"])
    return df.reset_index()


def build_selfshared_team(df):
    agg = df.groupby("team_name")[
        ["self_inv", "shared_inv", "self_ratio"] + OUTCOME_COLS
    ].mean().reset_index()
    return agg.merge(_furthest_stage(df), on="team_name")


def plot_selfshared(df_team, outcome_col):
    df_sorted = df_team.sort_values("self_ratio", ascending=False).reset_index(drop=True)

    fig_bar = px.bar(
        df_sorted, x="team_name", y=["self_inv", "shared_inv"],
        title="Self vs Shared Involvement per Team",
        labels={"value": "Total Involvement", "team_name": "Team"},
        barmode="stack",
    )
    fig_bar.update_xaxes(tickangle=45)

    fig_scatter = px.scatter(
        df_sorted, x="self_ratio", y=outcome_col,
        text="team_name", color="furthest_stage",
        color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
        title="Self-involvement ratio vs Defensive Performance",
        labels={"self_ratio": "Self ratio (self / total)", "furthest_stage": "Furthest Stage"},
    )
    fig_scatter.update_traces(textposition="top center")

    return fig_bar, fig_scatter, df_sorted


# ── Defensive Style ───────────────────────────────────────────────────────────
def _best_stage(stages):
    ranked = [s for s in STYLE_STAGE_ORDER if s in stages.values]
    return ranked[-1] if ranked else stages.iloc[0]


def build_style_team(edge_df, x_col, y_col, size_col):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)

    match_level = (
        edge_df.groupby("match_team_id")[[x_col, y_col]].sum()
        .merge(outcomes[["match_team_id", "team_name", "competition_stage"] + OUTCOME_COLS], on="match_team_id")
        .merge(self_inv_match[["match_team_id", x_col + "_self_inv", y_col + "_self_inv"]], on="match_team_id", how="left")
    )
    match_level["x_total"] = match_level[x_col] + match_level[x_col + "_self_inv"]
    match_level["y_total"] = match_level[y_col] + match_level[y_col + "_self_inv"]

    team_df = (
        match_level.groupby("team_name")
        .agg(
            x=(x_col, "mean"), y=(y_col, "mean"),
            x_total=("x_total", "mean"), y_total=("y_total", "mean"),
            size=(size_col, "mean"),
            n_matches=("match_team_id", "count"),
            best_stage=("competition_stage", _best_stage),
        )
        .reset_index()
    )
    team_df["best_stage"] = pd.Categorical(team_df["best_stage"], categories=STYLE_STAGE_ORDER, ordered=True)
    return team_df


def plot_style(team_df, x_col, y_col, size_col, size_scale):
    shared = dict(
        size="size", size_max=size_scale,
        color="best_stage",
        color_discrete_map=STYLE_STAGE_PALETTE,
        category_orders={"best_stage": STYLE_STAGE_ORDER},
        hover_name="team_name",
    )

    fig1 = px.scatter(
        team_df, x="x", y="y",
        title="Network contribution vs fault  |  color = best stage",
        labels={"x": x_col, "y": y_col, "size": size_col, "best_stage": "Best stage"},
        hover_data={"x": ":.3f", "y": ":.3f", "size": ":.3f", "n_matches": True, "best_stage": True},
        **shared,
    )
    _add_quadrant_lines(fig1, team_df, "x", "y", opacity=0.5)
    fig1.update_layout(height=600)

    fig2 = px.scatter(
        team_df, x="x_total", y="y_total",
        title="Total contribution vs fault (network + self-inv)  |  color = best stage",
        labels={"x_total": f"{x_col} (network + self-inv)", "y_total": f"{y_col} (network + self-inv)",
                "size": size_col, "best_stage": "Best stage"},
        hover_data={"x_total": ":.3f", "y_total": ":.3f", "size": ":.3f", "n_matches": True, "best_stage": True},
        **shared,
    )
    _add_quadrant_lines(fig2, team_df, "x_total", "y_total", opacity=0.5)
    fig2.update_layout(height=600)

    return fig1, fig2


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Defensive Network Analysis")

with st.sidebar:
    method = st.selectbox("Edge weight method", list(edge_dfs))

    st.subheader("Concentrated / Self vs Shared")
    metric = st.selectbox("Metric", WEIGHT_COLS)
    outcome_col = st.selectbox("Outcome metric", OUTCOME_COLS)

    st.subheader("Defensive Style")
    x_col = st.selectbox("X axis (contribution)", CONTRIBUTION_COLS)
    y_col = st.selectbox("Y axis (fault)", FAULT_COLS)
    size_col = st.selectbox("Bubble size", OUTCOME_COLS, key="style_size")
    size_scale = st.slider("Bubble size scale", min_value=10, max_value=80, value=40)

df_conc_match = build_conc_match(edge_dfs[method], metric)
df_conc_team = build_conc_team(df_conc_match)
df_self_match = build_selfshared_match(edge_dfs[method], metric)
df_self_team = build_selfshared_team(df_self_match)
df_style_team = build_style_team(edge_dfs[method], x_col, y_col, size_col)

tab_conc, tab_self, tab_style, tab_data = st.tabs([
    "Concentrated vs Balanced", "Self vs Shared", "Defensive Style", "Data"
])

with tab_conc:
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_conc_match(df_conc_match, outcome_col), use_container_width=True)
    c2.plotly_chart(plot_conc_team(df_conc_team, outcome_col), use_container_width=True)

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
        df_style_team[["team_name", "x", "x_total", "y", "y_total", "size", "n_matches", "best_stage"]]
        .rename(columns={"x": f"network_{x_col}", "x_total": f"total_{x_col}",
                         "y": f"network_{y_col}", "y_total": f"total_{y_col}", "size": size_col})
        .sort_values("best_stage", ascending=False)
        .round(3),
        use_container_width=True,
    )

with tab_data:
    st.subheader("Concentrated vs Balanced — team-match level")
    st.dataframe(df_conc_match)
    st.subheader("Concentrated vs Balanced — team level")
    st.dataframe(df_conc_team)
    st.subheader("Self vs Shared — team-match level")
    st.dataframe(df_self_match)
    st.subheader("Self vs Shared — team level")
    st.dataframe(df_self_sorted)
