import pandas as pd
import streamlit as st
import plotly.express as px

edges    = pd.read_csv("scripts/2026-05-05_player_net_m2_edges.csv")
outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")

contribution_cols = ["raw_contribution", "valued_contribution"]
fault_cols        = ["raw_fault",        "valued_fault"]

STAGE_ORDER = [
    "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals",
    "Third-place match", "Final",
]
STAGE_COLORS = {
    "Group Stage":       "#adb5bd",
    "Round of 16":       "#4cc9f0",
    "Quarter-finals":    "#4361ee",
    "Semi-finals":       "#f77f00",
    "Third-place match": "#9b2226",
    "Final":             "#2dc653",
}


def best_stage(stages):
    ranked = [s for s in STAGE_ORDER if s in stages.values]
    return ranked[-1] if ranked else stages.iloc[0]


def build_player_df(x_col, y_col, min_edges):
    # join match_team_id and outcome/stage info
    df = edges.copy()
    df["match_team_id"] = df["match_id"].astype(str) + "_" + df["defending_team"].astype(str)
    df = df.merge(
        outcomes[["match_team_id", "team_name", "competition_stage"]],
        on="match_team_id", how="left",
    )

    # aggregate to player level
    player_df = (
        df.groupby(["defender_id", "defender_name"])
        .agg(
            x=(x_col, "sum"),
            y=(y_col, "sum"),
            n_edges=("passer_id", "count"),
            team_name=("team_name", "first"),
            best_stage=("competition_stage", best_stage),
        )
        .reset_index()
    )

    # filter players with too few edges
    player_df = player_df[player_df["n_edges"] >= min_edges]

    player_df["best_stage"] = pd.Categorical(
        player_df["best_stage"], categories=STAGE_ORDER, ordered=True
    )
    return player_df


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Defensive Style — Player Scatter (Method 2)")

with st.sidebar:
    x_col      = st.selectbox("X axis (contribution)", contribution_cols, index=0)
    y_col      = st.selectbox("Y axis (fault)",        fault_cols,        index=0)
    min_edges  = st.slider("Min edges per player", min_value=1, max_value=20, value=5)
    size_scale = st.slider("Bubble size scale",    min_value=10, max_value=80, value=20)

player_df = build_player_df(x_col, y_col, min_edges)

st.caption(f"{len(player_df)} players after filtering (≥ {min_edges} edges)")

fig = px.scatter(
    player_df,
    x="x", y="y",
    color="best_stage",
    category_orders={"best_stage": STAGE_ORDER},
    color_discrete_map=STAGE_COLORS,
    hover_name="defender_name",
    hover_data={"x": ":.3f", "y": ":.3f", "n_edges": True,
                "team_name": True, "best_stage": True},
    labels={"x": x_col, "y": y_col, "best_stage": "Best stage"},
    title=f"Player defensive style: {x_col} vs {y_col}",
)
fig.add_vline(x=player_df["x"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig.add_hline(y=player_df["y"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig.update_layout(height=650)
fig.update_traces(marker=dict(size=size_scale / 4, opacity=0.8))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Player-level values")
st.dataframe(
    player_df[["defender_name", "team_name", "best_stage", "x", "y", "n_edges"]]
    .rename(columns={"x": x_col, "y": y_col})
    .sort_values(x_col, ascending=False)
    .round(3),
    use_container_width=True,
)
