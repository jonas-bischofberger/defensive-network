import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from mplsoccer import Pitch

st.set_page_config(layout="wide")
st.title("Shared Defensive Network Viewer")


# 1. data
@st.cache_data
def load_data():
    edge_df = pd.read_csv("scripts/2026-04-03-shared_defensive_edge_list_without_blocked_respon1.csv")
    player_df = pd.read_csv("scripts/2026-04-09_player_average_defensive_positions_all_matches.csv")
    meta_df = pd.read_csv("scripts/meta.csv")
    return edge_df, player_df, meta_df

edge_df, player_df, meta_df = load_data()

# 2. the thickness of the weight
def scale_series(series, min_width=1.5, max_width=8.0):
    s = series.astype(float)
    if s.nunique() == 1:
        return np.array([(min_width + max_width) / 2] * len(s))
    return min_width + (s - s.min()) / (s.max() - s.min()) * (max_width - min_width)

# 3. player location
def get_player_positions(player_df, match_id, defending_team, players, metric):
    df = player_df[
        (player_df["match_id"] == match_id) &
        (player_df["defending_team"] == defending_team) &
        (player_df["defender_name"].isin(players))
    ].copy()

    if metric == "raw_responsibility":
        x_col = "raw_involvement_avg_x"
        y_col = "raw_involvement_avg_y"
    else:
        x_col = f"{metric}_avg_x"
        y_col = f"{metric}_avg_y"

    df["x"] = df[x_col].fillna(df["raw_involvement_avg_x"])   #if the player miss the metric, use raw_involvement_avg as fallback
    df["y"] = df[y_col].fillna(df["raw_involvement_avg_y"])

    # transfer to statsbomb pitch coordinates (0,0) at top left, (120,80) at bottom right
    df["plot_x"] = df["x"] + 60
    df["plot_y"] = df["y"] + 40

    return {
        row["defender_name"]: (row["plot_x"], row["plot_y"])
        for _, row in df.dropna(subset=["plot_x", "plot_y"]).iterrows()
    }


# 4. plotting function
# =========================
def plot_defensive_network(
    edge_df,
    player_df,
    match_id,
    defending_team,
    edge_metric,
    width_metric,
    color_metric,
    position_metric,
    min_edge_count=1,
    cmap_name="magma_r",
    node_size=200,
):
    edge_count_col = f"{edge_metric}_edge_count"

    df_plot = edge_df[
        (edge_df["match_id"] == match_id) &
        (edge_df["defending_team"] == defending_team) &
        (edge_df[edge_count_col].fillna(0) >= min_edge_count)
        ].copy()

    if df_plot.empty:
        return None

    players = sorted(set(df_plot["player_1"]).union(set(df_plot["player_2"])))
    player_pos = get_player_positions(
        player_df, match_id, defending_team, players, position_metric
    )

    # 没找到位置的球员，放在中间兜底
    for p in players:
        if p not in player_pos:
            player_pos[p] = (60, 40)

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#f7f7f7",
        line_color="#999999"
    )
    fig, ax = pitch.draw(figsize=(12, 8))

    width_values = scale_series(df_plot[width_metric])

    color_values = df_plot[color_metric].astype(float)
    vmin, vmax = color_values.min(), color_values.max()
    norm = Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6) if vmin == vmax else Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.linspace(0.05, 0.90, 256))
    sm = ScalarMappable(cmap=ListedColormap(colors), norm=norm)
    sm.set_array([])

    for (_, row), lw in zip(df_plot.iterrows(), width_values):
        x1, y1 = player_pos[row["player_1"]]
        x2, y2 = player_pos[row["player_2"]]

        ax.plot(
            [x1, x2], [y1, y2],
            color=sm.to_rgba(row[color_metric]),
            linewidth=lw,
            alpha=0.75,
            zorder=1
        )

    xs = [player_pos[p][0] for p in players]
    ys = [player_pos[p][1] for p in players]

    pitch.scatter(
        xs, ys,
        s=node_size,
        color="#dbe9f6",
        edgecolors="black",
        linewidth=1.2,
        ax=ax,
        zorder=2
    )

    for p in players:
        x, y = player_pos[p]
        ax.text(x, y, p, ha="center", va="center", fontsize=9, zorder=3)

    cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label(color_metric, fontsize=11)

    ax.set_title(
        f"Shared Defensive Network\n"
        f"Match {match_id} | Team {defending_team}\n"
        f"Position = {position_metric} | Width = {width_metric} | Color = {color_metric}",
        fontsize=14
    )

    return fig


# 5. Sidebar
st.sidebar.header("Filters")

match_ids = sorted(edge_df["match_id"].dropna().unique())
match_id_2_title = dict(zip(meta_df["match_id"], meta_df["match_title"]))

selected_match = st.sidebar.selectbox("Match ID", match_ids, format_func=lambda x: str(match_id_2_title.get(x, x)))

team_options = sorted(edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"].dropna().unique())
team_df = player_df[player_df["match_id"] == selected_match]
team_id_2_name = dict(zip(team_df["defending_team"], team_df["team_name"]))

selected_team = st.sidebar.selectbox("Defending Team", team_options, format_func=lambda x: str(team_id_2_name.get(x, x)))

edge_metric_options = [
    "raw_involvement",
    "raw_contribution",
    "raw_fault",
    "valued_involvement",
    "valued_contribution",
    "valued_fault",
    "raw_responsibility",
]

position_metric_options = [
    "raw_involvement",
    "raw_contribution",
    "raw_fault",
    "valued_involvement",
    "valued_contribution",
    "valued_fault",
    "raw_responsibility",
]

selected_edge_metric = st.sidebar.selectbox("Edge metric", edge_metric_options, index=0)
selected_width_metric = st.sidebar.selectbox("Edge width metric", edge_metric_options, index=0)
selected_color_metric = st.sidebar.selectbox("Edge color metric", edge_metric_options, index=1)
selected_position_metric = st.sidebar.selectbox("Node position metric", position_metric_options, index=0)

min_edge_count = st.sidebar.slider("Minimum edge count", 1, 10, 1)
cmap_name = st.sidebar.selectbox("Color map", ["magma_r", "viridis", "plasma", "cividis", "coolwarm"], index=0)


# 6. Main display
fig = plot_defensive_network(
    edge_df=edge_df,
    player_df=player_df,
    match_id=selected_match,
    defending_team=selected_team,
    edge_metric=selected_edge_metric,
    width_metric=selected_width_metric,
    color_metric=selected_color_metric,
    position_metric=selected_position_metric,
    min_edge_count=min_edge_count,
    cmap_name=cmap_name
)

if fig is not None:
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("No edges available for this selection.")