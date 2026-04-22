import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from mplsoccer import Pitch

st.set_page_config(layout="wide")
st.title("Shared Defensive Network")


# 1. data
# edge_dfs = {
#     "average": pd.read_csv("scripts/2026-04-13_defensive_network_edge(average).csv"),
#     "min": pd.read_csv("scripts/2026-04-13_defensive_network_edge(min).csv"),
#     "product": pd.read_csv("scripts/2026-04-13_defensive_network_edge(product).csv"),
#     "sum": pd.read_csv("scripts/2026-04-13_defensive_network_edge(sum).csv")}
edge_dfs = {
    "average": pd.read_csv("scripts/2026-04-22_test.csv"),
    "min": pd.read_csv("scripts/2026-04-22_test.csv"),
    "product": pd.read_csv("scripts/2026-04-22_test.csv"),
    "sum": pd.read_csv("scripts/2026-04-22_test.csv")}


player_df = pd.read_csv("scripts/starter.csv")
meta_df = pd.read_csv("scripts/meta_worldcup.csv")


# 2. player location
def get_player_positions(player_df, match_id, defending_team, players, metric):
    df = player_df[
        (player_df["match_id"] == match_id) &
        (player_df["defending_team"] == defending_team) &
        (player_df["defender_name"].isin(players))
    ].copy()

    if metric in ["raw_responsibility", "raw_fault_r", "raw_contribution_r","valued_responsibility",
                  "valued_contribution_r", "valued_fault_r", "respon-inv"]:

        df["x"] = df["raw_involvement_avg_x"].fillna(df["overall_avg_x"])
        df["y"] = df["raw_involvement_avg_y"].fillna(df["overall_avg_y"])

    else:
        df["x"] = df[f"{metric}_avg_x"]
        df["y"] = df[f"{metric}_avg_y"]

    df = df.dropna(subset=["x", "y"])

    df["plot_x"] = df["x"] + 60
    df["plot_y"] = df["y"] + 40
    #
    #  self_inv
    self_inv_col = f"{metric}_self_inv"
    positions = {}
    for _, row in df.iterrows():
        self_inv = row[self_inv_col] if self_inv_col in df.columns else 0.0
        starter = row["starter"] if "starter" in df.columns else 0
        positions[row["defender_name"]] = (row["plot_x"], row["plot_y"], self_inv, starter)

    return positions


def plot_defensive_network(edge_df, player_df, match_id, defending_team, metric, min_edge_count=1, cmap_name="magma_r",
                           node_size_option="self_inv", node_size=50):

    edge_count_col = f"{metric}_edge_count"
    df_plot = edge_df[(edge_df["match_id"] == match_id) & (edge_df["defending_team"] == defending_team) &
                      (edge_df[edge_count_col].fillna(0) >= min_edge_count)].copy()

    players = sorted(set(df_plot["player_1"]).union(set(df_plot["player_2"])))
    player_pos = get_player_positions(player_df, match_id, defending_team, players, metric)

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f7f7f7", line_color="#999999")
    fig, ax = pitch.draw(figsize=(12, 8))

    edge_count_col = f"{metric}_edge_count"
    s = df_plot[edge_count_col].astype(float)
    width_values = 1.5 + (s - s.min()) / (s.max() - s.min()) * (8.0 - 1.5)

    color_values = df_plot[metric].astype(float)
    vmin, vmax = color_values.min(), color_values.max()
    norm = Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6) if vmin == vmax else Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.linspace(0.05, 0.90, 256))
    sm = ScalarMappable(cmap=ListedColormap(colors), norm=norm)
    sm.set_array([])

    for (_, row), lw in zip(df_plot.iterrows(), width_values):
        x1, y1 = player_pos[row["player_1"]][:2]
        x2, y2 = player_pos[row["player_2"]][:2]

        ax.plot(
            [x1, x2], [y1, y2],
            color=sm.to_rgba(row[metric]),
            linewidth=lw,
            alpha=0.75,
            zorder=1
        )

    xs = [player_pos[p][0] for p in players]
    ys = [player_pos[p][1] for p in players]

    # node size
    if node_size_option == "self_inv":
        self_inv_vals = np.array([0.0 if pd.isna(player_pos[p][2]) else player_pos[p][2] for p in players])
    else:
        self_inv_vals = np.array([
            0.0 if pd.isna(
                player_df.loc[
                    (player_df["match_id"] == match_id) &
                    (player_df["defending_team"] == defending_team) &
                    (player_df["defender_name"] == p),
                    f"{metric}_n"
                ].values[0]
            )
            else player_df.loc[
                (player_df["match_id"] == match_id) &
                (player_df["defending_team"] == defending_team) &
                (player_df["defender_name"] == p),
                f"{metric}_n"
            ].values[0]
            for p in players
        ])
    min_size, max_size = 100, 400
    if self_inv_vals.max() > self_inv_vals.min():
        node_sizes = min_size + (self_inv_vals - self_inv_vals.min()) / (self_inv_vals.max() - self_inv_vals.min()) * (
                    max_size - min_size)
    else:
        node_sizes = np.full(len(players), node_size)

    # 替换原来的 pitch.scatter 部分
    starter_flags = [player_pos[p][3] for p in players]
    node_colors = ["#feffce" if s == 1 else "#bebada" for s in starter_flags]

    pitch.scatter(xs, ys, s=node_sizes, color=node_colors, edgecolors="black", linewidth=1.2, ax=ax, zorder=2)

    # # 加图例
    # from matplotlib.patches import Patch
    # ax.legend(handles=[
    #     Patch(color="#d4edda", label="Starter"),
    #     Patch(color="#dbe9f6", label="Substitute")
    # ], loc="upper right", fontsize=9)

    for p in players:
        x, y = player_pos[p][:2]
        ax.text(x, y, p, ha="center", va="center", fontsize=9, zorder=3)

    cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label(metric, fontsize=11)

    ax.set_title(
        f"Shared Defensive Network\n"
        f"Match {match_id} | Team {defending_team}\n"
        f"Position = {metric} | Width = {metric} | Color = {metric}",
        fontsize=14)

    return fig


# 5. Sidebar
st.sidebar.header("Filters")

edge_method = st.sidebar.selectbox("Edge weight method", ["average", "min", "product", "sum"], index=2)
edge_df = edge_dfs[edge_method]

node_size_option = st.sidebar.selectbox("Node size", ["self_inv", "inv_number"], index=0)

match_ids = sorted(edge_df["match_id"].dropna().unique())
match_id_2_title = dict(zip(meta_df["match_id"], meta_df["home_team_name"] + " vs " + meta_df["guest_team_name"]))

selected_match = st.sidebar.selectbox("Match ID", match_ids, format_func=lambda x: f"{x}: {match_id_2_title[x]}")

team_options = sorted(edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"].dropna().unique())
team_df = player_df[player_df["match_id"] == selected_match]
team_id_2_name = dict(zip(team_df["defending_team"], team_df["team_name"]))
selected_team = \
    st.sidebar.selectbox("Defending Team", team_options, format_func=lambda x: str(team_id_2_name.get(x, x)))

metric_options = ["raw_involvement", "raw_contribution", "raw_fault", "valued_involvement", "valued_contribution",
                  "valued_fault", "raw_responsibility", "raw_fault_r", "raw_contribution_r",
                  "valued_responsibility", "valued_contribution_r", "valued_fault_r", "respon-inv"]

selected_metric = st.sidebar.selectbox("Metric", metric_options, index=0)

# min_edge_count = st.sidebar.slider("Minimum edge count", 1, 200, 1)  # fixed number

edge_count_col = f"{selected_metric}_edge_count"
max_count = int(edge_df[edge_count_col].fillna(0).max())
min_edge_count = st.sidebar.slider("Minimum edge count", 1, max_count, 1)  # flexible number

cmap_name = st.sidebar.selectbox("Color map", ["magma_r", "viridis", "plasma", "cividis", "coolwarm"], index=0)


# 6. Main display
fig = plot_defensive_network(edge_df=edge_df, player_df=player_df, match_id=selected_match,
                             defending_team=selected_team, metric=selected_metric, min_edge_count=min_edge_count,
                             cmap_name=cmap_name, node_size_option=node_size_option)

if fig is not None:
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("No edges available for this selection.")
