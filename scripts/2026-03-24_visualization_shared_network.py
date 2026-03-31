# import matplotlib
# matplotlib.use("TkAgg")
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize, ListedColormap
# from matplotlib import cm
#
# # ---- 画足球场 ----
# from mplsoccer import Pitch
#
#
# # =========================
# # 1. 读取 edge list
# # =========================
# edge_df = pd.read_csv("shared_defensive_edge_list_total.csv")
#
#
# # =========================
# # 2. 固定 4-3-3 阵型坐标
# #    这里只是占位，不代表真实位置
# # =========================
# def get_fixed_positions():
#     # 基础11人 + 替补/额外球员备用位置
#     return [
#         (8, 40),    # GK
#         (25, 12),   # RB
#         (22, 30),   # RCB
#         (22, 50),   # LCB
#         (25, 68),   # LB
#         (45, 18),   # RCM
#         (48, 40),   # CM
#         (45, 62),   # LCM
#         (72, 15),   # RW
#         (82, 40),   # ST
#         (72, 65),   # LW
#
#         # extra players / substitutes
#         (15, 5),
#         (15, 75),
#         (35, 5),
#         (35, 75),
#         (55, 5),
#         (55, 75),
#         (90, 10),
#         (90, 70),
#         (102, 25),
#         (102, 55),
#     ]
#
#
# def assign_fixed_positions(players):
#     positions = get_fixed_positions()
#
#     if len(players) > len(positions):
#         raise ValueError(
#             f"Not enough fixed positions for {len(players)} players. "
#             f"Only {len(positions)} positions defined."
#         )
#
#     return {player: pos for player, pos in zip(players, positions)}
#
#
# # =========================
# # 3. 画单场单队网络
# # =========================
# def plot_defensive_network(
#     edge_df,
#     match_id,
#     defending_team,
#     min_edge_count=1,
#     node_size=1400,
#     pitch_color="#f2f2f2",
#     line_color="#6e6e6e",
#     cmap_name="magma_r"
# ):
#     # 筛选比赛和球队
#     df_plot = edge_df[
#         (edge_df["match_id"] == match_id) &
#         (edge_df["defending_team"] == defending_team)
#     ].copy()
#
#     # 过滤弱边，避免太乱
#     df_plot = df_plot[df_plot["edge_count"] >= min_edge_count].copy()
#
#     if df_plot.empty:
#         print("No edges found for this match/team.")
#         return
#
#     # 所有节点
#     players = sorted(set(df_plot["player_1"]).union(set(df_plot["player_2"])))
#
#     # 固定阵型坐标
#     player_pos = assign_fixed_positions(players)
#
#     # 建图时不一定需要 networkx，这里直接画就行
#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color=pitch_color,
#         line_color=line_color
#     )
#     fig, ax = pitch.draw(figsize=(12, 8), constrained_layout=True, tight_layout=False)
#
#     # -------------------------
#     # 4. 边宽：edge_count
#     # -------------------------
#     min_count = df_plot["edge_count"].min()
#     max_count = df_plot["edge_count"].max()
#
#     if min_count == max_count:
#         width_values = np.array([4.0] * len(df_plot))
#     else:
#         width_values = 1.5 + 8 * (df_plot["edge_count"] - min_count) / (max_count - min_count)
#
#     # -------------------------
#     # 5. 边颜色：raw_involvement
#     # -------------------------
#     min_raw = df_plot["raw_involvement"].min()
#     max_raw = df_plot["raw_involvement"].max()
#
#     if min_raw == max_raw:
#         norm_raw = Normalize(vmin=min_raw - 1e-6, vmax=max_raw + 1e-6)
#     else:
#         norm_raw = Normalize(vmin=min_raw, vmax=max_raw)
#
#     base_cmap = cm.get_cmap(cmap_name)
#     new_colors = base_cmap(np.linspace(0.05, 0.90, 256))
#     truncated_cmap = ListedColormap(new_colors)
#     sm = ScalarMappable(cmap=truncated_cmap, norm=norm_raw)
#     sm.set_array([])
#
#     # -------------------------
#     # 6. 画边（无向）
#     # -------------------------
#     for (_, row), lw in zip(df_plot.iterrows(), width_values):
#         p1 = row["player_1"]
#         p2 = row["player_2"]
#
#         x1, y1 = player_pos[p1]
#         x2, y2 = player_pos[p2]
#
#         color = sm.to_rgba(row["raw_involvement"])
#
#         ax.plot(
#             [x1, x2],
#             [y1, y2],
#             color=color,
#             linewidth=lw,
#             alpha=0.75,
#             zorder=1
#         )
#
#     # -------------------------
#     # 7. 画节点
#     # -------------------------
#     xs = [player_pos[p][0] for p in players]
#     ys = [player_pos[p][1] for p in players]
#
#     pitch.scatter(
#         xs, ys,
#         s=node_size,
#         color="#d9e6f2",
#         edgecolors="black",
#         linewidth=1.2,
#         alpha=1,
#         ax=ax,
#         zorder=2
#     )
#
#     # 标签
#     for p in players:
#         x, y = player_pos[p]
#         ax.text(
#             x, y,
#             p,
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="black",
#             zorder=3
#         )
#
#     # colorbar
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
#     cbar.set_label("Raw involvement", fontsize=11)
#
#     ax.set_title(
#         f"Shared Defensive Network\nMatch {match_id} | Team {defending_team}\n"
#         f"Edge width = edge count | Edge color = raw involvement",
#         fontsize=14
#     )
#
#     return fig, ax
#
#
# # =========================
# # 4. 使用示例
# # =========================
# print(edge_df[["match_id", "defending_team"]].drop_duplicates().head(20))
#
# fig, ax = plot_defensive_network(
#     edge_df=edge_df,
#     match_id=3833,
#     defending_team=376,
#     min_edge_count=1   # 太乱的话改成 2 或 3
# )
#
# plt.show()


'''
streamlit app to visualize the shared defensive network for a selected match and team, with options to choose edge
width and color metrics.
'''

#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize, ListedColormap
# from mplsoccer import Pitch
#
#
# st.set_page_config(layout="wide")
# st.title("Shared Defensive Network Viewer")
#
#
# # =========================
# # 1. 读取数据
# # =========================
# @st.cache_data
# def load_data():
#     df = pd.read_csv("scripts/shared_defensive_edge_list_total222.csv")
#     return df
#
# edge_df = load_data()
#
#
# # =========================
# # 2. 固定位置
# # =========================
# def get_fixed_positions():
#     return [
#         (8, 40),    # 1
#         (25, 12),   # 2
#         (22, 30),   # 3
#         (22, 50),   # 4
#         (25, 68),   # 5
#         (45, 18),   # 6
#         (48, 40),   # 7
#         (45, 62),   # 8
#         (72, 15),   # 9
#         (82, 40),   # 10
#         (72, 65),   # 11
#         (15, 5),    # extra
#         (15, 75),
#         (35, 5),
#         (35, 75),
#         (55, 5),
#         (55, 75),
#         (90, 10),
#         (90, 70),
#         (102, 25),
#         (102, 55),
#     ]
#
#
# def assign_fixed_positions(players):
#     positions = get_fixed_positions()
#
#     if len(players) > len(positions):
#         raise ValueError(
#             f"Not enough fixed positions for {len(players)} players."
#         )
#
#     return {player: pos for player, pos in zip(players, positions)}
#
#
# # =========================
# # 3. 数值缩放函数
# # =========================
# def scale_series(series, min_width=1.5, max_width=8.0):
#     s = series.astype(float)
#
#     if len(s) == 0:
#         return np.array([])
#
#     if s.nunique() == 1:
#         return np.array([(min_width + max_width) / 2] * len(s))
#
#     return min_width + (s - s.min()) / (s.max() - s.min()) * (max_width - min_width)
#
#
# # =========================
# # 4. 作图函数
# # =========================
# def plot_defensive_network(
#     edge_df,
#     match_id,
#     defending_team,
#     width_metric,
#     color_metric,
#     min_edge_count=1,
#     cmap_name="magma_r",
#     pitch_color="#f7f7f7",
#     line_color="#999999",
#     node_size=1400,
# ):
#     df_plot = edge_df[
#         (edge_df["match_id"] == match_id) &
#         (edge_df["defending_team"] == defending_team)
#     ].copy()
#
#     df_plot = df_plot[df_plot["edge_count"] >= min_edge_count].copy()
#
#     if df_plot.empty:
#         return None
#
#     players = sorted(set(df_plot["player_1"]).union(set(df_plot["player_2"])))
#     player_pos = assign_fixed_positions(players)
#
#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color=pitch_color,
#         line_color=line_color
#     )
#     fig, ax = pitch.draw(figsize=(12, 8))
#
#     # 线宽
#     width_values = scale_series(df_plot[width_metric], min_width=1.5, max_width=8.0)
#
#     # 颜色
#     color_values = df_plot[color_metric].astype(float)
#     vmin, vmax = color_values.min(), color_values.max()
#     if vmin == vmax:
#         norm = Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)
#     else:
#         norm = Normalize(vmin=vmin, vmax=vmax)
#
#     base_cmap = plt.colormaps[cmap_name]
#     new_colors = base_cmap(np.linspace(0.05, 0.90, 256))
#     truncated_cmap = ListedColormap(new_colors)
#     sm = ScalarMappable(cmap=truncated_cmap, norm=norm)
#     sm.set_array([])
#
#     # 画边
#     for (_, row), lw in zip(df_plot.iterrows(), width_values):
#         p1, p2 = row["player_1"], row["player_2"]
#         x1, y1 = player_pos[p1]
#         x2, y2 = player_pos[p2]
#
#         ax.plot(
#             [x1, x2], [y1, y2],
#             color=sm.to_rgba(row[color_metric]),
#             linewidth=lw,
#             alpha=0.75,
#             zorder=1
#         )
#
#     # 画点
#     xs = [player_pos[p][0] for p in players]
#     ys = [player_pos[p][1] for p in players]
#
#     pitch.scatter(
#         xs, ys,
#         s=node_size,
#         color="#dbe9f6",
#         edgecolors="black",
#         linewidth=1.2,
#         ax=ax,
#         zorder=2
#     )
#
#     # 标签
#     for p in players:
#         x, y = player_pos[p]
#         ax.text(
#             x, y, p,
#             ha="center", va="center",
#             fontsize=9,
#             color="black",
#             zorder=3
#         )
#
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
#     cbar.set_label(color_metric, fontsize=11)
#
#     ax.set_title(
#         f"Shared Defensive Network\n"
#         f"Match {match_id} | Team {defending_team}\n"
#         f"Width = {width_metric} | Color = {color_metric}",
#         fontsize=14
#     )
#
#     return fig
#
#
# # =========================
# # 5. Sidebar 控件
# # =========================
# st.sidebar.header("Filters")
#
# match_ids = sorted(edge_df["match_id"].dropna().unique().tolist())
# selected_match = st.sidebar.selectbox("Match ID", match_ids)
#
# team_options = sorted(
#     edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"]
#     .dropna().unique().tolist()
# )
# selected_team = st.sidebar.selectbox("Defending Team", team_options)
#
# metric_options = [
#     "edge_count",
#     "raw_involvement",
#     "raw_contribution",
#     "raw_fault",
#     "valued_involvement",
#     "valued_contribution",
#     "valued_fault",
# ]
#
# selected_width_metric = st.sidebar.selectbox(
#     "Edge width metric",
#     metric_options,
#     index=0
# )
#
# selected_color_metric = st.sidebar.selectbox(
#     "Edge color metric",
#     metric_options,
#     index=1
# )
#
# min_edge_count = st.sidebar.slider("Minimum edge count", 1, 10, 1)
#
# cmap_name = st.sidebar.selectbox(
#     "Color map",
#     ["magma_r", "viridis", "plasma", "cividis", "coolwarm"],
#     index=0
# )
#
#
# # =========================
# # 6. 主界面显示
# # =========================
# st.write("Preview of selected data:")
# df_preview = edge_df[
#     (edge_df["match_id"] == selected_match) &
#     (edge_df["defending_team"] == selected_team)
# ].copy()
# st.dataframe(df_preview.head(20), use_container_width=True)
#
# fig = plot_defensive_network(
#     edge_df=edge_df,
#     match_id=selected_match,
#     defending_team=selected_team,
#     width_metric=selected_width_metric,
#     color_metric=selected_color_metric,
#     min_edge_count=min_edge_count,
#     cmap_name=cmap_name
# )
#
# if fig is not None:
#     st.pyplot(fig, use_container_width=True)
# else:
#     st.warning("No edges available for this selection.")
'''
streamlit version 2.0
'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from mplsoccer import Pitch

st.set_page_config(layout="wide")
st.title("Shared Defensive Network Viewer")


# =========================
# 1. 读取数据
# =========================
@st.cache_data
def load_data():
    edge_df = pd.read_csv("scripts/shared_defensive_edge_list_total222.csv")
    player_df = pd.read_csv("scripts/player_info_all_matches.csv")
    return edge_df, player_df

edge_df, player_df = load_data()


# =========================
# 2. 缩放函数
# =========================
def scale_series(series, min_width=1.5, max_width=8.0):
    s = series.astype(float)
    if s.nunique() == 1:
        return np.array([(min_width + max_width) / 2] * len(s))
    return min_width + (s - s.min()) / (s.max() - s.min()) * (max_width - min_width)


# =========================
# 3. 读取球员位置
# =========================
def get_player_positions(player_df, match_id, defending_team, players, metric):
    df = player_df[
        (player_df["match_id"] == match_id) &
        (player_df["defending_team"] == defending_team) &
        (player_df["defender_name"].isin(players))
    ].copy()

    x_col = f"{metric}_avg_x"
    y_col = f"{metric}_avg_y"

    df["x"] = df[x_col].fillna(df["raw_involvement_avg_x"])
    df["y"] = df[y_col].fillna(df["raw_involvement_avg_y"])

    # 中心坐标 -> statsbomb坐标
    df["plot_x"] = df["x"] + 60
    df["plot_y"] = df["y"] + 40

    return {
        row["defender_name"]: (row["plot_x"], row["plot_y"])
        for _, row in df.dropna(subset=["plot_x", "plot_y"]).iterrows()
    }


# =========================
# 4. 作图函数
# =========================
def plot_defensive_network(
    edge_df,
    player_df,
    match_id,
    defending_team,
    width_metric,
    color_metric,
    position_metric,
    min_edge_count=1,
    cmap_name="magma_r",
    node_size=1400,
):
    df_plot = edge_df[
        (edge_df["match_id"] == match_id) &
        (edge_df["defending_team"] == defending_team) &
        (edge_df["edge_count"] >= min_edge_count)
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


# =========================
# 5. Sidebar
# =========================
st.sidebar.header("Filters")

match_ids = sorted(edge_df["match_id"].dropna().unique())
selected_match = st.sidebar.selectbox("Match ID", match_ids)

team_options = sorted(
    edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"].dropna().unique()
)
selected_team = st.sidebar.selectbox("Defending Team", team_options)

edge_metric_options = [
    "edge_count",
    "raw_involvement",
    "raw_contribution",
    "raw_fault",
    "valued_involvement",
    "valued_contribution",
    "valued_fault",
]

position_metric_options = [
    "raw_involvement",
    "raw_contribution",
    "raw_fault",
    "valued_involvement",
    "valued_contribution",
    "valued_fault",
]

selected_width_metric = st.sidebar.selectbox("Edge width metric", edge_metric_options, index=0)
selected_color_metric = st.sidebar.selectbox("Edge color metric", edge_metric_options, index=1)
selected_position_metric = st.sidebar.selectbox("Node position metric", position_metric_options, index=0)

min_edge_count = st.sidebar.slider("Minimum edge count", 1, 10, 1)
cmap_name = st.sidebar.selectbox("Color map", ["magma_r", "viridis", "plasma", "cividis", "coolwarm"], index=0)


# =========================
# 6. 显示
# =========================
fig = plot_defensive_network(
    edge_df=edge_df,
    player_df=player_df,
    match_id=selected_match,
    defending_team=selected_team,
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