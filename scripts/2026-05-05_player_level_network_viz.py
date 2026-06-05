
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm as mplcm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import defensive_network.utility.pitch

st.set_page_config(layout="wide")
st.title("Player Defensive Network")

HERE = os.path.dirname(__file__)
METRICS = ["raw_involvement", "valued_involvement", "raw_fault", "valued_fault",
           "raw_contribution", "valued_contribution"]

COLOR_STARTER = "#FFD700"   # gold
COLOR_SUB = "#87CEEB"   # sky blue


@st.cache_data
def load_data():
    m1_n = pd.read_csv(os.path.join(HERE, "2026-05-05_player_net_m1_nodes.csv"))
    m1_e = pd.read_csv(os.path.join(HERE, "2026-05-05_player_net_m1_edges.csv"))
    m2_n = pd.read_csv(os.path.join(HERE, "2026-05-05_player_net_m2_nodes.csv"))
    m2_e = pd.read_csv(os.path.join(HERE, "2026-05-05_player_net_m2_edges.csv"))
    return m1_n, m1_e, m2_n, m2_e


m1_n, m1_e, m2_n, m2_e = load_data()

match_id_2_title = m1_n.drop_duplicates("match_id").set_index("match_id")["match_name"].to_dict()
team_name_lookup = (m1_n.drop_duplicates("defending_team")
                    .set_index("defending_team")["defending_team_name"].to_dict())


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

method = st.sidebar.selectbox("Method", ["Method 1 (successful passes)", "Method 2 (all passes)"])
use_m2 = method.startswith("Method 2")
nodes_all = m2_n if use_m2 else m1_n
edges_all = m2_e if use_m2 else m1_e

match_ids = sorted(edges_all["match_id"].dropna().unique())
selected_match = st.sidebar.selectbox(
    "Match", match_ids,
    format_func=lambda x: match_id_2_title.get(int(x), str(x))
)

team_ids = sorted(edges_all.loc[edges_all["match_id"] == selected_match,
                                "defending_team"].dropna().unique())
selected_team = st.sidebar.selectbox(
    "Defending Team", team_ids,
    format_func=lambda x: team_name_lookup.get(int(x), str(x))
)

defenders = sorted(edges_all.loc[
    (edges_all["match_id"] == selected_match) &
    (edges_all["defending_team"] == selected_team),
    "defender_name"
].dropna().unique())
selected_defender = st.sidebar.selectbox("Defender", defenders)

selected_metric = st.sidebar.selectbox("Metric", METRICS)

e_sub = edges_all[
    (edges_all["match_id"] == selected_match) &
    (edges_all["defending_team"] == selected_team) &
    (edges_all["defender_name"] == selected_defender)
]
max_count = int(e_sub["n_passes"].max()) if not e_sub.empty else 1
min_passes = st.sidebar.slider("Min passes per edge", 1, max_count, 1) if max_count > 1 else 1

cmap_name = st.sidebar.selectbox("Edge colormap", ["YlOrBr", "magma_r", "viridis", "plasma", "coolwarm"])


# ── Plot ──────────────────────────────────────────────────────────────────────
def node_color(starter_val):
    return COLOR_STARTER if int(starter_val) == 1 else COLOR_SUB

def plot_player_network(edges_all, nodes_all, match_id, team_id, defender,
                        metric, min_passes, cmap_name, use_m2):
    metric_avg_col = f"{metric}_avg"

    edges = edges_all[
        (edges_all["match_id"] == match_id) &
        (edges_all["defending_team"] == team_id) &
        (edges_all["defender_name"] == defender) &
        (edges_all["n_passes"] >= min_passes)
    ].copy()
    if "involvement" not in metric:
        edges = edges[edges[metric_avg_col] > 0]

    node_rows = nodes_all[
        (nodes_all["match_id"] == match_id) &
        (nodes_all["defending_team"] == team_id) &
        (nodes_all["defender_name"] == defender)
    ].copy()

    if edges.empty:
        return None

    # position and metadata from node table
    pos = (node_rows.dropna(subset=["player_x", "player_y"])
           .set_index("player_name")
           .to_dict("index"))

    # edge color scale
    all_edge_vals = edges[metric_avg_col].dropna().astype(float)
    if all_edge_vals.empty:
        return None
    vmin, vmax = all_edge_vals.min(), all_edge_vals.max()
    norm = Normalize(vmin=vmin - 1e-9, vmax=vmax + 1e-9) if vmin == vmax else Normalize(vmin=vmin, vmax=vmax)
    cmap = mplcm.get_cmap(cmap_name)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # edge width scaled to n_passes
    n_vals = edges["n_passes"].astype(float)
    span = n_vals.max() - n_vals.min()
    widths = (1.5 + (n_vals - n_vals.min()) / span * 5.5) if span > 0 else pd.Series(
        [3.0] * len(edges), index=edges.index)

    fig, ax = defensive_network.utility.pitch.plot_football_pitch(figsize=(14, 9))

    # draw edges
    edge_players = set()
    for _, row in edges.sort_values(metric_avg_col).iterrows():
        pn, rn = str(row["passer_name"]), str(row["receiver_name"])
        if pn not in pos or rn not in pos:
            continue
        x1, y1 = pos[pn]["player_x"], pos[pn]["player_y"]
        x2, y2 = pos[rn]["player_x"], pos[rn]["player_y"]
        arrow = matplotlib.patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->", mutation_scale=25,
            connectionstyle="arc3,rad=0.2",
            linewidth=widths[row.name],
            color=cmap(norm(row[metric_avg_col])),
            zorder=2,
        )
        ax.add_patch(arrow)
        edge_players.add(pn)
        edge_players.add(rn)

    # build smart labels: last name only, but add first initial if last name is duplicated
    last_names = [pn.split()[-1] for pn in pos]
    duplicate_last = {ln for ln in last_names if last_names.count(ln) > 1}

    def make_label(name):
        last = name.split()[-1]
        if last in duplicate_last and len(name.split()) > 1:
            return name.split()[0][0] + ". " + last
        return last

    # draw nodes
    NODE_MULT = 200
    max_passer = max(edges.groupby("passer_name")["n_passes"].sum().max(), 1) if use_m2 else None

    for pn, info in pos.items():
        x, y = info["player_x"], info["player_y"]
        in_edge = pn in edge_players

        # Method 1: size from node_size (B/D count); skip if isolated with no B/D
        if not use_m2:
            n_bd = info.get("node_size", np.nan)
            n_bd = n_bd if pd.notna(n_bd) else 0
            if not in_edge and n_bd == 0:
                continue
            s = max(50, n_bd ** 1.5 * NODE_MULT / 15) if n_bd > 0 else 50
        else:
            # Method 2: size normalized by max passer in this network
            passer_n = edges.loc[edges["passer_name"] == pn, "n_passes"].sum()
            s = 50 + (passer_n / max_passer) * 200

        color = node_color(info.get("starter", np.nan))
        ax.scatter(x, y, s=s, c=[color], edgecolors="black",
                   linewidths=1, marker="o", zorder=3)

        ydelta = 2.75 + math.sqrt(s) / 20
        ax.text(x, y - ydelta, make_label(pn),
                ha="center", va="top", fontsize=9, color="black", zorder=4)

    # legend: only show starter/sub (no unknown — starter is always 0 or 1 in data)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_STARTER,
               markeredgecolor="black", markersize=10, label="Starter"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SUB,
               markeredgecolor="black", markersize=10, label="Sub"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.7)

    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(f"{metric} (avg per pass)", fontsize=10)

    team_name = team_name_lookup.get(int(team_id), str(team_id))
    match_title = match_id_2_title.get(int(match_id), str(match_id))
    ax.set_title(
        f"{defender}  ·  {match_title}\n{team_name}  |  {metric}",
        fontsize=12, color="black"
    )
    return fig


# ── Render ────────────────────────────────────────────────────────────────────
fig = plot_player_network(
    edges_all=edges_all,
    nodes_all=nodes_all,
    match_id=selected_match,
    team_id=selected_team,
    defender=selected_defender,
    metric=selected_metric,
    min_passes=min_passes,
    cmap_name=cmap_name,
    use_m2=use_m2,
)

if fig is not None:
    st.pyplot(fig, use_container_width=True)
else:
    st.warning("No data available for this selection.")
