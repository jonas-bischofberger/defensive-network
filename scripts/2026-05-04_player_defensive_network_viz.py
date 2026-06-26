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


@st.cache_data
def load_data():
    m1_e = pd.read_csv(os.path.join(HERE, "2026-05-01_player_net_m1_edges.csv"))
    m1_n = pd.read_csv(os.path.join(HERE, "2026-05-01_player_net_m1_nodes.csv"))
    m2_e = pd.read_csv(os.path.join(HERE, "2026-05-01_player_net_m2_edges.csv"))
    meta = pd.read_csv(os.path.join(HERE, "meta_worldcup.csv"))
    return m1_e, m1_n, m2_e, meta


m1_e, m1_n, m2_e, meta = load_data()

match_id_2_title = dict(zip(meta["match_id"], meta["home_team_name"] + " vs " + meta["guest_team_name"]))
team_name_lookup = {}
for _, row in meta.iterrows():
    team_name_lookup[int(row["home_team_id"])] = row["home_team_name"]
    team_name_lookup[int(row["guest_team_id"])] = row["guest_team_name"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

method = st.sidebar.selectbox("Method", ["Method 1 (C edges + B/D nodes)", "Method 2 (all edges)"])
use_m2 = method.startswith("Method 2")
edge_df = m2_e if use_m2 else m1_e

match_ids = sorted(edge_df["match_id"].dropna().unique())
selected_match = st.sidebar.selectbox(
    "Match", match_ids,
    format_func=lambda x: match_id_2_title.get(int(x), str(x))
)

team_ids = sorted(edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"].dropna().unique())
selected_team = st.sidebar.selectbox(
    "Defending Team", team_ids,
    format_func=lambda x: team_name_lookup.get(int(x), str(x))
)

defenders = sorted(edge_df.loc[
    (edge_df["match_id"] == selected_match) &
    (edge_df["defending_team"] == selected_team),
    "defender_name"
].dropna().unique())
selected_defender = st.sidebar.selectbox("Defender", defenders)

selected_metric = st.sidebar.selectbox("Metric", METRICS)

e_sub = edge_df[
    (edge_df["match_id"] == selected_match) &
    (edge_df["defending_team"] == selected_team) &
    (edge_df["defender_name"] == selected_defender)
]
max_count = int(e_sub["n_passes"].max()) if not e_sub.empty else 1
if max_count > 1:
    min_passes = st.sidebar.slider("Minimum passes per edge", 1, max_count, 1)
else:
    min_passes = 1

cmap_name = st.sidebar.selectbox("Colormap", ["YlOrBr", "magma_r", "viridis", "plasma", "coolwarm"])


# ── Core logic ────────────────────────────────────────────────────────────────
def build_player_positions(edges, bd_nodes=None):
    """
    Returns: {player_name: {"x": float, "y": float, "n": int, "has_edge": bool}}

    Position priority:
      1. passer_x/y from edges (average over all C+B+D, pre-computed, most reliable)
      2. passer_x/y from bd_nodes (same computation, for B/D-only passers)
      3. receiver_x/y from edges (fallback for players who only ever receive)
    """
    pos = {}

    # Priority 3: receivers (lowest priority, overwritten if better source available)
    for _, row in edges.iterrows():
        rn = str(row["receiver_name"])
        if pd.isna(row.get("receiver_x")) or pd.isna(row.get("receiver_y")):
            continue
        if rn not in pos:
            pos[rn] = {"x": row["receiver_x"], "y": row["receiver_y"],
                       "n": 0, "has_edge": True}

    # Priority 2: B/D passer positions
    if bd_nodes is not None and not bd_nodes.empty:
        for _, row in bd_nodes.iterrows():
            pn = str(row["passer_name"])
            if pn not in pos:
                pos[pn] = {"x": row["passer_x"], "y": row["passer_y"],
                           "n": 0, "has_edge": False}
            pos[pn]["n"] += row["n_passes"]

    # Priority 1: passer positions from edges (most reliable, mark has_edge=True)
    passer_canonical = (
        edges.groupby("passer_name")
        .agg(x=("passer_x", "first"), y=("passer_y", "first"), n=("n_passes", "sum"))
        .reset_index()
    )
    for _, row in passer_canonical.iterrows():
        pn = str(row["passer_name"])
        if pn in pos:
            pos[pn]["x"] = row["x"]
            pos[pn]["y"] = row["y"]
            pos[pn]["n"] += row["n"]
            pos[pn]["has_edge"] = True
        else:
            pos[pn] = {"x": row["x"], "y": row["y"], "n": row["n"], "has_edge": True}

    return pos


def plot_player_network(edge_df, node_df, match_id, team_id, defender,
                        metric, min_passes, cmap_name, use_m2):
    metric_avg_col = f"{metric}_avg"

    # ── Filter ──
    edges = edge_df[
        (edge_df["match_id"] == match_id) &
        (edge_df["defending_team"] == team_id) &
        (edge_df["defender_name"] == defender) &
        (edge_df["n_passes"] >= min_passes)
    ].copy()

    bd_nodes = None
    if not use_m2 and node_df is not None:
        bd_nodes = node_df[
            (node_df["match_id"] == match_id) &
            (node_df["defending_team"] == team_id) &
            (node_df["defender_name"] == defender)
        ].copy().dropna(subset=["passer_x", "passer_y"])

    if edges.empty:
        return None

    edges = edges.dropna(subset=["passer_x", "passer_y"])

    # ── Player canonical positions ──
    player_pos = build_player_positions(edges, bd_nodes)
    if not player_pos:
        return None

    # ── Color scale (based on edge metric values) ──
    all_edge_vals = edges[metric_avg_col].dropna().astype(float)
    vmin, vmax = all_edge_vals.min(), all_edge_vals.max()
    if vmin == vmax:
        norm = Normalize(vmin=vmin - 1e-9, vmax=vmax + 1e-9)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mplcm.get_cmap(cmap_name)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ── Compute per-player metric average (for node color) ──
    passer_val = edges.groupby("passer_name")[metric_avg_col].mean().to_dict()
    recv_val   = edges.groupby("receiver_name")[metric_avg_col].mean().to_dict()
    def player_val(name):
        if name in passer_val and name in recv_val:
            return (passer_val[name] + recv_val[name]) / 2
        return passer_val.get(name, recv_val.get(name, (vmin + vmax) / 2))

    # ── Edge width scale ──
    n_vals = edges["n_passes"].astype(float)
    span = n_vals.max() - n_vals.min()
    if span > 0:
        widths = 1.5 + (n_vals - n_vals.min()) / span * 5.5
    else:
        widths = pd.Series([3.0] * len(edges), index=edges.index)

    # ── Draw pitch ──
    fig, ax = defensive_network.utility.pitch.plot_football_pitch(figsize=(14, 9))

    # ── Draw edges (low-value first so high-value renders on top) ──
    for _, row in edges.sort_values(metric_avg_col).iterrows():
        pn = str(row["passer_name"])
        rn = str(row["receiver_name"])
        if pn not in player_pos or rn not in player_pos:
            continue
        x1, y1 = player_pos[pn]["x"], player_pos[pn]["y"]
        x2, y2 = player_pos[rn]["x"], player_pos[rn]["y"]
        lw    = widths[row.name]
        color = cmap(norm(row[metric_avg_col]))
        arrow = matplotlib.patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->", mutation_scale=25,
            connectionstyle="arc3,rad=0.2",
            linewidth=lw, color=color, zorder=2,
        )
        ax.add_patch(arrow)

    # ── B/D count per player (for node size in Method 1) ──
    bd_count = {}
    if not use_m2 and bd_nodes is not None and not bd_nodes.empty:
        bd_count = bd_nodes.groupby("passer_name")["n_passes"].sum().to_dict()

    # ── Draw nodes ──
    NODE_MULT = 200
    c_players = set(edges["passer_name"].astype(str)) | set(edges["receiver_name"].astype(str))

    for pn, info in player_pos.items():
        x, y = info["x"], info["y"]
        val   = player_val(pn)
        color = cmap(norm(val))

        # B/D-only: never appeared in any C edge (neither as passer nor receiver)
        bd_only = (not use_m2) and (pn not in c_players)

        if not use_m2:
            # Method 1: node size = number of B/D passes this player had
            n_bd = bd_count.get(pn, 0)
            s = max(50, n_bd ** 1.5 * NODE_MULT / 15) if n_bd > 0 else 50
        else:
            # Method 2: node size = total passes as passer in this network
            n = max(1, info["n"])
            s = max(50, n ** 1.5 * NODE_MULT / 15)

        if bd_only:
            ax.scatter(x, y, s=s, c=["#cccccc"], edgecolors="#555555",
                       linewidths=1.5, marker="s", zorder=3)
            label_color = "#555555"
        else:
            ax.scatter(x, y, s=s, c=[color], edgecolors="black",
                       linewidths=1, marker="o", zorder=3)
            label_color = "black"

        ydelta = 2.75 + math.sqrt(s) / 20
        ax.text(x, y - ydelta, pn.split()[-1],
                ha="center", va="top", fontsize=9, color=label_color, zorder=4)

    # ── Colorbar + title ──
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(f"{metric} (avg per pass)", fontsize=10)

    team_name   = team_name_lookup.get(int(team_id), str(team_id))
    match_title = match_id_2_title.get(int(match_id), str(match_id))
    ax.set_title(
        f"{defender}  ·  {match_title}\n{team_name}  |  {metric}",
        fontsize=12, color="black"
    )

    return fig


# ── Render ────────────────────────────────────────────────────────────────────
fig = plot_player_network(
    edge_df=edge_df,
    node_df=m1_n if not use_m2 else None,
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
