import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from mplsoccer import Pitch

st.set_page_config(layout="wide")
st.title("Shared Defensive Network")

ZONE_KEYS   = ["high_press", "mid", "own"]
ZONE_LABELS = {"high_press": "High Press", "mid": "Mid Block", "own": "Low Block"}


# 1. data
edge_dfs = {
    "average": pd.read_csv("scripts/2026-04-28_defensive_network_edge(average).csv"),
    "min":     pd.read_csv("scripts/2026-04-28_defensive_network_edge(min).csv"),
    "product": pd.read_csv("scripts/2026-04-28_defensive_network_edge(product).csv"),
    "sum":     pd.read_csv("scripts/2026-04-28_defensive_network_edge(sum).csv"),
}

zone_edge_dfs = {}
missing_zone_files = []
for m in ["average", "min", "product", "sum"]:
    path = f"scripts/2026-06-18_zone_network_edge({m}).csv"
    try:
        zone_edge_dfs[m] = pd.read_csv(path)
    except FileNotFoundError:
        missing_zone_files.append(path)
has_zone_data = len(missing_zone_files) == 0

try:
    zone_pos_df = pd.read_csv("scripts/2026-06-18_zone_network_positions.csv")
    has_zone_pos = True
except FileNotFoundError:
    zone_pos_df = None
    has_zone_pos = False

player_df = pd.read_csv("scripts/2026-05-06_node_level_metrics_with_mins.csv")
meta_df   = pd.read_csv("scripts/meta_worldcup.csv")

BASE_METRICS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]

def _base_metric(metric):
    """Strip _per_pass suffix to get the underlying metric name."""
    return metric.removesuffix("_per_pass")

def _add_per_pass_cols(df):
    for m in BASE_METRICS:
        count_col = f"{m}_edge_count"
        if m in df.columns and count_col in df.columns:
            df[f"{m}_per_pass"] = df[m] / df[count_col].replace(0, np.nan)
    return df

for _d in edge_dfs.values():
    _add_per_pass_cols(_d)
for _d in zone_edge_dfs.values():
    _add_per_pass_cols(_d)


# 2. player location
def get_player_positions(player_df, match_id, defending_team, players, metric, zone_pos_df=None):
    # zone_pos_df: already pre-filtered to a specific zone; None means full-match
    def _filter(source):
        return source[
            (source["match_id"] == match_id) &
            (source["defending_team"] == defending_team) &
            (source["defender_name"].isin(players))
        ].copy()

    if zone_pos_df is not None:
        df = _filter(zone_pos_df)
        # fall back to full-match positions for players absent from this zone
        missing = set(players) - set(df["defender_name"])
        if missing:
            df = pd.concat([df, _filter(player_df).query("defender_name in @missing")],
                           ignore_index=True)
    else:
        df = _filter(player_df)

    bm = _base_metric(metric)
    use_involvement_pos = bm in [
        "raw_responsibility", "raw_fault_r", "raw_contribution_r",
        "valued_responsibility", "valued_contribution_r", "valued_fault_r", "respon-inv",
    ]
    if use_involvement_pos:
        df["x"] = df["raw_involvement_avg_x"].fillna(df["overall_avg_x"])
        df["y"] = df["raw_involvement_avg_y"].fillna(df["overall_avg_y"])
    else:
        x_col, y_col = f"{bm}_avg_x", f"{bm}_avg_y"
        df["x"] = (df[x_col] if x_col in df.columns else np.nan)
        df["y"] = (df[y_col] if y_col in df.columns else np.nan)
        df["x"] = df["x"].fillna(df.get("raw_involvement_avg_x", np.nan)).fillna(df["overall_avg_x"])
        df["y"] = df["y"].fillna(df.get("raw_involvement_avg_y", np.nan)).fillna(df["overall_avg_y"])

    df = df.dropna(subset=["x", "y"])
    df["plot_x"] = df["x"] + 60
    df["plot_y"] = df["y"] + 40

    # self_inv and starter always come from the full-match player_df
    pinfo = _filter(player_df).set_index("defender_name")
    self_inv_col = f"{bm}_self_inv"
    positions = {}
    for _, row in df.iterrows():
        p = row["defender_name"]
        self_inv = float(pinfo.at[p, self_inv_col]) if p in pinfo.index and self_inv_col in pinfo.columns else 0.0
        self_inv = 0.0 if pd.isna(self_inv) else self_inv
        starter  = int(pinfo.at[p, "starter"]) if p in pinfo.index and "starter" in pinfo.columns else 0
        positions[p] = (row["plot_x"], row["plot_y"], self_inv, starter)

    return positions


def plot_defensive_network(edge_df, player_df, match_id, defending_team, metric,
                           min_edge_count=1, cmap_name="magma_r",
                           node_size_option="self_inv", node_size=100,
                           zone_label=None, zone_pos_df=None,
                           vmin=None, vmax=None):

    edge_count_col = f"{_base_metric(metric)}_edge_count"
    df_plot = edge_df[
        (edge_df["match_id"] == match_id) &
        (edge_df["defending_team"] == defending_team) &
        (edge_df[edge_count_col].fillna(0) >= min_edge_count)
    ].copy()

    if df_plot.empty:
        return None

    players = sorted(set(df_plot["player_1"]).union(set(df_plot["player_2"])))
    player_pos = get_player_positions(player_df, match_id, defending_team, players, metric,
                                      zone_pos_df=zone_pos_df)

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f7f7f7", line_color="#999999")
    fig, ax = pitch.draw(figsize=(12, 8))

    s = df_plot[edge_count_col].astype(float)
    width_values = 1.5 + (s - s.min()) / (s.max() - s.min() + 1e-9) * (8.0 - 1.5)

    color_values = df_plot[metric].astype(float)
    _vmin = vmin if vmin is not None else color_values.min()
    _vmax = vmax if vmax is not None else color_values.max()
    norm = Normalize(vmin=_vmin - 1e-6, vmax=_vmax + 1e-6) if _vmin == _vmax else Normalize(vmin=_vmin, vmax=_vmax)

    cmap   = plt.colormaps[cmap_name]
    colors = cmap(np.linspace(0.05, 0.90, 256))
    sm     = ScalarMappable(cmap=ListedColormap(colors), norm=norm)
    sm.set_array([])

    for (_, row), lw in zip(df_plot.iterrows(), width_values):
        if row["player_1"] not in player_pos or row["player_2"] not in player_pos:
            continue
        x1, y1 = player_pos[row["player_1"]][:2]
        x2, y2 = player_pos[row["player_2"]][:2]
        ax.plot([x1, x2], [y1, y2],
                color=sm.to_rgba(row[metric]),
                linewidth=lw, alpha=0.75, zorder=1)

    xs = [player_pos[p][0] for p in players]
    ys = [player_pos[p][1] for p in players]

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
        node_sizes = min_size + (self_inv_vals - self_inv_vals.min()) / (self_inv_vals.max() - self_inv_vals.min()) * (max_size - min_size)
    else:
        node_sizes = np.full(len(players), node_size)

    starter_flags = [player_pos[p][3] for p in players]
    node_colors   = ["#feffce" if s == 1 else "#bebada" for s in starter_flags]
    pitch.scatter(xs, ys, s=node_sizes, color=node_colors, edgecolors="black", linewidth=1.2, ax=ax, zorder=2)

    for p in players:
        x, y = player_pos[p][:2]
        ax.text(x, y, p, ha="center", va="center", fontsize=9, zorder=3)

    cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label(metric, fontsize=11)

    zone_str = f" | Zone: {zone_label}" if zone_label else ""
    ax.set_title(
        f"Shared Defensive Network{zone_str}\n"
        f"Match {match_id} | Team {defending_team}\n"
        f"Position = {metric} | Width = {metric} | Color = {metric}",
        fontsize=14)

    return fig


# 5. Sidebar
st.sidebar.header("Filters")

edge_method = st.sidebar.selectbox("Edge weight method", ["average", "min", "product", "sum"], index=2)
edge_df     = edge_dfs[edge_method]

node_size_option = st.sidebar.selectbox("Node size", ["self_inv", "inv_number"], index=0)

match_ids         = sorted(edge_df["match_id"].dropna().unique())
match_id_2_title  = dict(zip(meta_df["match_id"], meta_df["home_team_name"] + " vs " + meta_df["guest_team_name"]))
selected_match    = st.sidebar.selectbox("Match ID", match_ids, format_func=lambda x: f"{x}: {match_id_2_title[x]}")

team_options   = sorted(edge_df.loc[edge_df["match_id"] == selected_match, "defending_team"].dropna().unique())
team_df        = player_df[player_df["match_id"] == selected_match]
team_id_2_name = dict(zip(team_df["defending_team"], team_df["team_name"]))
selected_team  = st.sidebar.selectbox("Defending Team", team_options, format_func=lambda x: str(team_id_2_name.get(x, x)))

metric_options = [
    "raw_involvement", "raw_contribution", "raw_fault",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r", "respon-inv",
] + [f"{m}_per_pass" for m in BASE_METRICS]
selected_metric = st.sidebar.selectbox("Metric", metric_options, index=0)

edge_count_col = f"{selected_metric}_edge_count"
max_count      = int(edge_df[edge_count_col].fillna(0).max())
min_edge_count = st.sidebar.slider("Minimum edge count", 1, max_count, 1)

cmap_name = st.sidebar.selectbox("Color map", ["magma_r", "viridis", "plasma", "cividis", "coolwarm"], index=0)


# 6. Main display — tabs: full match + one per zone

def _scale_for(df, metric):
    """Global vmin/vmax across the entire dataset for this metric."""
    vals = df[metric].dropna()
    return (float(vals.min()), float(vals.max())) if not vals.empty else (0.0, 1.0)


tab_labels = ["Full match"] + [ZONE_LABELS[z] for z in ZONE_KEYS]
tabs = st.tabs(tab_labels)

_full_vmin, _full_vmax = _scale_for(edge_df, selected_metric)

with tabs[0]:
    fig = plot_defensive_network(
        edge_df=edge_df, player_df=player_df,
        match_id=selected_match, defending_team=selected_team,
        metric=selected_metric, min_edge_count=min_edge_count,
        cmap_name=cmap_name, node_size_option=node_size_option,
        vmin=_full_vmin, vmax=_full_vmax,
    )
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No edges available for this selection.")

if not has_zone_data:
    for i in range(1, 4):
        with tabs[i]:
            st.warning(
                "Zone-filtered edge files not found. "
                "Run `scripts/2026-06-18_zone_network_edges.py` to generate them.\n\n"
                "Missing:\n" + "\n".join(f"- `{p}`" for p in missing_zone_files)
            )
else:
    zone_edge_df = zone_edge_dfs[edge_method]
    for i, zone_key in enumerate(ZONE_KEYS):
        with tabs[i + 1]:
            df_zone_edges = zone_edge_df[zone_edge_df["zone"] == zone_key]
            df_zone_pos = (
                zone_pos_df[zone_pos_df["zone"] == zone_key] if has_zone_pos else None
            )
            _zone_vmin, _zone_vmax = _scale_for(df_zone_edges, selected_metric)
            fig = plot_defensive_network(
                edge_df=df_zone_edges, player_df=player_df,
                match_id=selected_match, defending_team=selected_team,
                metric=selected_metric, min_edge_count=min_edge_count,
                cmap_name=cmap_name, node_size_option=node_size_option,
                zone_label=ZONE_LABELS[zone_key],
                zone_pos_df=df_zone_pos,
                vmin=_zone_vmin, vmax=_zone_vmax,
            )
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning(f"No edges in the {ZONE_LABELS[zone_key]} zone for this selection.")
