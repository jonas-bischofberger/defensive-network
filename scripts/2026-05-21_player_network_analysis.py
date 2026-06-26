"""
Player-level defensive network analysis — correct framing.

Each player's defensive map:
  Nodes = attacking players (passers + receivers from the opposing team)
  Edges = attacking passes the defender was involved in stopping
  Weight = valued_involvement on that pass combination

Analyses:
  1. Breadth       — avg unique attackers defended per match (how wide is the coverage?)
  2. Concentration — Gini of involvement across attacking passes (specialist vs generalist)
  3. Corr vs FIFA  — breadth + Gini + per-90 metrics correlated with FIFA ratings
  4. Network Depth — advanced: eigenvector / betweenness of attacked passing combinations
  5. Player Map    — inspect a single player's defensive map

Note on multi-match players:
  Per-90 metrics are minutes-weighted averaged across all of a player's matches.
  FIFA rating is a single fixed value per player.
  Each player = one data point in any correlation analysis.
"""
import unicodedata
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rapidfuzz import process, fuzz
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── File paths ─────────────────────────────────────────────────────────────────
EDGE_FILE  = "scripts/2026-05-05_player_net_m2_edges.csv"
PER90_FILE = "scripts/2026-05-20_player_level_per90.csv"
FIFA_FILE  = "scripts/fifa_ratings.csv"
NICK_FILE  = "scripts/nickname_map.csv"

PER90_METRICS = [
    "raw_involvement_per90", "valued_involvement_per90",
    "raw_fault_per90",       "valued_fault_per90",
    "raw_contribution_per90","valued_contribution_per90",
    "passes_defended_per90",
]
FIFA_COLS = ["overall rating", "defending rating", "def_awareness_rating"]


# ── Name normalisation ─────────────────────────────────────────────────────────
_LIGATURES = str.maketrans({
    "æ": "ae", "Æ": "Ae", "ø": "o",  "Ø": "O",
    "å": "a",  "Å": "A",  "ß": "ss", "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "Oe",
})

def _normalize(name: str) -> str:
    name = name.translate(_LIGATURES)
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace("-", " ")


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading edges…")
def load_edges() -> pd.DataFrame:
    return pd.read_csv(EDGE_FILE)

@st.cache_data(show_spinner="Loading per-90 data…")
def load_per90() -> pd.DataFrame:
    return pd.read_csv(PER90_FILE)

@st.cache_data(show_spinner="Loading FIFA ratings…")
def load_fifa() -> pd.DataFrame:
    df = pd.read_csv(FIFA_FILE)
    df = df[df["comp"].str.contains("World Cup", na=False)]
    return df.sort_values("overall rating", ascending=False).drop_duplicates("name")[["name"] + FIFA_COLS]

@st.cache_data(show_spinner="Loading nickname map…")
def load_nickname_map() -> dict:
    df = pd.read_csv(NICK_FILE).drop_duplicates(subset=["player_nickname"])
    return dict(zip(df["player_nickname"], df["Player"]))

@st.cache_data(show_spinner="Building name map…")
def build_name_map(per90_names: tuple, fifa_names: tuple, threshold: int = 88) -> dict:
    """1. Exact  2. Norm exact  3. Nickname → exact → norm exact  4. Fuzzy"""
    fifa_set = set(fifa_names)
    nick_map = load_nickname_map()
    norm_fifa: dict[str, str] = {}
    for n in fifa_names:
        k = _normalize(n)
        if k not in norm_fifa:
            norm_fifa[k] = n
    norm_list = list(norm_fifa.keys())
    out: dict[str, str] = {}
    for name in per90_names:
        if name in fifa_set:
            out[name] = name; continue
        norm = _normalize(name)
        if norm in norm_fifa:
            out[name] = norm_fifa[norm]; continue
        full = nick_map.get(name)
        if full:
            if full in fifa_set:
                out[name] = full; continue
            nf = _normalize(full)
            if nf in norm_fifa:
                out[name] = norm_fifa[nf]; continue
        r = process.extractOne(norm, norm_list, scorer=fuzz.WRatio, score_cutoff=threshold)
        if r:
            out[name] = norm_fifa[r[0]]
    return out


# ── Player filtering ───────────────────────────────────────────────────────────
def get_qualifying(per90: pd.DataFrame, starters_only: bool,
                   min_minutes: int, min_matches: int) -> tuple:
    df = per90[per90["starter"] == 1] if starters_only else per90
    mins    = df.groupby("defender_name")["mins_played"].sum()
    matches = df.groupby("defender_name")["match_id"].nunique()
    return tuple(sorted(mins[(mins >= min_minutes) & (matches >= min_matches)].index))


# ── Per-player aggregation (minutes-weighted mean across matches) ───────────────
@st.cache_data
def aggregate_player(players: tuple) -> pd.DataFrame:
    """One row per player. Per-90 metrics = minutes-weighted mean across all matches."""
    df = load_per90()
    df = df[df["defender_name"].isin(players)].copy()

    def wmean(g):
        w = g["mins_played"].values
        return pd.Series({m: np.average(g[m].values, weights=w) for m in PER90_METRICS})

    agg = df.groupby("defender_name").apply(wmean, include_groups=False).reset_index()
    agg = agg.merge(df.groupby("defender_name")["mins_played"].sum().rename("total_mins"), on="defender_name")
    agg = agg.merge(df.groupby("defender_name")["match_id"].nunique().rename("n_matches"), on="defender_name")
    agg = agg.merge(
        df.groupby("defender_name")["defending_team_name"].agg(lambda x: x.mode()[0]).rename("team"),
        on="defender_name",
    )
    return agg

@st.cache_data
def merge_with_fifa(players: tuple) -> pd.DataFrame:
    agg  = aggregate_player(players)
    fifa = load_fifa()
    nmap = build_name_map(players, tuple(fifa["name"].unique()))
    agg  = agg.copy()
    agg["fifa_name"] = agg["defender_name"].map(nmap)
    return (agg.dropna(subset=["fifa_name"])
               .merge(fifa.rename(columns={"name": "fifa_name"}), on="fifa_name", how="inner"))


# ── Defensive map metrics (from attacking network) ────────────────────────────
@st.cache_data
def compute_defensive_map_metrics(players: tuple) -> pd.DataFrame:
    """
    For each player's defensive map (nodes=attackers, edges=attacking passes):
      - n_unique_attackers      : raw unique passer+receiver count per match (avg)
      - n_unique_attackers_per90: normalised by minutes played × 90
      - degree                  : raw unique (passer→receiver) pairs per match (avg)
      - degree_per90            : normalised by minutes played × 90
      - gini                    : concentration of valued_involvement (avg, minutes-weighted)

    Raw counts are confounded by playing time; use _per90 versions for cross-player comparison.
    """
    edges = load_edges()
    per90 = load_per90()
    df    = edges[edges["defender_name"].isin(players)].copy()

    # minutes played per (defender, match) for normalisation
    mins_lookup = (per90[per90["defender_name"].isin(players)]
                   .set_index(["defender_name", "match_id"])["mins_played"]
                   .to_dict())

    rows = []
    for (defender, match_id), g in df.groupby(["defender_name", "match_id"]):
        passers  = set(g["passer_id"].dropna().astype(int))
        receivers = set(g["receiver_id"].dropna().astype(int))
        n_unique = len(passers | receivers)

        pass_inv = g.groupby(["passer_id", "receiver_id"])["valued_involvement"].sum()
        degree   = len(pass_inv)

        x = pass_inv.values
        x = x[x > 0]
        n = len(x)
        if n >= 2:
            xs   = np.sort(x)
            gini = (2 * np.dot(np.arange(1, n + 1), xs) / (n * xs.sum())) - (n + 1) / n
        else:
            gini = np.nan

        mins = mins_lookup.get((defender, match_id), np.nan)
        rows.append({
            "defender_name":           defender,
            "match_id":                match_id,
            "mins_played":             mins,
            "n_unique_attackers":      n_unique,
            "degree":                  degree,
            "gini":                    gini,
        })

    per_match = pd.DataFrame(rows)

    # per-90 normalisation per match before averaging
    per_match["n_unique_attackers_per90"] = per_match["n_unique_attackers"] / per_match["mins_played"] * 90
    per_match["degree_per90"]             = per_match["degree"]             / per_match["mins_played"] * 90

    # minutes-weighted average across matches
    def wavg(g):
        w = g["mins_played"].values
        w = np.where(np.isnan(w), 0, w)
        total = w.sum()
        if total == 0:
            return pd.Series({c: np.nan for c in
                              ["n_unique_attackers", "n_unique_attackers_per90",
                               "degree", "degree_per90", "gini"]})
        return pd.Series({
            "n_unique_attackers":       np.average(g["n_unique_attackers"].values,      weights=w),
            "n_unique_attackers_per90": np.average(g["n_unique_attackers_per90"].values, weights=w),
            "degree":                   np.average(g["degree"].values,                  weights=w),
            "degree_per90":             np.average(g["degree_per90"].values,            weights=w),
            "gini":                     np.average(g["gini"].fillna(0).values,          weights=w),
        })

    return per_match.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


@st.cache_data
def compute_centrality_scores(players: tuple) -> pd.DataFrame:
    """
    Build the full attacking network from all passes in the dataset.
    Compute eigenvector + betweenness centrality of attacking nodes.
    For each defender, compute weighted-avg centrality of the attackers they defended.

    Interpretation:
      High eigenvector score → tends to defend against 'hub' attackers central
                               to their team's passing flow.
      High betweenness score → tends to defend 'bridge' attackers who link
                               different zones of the attacking network.
    """
    edges = load_edges()

    # full attacking network (all passes across all matches)
    atk = (edges.groupby(["passer_id", "receiver_id"])["n_passes"]
                .sum().reset_index())
    G = nx.DiGraph()
    for _, r in atk.iterrows():
        G.add_edge(int(r["passer_id"]), int(r["receiver_id"]), weight=r["n_passes"])

    # compute centrality on the undirected version for stability
    Gu = G.to_undirected()
    try:
        eig_cent = nx.eigenvector_centrality_numpy(Gu, weight="weight")
    except Exception:
        eig_cent = dict.fromkeys(Gu.nodes(), np.nan)
    btw_cent = nx.betweenness_centrality(Gu, weight="weight", normalized=True)

    # for each defender: weighted avg centrality of attackers they faced
    df = edges[edges["defender_name"].isin(players)].copy()
    df["eig_node"] = df["passer_id"].map(eig_cent).fillna(0) + df["receiver_id"].map(eig_cent).fillna(0)
    df["btw_node"] = df["passer_id"].map(btw_cent).fillna(0) + df["receiver_id"].map(btw_cent).fillna(0)

    def wavg(g):
        w = g["valued_involvement"].values
        if w.sum() == 0:
            return pd.Series({"eig_centrality": np.nan, "btw_centrality": np.nan})
        return pd.Series({
            "eig_centrality": np.average(g["eig_node"].values, weights=w),
            "btw_centrality": np.average(g["btw_node"].values, weights=w),
        })

    return df.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Player Network Analysis", layout="wide")
st.title("Player-Level Defensive Network Analysis")
st.caption(
    "Each player's **defensive map**: nodes = attacking players, edges = attacking passes, "
    "weight = player's involvement in defending that pass. "
    "**Per-90 metrics** are minutes-weighted averaged across all of a player's matches. "
    "**FIFA rating** is a single value per player → each player = one data point in correlations."
)

per90 = load_per90()

with st.sidebar:
    st.header("Filters")
    starters_only = st.checkbox("Starters only", value=True)
    max_mins    = int(per90.groupby("defender_name")["mins_played"].sum().max())
    max_matches = int(per90.groupby("defender_name")["match_id"].nunique().max())
    min_minutes = st.slider("Min total minutes", 0, max_mins, 90, 10)
    min_matches = st.slider("Min matches played", 1, max_matches, 2)

players = get_qualifying(per90, starters_only, min_minutes, min_matches)
st.caption(f"**{len(players)} players** pass filters")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Breadth", "Concentration (Gini)", "Correlation vs FIFA",
    "Network Depth", "Player Map", "Archetypes", "Over/Underrated",
])


# ── Tab 1: Breadth ────────────────────────────────────────────────────────────
with t1:
    st.markdown(
        "**Avg unique attackers defended per match** — how many different attacking players "
        "appear in this player's defensive map on average. "
        "High = roaming/pressing defender covering many attackers. "
        "Low = positional/zonal defender focused on specific opponents."
    )
    dm  = compute_defensive_map_metrics(players)
    mf  = merge_with_fifa(players)[["defender_name", "team", "defending rating", "def_awareness_rating"]]
    sub = mf.merge(dm, on="defender_name", how="inner")

    fifa_col = st.selectbox("FIFA rating (colour)", FIFA_COLS, index=1, key="br_fifa")
    fig = px.scatter(
        sub, x="n_unique_attackers", y="degree",
        color=fifa_col, color_continuous_scale="RdYlGn",
        hover_name="defender_name",
        hover_data={"team": True, "n_unique_attackers": ":.1f", "degree": ":.1f"},
        title="Attacking breadth: unique attackers vs unique pass combinations",
        labels={"n_unique_attackers": "Avg unique attackers / match",
                "degree": "Avg unique pass combinations / match"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Distribution of avg unique attackers defended**")
    fig2 = px.histogram(dm, x="n_unique_attackers", nbins=30,
                        labels={"n_unique_attackers": "Avg unique attackers / match"})
    st.plotly_chart(fig2, use_container_width=True)

    n_show = st.slider("Top / bottom N", 5, 20, 10, key="br_n")
    c1, c2 = st.columns(2)
    c1.markdown("**Widest coverage** (many attackers)")
    c1.dataframe(sub.nlargest(n_show, "n_unique_attackers")
                    [["defender_name", "team", "n_unique_attackers", "degree", fifa_col]]
                    .reset_index(drop=True), use_container_width=True)
    c2.markdown("**Narrowest coverage** (few attackers)")
    c2.dataframe(sub.nsmallest(n_show, "n_unique_attackers")
                    [["defender_name", "team", "n_unique_attackers", "degree", fifa_col]]
                    .reset_index(drop=True), use_container_width=True)


# ── Tab 2: Concentration (Gini) ───────────────────────────────────────────────
with t2:
    st.markdown(
        "**Gini coefficient of involvement across attacking pass combinations.** "
        "Low Gini (≈0) = involvement spread evenly across many passes → **generalist**. "
        "High Gini (≈1) = involvement concentrated on a few specific pass combinations → **specialist**."
    )
    dm   = compute_defensive_map_metrics(players)
    mf   = merge_with_fifa(players)[["defender_name", "team"] + FIFA_COLS]
    sub2 = mf.merge(dm, on="defender_name", how="inner").dropna(subset=["gini"])

    fifa_col2 = st.selectbox("FIFA rating (colour)", FIFA_COLS, index=1, key="gini_fifa")
    fig = px.scatter(
        sub2, x="n_unique_attackers", y="gini",
        color=fifa_col2, color_continuous_scale="RdYlGn",
        hover_name="defender_name",
        hover_data={"team": True, "gini": ":.3f", "n_unique_attackers": ":.1f"},
        title="Concentration vs breadth — coloured by FIFA rating",
        labels={"n_unique_attackers": "Avg unique attackers / match",
                "gini": "Gini (concentration)"},
    )
    xm, ym = sub2["n_unique_attackers"].median(), sub2["gini"].median()
    fig.add_vline(x=xm, line_dash="dot", line_color="grey")
    fig.add_hline(y=ym, line_dash="dot", line_color="grey")
    for label, qx, qy in [
        ("Broad + specialist",  0.97, 0.97),
        ("Broad + generalist",  0.97, 0.03),
        ("Narrow + specialist", 0.03, 0.97),
        ("Narrow + generalist", 0.03, 0.03),
    ]:
        fig.add_annotation(
            x=sub2["n_unique_attackers"].quantile(qx),
            y=sub2["gini"].quantile(qy),
            text=label, showarrow=False, font=dict(size=10, color="grey"),
        )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(sub2, x="gini", nbins=30, title="Distribution of Gini across players")
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Correlation vs FIFA ────────────────────────────────────────────────
with t3:
    st.markdown(
        "Correlate player-level network metrics and per-90 metrics with FIFA ratings. "
        "Each point = one player (per-90 = minutes-weighted avg across matches, FIFA = single value)."
    )
    dm3  = compute_defensive_map_metrics(players)
    mf3  = merge_with_fifa(players)
    sub3 = mf3.merge(dm3, on="defender_name", how="inner")

    all_x_cols = ["n_unique_attackers_per90", "degree_per90", "gini",
                  "n_unique_attackers", "degree"] + PER90_METRICS
    c1, c2, c3 = st.columns(3)
    x_col    = c1.selectbox("X metric", all_x_cols, index=0, key="corr_x")
    fifa_col3 = c2.selectbox("FIFA rating (Y)", FIFA_COLS, index=1, key="corr_fifa")
    method   = c3.radio("Correlation", ["Pearson", "Spearman"], horizontal=True, key="corr_m")

    sub3_plot = sub3[[x_col, fifa_col3, "defender_name", "team"]].dropna()
    if len(sub3_plot) >= 5:
        x, y = sub3_plot[x_col].values.astype(float), sub3_plot[fifa_col3].values.astype(float)
        if method == "Pearson":
            r_val, p_val = pearsonr(x, y)
        else:
            from scipy.stats import spearmanr
            r_val, p_val = spearmanr(x, y)
        slope, intercept, *_ = linregress(x, y)
        sub3_plot = sub3_plot.copy()
        sub3_plot["residual"] = y - (slope * x + intercept)
        x_line = np.linspace(x.min(), x.max(), 100)
        fig = px.scatter(
            sub3_plot, x=x_col, y=fifa_col3,
            color="residual", color_continuous_scale="RdYlGn",
            hover_name="defender_name", hover_data={"team": True, "residual": ":.3f"},
            title=f"{x_col}  vs  {fifa_col3}   ({method} r={r_val:.3f}, p={p_val:.3f})",
        )
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept,
                                 mode="lines", line=dict(color="black", dash="dash"),
                                 name="regression"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full correlation table** (all network + per-90 metrics vs all FIFA ratings)")
    rows = []
    for xc in all_x_cols:
        row = {"metric": xc}
        for fc in FIFA_COLS:
            tmp = sub3[[xc, fc]].dropna()
            if len(tmp) < 5:
                row[fc] = "—"
            else:
                if method == "Pearson":
                    r, p = pearsonr(tmp[xc].values.astype(float), tmp[fc].values.astype(float))
                else:
                    r, p = spearmanr(tmp[xc].values.astype(float), tmp[fc].values.astype(float))
                row[fc] = f"{r:.3f} ({p:.3f})"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("metric"), use_container_width=True)


# ── Tab 4: Network Depth ──────────────────────────────────────────────────────
with t4:
    st.markdown("""
**Advanced network metrics** — the attacking network is built from all passes in the dataset.
Centrality is computed once for all attacking players, then each defender gets a
weighted-average centrality score based on the attacking players they defended against.

| Metric | Interpretation |
|---|---|
| **Eigenvector centrality** | How often does this player defend against **hub attackers** — players who are central to their team's passing flow? High = tends to mark the most influential players. |
| **Betweenness centrality** | How often does this player defend against **bridge attackers** — players who connect different zones of the attacking network? High = disrupts the key linking players. |
    """)
    cent = compute_centrality_scores(players)
    mf4  = merge_with_fifa(players)[["defender_name", "team"] + FIFA_COLS]
    dm4  = compute_defensive_map_metrics(players)
    sub4 = mf4.merge(cent, on="defender_name", how="inner").merge(dm4, on="defender_name", how="left")

    fifa_col4 = st.selectbox("FIFA rating (colour)", FIFA_COLS, index=1, key="net_fifa")
    cent_metric = st.radio("Centrality metric", ["eig_centrality", "btw_centrality"],
                           format_func=lambda x: {"eig_centrality": "Eigenvector", "btw_centrality": "Betweenness"}[x],
                           horizontal=True, key="cent_m")

    fig = px.scatter(
        sub4, x=fifa_col4, y=cent_metric,
        color=cent_metric, color_continuous_scale="Viridis",
        hover_name="defender_name",
        hover_data={"team": True, cent_metric: ":.4f"},
        title=f"{cent_metric} vs {fifa_col4}",
        labels={cent_metric: {"eig_centrality": "Eigenvector centrality score",
                               "btw_centrality": "Betweenness centrality score"}[cent_metric]},
    )
    st.plotly_chart(fig, use_container_width=True)

    n_show4 = st.slider("Top / bottom N", 5, 20, 10, key="net_n")
    c1, c2  = st.columns(2)
    show_cols4 = ["defender_name", "team", cent_metric, fifa_col4]
    c1.markdown("**Highest** (defends most central/bridge attackers)")
    c1.dataframe(sub4.nlargest(n_show4, cent_metric)[show_cols4].reset_index(drop=True),
                 use_container_width=True)
    c2.markdown("**Lowest** (defends more peripheral attackers)")
    c2.dataframe(sub4.nsmallest(n_show4, cent_metric)[show_cols4].reset_index(drop=True),
                 use_container_width=True)


# ── Tab 5: Player Map ─────────────────────────────────────────────────────────
with t5:
    st.markdown(
        "Inspect a single player's defensive map. "
        "Node size = total involvement with that attacking player. "
        "Edge weight = involvement on that specific pass combination."
    )
    edges_all  = load_edges()
    player_sel = st.selectbox("Select player", sorted(players), key="map_player")
    df_p = edges_all[edges_all["defender_name"] == player_sel].copy()

    if df_p.empty:
        st.warning("No edge data for this player.")
    else:
        pass_inv = (df_p.groupby(["passer_name", "receiver_name"])["valued_involvement"]
                       .sum().reset_index())
        pass_inv = pass_inv[pass_inv["valued_involvement"] > 0]

        # aggregate node sizes
        passer_inv   = pass_inv.groupby("passer_name")["valued_involvement"].sum().rename("inv")
        receiver_inv = pass_inv.groupby("receiver_name")["valued_involvement"].sum().rename("inv")
        node_inv = pd.concat([passer_inv, receiver_inv]).groupby(level=0).sum().reset_index()
        node_inv.columns = ["player", "inv"]

        top_n = st.slider("Show top N attacking edges (by involvement)", 5, 50, 20, key="map_n")
        top_edges = pass_inv.nlargest(top_n, "valued_involvement")

        fig = px.scatter(
            node_inv,
            x=node_inv.index, y=np.zeros(len(node_inv)),
            size="inv", hover_name="player",
            title=f"{player_sel}'s defensive map — top {top_n} edges",
        )

        # show as heatmap: passer vs receiver involvement matrix
        pivot = top_edges.pivot(index="passer_name", columns="receiver_name",
                                values="valued_involvement").fillna(0)
        fig_heat = px.imshow(
            pivot, color_continuous_scale="Blues",
            title=f"{player_sel} — involvement on attacking pass combinations (top {top_n})",
            labels={"x": "Receiver", "y": "Passer", "color": "Valued involvement"},
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Top attacking pass combinations**")
        st.dataframe(top_edges.reset_index(drop=True).round(4), use_container_width=True)

        st.markdown("**Most defended attacking players**")
        st.dataframe(node_inv.sort_values("inv", ascending=False).head(15).reset_index(drop=True).round(4),
                     use_container_width=True)


# ── Tab 6: Archetypes ─────────────────────────────────────────────────────────
with t6:
    st.markdown(
        "**K-means clustering on per-90 metrics + network metrics**, projected onto 2D via PCA. "
        "Features: 7 per-90 activity metrics + Gini (concentration) + degree/90 (unweighted pass breadth)."
    )

    ARCH_COLS = PER90_METRICS + ["gini", "degree_per90"]

    agg6 = aggregate_player(players)
    dm6  = compute_defensive_map_metrics(players)
    arch = agg6.merge(dm6[["defender_name", "gini", "degree_per90"]], on="defender_name", how="inner")
    arch = arch.merge(
        merge_with_fifa(players)[["defender_name", "defending rating"]],
        on="defender_name", how="left",
    )

    c1, c2 = st.columns(2)
    k        = c1.slider("Clusters (k)", 2, 6, 4, key="arch_k")
    sel_cols = c2.multiselect("Features for clustering", ARCH_COLS, default=ARCH_COLS, key="arch_m")

    if len(sel_cols) < 2:
        st.warning("Select at least 2 features.")
    else:
        valid = arch.dropna(subset=sel_cols).copy()
        Xs    = StandardScaler().fit_transform(valid[sel_cols])
        valid["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs).astype(str)

        pca    = PCA(n_components=2)
        xy     = pca.fit_transform(Xs)
        valid["PC1"], valid["PC2"] = xy[:, 0], xy[:, 1]

        fig = px.scatter(
            valid, x="PC1", y="PC2", color="cluster",
            hover_name="defender_name",
            hover_data={"team": True, "cluster": True, "defending rating": ":.0f"},
            title=f"Defensive archetypes — k={k}  (PCA projection)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"PC1 {pca.explained_variance_ratio_[0]:.1%} · "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%} of variance explained"
        )

        # PCA loadings — which features drive each component
        loadings = pd.DataFrame(
            pca.components_.T, index=sel_cols, columns=["PC1", "PC2"]
        ).round(3)
        with st.expander("PCA loadings (feature contributions to each component)"):
            st.dataframe(
                loadings.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1),
                use_container_width=True,
            )

        st.markdown("**Defending rating distribution per cluster**")
        fig_box = px.box(
            valid.dropna(subset=["defending rating"]),
            x="cluster", y="defending rating",
            color="cluster", points="all",
            hover_name="defender_name",
            hover_data={"team": True, "defending rating": True},
            title="FIFA defending rating by archetype cluster",
            labels={"cluster": "Cluster", "defending rating": "FIFA defending rating"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Cluster mean defending rating**")
        rating_summary = (valid.dropna(subset=["defending rating"])
                               .groupby("cluster")["defending rating"]
                               .agg(mean="mean", median="median", std="std", n="count")
                               .round(1).reset_index())
        st.dataframe(rating_summary, use_container_width=True)

        st.markdown("**Cluster profiles — mean of each feature**")
        profile = valid.groupby("cluster")[sel_cols].mean().round(3)
        st.dataframe(
            profile.style.background_gradient(cmap="YlGn", axis=0),
            use_container_width=True,
        )

        st.markdown("**Players per cluster**")
        for cl, grp in valid.groupby("cluster"):
            with st.expander(f"Cluster {cl}  ({len(grp)} players)"):
                st.dataframe(
                    grp[["defender_name", "team", "defending rating"] + sel_cols]
                      .sort_values("defending rating", ascending=False)
                      .reset_index(drop=True)
                      .round(3),
                    use_container_width=True,
                )


# ── Tab 7: Over / Underrated ──────────────────────────────────────────────────
with t7:
    df_fifa = merge_with_fifa(players)
    c1, c2  = st.columns(2)
    fifa_col   = c1.selectbox("FIFA rating", FIFA_COLS, index=1, key="ou_fifa")
    metric_col = c2.selectbox("Network metric", PER90_METRICS, index=1, key="ou_metric")

    sub = df_fifa[[metric_col, fifa_col, "defender_name", "team"]].dropna()
    if len(sub) < 5:
        st.warning("Not enough matched players — loosen filters.")
    else:
        x, y = sub[fifa_col].values.astype(float), sub[metric_col].values.astype(float)
        slope, intercept, r_val, p_val, _ = linregress(x, y)
        sub = sub.copy()
        sub["residual"] = y - (slope * x + intercept)

        x_line = np.linspace(x.min(), x.max(), 100)
        fig = px.scatter(
            sub, x=fifa_col, y=metric_col,
            color="residual", color_continuous_scale="RdYlGn",
            hover_name="defender_name",
            hover_data={"team": True, "residual": ":.3f"},
            title=f"{metric_col}  vs  {fifa_col}   (r = {r_val:.3f}, p = {p_val:.3f})",
        )
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", line=dict(color="black", dash="dash"), name="regression",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = above regression (underrated by FIFA) · Red = below (overrated)")

        n_show = st.slider("Top / bottom N", 3, 20, 5, key="ou_n")
        col_a, col_b = st.columns(2)
        cols_show = ["defender_name", "team", fifa_col, metric_col, "residual"]
        col_a.markdown("**Most underrated** (high network, low FIFA)")
        col_a.dataframe(sub.nlargest(n_show, "residual")[cols_show].reset_index(drop=True),
                        use_container_width=True)
        col_b.markdown("**Most overrated** (low network, high FIFA)")
        col_b.dataframe(sub.nsmallest(n_show, "residual")[cols_show].reset_index(drop=True),
                        use_container_width=True)
