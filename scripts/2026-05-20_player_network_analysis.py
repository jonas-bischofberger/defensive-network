"""
Player-level defensive network: 7-tab analysis.

  1. Over/Underrated  — scatter: per-90 metric vs FIFA rating + regression line
  2. Archetypes       — k-means on per-90 metrics, PCA scatter + cluster profiles
  3. Consistency      — mean vs CV across matches, coloured by FIFA rating
  4. Self vs Team     — proportion of involvement from solo vs coordinated actions
  5. Network Metrics  — degree, avg co-defenders, attacker diversity
  6. Partnerships     — top co-defending pairs + per-team heatmap
  7. Centrality       — eigenvector centrality in team co-defending network
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
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── file paths ─────────────────────────────────────────────────────────────────
EDGE_FILE  = "scripts/2026-05-05_player_net_m2_edges.csv"
PER90_FILE = "scripts/2026-05-20_player_level_per90.csv"
FIFA_FILE  = "scripts/fifa_ratings.csv"
NICK_FILE  = "scripts/nickname_map.csv"

METRICS = [
    "raw_involvement_per90",   "valued_involvement_per90",
    "raw_fault_per90",         "valued_fault_per90",
    "raw_contribution_per90",  "valued_contribution_per90",
    "passes_defended_per90",
]
FIFA_COLS = ["overall rating", "defending rating", "def_awareness_rating"]


# ── name normalisation ─────────────────────────────────────────────────────────
_LIGATURES = str.maketrans({
    "æ": "ae", "Æ": "Ae", "ø": "o",  "Ø": "O",
    "å": "a",  "Å": "A",  "ß": "ss", "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "Oe",
})

def _normalize(name: str) -> str:
    """Strip accents, expand ligatures, replace hyphens with spaces."""
    name = name.translate(_LIGATURES)
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace("-", " ")


# ── data loading (all cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading per-90 data…")
def load_per90() -> pd.DataFrame:
    return pd.read_csv(PER90_FILE)

@st.cache_data(show_spinner="Loading edges…")
def load_edges() -> pd.DataFrame:
    return pd.read_csv(EDGE_FILE)

@st.cache_data(show_spinner="Loading FIFA ratings…")
def load_fifa() -> pd.DataFrame:
    df = pd.read_csv(FIFA_FILE)
    df = df[df["comp"].str.contains("World Cup", na=False)]
    return (df.sort_values("overall rating", ascending=False)
              .drop_duplicates("name")[["name"] + FIFA_COLS])

@st.cache_data(show_spinner="Loading nickname map…")
def load_nickname_map() -> dict:
    df = pd.read_csv(NICK_FILE).drop_duplicates(subset=["player_nickname"])
    return dict(zip(df["player_nickname"], df["Player"]))

@st.cache_data(show_spinner="Building name map…")
def build_name_map(per90_names: tuple, fifa_names: tuple, threshold: int = 88) -> dict:
    """
    Matching priority:
      1. Exact match
      2. Normalized exact  (accents, ligatures, hyphens)
      3. Nickname map → exact → normalized exact
      4. Fuzzy fallback
    """
    fifa_set = set(fifa_names)
    nick_map = load_nickname_map()
    norm_fifa: dict[str, str] = {}
    for n in fifa_names:
        key = _normalize(n)
        if key not in norm_fifa:
            norm_fifa[key] = n
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


# ── player aggregation ────────────────────────────────────────────────────────
def get_qualifying(per90: pd.DataFrame, starters_only: bool,
                   min_minutes: int, min_matches: int) -> tuple:
    df = per90[per90["starter"] == 1] if starters_only else per90
    mins    = df.groupby("defender_name")["mins_played"].sum()
    matches = df.groupby("defender_name")["match_id"].nunique()
    players = mins[(mins >= min_minutes) & (matches >= min_matches)].index
    return tuple(sorted(players))


@st.cache_data
def aggregate_player(players: tuple) -> pd.DataFrame:
    df = load_per90()
    df = df[df["defender_name"].isin(players)].copy()

    def wmean(g):
        w = g["mins_played"].values
        return pd.Series({m: np.average(g[m].values, weights=w) for m in METRICS})

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


@st.cache_data
def compute_consistency(players: tuple) -> pd.DataFrame:
    """Per-player mean and CV (std/mean) across matches."""
    df = load_per90()
    df = df[df["defender_name"].isin(players)]
    mean = df.groupby("defender_name")[METRICS].mean()
    std  = df.groupby("defender_name")[METRICS].std().fillna(0)
    cv   = std / (mean.abs() + 1e-9)
    mean.columns = [c + "_mean" for c in mean.columns]
    cv.columns   = [c + "_cv"   for c in cv.columns]
    return pd.concat([mean, cv], axis=1).reset_index()


# ── network computations ──────────────────────────────────────────────────────
@st.cache_data
def compute_self_team(players: tuple) -> pd.DataFrame:
    """
    Self = passes where this player was the sole defender.
    Team = passes co-defended with at least one teammate.
    """
    df = load_edges()
    df = df[df["defender_name"].isin(players)].copy()
    n_def = df.groupby(["match_id", "defending_team", "passer_id", "receiver_id"]
                        )["defender_id"].transform("nunique")
    df["self_vi"] = df["valued_involvement"].where(n_def == 1, 0.0)
    df["team_vi"] = df["valued_involvement"].where(n_def >  1, 0.0)
    grp   = df.groupby("defender_name")
    s, t  = grp["self_vi"].sum(), grp["team_vi"].sum()
    total = (s + t).replace(0, np.nan)
    return pd.DataFrame({
        "defender_name": s.index,
        "self_ratio":    (s / total).values,
        "team_ratio":    (t / total).values,
        "self_vi":       s.values,
        "team_vi":       t.values,
    })


@st.cache_data
def compute_network_metrics(players: tuple) -> pd.DataFrame:
    """Degree, avg co-defenders, attacker diversity — averaged across matches."""
    df = load_edges()
    df = df[df["defender_name"].isin(players)].copy()
    n_def      = df.groupby(["match_id", "defending_team", "passer_id", "receiver_id"]
                              )["defender_id"].transform("nunique")
    df["n_co"] = (n_def - 1).clip(lower=0)

    # unique pass situations (passer→receiver) per player per match
    degree = (df.groupby(["defender_name", "match_id", "passer_id", "receiver_id"])
                .size().reset_index()
                .groupby(["defender_name", "match_id"]).size()
                .reset_index(name="degree"))

    co_def = (df.groupby(["defender_name", "match_id"])["n_co"]
                .mean().reset_index(name="avg_co_defenders"))

    att_div = (df.groupby(["defender_name", "match_id"])["passer_id"]
                 .nunique().reset_index(name="attacker_diversity"))

    merged = (degree.merge(co_def, on=["defender_name", "match_id"])
                    .merge(att_div, on=["defender_name", "match_id"]))
    return (merged.groupby("defender_name")[["degree", "avg_co_defenders", "attacker_diversity"]]
                  .mean().reset_index())


@st.cache_data
def compute_partnerships(players: tuple) -> pd.DataFrame:
    """Count how many times each pair of players co-defended the same pass."""
    df = load_edges()
    df = df[df["defender_name"].isin(players)]
    groups = (df.groupby(["match_id", "defending_team", "passer_id", "receiver_id"])
               ["defender_name"].apply(list).reset_index())
    rows = []
    for _, row in groups.iterrows():
        unique = sorted(set(row["defender_name"]))
        for a, b in combinations(unique, 2):
            rows.append({"player_a": a, "player_b": b})
    if not rows:
        return pd.DataFrame(columns=["player_a", "player_b", "co_defenses"])
    return (pd.DataFrame(rows)
              .groupby(["player_a", "player_b"]).size()
              .reset_index(name="co_defenses")
              .sort_values("co_defenses", ascending=False))


@st.cache_data
def compute_centrality(players: tuple) -> pd.DataFrame:
    """
    Build a co-defending network per team (across all matches).
    Nodes = defenders, edge weight = co-defending frequency.
    Return mean eigenvector centrality per player.
    """
    df = load_edges()
    df = df[df["defender_name"].isin(players)]
    groups = (df.groupby(["defending_team", "match_id", "passer_id", "receiver_id"])
               ["defender_name"].apply(list).reset_index())

    pair_rows = []
    for _, row in groups.iterrows():
        unique = sorted(set(row["defender_name"]))
        for a, b in combinations(unique, 2):
            pair_rows.append({"team": row["defending_team"], "player_a": a, "player_b": b})

    if not pair_rows:
        return pd.DataFrame(columns=["defender_name", "centrality"])

    ew = (pd.DataFrame(pair_rows)
            .groupby(["team", "player_a", "player_b"]).size()
            .reset_index(name="weight"))

    records = []
    for team, grp in ew.groupby("team"):
        G = nx.Graph()
        for _, r in grp.iterrows():
            G.add_edge(r["player_a"], r["player_b"], weight=r["weight"])
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight="weight")
            for node, c in cent.items():
                records.append({"defender_name": node, "centrality": c})
        except Exception:
            pass

    if not records:
        return pd.DataFrame(columns=["defender_name", "centrality"])
    return (pd.DataFrame(records)
              .groupby("defender_name")["centrality"].mean()
              .reset_index())


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Player Network Analysis", layout="wide")
st.title("Player-Level Defensive Network Analysis")

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

(t1, t2, t3, t4, t5, t6, t7) = st.tabs([
    "Over/Underrated", "Archetypes", "Consistency",
    "Self vs Team", "Network Metrics", "Partnerships", "Centrality",
])


# ── Tab 1: Over / Underrated ──────────────────────────────────────────────────
with t1:
    df_fifa = merge_with_fifa(players)
    c1, c2  = st.columns(2)
    fifa_col   = c1.selectbox("FIFA rating", FIFA_COLS, index=1, key="ou_fifa")
    metric_col = c2.selectbox("Network metric", METRICS, index=1, key="ou_metric")

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


# ── Tab 2: Archetypes ─────────────────────────────────────────────────────────
with t2:
    agg   = aggregate_player(players)
    c1, c2 = st.columns(2)
    k         = c1.slider("Clusters (k)", 2, 6, 4, key="arch_k")
    sel_cols  = c2.multiselect("Metrics for clustering", METRICS, default=METRICS, key="arch_m")

    if len(sel_cols) < 2:
        st.warning("Select at least 2 metrics.")
    else:
        valid  = agg.dropna(subset=sel_cols).copy()
        Xs     = StandardScaler().fit_transform(valid[sel_cols])
        valid["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs).astype(str)
        coords = PCA(n_components=2).fit(Xs)
        xy     = coords.transform(Xs)
        valid["PC1"], valid["PC2"] = xy[:, 0], xy[:, 1]

        fig = px.scatter(
            valid, x="PC1", y="PC2", color="cluster",
            hover_name="defender_name",
            hover_data={"team": True, "cluster": True},
            title=f"Defensive archetypes — k={k}  (PCA projection)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"PC1 {coords.explained_variance_ratio_[0]:.1%} · "
            f"PC2 {coords.explained_variance_ratio_[1]:.1%} of variance"
        )

        st.markdown("**Cluster profiles — mean per-90 metrics**")
        profile = valid.groupby("cluster")[sel_cols].mean().round(3)
        st.dataframe(
            profile.style.background_gradient(cmap="YlGn", axis=0),
            use_container_width=True,
        )


# ── Tab 3: Consistency ────────────────────────────────────────────────────────
with t3:
    cons     = compute_consistency(players)
    df_fifa3 = merge_with_fifa(players)[["defender_name", "team", "defending rating"]]
    cons     = cons.merge(df_fifa3, on="defender_name", how="left")

    metric_sel = st.selectbox("Metric", METRICS, index=1, key="cons_m")
    mean_col, cv_col = metric_sel + "_mean", metric_sel + "_cv"

    sub = cons[[mean_col, cv_col, "defender_name", "team", "defending rating"]].dropna()
    xm, ym = sub[mean_col].median(), sub[cv_col].median()

    fig = px.scatter(
        sub, x=mean_col, y=cv_col,
        color="defending rating", color_continuous_scale="RdYlGn",
        hover_name="defender_name",
        hover_data={"team": True, "defending rating": True},
        title="Consistency: mean vs CV — coloured by FIFA defending rating",
        labels={mean_col: f"Mean {metric_sel}", cv_col: "CV (std / mean)"},
    )
    fig.add_vline(x=xm, line_dash="dot", line_color="grey")
    fig.add_hline(y=ym, line_dash="dot", line_color="grey")
    for text, ax, ay in [
        ("Consistent high", sub[mean_col].quantile(0.97), sub[cv_col].quantile(0.03)),
        ("Volatile high",   sub[mean_col].quantile(0.97), sub[cv_col].quantile(0.97)),
        ("Consistent low",  sub[mean_col].quantile(0.03), sub[cv_col].quantile(0.03)),
        ("Volatile low",    sub[mean_col].quantile(0.03), sub[cv_col].quantile(0.97)),
    ]:
        fig.add_annotation(x=ax, y=ay, text=text, showarrow=False,
                           font=dict(size=10, color="grey"))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Self vs Team ───────────────────────────────────────────────────────
with t4:
    st.caption(
        "**Self** = passes where this player was the **only** defender involved. "
        "**Team** = passes co-defended with at least one teammate. "
        "~97% of involvements in this dataset are team actions."
    )
    st_df    = compute_self_team(players)
    df_fifa4 = merge_with_fifa(players)[["defender_name", "team", "defending rating"]]
    merged4  = df_fifa4.merge(st_df, on="defender_name", how="inner")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            merged4, x="defending rating", y="self_ratio",
            color="self_ratio", color_continuous_scale="RdYlGn_r",
            hover_name="defender_name",
            hover_data={"team": True, "self_ratio": ":.3f"},
            title="Self ratio vs FIFA defending rating",
            labels={"self_ratio": "Self ratio (solo / total)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(st_df, x="self_ratio", nbins=40,
                            title="Distribution of self ratio across players")
        st.plotly_chart(fig2, use_container_width=True)

    n_show4 = st.slider("Top / bottom N", 3, 20, 10, key="st_n")
    col_a, col_b = st.columns(2)
    show_cols = ["defender_name", "team", "self_ratio", "self_vi", "team_vi"]
    merged4_show = merged4[show_cols].round(4)
    col_a.markdown("**Highest self ratio** (more isolated)")
    col_a.dataframe(merged4_show.nlargest(n_show4, "self_ratio").reset_index(drop=True),
                    use_container_width=True)
    col_b.markdown("**Lowest self ratio** (most coordinated)")
    col_b.dataframe(merged4_show.nsmallest(n_show4, "self_ratio").reset_index(drop=True),
                    use_container_width=True)


# ── Tab 5: Network Metrics ────────────────────────────────────────────────────
with t5:
    nm       = compute_network_metrics(players)
    df_fifa5 = merge_with_fifa(players)[["defender_name", "team", "defending rating"]]
    merged5  = df_fifa5.merge(nm, on="defender_name", how="inner")

    net_metric = st.selectbox(
        "Network metric",
        ["degree", "avg_co_defenders", "attacker_diversity"],
        format_func=lambda x: {
            "degree":             "Degree (unique pass situations / match)",
            "avg_co_defenders":   "Avg co-defenders per pass",
            "attacker_diversity": "Attacker diversity (unique passers / match)",
        }[x],
        key="nm_sel",
    )
    fig = px.scatter(
        merged5, x="defending rating", y=net_metric,
        color=net_metric, color_continuous_scale="Blues",
        hover_name="defender_name",
        hover_data={"team": True},
        title=f"{net_metric} vs FIFA defending rating",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Distributions**")
    c1, c2, c3 = st.columns(3)
    for col_name, label, container in [
        ("degree",             "Degree",             c1),
        ("avg_co_defenders",   "Avg co-defenders",   c2),
        ("attacker_diversity", "Attacker diversity", c3),
    ]:
        fig = px.histogram(nm, x=col_name, nbins=30, title=label)
        container.plotly_chart(fig, use_container_width=True)


# ── Tab 6: Partnerships ───────────────────────────────────────────────────────
with t6:
    pairs = compute_partnerships(players)
    if pairs.empty:
        st.warning("No partnership data for current filters.")
    else:
        n_top = st.slider("Top N pairs", 10, 50, 20, key="pair_n")
        top   = pairs.head(n_top).copy()
        top["pair"] = top["player_a"] + "  +  " + top["player_b"]
        fig = px.bar(
            top, x="co_defenses", y="pair", orientation="h",
            title=f"Top {n_top} co-defending pairs (all teams)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        team_sel    = st.selectbox("Team heatmap", sorted(per90["defending_team_name"].unique()), key="pair_team")
        team_players = set(per90[per90["defending_team_name"] == team_sel]["defender_name"].unique())
        team_pairs   = pairs[pairs["player_a"].isin(team_players) & pairs["player_b"].isin(team_players)]

        if team_pairs.empty:
            st.info(f"No co-defending pairs found for {team_sel} with current filters.")
        else:
            all_p = sorted(team_players & (set(team_pairs["player_a"]) | set(team_pairs["player_b"])))
            mat   = pd.DataFrame(0, index=all_p, columns=all_p, dtype=float)
            for _, r in team_pairs.iterrows():
                mat.loc[r["player_a"], r["player_b"]] = r["co_defenses"]
                mat.loc[r["player_b"], r["player_a"]] = r["co_defenses"]
            fig2 = px.imshow(mat, color_continuous_scale="Blues",
                             title=f"{team_sel} — co-defending heatmap")
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 7: Centrality ─────────────────────────────────────────────────────────
with t7:
    st.caption(
        "Eigenvector centrality in the team co-defending network. "
        "High centrality = player co-defends frequently with many highly-connected teammates."
    )
    cent = compute_centrality(players)
    if cent.empty:
        st.warning("No centrality data — loosen filters.")
    else:
        df_fifa7 = merge_with_fifa(players)[["defender_name", "team", "defending rating"]]
        merged7  = df_fifa7.merge(cent, on="defender_name", how="inner")

        fig = px.scatter(
            merged7, x="defending rating", y="centrality",
            color="centrality", color_continuous_scale="Viridis",
            hover_name="defender_name",
            hover_data={"team": True, "centrality": ":.4f"},
            title="Eigenvector centrality vs FIFA defending rating",
        )
        st.plotly_chart(fig, use_container_width=True)

        n_show7 = st.slider("Top / bottom N", 5, 20, 10, key="cent_n")
        c1, c2  = st.columns(2)
        show7   = ["defender_name", "team", "centrality", "defending rating"]
        c1.markdown("**Highest centrality** (network hubs)")
        c1.dataframe(merged7.nlargest(n_show7, "centrality")[show7].reset_index(drop=True),
                     use_container_width=True)
        c2.markdown("**Lowest centrality** (peripheral defenders)")
        c2.dataframe(merged7.nsmallest(n_show7, "centrality")[show7].reset_index(drop=True),
                     use_container_width=True)
