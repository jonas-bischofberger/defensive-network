"""
Player-level defensive network analysis — v3 (unified).

Source data: 2026-05-25_player_level_enriched_with_ratings.csv
  - FIFA ratings embedded; no external name matching required.
  - Three normalization families selectable from sidebar:
      * Per 90 min
      * Per pass against  (opportunity-adjusted)
      * Per pass defended (efficiency)

Edge data: 2026-05-05_player_net_m2_edges.csv
  - Used for structural metrics: breadth, Gini, centrality, Jaccard stability.

Tabs:
  1  Breadth           — unique attackers / pass combinations defended
  2  Concentration     — Gini of involvement distribution
  3  Correlation       — activity + network metrics vs FIFA ratings
                         (option: partial out overall_position)
  4  ICC               — player consistency across matches
  5  Network Depth     — eigenvector / betweenness of attacked passing nodes
  6  Stability         — match-to-match Jaccard similarity of defensive maps
  7  Player Map        — single-player heatmap
  8  Archetypes        — k-means + PCA clustering
  9  Over/Underrated   — regression residuals vs FIFA
"""
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ── Constants ──────────────────────────────────────────────────────────────────
ENRICHED_FILE = "scripts/2026-05-25_player_level_enriched_with_ratings.csv"
EDGE_FILE     = "scripts/2026-05-05_player_net_m2_edges.csv"

FIFA_COLS = ["overall_rating", "defending_rating", "def_awareness_rating", "interceptions_rating"]

METRIC_FAMILIES: dict[str, list[str]] = {
    "Per 90 min": [
        "raw_involvement_per90",  "valued_involvement_per90",
        "raw_fault_per90",        "valued_fault_per90",
        "raw_contribution_per90", "valued_contribution_per90",
        "passes_defended_per90",
    ],
    "Per pass against (opportunity-adjusted)": [
        "raw_involvement_per_pass_against",  "valued_involvement_per_pass_against",
        "raw_fault_per_pass_against",        "valued_fault_per_pass_against",
        "raw_contribution_per_pass_against", "valued_contribution_per_pass_against",
        "passes_defended_per_pass_against",
    ],
    "Per pass against — on field (opportunity-adjusted)": [
        "raw_involvement_per_pass_against_onfield",  "valued_involvement_per_pass_against_onfield",
        "raw_fault_per_pass_against_onfield",        "valued_fault_per_pass_against_onfield",
        "raw_contribution_per_pass_against_onfield", "valued_contribution_per_pass_against_onfield",
        "passes_defended_per_pass_against_onfield",
    ],
    "Per pass defended (efficiency)": [
        "raw_involvement_per_pass_defended",  "valued_involvement_per_pass_defended",
        "raw_fault_per_pass_defended",        "valued_fault_per_pass_defended",
        "raw_contribution_per_pass_defended", "valued_contribution_per_pass_defended",
    ],
    "Absolute (raw per-match totals)": [
        "raw_involvement",  "valued_involvement",
        "raw_fault",        "valued_fault",
        "raw_contribution", "valued_contribution",
        "passes_defended",
    ],
}

BASIC_6_PER90 = [
    "raw_involvement_per90",   "valued_involvement_per90",
    "raw_fault_per90",         "valued_fault_per90",
    "raw_contribution_per90",  "valued_contribution_per90",
]
PRESENTATION_FIFA = ["defending_rating", "def_awareness_rating"]


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading enriched data…")
def load_enriched() -> pd.DataFrame:
    return pd.read_csv(ENRICHED_FILE)


@st.cache_data(show_spinner="Loading edges…")
def load_edges() -> pd.DataFrame:
    return pd.read_csv(EDGE_FILE)


# ── Player filtering ───────────────────────────────────────────────────────────
def get_qualifying(df: pd.DataFrame, starters_only: bool,
                   min_minutes: int, min_matches: int) -> tuple:
    sub     = df[df["starter"] == 1] if starters_only else df
    mins    = sub.groupby("defender_name")["mins_played"].sum()
    matches = sub.groupby("defender_name")["match_id"].nunique()
    return tuple(sorted(mins[(mins >= min_minutes) & (matches >= min_matches)].index))


# ── Player-level aggregation ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Aggregating player data…")
def aggregate_player(players: tuple, norm_method: str) -> pd.DataFrame:
    """
    One row per player. Activity metrics = minutes-weighted mean across matches.
    FIFA ratings are constant per player in the enriched table → take mean (any aggregation works).
    """
    df      = load_enriched()
    df      = df[df["defender_name"].isin(players)].copy()
    metrics = METRIC_FAMILIES[norm_method]

    rows = []
    for name, g in df.groupby("defender_name"):
        w   = g["mins_played"].values.astype(float)
        row = {"defender_name": name}
        for m in metrics:
            if m in g.columns:
                row[m] = float(np.average(g[m].values.astype(float), weights=w))
        row["total_mins"]           = float(w.sum())
        row["n_matches"]            = int(g["match_id"].nunique())
        def _mode(s):
            m = s.dropna().mode()
            return m.iloc[0] if len(m) > 0 else None

        row["team"]                 = _mode(g["defending_team_name"])
        row["overall_position"]     = _mode(g["overall_position"])
        row["most_common_position"] = _mode(g["most_common_position"])
        for fc in FIFA_COLS:
            row[fc] = float(g[fc].mean())
        rows.append(row)

    return pd.DataFrame(rows)


# ── Defensive map topology (from edge file) ────────────────────────────────────
@st.cache_data(show_spinner="Computing breadth / Gini…")
def compute_defensive_map_metrics(players: tuple) -> pd.DataFrame:
    edges = load_edges()
    enr   = load_enriched()
    df    = edges[edges["defender_name"].isin(players)].copy()

    mins_lookup = (enr[enr["defender_name"].isin(players)]
                   .set_index(["defender_name", "match_id"])["mins_played"]
                   .to_dict())

    rows = []
    for (defender, match_id), g in df.groupby(["defender_name", "match_id"]):
        g_clean  = g.dropna(subset=["passer_id", "receiver_id"])
        passers  = set(g_clean["passer_id"].astype(int))
        receivers = set(g_clean["receiver_id"].astype(int))
        n_unique = len(passers | receivers)

        pass_inv = g_clean.groupby(["passer_id", "receiver_id"])["valued_involvement"].sum()
        degree   = len(pass_inv)

        x = pass_inv.values[pass_inv.values > 0]
        n = len(x)
        if n >= 2:
            xs   = np.sort(x)
            gini = (2 * np.dot(np.arange(1, n + 1), xs) / (n * xs.sum())) - (n + 1) / n
        else:
            gini = np.nan

        mins = mins_lookup.get((defender, match_id), np.nan)
        rows.append({
            "defender_name": defender, "match_id": match_id,
            "mins_played": mins, "n_unique_attackers": n_unique,
            "degree": degree, "gini": gini,
        })

    pm = pd.DataFrame(rows)
    pm["n_unique_attackers_per90"] = pm["n_unique_attackers"] / pm["mins_played"] * 90
    pm["degree_per90"]             = pm["degree"]             / pm["mins_played"] * 90

    def wavg(g):
        w = np.where(np.isnan(g["mins_played"].values), 0.0, g["mins_played"].values)
        if w.sum() == 0:
            return pd.Series({c: np.nan for c in
                              ["n_unique_attackers", "n_unique_attackers_per90",
                               "degree", "degree_per90", "gini"]})
        return pd.Series({
            "n_unique_attackers":       np.average(g["n_unique_attackers"].values, weights=w),
            "n_unique_attackers_per90": np.average(g["n_unique_attackers_per90"].values, weights=w),
            "degree":                   np.average(g["degree"].values, weights=w),
            "degree_per90":             np.average(g["degree_per90"].values, weights=w),
            "gini":                     np.average(g["gini"].fillna(0).values, weights=w),
        })

    return pm.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


@st.cache_data(show_spinner="Computing position-based breadth…")
def compute_position_breadth(players: tuple, pos_col: str) -> pd.DataFrame:
    """
    Same as compute_defensive_map_metrics but uses (passer_position, receiver_position)
    instead of player IDs — making breadth comparable across matches and opponents.

    pos_col: "overall_position" (5 broad) or "most_common_position" (specific roles).
    """
    edges = load_edges()
    enr   = load_enriched()

    id_pos = (enr[["defender_id", pos_col]].dropna()
              .drop_duplicates("defender_id")
              .set_index("defender_id")[pos_col]
              .to_dict())

    df = edges[edges["defender_name"].isin(players)].copy()
    df["passer_pos"]   = df["passer_id"].map(id_pos)
    df["receiver_pos"] = df["receiver_id"].map(id_pos)
    df = df.dropna(subset=["passer_pos", "receiver_pos"])

    mins_lookup = (enr[enr["defender_name"].isin(players)]
                   .set_index(["defender_name", "match_id"])["mins_played"]
                   .to_dict())

    rows = []
    for (defender, match_id), g in df.groupby(["defender_name", "match_id"]):
        positions    = set(g["passer_pos"]) | set(g["receiver_pos"])
        n_unique_pos = len(positions)
        degree_pos   = len(set(zip(g["passer_pos"], g["receiver_pos"])))
        mins = mins_lookup.get((defender, match_id), np.nan)
        rows.append({
            "defender_name":     defender,
            "match_id":          match_id,
            "mins_played":       mins,
            "n_unique_positions": n_unique_pos,
            "degree_positions":   degree_pos,
        })

    pm = pd.DataFrame(rows)
    pm["n_unique_positions_per90"] = pm["n_unique_positions"] / pm["mins_played"] * 90
    pm["degree_positions_per90"]   = pm["degree_positions"]   / pm["mins_played"] * 90

    def wavg(g):
        w = np.where(np.isnan(g["mins_played"].values), 0.0, g["mins_played"].values)
        if w.sum() == 0:
            return pd.Series({c: np.nan for c in
                              ["n_unique_positions", "n_unique_positions_per90",
                               "degree_positions", "degree_positions_per90"]})
        return pd.Series({
            "n_unique_positions":       np.average(g["n_unique_positions"].values, weights=w),
            "n_unique_positions_per90": np.average(g["n_unique_positions_per90"].values, weights=w),
            "degree_positions":         np.average(g["degree_positions"].values, weights=w),
            "degree_positions_per90":   np.average(g["degree_positions_per90"].values, weights=w),
        })

    return pm.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


@st.cache_data(show_spinner="Computing centrality scores…")
def compute_centrality_scores(players: tuple) -> pd.DataFrame:
    """
    Build the full attacking network from all passes in the dataset.
    For each defender: weighted-avg centrality of the attacking nodes they defended.

    Eigenvector → defends hub attackers (most connected in passing network).
    Betweenness → defends bridge attackers (linking different zones).
    """
    edges = load_edges()

    atk = edges.groupby(["passer_id", "receiver_id"])["n_passes"].sum().reset_index()
    G   = nx.DiGraph()
    for _, r in atk.iterrows():
        G.add_edge(int(r["passer_id"]), int(r["receiver_id"]), weight=r["n_passes"])
    Gu = G.to_undirected()

    try:
        eig_cent = nx.eigenvector_centrality_numpy(Gu, weight="weight")
    except Exception:
        eig_cent = dict.fromkeys(Gu.nodes(), np.nan)
    btw_cent = nx.betweenness_centrality(Gu, weight="weight", normalized=True)

    df = edges[edges["defender_name"].isin(players)].copy()
    df["eig_node"] = df["passer_id"].map(eig_cent).fillna(0) + df["receiver_id"].map(eig_cent).fillna(0)
    df["btw_node"] = df["passer_id"].map(btw_cent).fillna(0) + df["receiver_id"].map(btw_cent).fillna(0)

    def wavg(g):
        w = g["valued_involvement"].values
        if w.sum() == 0:
            return pd.Series({"eig_centrality": np.nan, "btw_centrality": np.nan})
        return pd.Series({
            "eig_centrality": float(np.average(g["eig_node"].values, weights=w)),
            "btw_centrality": float(np.average(g["btw_node"].values, weights=w)),
        })

    return df.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


@st.cache_data(show_spinner="Computing match-to-match stability…")
def compute_jaccard_stability(players: tuple, pos_col: str = "overall_position") -> pd.DataFrame:
    """
    Position-based structural stability of each player's defensive map.

    pos_col: "overall_position" (5 broad categories) or "most_common_position" (specific roles).

    For each player: average pairwise Jaccard similarity between defensive maps
    across all pairs of matches, where each map is the set of
    (passer_position, receiver_position) combinations defended.

    Using position instead of player ID abstracts away opponent identity.
    """
    edges = load_edges()
    enr   = load_enriched()

    id_pos = (enr[["defender_id", pos_col]].dropna()
              .drop_duplicates("defender_id")
              .set_index("defender_id")[pos_col]
              .to_dict())

    df = edges[edges["defender_name"].isin(players)].copy()
    df["passer_pos"]   = df["passer_id"].map(id_pos)
    df["receiver_pos"] = df["receiver_id"].map(id_pos)
    df = df.dropna(subset=["passer_pos", "receiver_pos"])

    rows = []
    for defender, grp in df.groupby("defender_name"):
        match_sets: dict = {}
        for mid, mg in grp.groupby("match_id"):
            match_sets[mid] = set(zip(mg["passer_pos"], mg["receiver_pos"]))

        n = len(match_sets)
        if n < 2:
            rows.append({"defender_name": defender, "jaccard_stability": np.nan,
                         "stability_n_matches": n})
            continue

        sets_list = list(match_sets.values())
        jaccards  = []
        for i in range(len(sets_list)):
            for j in range(i + 1, len(sets_list)):
                s1, s2 = sets_list[i], sets_list[j]
                union  = len(s1 | s2)
                jaccards.append(len(s1 & s2) / union if union > 0 else 0.0)

        rows.append({
            "defender_name":      defender,
            "jaccard_stability":  float(np.mean(jaccards)),
            "stability_n_matches": n,
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing entropy breadth…")
def compute_entropy_breadth(players: tuple, pos_col: str) -> pd.DataFrame:
    """
    Shannon entropy of valued_involvement across (passer_pos, receiver_pos) combos per match.
    High = effort spread evenly across many combos; low = concentrated on a few.
    """
    edges = load_edges()
    enr   = load_enriched()

    id_pos = (enr[["defender_id", pos_col]].dropna()
              .drop_duplicates("defender_id")
              .set_index("defender_id")[pos_col].to_dict())

    df = edges[edges["defender_name"].isin(players)].copy()
    df["passer_pos"]   = df["passer_id"].map(id_pos)
    df["receiver_pos"] = df["receiver_id"].map(id_pos)
    df = df.dropna(subset=["passer_pos", "receiver_pos"])

    mins_lookup = (enr[enr["defender_name"].isin(players)]
                   .set_index(["defender_name", "match_id"])["mins_played"].to_dict())

    rows = []
    for (defender, match_id), g in df.groupby(["defender_name", "match_id"]):
        inv = g.groupby(["passer_pos", "receiver_pos"])["valued_involvement"].sum()
        inv = inv.values[inv.values > 0]
        if len(inv) > 1:
            p   = inv / inv.sum()
            ent = float(-np.sum(p * np.log(p)))
        else:
            ent = 0.0
        mins = mins_lookup.get((defender, match_id), np.nan)
        rows.append({"defender_name": defender, "match_id": match_id,
                     "mins_played": mins, "entropy_breadth": ent})

    pm = pd.DataFrame(rows)
    pm["entropy_breadth_per90"] = pm["entropy_breadth"] / pm["mins_played"] * 90

    def wavg(g):
        w = np.where(np.isnan(g["mins_played"].values), 0.0, g["mins_played"].values)
        if w.sum() == 0:
            return pd.Series({"entropy_breadth": np.nan, "entropy_breadth_per90": np.nan})
        return pd.Series({
            "entropy_breadth":       np.average(g["entropy_breadth"].values, weights=w),
            "entropy_breadth_per90": np.average(g["entropy_breadth_per90"].values, weights=w),
        })

    return pm.groupby("defender_name").apply(wavg, include_groups=False).reset_index()


@st.cache_data(show_spinner="Computing cosine stability…")
def compute_cosine_stability(players: tuple, pos_col: str) -> pd.DataFrame:
    """
    Pairwise cosine similarity of per-match involvement vectors across (passer_pos, receiver_pos) combos.
    High = same combos at similar intensities across matches; low = different effort distribution.
    """
    edges = load_edges()
    enr   = load_enriched()

    id_pos = (enr[["defender_id", pos_col]].dropna()
              .drop_duplicates("defender_id")
              .set_index("defender_id")[pos_col].to_dict())

    df = edges[edges["defender_name"].isin(players)].copy()
    df["passer_pos"]   = df["passer_id"].map(id_pos)
    df["receiver_pos"] = df["receiver_id"].map(id_pos)
    df = df.dropna(subset=["passer_pos", "receiver_pos"])

    rows = []
    for defender, grp in df.groupby("defender_name"):
        all_combos: set = set()
        match_invs: dict = {}
        for mid, mg in grp.groupby("match_id"):
            ci = mg.groupby(["passer_pos", "receiver_pos"])["valued_involvement"].sum()
            match_invs[mid] = ci
            all_combos.update(ci.index.tolist())

        n = len(match_invs)
        if n < 2:
            rows.append({"defender_name": defender,
                         "cosine_stability": np.nan, "cosine_n_matches": n})
            continue

        combos_list = sorted(all_combos)
        vecs = [np.array([match_invs[mid].get(c, 0.0) for c in combos_list])
                for mid in match_invs]

        cosines = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a, b = vecs[i], vecs[j]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                cosines.append(float(np.dot(a, b) / denom) if denom > 0 else 0.0)

        rows.append({"defender_name": defender,
                     "cosine_stability": float(np.mean(cosines)),
                     "cosine_n_matches": n})

    return pd.DataFrame(rows)


# ── Statistics ─────────────────────────────────────────────────────────────────
def _residualise_position(df: pd.DataFrame, col: str) -> np.ndarray:
    """Partial out overall_position using OLS on dummy coding. Returns residuals."""
    sub  = df[[col, "overall_position"]].dropna()
    X    = pd.get_dummies(sub["overall_position"], drop_first=True).values.astype(float)
    y    = sub[col].values.astype(float)
    pred = LinearRegression().fit(X, y).predict(X)
    result                = np.full(len(df), np.nan)
    valid_mask            = df[col].notna().values & df["overall_position"].notna().values
    result[valid_mask]    = y - pred
    return result


def basic6_corr_tbl(df: pd.DataFrame, metrics: list | None = None,
                    corr_method: str = "Pearson", partial: bool = True) -> None:
    """basic metrics × (defending_rating + def_awareness_rating) × (raw / partial).

    metrics : list of metric columns (defaults to BASIC_6_PER90). Pass the 6
              basic metrics of the currently selected normalization family.
    partial : if True, also compute position-residualised correlations; if
              False (e.g. a single position is already selected) the partial
              columns are shown as —.
    """
    if metrics is None:
        metrics = BASIC_6_PER90
    fn = pearsonr if corr_method == "Pearson" else spearmanr
    results: dict = {}
    for fc in PRESENTATION_FIFA:
        for m in metrics:
            sub = (df[[m, fc, "overall_position"]].dropna()
                   if m in df.columns else df.iloc[0:0])
            if len(sub) < 5:
                results[(fc, "raw", m)] = (np.nan, np.nan)
                results[(fc, "partial", m)] = (np.nan, np.nan)
                continue
            r, p = fn(sub[m].values.astype(float), sub[fc].values.astype(float))
            results[(fc, "raw", m)] = (float(r), float(p))
            if partial:
                x_r = _residualise_position(sub, m)
                y_r = _residualise_position(sub, fc)
                mask = ~(np.isnan(x_r) | np.isnan(y_r))
                if mask.sum() < 5:
                    results[(fc, "partial", m)] = (np.nan, np.nan)
                else:
                    rp, pp = fn(x_r[mask], y_r[mask])
                    results[(fc, "partial", m)] = (float(rp), float(pp))
            else:
                results[(fc, "partial", m)] = (np.nan, np.nan)

    def _fmt(r, p):
        if np.isnan(r):
            return "—"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"{r:.2f}{sig}"

    mi_cols = pd.MultiIndex.from_tuples(
        [(fc, kind) for fc in PRESENTATION_FIFA for kind in ("raw", "partial")]
    )
    rows = [
        [_fmt(*results[(fc, kind, m)]) for fc in PRESENTATION_FIFA for kind in ("raw", "partial")]
        for m in metrics
    ]
    disp = pd.DataFrame(rows, index=metrics, columns=mi_cols)
    gmap = np.array([
        [results[(fc, kind, m)][0] for fc in PRESENTATION_FIFA for kind in ("raw", "partial")]
        for m in metrics
    ])
    st.dataframe(
        disp.style.background_gradient(cmap="RdYlGn", gmap=gmap, axis=None, vmin=-1, vmax=1),
        use_container_width=True,
    )


def compute_icc(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """ICC(1,1) one-way random effects model on match-level data."""
    rows = []
    for m in metrics:
        s = df[["defender_name", m]].dropna()
        if s["defender_name"].nunique() < 2:
            continue
        g        = s.groupby("defender_name")[m]
        nt, ng   = len(s), g.ngroups
        sz       = g.count()
        mn       = g.mean()
        grand_mn = s[m].mean()
        msb      = (sz * (mn - grand_mn) ** 2).sum() / (ng - 1)
        msw      = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0       = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        icc      = (msb - msw) / (msb + (k0 - 1) * msw)
        rows.append({
            "metric":    m,
            "ICC":       round(float(icc), 3),
            "n_players": ng,
            "n_obs":     nt,
            "stability": "stable" if icc > 0.5 else ("moderate" if icc > 0.25 else "low"),
        })
    return pd.DataFrame(rows)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Player Network Analysis", layout="wide")
st.title("Player-Level Defensive Network Analysis")
st.caption(
    "Source: **2026-05-25_player_level_enriched_with_ratings.csv** · "
    "FIFA ratings embedded, no external matching required. "
    "Switch normalization method in the sidebar."
)

enriched = load_enriched()

with st.sidebar:
    st.header("Filters")
    starters_only = st.checkbox("Starters only", value=True)
    max_mins    = int(enriched.groupby("defender_name")["mins_played"].sum().max())
    max_matches = int(enriched.groupby("defender_name")["match_id"].nunique().max())
    min_minutes = st.slider("Min total minutes", 0, max_mins, 90, 10)
    min_matches = st.slider("Min matches played", 1, max_matches, 2)

    st.divider()
    st.header("Normalization")
    norm_method = st.selectbox(
        "Activity metric normalization",
        list(METRIC_FAMILIES.keys()),
        index=0,
        help=(
            "**Per 90 min** — volume relative to playing time.\n\n"
            "**Per pass against** — rate per passing opportunity faced. Controls for "
            "how many passes the team conceded (opportunity-adjusted), but uses the "
            "team's *full-match* opponent passes for everyone.\n\n"
            "**Per pass against — on field** — same idea, but the denominator is only "
            "the opponent passes that happened *while this player was on the pitch*. "
            "More accurate for substitutes / partial-match players.\n\n"
            "**Per pass defended** — rate per pass the player actually stopped (efficiency). "
            "High values = high involvement *on the passes they did stop*."
        ),
    )

players = get_qualifying(enriched, starters_only, min_minutes, min_matches)

st.caption(f"**{len(players)} players** pass filters")

ACTIVE_METRICS = METRIC_FAMILIES[norm_method]

# Compute once, reuse across all tabs
# (stability / entropy / breadth computed inside Tab 1 — depend on pos_col selector)
agg  = aggregate_player(players, norm_method)
dm   = compute_defensive_map_metrics(players)
cent = compute_centrality_scores(players)

full_df = (
    agg
    .merge(dm,   on="defender_name", how="left")
    .merge(cent, on="defender_name", how="left")
)

tabs = st.tabs([
    "Defensive Style", "Concentration", "Correlation vs FIFA",
    "ICC", "Network Depth",
    "Player Map", "Archetypes", "Over/Underrated",
    "Volume × Quality",
])
t_style, t_gini, t_corr, t_icc, t_depth, t_map, t_arch, t_ou, t_vq = tabs


# ── Tab 1: Defensive Style (Breadth + Stability) ──────────────────────────────
with t_style:
    st.markdown(
        "Defensive style = **how many** different passing combinations a player covers (breadth) "
        "× **how consistently** the same types of combinations appear across matches (stability). "
        "The quadrant plot at the bottom combines both dimensions."
    )

    c_pos, c_fifa_s = st.columns(2)
    pos_gran = c_pos.radio(
        "Position granularity (for position-based metrics)",
        ["overall_position (5 broad)", "most_common_position (specific roles)"],
        horizontal=True, key="style_pos_gran",
    )
    pos_col  = "overall_position" if pos_gran.startswith("overall") else "most_common_position"
    fifa_col_s = c_fifa_s.selectbox("FIFA rating (colour)", FIFA_COLS, index=1, key="style_fifa")

    c_brw, c_stab_t = st.columns(2)
    br_weight = c_brw.radio(
        "Breadth weighting (position-based only)",
        ["Structural (count)", "Weighted (entropy of involvement)"],
        horizontal=True, key="br_weight",
    )
    stab_type = c_stab_t.radio(
        "Stability metric",
        ["Structural (Jaccard)", "Weighted (Cosine similarity)"],
        horizontal=True, key="stab_type",
    )
    stab_col      = "jaccard_stability" if stab_type.startswith("Structural") else "cosine_stability"
    stab_label    = "Avg pairwise Jaccard similarity" if stab_col == "jaccard_stability" else "Avg pairwise cosine similarity"
    n_matches_col = "stability_n_matches" if stab_col == "jaccard_stability" else "cosine_n_matches"

    # Compute position-based breadth and stability (depend on pos_col)
    pb  = compute_position_breadth(players, pos_col)
    jac = compute_jaccard_stability(players, pos_col)
    eb  = compute_entropy_breadth(players, pos_col)
    cs  = compute_cosine_stability(players, pos_col)

    style_df = (
        full_df
        .merge(pb,  on="defender_name", how="left")
        .merge(jac[["defender_name", "jaccard_stability", "stability_n_matches"]],
               on="defender_name", how="left")
        .merge(eb[["defender_name", "entropy_breadth", "entropy_breadth_per90"]],
               on="defender_name", how="left")
        .merge(cs[["defender_name", "cosine_stability", "cosine_n_matches"]],
               on="defender_name", how="left")
    )

    pos_filt = st.selectbox(
        "Filter by position",
        ["All"] + sorted(style_df["overall_position"].dropna().unique()),
        index=0, key="style_pos_filt",
    )
    if pos_filt != "All":
        style_df = style_df[style_df["overall_position"] == pos_filt]

    # ── Section A: Breadth ────────────────────────────────────────────────────
    st.subheader("Breadth")
    st.caption(
        "**Player-ID based** uses exact player identities (confounded by opponent). "
        "**Position-based** abstracts to role types — comparable across matches and opponents."
    )

    br_mode = st.radio(
        "Breadth metric",
        ["Player-ID based", "Position-based"],
        horizontal=True, key="br_mode",
    )

    if br_mode == "Player-ID based":
        x_br, y_br = "n_unique_attackers", "degree"
        xl_br, yl_br = "Avg unique attackers / match", "Avg unique pass combinations / match"
    elif br_weight == "Weighted (entropy of involvement)":
        x_br, y_br = "entropy_breadth", "degree_positions"
        xl_br, yl_br = "Entropy of involvement (nats)", "Avg unique position combos / match"
    else:
        x_br, y_br = "n_unique_positions", "degree_positions"
        xl_br, yl_br = f"Avg unique positions / match ({pos_col})", "Avg unique position combos / match"

    sub_br = style_df.dropna(subset=[x_br, y_br])
    fig_br = px.scatter(
        sub_br, x=x_br, y=y_br,
        color=fifa_col_s, color_continuous_scale="RdYlGn",
        hover_name="defender_name",
        hover_data={"team": True, "overall_position": True,
                    x_br: ":.2f", y_br: ":.2f"},
        title=f"Breadth ({br_mode}): unique positions vs unique position combinations",
        labels={x_br: xl_br, y_br: yl_br},
    )
    st.plotly_chart(fig_br, use_container_width=True)

    fig_box_br = px.box(
        sub_br, x="overall_position", y=x_br,
        color="overall_position", points="all", hover_name="defender_name",
        title=f"{x_br} by position",
        labels={x_br: xl_br, "overall_position": "Position"},
    )
    st.plotly_chart(fig_box_br, use_container_width=True)

    n_show_br = st.slider("Top / bottom N", 5, 20, 10, key="br_n")
    c1, c2 = st.columns(2)
    show_br = ["defender_name", "team", "overall_position", x_br, y_br, fifa_col_s]
    c1.markdown("**Widest coverage**")
    c1.dataframe(sub_br.nlargest(n_show_br, x_br)[show_br].reset_index(drop=True),
                 use_container_width=True)
    c2.markdown("**Narrowest coverage**")
    c2.dataframe(sub_br.nsmallest(n_show_br, x_br)[show_br].reset_index(drop=True),
                 use_container_width=True)

    # ── Section B: Stability ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Stability")
    if stab_col == "jaccard_stability":
        st.caption(
            "**Structural (Jaccard):** avg pairwise Jaccard similarity of **(passer_pos, receiver_pos)** sets "
            "across matches — treats all combos equally (binary presence/absence)."
        )
    else:
        st.caption(
            "**Weighted (Cosine):** avg pairwise cosine similarity of involvement vectors "
            "across matches — combos weighted by `valued_involvement`, so intensity differences matter."
        )

    sub_stab = style_df.dropna(subset=[stab_col])
    fig_stab = px.scatter(
        sub_stab, x=stab_col, y=fifa_col_s,
        color="overall_position",
        hover_name="defender_name",
        hover_data={"team": True, n_matches_col: True, stab_col: ":.3f"},
        title="Stability vs FIFA rating",
        labels={stab_col: stab_label},
    )
    valid_stab = sub_stab[[fifa_col_s, stab_col]].dropna()
    if len(valid_stab) >= 5:
        r_s, p_s = pearsonr(valid_stab[stab_col].values.astype(float),
                            valid_stab[fifa_col_s].values.astype(float))
        sl_s, ic_s, *_ = linregress(valid_stab[stab_col].values.astype(float),
                                     valid_stab[fifa_col_s].values.astype(float))
        x_line_s = np.linspace(valid_stab[stab_col].min(), valid_stab[stab_col].max(), 100)
        fig_stab.add_trace(go.Scatter(
            x=x_line_s, y=sl_s * x_line_s + ic_s,
            mode="lines", line=dict(color="black", dash="dash"),
            name=f"r={r_s:.3f}, p={p_s:.3f}",
        ))
    st.plotly_chart(fig_stab, use_container_width=True)

    fig_box_stab = px.box(
        sub_stab, x="overall_position", y=stab_col,
        color="overall_position", points="all", hover_name="defender_name",
        title="Stability by position",
        labels={stab_col: stab_label, "overall_position": "Position"},
    )
    st.plotly_chart(fig_box_stab, use_container_width=True)

    n_show_stab = st.slider("Top / bottom N", 5, 20, 10, key="stab_n")
    c1, c2 = st.columns(2)
    show_stab = ["defender_name", "team", "overall_position",
                 stab_col, n_matches_col, fifa_col_s]
    c1.markdown("**Most stable**")
    c1.dataframe(sub_stab.nlargest(n_show_stab, stab_col)[show_stab].reset_index(drop=True),
                 use_container_width=True)
    c2.markdown("**Least stable**")
    c2.dataframe(sub_stab.nsmallest(n_show_stab, stab_col)[show_stab].reset_index(drop=True),
                 use_container_width=True)

    # ── Section C: Breadth × Stability quadrant ───────────────────────────────
    st.divider()
    st.subheader("Breadth × Stability — Defensive Style Quadrants")
    st.markdown(
        "Combining both dimensions reveals four defensive archetypes:\n\n"
        "| | Low stability | High stability |\n"
        "|---|---|---|\n"
        "| **High breadth** | Flexible / pressing (covers many role-types, changes each match) "
        "| Structurally exposed (many role-types, always the same — position-driven) |\n"
        "| **Low breadth** | Situational / niche (few role-types, changes each match) "
        "| Role-based marker (few role-types, always the same → man-marking) |"
    )

    # X-axis options: attackers vs pass combinations, raw vs per90 vs entropy
    if br_mode == "Player-ID based":
        _x_opts = {
            "Unique attackers (raw)":      "n_unique_attackers",
            "Unique attackers (per 90)":   "n_unique_attackers_per90",
            "Unique pass combos (raw)":    "degree",
            "Unique pass combos (per 90)": "degree_per90",
        }
    elif br_weight == "Weighted (entropy of involvement)":
        _x_opts = {
            "Entropy of involvement (avg)":    "entropy_breadth",
            "Entropy of involvement (per 90)": "entropy_breadth_per90",
        }
    else:
        _x_opts = {
            "Unique positions (raw)":          "n_unique_positions",
            "Unique positions (per 90)":        "n_unique_positions_per90",
            "Unique position combos (raw)":     "degree_positions",
            "Unique position combos (per 90)":  "degree_positions_per90",
        }

    _y_opts = {
        "Structural (Jaccard)":   "jaccard_stability",
        "Weighted (Cosine sim.)": "cosine_stability",
    }

    c1, c2, c3, c4 = st.columns(4)
    x_q_label = c1.selectbox("X axis (breadth)", list(_x_opts.keys()), index=0, key="quad_x")
    x_q = _x_opts[x_q_label]
    y_q_label = c2.selectbox("Y axis (stability)", list(_y_opts.keys()),
                              index=0 if stab_col == "jaccard_stability" else 1,
                              key="quad_y")
    y_q = _y_opts[y_q_label]
    y_q_axis_label = "Stability (Jaccard)" if y_q == "jaccard_stability" else "Stability (Cosine)"
    size_q = c3.selectbox("Node size", ["uniform"] + FIFA_COLS, index=1, key="quad_size")
    _quad_pos_opts = ["All"] + sorted(style_df["overall_position"].dropna().unique())
    pos_filt_q = c4.selectbox(
        "Filter position (this plot)",
        _quad_pos_opts,
        index=0,
        key="quad_pos_filt",
        help="Further filter within this plot only.",
    )

    sub_q = style_df.dropna(subset=[x_q, y_q])
    if pos_filt_q != "All":
        sub_q = sub_q[sub_q["overall_position"] == pos_filt_q]

    # Node size: scale FIFA rating to [3, 18]; fill NaN with minimum
    sub_q = sub_q.copy()
    if size_q != "uniform" and size_q in sub_q.columns:
        _sz = sub_q[size_q].fillna(sub_q[size_q].min())
        sub_q["_node_size"] = 3 + 15 * (_sz - _sz.min()) / (_sz.max() - _sz.min() + 1e-9)
    else:
        sub_q["_node_size"] = 8

    # When a specific position is selected, color by sub-role (strip Left/Right prefix)
    if pos_filt_q != "All":
        def _strip_lr(p):
            if pd.isna(p):
                return p
            if p.startswith("Left "):
                return p[5:]
            if p.startswith("Right "):
                return p[6:]
            return p
        sub_q["_role"] = sub_q["most_common_position"].map(_strip_lr)
        _color_col   = "_role"
        _color_label = f"Role ({pos_filt_q})"
    else:
        _color_col   = "overall_position"
        _color_label = "Position"

    fig_q = px.scatter(
        sub_q, x=x_q, y=y_q,
        color=_color_col,
        size="_node_size", size_max=18,
        symbol=_color_col,
        hover_name="defender_name",
        hover_data={"team": True, "overall_position": True,
                    "most_common_position": True,
                    x_q: ":.2f", y_q: ":.3f",
                    "_node_size": False,
                    "_role" if pos_filt_q != "All" else "overall_position": False,
                    **({size_q: ":.1f"} if size_q != "uniform" else {})},
        title=f"Defensive style quadrants — {x_q_label} × {y_q_label}"
              + (f"  |  size = {size_q}" if size_q != "uniform" else ""),
        labels={x_q: x_q_label, y_q: y_q_axis_label, _color_col: _color_label},
    )
    if len(sub_q) >= 4:
        xm_q = sub_q[x_q].median()
        ym_q  = sub_q[y_q].median()
        fig_q.add_vline(x=xm_q, line_dash="dot", line_color="grey")
        fig_q.add_hline(y=ym_q, line_dash="dot", line_color="grey")
        for label, qx, qy in [
            ("Flexible / pressing",    0.97, 0.03),
            ("Structurally exposed",   0.97, 0.97),
            ("Situational / niche",    0.03, 0.03),
            ("Role-based marker",      0.03, 0.97),
        ]:
            fig_q.add_annotation(
                x=sub_q[x_q].quantile(qx),
                y=sub_q[y_q].quantile(qy),
                text=label, showarrow=False,
                font=dict(size=10, color="grey"),
            )
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Rating analysis per quadrant (position-specific medians) ─────────────
    st.divider()
    st.subheader("FIFA Rating by Quadrant — per position")
    st.caption(
        "Quadrants defined by **position-specific medians** of breadth and stability. "
        "Pairwise Mann-Whitney U test with Bonferroni correction (×6 comparisons per position)."
    )

    from itertools import combinations as _combinations
    from scipy.stats import kruskal as _kruskal, mannwhitneyu as _mwu

    _fifa_col_q = st.selectbox("FIFA rating", FIFA_COLS, index=1, key="quad_rating_col")

    quad_results = []
    for _pos, _g in sub_q.dropna(subset=[x_q, y_q, _fifa_col_q]).groupby("overall_position"):
        if len(_g) < 8:
            continue
        _med_x   = _g[x_q].median()
        _med_stab = _g[y_q].median()
        _g = _g.copy()
        _g["quadrant"] = np.where(
            (_g[x_q] >= _med_x) & (_g[y_q] >= _med_stab), "Broad+Stable",
            np.where(
                (_g[x_q] >= _med_x) & (_g[y_q] <  _med_stab), "Broad+Flexible",
                np.where(
                    (_g[x_q] <  _med_x) & (_g[y_q] >= _med_stab), "Narrow+Stable",
                    "Narrow+Flexible"
                )
            )
        )
        quad_results.append((_pos, _g))

    for _pos, _g in quad_results:
        st.markdown(f"**{_pos}** (n={len(_g)})")
        _summary = (_g.groupby("quadrant")[_fifa_col_q]
                     .agg(n="count", mean="mean", median="median", std="std")
                     .round(1).sort_values("mean", ascending=False))

        # Kruskal-Wallis
        _grps = [v[_fifa_col_q].values for _, v in _g.groupby("quadrant") if len(v) >= 3]
        _kw_p = np.nan
        if len(_grps) >= 2:
            _, _kw_p = _kruskal(*_grps)
        _summary["KW p"] = f"{_kw_p:.3f}" if not np.isnan(_kw_p) else "—"

        st.dataframe(
            _summary.style.background_gradient(cmap="RdYlGn", subset=["mean"], vmin=50, vmax=90),
            use_container_width=True,
        )

        # Pairwise Mann-Whitney U + Bonferroni
        _quads = [q for q in ["Narrow+Stable", "Narrow+Flexible",
                               "Broad+Stable",  "Broad+Flexible"]
                  if q in _g["quadrant"].values]
        _pairs = list(_combinations(_quads, 2))
        _n_comp = len(_pairs)
        _pw_rows = []
        for q1, q2 in _pairs:
            a = _g[_g["quadrant"] == q1][_fifa_col_q].dropna().values
            b = _g[_g["quadrant"] == q2][_fifa_col_q].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue
            _, _p = _mwu(a, b, alternative="two-sided")
            _p_adj = min(_p * _n_comp, 1.0)
            _pw_rows.append({
                "Comparison":    f"{q1}  vs  {q2}",
                "p (raw)":       round(_p, 4),
                "p (Bonferroni)": round(_p_adj, 4),
                "significant":   "✓" if _p_adj < 0.05 else "",
            })
        if _pw_rows:
            with st.expander(f"Pairwise comparisons — {_pos}"):
                st.dataframe(pd.DataFrame(_pw_rows), use_container_width=True)

    st.markdown("**Correlate stability with activity metrics**")
    stab_corr_col = st.selectbox(
        "Metric", ACTIVE_METRICS + [x_br, "gini"], key="stab_corr",
    )
    sub_sc = style_df[[stab_corr_col, stab_col, "defender_name", "team"]].dropna()
    if len(sub_sc) >= 5:
        r_sc, p_sc = pearsonr(sub_sc[stab_corr_col].values.astype(float),
                              sub_sc[stab_col].values.astype(float))
        sl_sc, ic_sc, *_ = linregress(sub_sc[stab_corr_col].values.astype(float),
                                      sub_sc[stab_col].values.astype(float))
        fig_sc = px.scatter(
            sub_sc, x=stab_corr_col, y=stab_col,
            hover_name="defender_name", hover_data={"team": True},
            title=f"{stab_corr_col}  vs  stability ({stab_type})   (r={r_sc:.3f}, p={p_sc:.3f})",
            labels={stab_col: stab_label},
        )
        x_sc_line = np.linspace(sub_sc[stab_corr_col].min(), sub_sc[stab_corr_col].max(), 100)
        fig_sc.add_trace(go.Scatter(
            x=x_sc_line, y=sl_sc * x_sc_line + ic_sc,
            mode="lines", line=dict(color="black", dash="dash"), name="regression",
        ))
        st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 2: Concentration ───────────────────────────────────────────────────────
with t_gini:
    st.markdown(
        "**Gini of involvement** across attacking pass combinations. "
        "Low (≈0) = generalist (spread evenly across many passes). "
        "High (≈1) = specialist (concentrated on a few specific combos)."
    )
    sub2 = full_df.dropna(subset=["gini"])

    c1, c2 = st.columns(2)
    fifa_col2 = c1.selectbox("FIFA rating (colour)", FIFA_COLS, index=1, key="gini_fifa")
    pos_filt2 = c2.multiselect("Filter by position",
                                sorted(sub2["overall_position"].dropna().unique()),
                                default=[], key="gini_pos")
    if pos_filt2:
        sub2 = sub2[sub2["overall_position"].isin(pos_filt2)]

    fig = px.scatter(
        sub2, x="n_unique_attackers", y="gini",
        color=fifa_col2, color_continuous_scale="RdYlGn",
        hover_name="defender_name",
        hover_data={"team": True, "overall_position": True, "gini": ":.3f"},
        title="Specialist vs Generalist",
        labels={"n_unique_attackers": "Avg unique attackers / match", "gini": "Gini (concentration)"},
    )
    xm, ym = sub2["n_unique_attackers"].median(), sub2["gini"].median()
    fig.add_vline(x=xm, line_dash="dot", line_color="grey")
    fig.add_hline(y=ym, line_dash="dot", line_color="grey")
    for label, qx, qy in [
        ("Broad + specialist", 0.97, 0.97), ("Broad + generalist", 0.97, 0.03),
        ("Narrow + specialist", 0.03, 0.97), ("Narrow + generalist", 0.03, 0.03),
    ]:
        fig.add_annotation(
            x=sub2["n_unique_attackers"].quantile(qx), y=sub2["gini"].quantile(qy),
            text=label, showarrow=False, font=dict(size=10, color="grey"),
        )
    st.plotly_chart(fig, use_container_width=True)

    fig3 = px.box(sub2, x="overall_position", y="gini", color="overall_position",
                  points="all", hover_name="defender_name",
                  title="Gini by position",
                  labels={"gini": "Gini", "overall_position": "Position"})
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 3: Correlation vs FIFA ─────────────────────────────────────────────────
with t_corr:
    st.subheader("Basic 6 metrics — defending & awareness ratings")
    st.caption(
        f"Normalization: **{norm_method}** (change in sidebar)  ·  "
        "raw and position-partialled (residualised on overall_position) shown side by side  ·  "
        "* p<0.05  ** p<0.01  *** p<0.001"
    )

    c1, c2 = st.columns(2)
    _corr_method_b6 = c1.radio("Method", ["Pearson", "Spearman"], horizontal=True, key="b6_method")
    _b6_pos_opts = ["All"] + sorted(full_df["overall_position"].dropna().unique())
    _b6_pos_filt = c2.selectbox("Filter by position", _b6_pos_opts, index=0, key="b6_pos")

    # 6 basic metrics of the currently selected normalization family (drop passes_defended)
    _basic6 = [m for m in ACTIVE_METRICS if not m.startswith("passes_defended")]

    _b6_df      = full_df if _b6_pos_filt == "All" else full_df[full_df["overall_position"] == _b6_pos_filt]
    _b6_partial = _b6_pos_filt == "All"
    if not _b6_partial:
        st.caption("ℹ️ Partial is disabled when a specific position is selected (partial columns show —).")

    basic6_corr_tbl(_b6_df, metrics=_basic6, corr_method=_corr_method_b6, partial=_b6_partial)


# ── Tab 4: ICC ─────────────────────────────────────────────────────────────────
with t_icc:
    st.markdown(
        "**ICC(1,1)** — intraclass correlation coefficient on match-level data. "
        "Measures what fraction of total variance in a metric is due to stable player identity "
        "vs match-to-match noise. "
        "> 0.5 = stable trait · 0.25–0.5 = moderate · < 0.25 = context-driven.\n\n"
        f"Currently using **{norm_method}** normalization (change in sidebar)."
    )

    df_icc = enriched.copy()
    if starters_only:
        df_icc = df_icc[df_icc["starter"] == 1]
    pm = df_icc.groupby("defender_name")["mins_played"].sum()
    pc = df_icc.groupby("defender_name")["match_id"].nunique()
    keep = pm[(pm >= min_minutes) & (pc >= min_matches)].index
    df_icc = df_icc[df_icc["defender_name"].isin(keep)]

    icc_df = compute_icc(df_icc, ACTIVE_METRICS)
    if icc_df.empty:
        st.warning("Not enough data — loosen filters.")
    else:
        _icc6 = icc_df[icc_df["metric"].isin(BASIC_6_PER90)].reset_index(drop=True)
        if not _icc6.empty:
            st.subheader("Basic 6 per-90 metrics")
            st.dataframe(
                _icc6.style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
                use_container_width=True,
            )
            st.divider()
        st.subheader("All active metrics")
        st.dataframe(
            icc_df.style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
            use_container_width=True,
        )
        fig_icc = px.bar(
            icc_df.sort_values("ICC"),
            x="ICC", y="metric", orientation="h",
            color="ICC", color_continuous_scale="RdYlGn", range_color=[0, 1],
            title="ICC(1,1) by metric — how stable is this metric as a player trait?",
        )
        fig_icc.add_vline(x=0.5,  line_dash="dot", line_color="green",
                          annotation_text="stable (0.5)")
        fig_icc.add_vline(x=0.25, line_dash="dot", line_color="orange",
                          annotation_text="moderate (0.25)")
        st.plotly_chart(fig_icc, use_container_width=True)

        st.markdown(
            "**Interpretation:** High ICC means the metric reliably differentiates players across "
            "matches — it's a stable trait. Low ICC means the value fluctuates with match context "
            "(opponent, formation, game state). Correlating a low-ICC metric with FIFA ratings "
            "is less meaningful as a player-quality measure."
        )


# ── Tab 5: Network Depth ──────────────────────────────────────────────────────
with t_depth:
    st.markdown("""
**Centrality of the attackers defended** — the attacking network is built from all passes in the dataset.
Centrality is computed once for all attacking players.
Each defender's score = weighted-avg centrality of the attacking nodes they defended (weighted by `valued_involvement`).

| Metric | Interpretation |
|---|---|
| **Eigenvector** | Defends against *hub* attackers — players central to their team's passing flow. |
| **Betweenness** | Defends against *bridge* attackers — players who link different zones of the attack. |
    """)

    sub4 = full_df.dropna(subset=["eig_centrality"])
    c1, c2 = st.columns(2)
    fifa_col4   = c1.selectbox("FIFA rating", FIFA_COLS, index=1, key="net_fifa")
    cent_metric = c2.radio("Centrality metric",
                           ["eig_centrality", "btw_centrality"],
                           format_func=lambda x: {"eig_centrality": "Eigenvector",
                                                   "btw_centrality": "Betweenness"}[x],
                           horizontal=True, key="cent_m")

    fig = px.scatter(
        sub4, x=fifa_col4, y=cent_metric,
        color="overall_position", symbol="overall_position",
        hover_name="defender_name",
        hover_data={"team": True, cent_metric: ":.4f"},
        title=f"{cent_metric} vs {fifa_col4}",
    )
    st.plotly_chart(fig, use_container_width=True)

    n_show4 = st.slider("Top / bottom N", 5, 20, 10, key="net_n")
    c1, c2  = st.columns(2)
    show4   = ["defender_name", "team", "overall_position", cent_metric, fifa_col4]
    c1.markdown("**Highest** — defends most central attackers")
    c1.dataframe(sub4.nlargest(n_show4, cent_metric)[show4].reset_index(drop=True),
                 use_container_width=True)
    c2.markdown("**Lowest** — defends most peripheral attackers")
    c2.dataframe(sub4.nsmallest(n_show4, cent_metric)[show4].reset_index(drop=True),
                 use_container_width=True)


# ── Tab 6: Player Map ─────────────────────────────────────────────────────────
with t_map:
    st.markdown(
        "Inspect a single player's full defensive map. "
        "Heatmap rows = passers, columns = receivers; colour = total `valued_involvement`."
    )
    edges_all  = load_edges()
    player_sel = st.selectbox("Select player", sorted(players), key="map_player")
    df_p       = edges_all[edges_all["defender_name"] == player_sel].copy()

    if df_p.empty:
        st.warning("No edge data for this player.")
    else:
        pass_inv = (df_p.groupby(["passer_name", "receiver_name"])["valued_involvement"]
                       .sum().reset_index())
        pass_inv = pass_inv[pass_inv["valued_involvement"] > 0]

        top_n     = st.slider("Top N edges (by involvement)", 5, 50, 20, key="map_n")
        top_edges = pass_inv.nlargest(top_n, "valued_involvement")

        pivot    = top_edges.pivot(index="passer_name", columns="receiver_name",
                                   values="valued_involvement").fillna(0)
        fig_heat = px.imshow(
            pivot, color_continuous_scale="Blues",
            title=f"{player_sel} — top {top_n} attacking pass combinations defended",
            labels={"x": "Receiver", "y": "Passer", "color": "Valued involvement"},
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        passer_inv   = pass_inv.groupby("passer_name")["valued_involvement"].sum()
        receiver_inv = pass_inv.groupby("receiver_name")["valued_involvement"].sum()
        node_inv     = (pd.concat([passer_inv, receiver_inv])
                        .groupby(level=0).sum()
                        .reset_index()
                        .rename(columns={0: "inv", "index": "player"}))
        node_inv.columns = ["player", "inv"]

        c1, c2 = st.columns(2)
        c1.markdown("**Top attacking pass combinations**")
        c1.dataframe(top_edges.reset_index(drop=True).round(4), use_container_width=True)
        c2.markdown("**Most defended attacking players**")
        c2.dataframe(node_inv.sort_values("inv", ascending=False).head(15)
                              .reset_index(drop=True).round(4), use_container_width=True)


# ── Tab 8: Archetypes ─────────────────────────────────────────────────────────
with t_arch:
    st.markdown(
        "**K-means clustering** on activity + network metrics, projected onto 2D via PCA. "
        f"Activity features use **{norm_method}** normalization."
    )

    ARCH_COLS = ACTIVE_METRICS + ["gini", "degree_per90", "n_unique_attackers_per90"]
    valid_arch = full_df.dropna(subset=ARCH_COLS).copy()

    c1, c2 = st.columns(2)
    k        = c1.slider("Clusters (k)", 2, 6, 4, key="arch_k")
    sel_cols = c2.multiselect("Features", ARCH_COLS, default=ARCH_COLS, key="arch_m")

    if len(sel_cols) < 2:
        st.warning("Select at least 2 features.")
    else:
        valid = valid_arch.dropna(subset=sel_cols).copy()
        Xs    = StandardScaler().fit_transform(valid[sel_cols])
        valid["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs).astype(str)

        pca = PCA(n_components=2)
        xy  = pca.fit_transform(Xs)
        valid["PC1"], valid["PC2"] = xy[:, 0], xy[:, 1]

        fig = px.scatter(
            valid, x="PC1", y="PC2", color="cluster",
            symbol="overall_position",
            hover_name="defender_name",
            hover_data={"team": True, "cluster": True, "defending_rating": ":.0f"},
            title=f"Defensive archetypes — k={k}  (PCA projection)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"PC1 {pca.explained_variance_ratio_[0]:.1%} · "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%} of variance explained"
        )

        with st.expander("PCA loadings — which features drive each component"):
            loadings = pd.DataFrame(
                pca.components_.T, index=sel_cols, columns=["PC1", "PC2"]
            ).round(3)
            st.dataframe(
                loadings.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1),
                use_container_width=True,
            )

        fig_box = px.box(
            valid.dropna(subset=["defending_rating"]),
            x="cluster", y="defending_rating", color="cluster", points="all",
            hover_name="defender_name",
            title="FIFA defending rating by archetype cluster",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Cluster profiles — mean of each feature**")
        st.dataframe(
            valid.groupby("cluster")[sel_cols].mean().round(3)
                 .style.background_gradient(cmap="YlGn", axis=0),
            use_container_width=True,
        )

        for cl, grp in valid.groupby("cluster"):
            with st.expander(f"Cluster {cl}  ({len(grp)} players)"):
                st.dataframe(
                    grp[["defender_name", "team", "overall_position", "defending_rating"] + sel_cols]
                      .sort_values("defending_rating", ascending=False)
                      .reset_index(drop=True)
                      .round(3),
                    use_container_width=True,
                )


# ── Tab 9: Over / Underrated ──────────────────────────────────────────────────
with t_ou:
    from scipy.stats import percentileofscore as _pct_score

    st.markdown(
        "**Percentile method, within position.** For each position group, every player is "
        "ranked on the chosen network metric and on the FIFA rating *separately*.\n\n"
        "Metric direction is handled automatically so a **high quality percentile always "
        "means good defending**: `involvement` and `contribution` are higher-is-better; "
        "`fault` is lower-is-better (inverted).\n\n"
        "**gap = FIFA percentile − quality percentile**  ·  "
        "negative → **underrated** (performance above recognition, green)  ·  "
        "positive → **overrated** (recognition above performance, red)."
    )

    c1, c2, c3 = st.columns(3)
    fifa_col_ou = c1.selectbox("FIFA rating", FIFA_COLS, index=1, key="ou_fifa")
    _basic6_ou  = [m for m in ACTIVE_METRICS if not m.startswith("passes_defended")]
    metric_col_ou = c2.selectbox("Network metric (6 basic)", _basic6_ou, index=0, key="ou_metric")
    _ou_pos_opts = ["All"] + sorted(full_df["overall_position"].dropna().unique())
    pos_filt_ou  = c3.selectbox("Filter by position", _ou_pos_opts, index=0, key="ou_pos")

    lower_better  = "fault" in metric_col_ou
    direction_lbl = "lower is better → inverted" if lower_better else "higher is better"
    st.caption(f"`{metric_col_ou}` — {direction_lbl}  ·  normalization: **{norm_method}**  ·  "
               "percentiles computed within each position group.")

    base = full_df[[metric_col_ou, fifa_col_ou, "defender_name", "team",
                    "overall_position"]].dropna()
    if pos_filt_ou != "All":
        base = base[base["overall_position"] == pos_filt_ou]

    if len(base) < 5:
        st.warning("Too few players — loosen filters.")
    else:
        rows_p = []
        for pos, grp in base.groupby("overall_position"):
            if len(grp) < 2:
                continue
            mvals  = grp[metric_col_ou].values.astype(float)
            signed = -mvals if lower_better else mvals   # direction-corrected
            fvals  = grp[fifa_col_ou].values.astype(float)
            for i, (_, row) in enumerate(grp.iterrows()):
                q_pct = _pct_score(signed, signed[i], kind="rank")
                f_pct = _pct_score(fvals,  fvals[i],  kind="rank")
                rows_p.append({
                    "defender_name":    row["defender_name"],
                    "team":             row["team"],
                    "overall_position": pos,
                    fifa_col_ou:        row[fifa_col_ou],
                    metric_col_ou:      row[metric_col_ou],
                    "quality_pct":      round(q_pct, 1),
                    "fifa_pct":         round(f_pct, 1),
                    "gap":              round(f_pct - q_pct, 1),
                })

        pct_df = pd.DataFrame(rows_p)

        fig_pct = px.scatter(
            pct_df, x="quality_pct", y="fifa_pct",
            color="gap", color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0,
            hover_name="defender_name",
            hover_data={"team": True, "overall_position": True,
                        metric_col_ou: ":.3f",
                        "quality_pct": ":.1f", "fifa_pct": ":.1f", "gap": ":.1f"},
            title=f"Within-position percentile: {metric_col_ou}  vs  {fifa_col_ou}",
            labels={"quality_pct": f"Defensive-quality percentile ({metric_col_ou})",
                    "fifa_pct":    f"FIFA percentile ({fifa_col_ou})"},
        )
        fig_pct.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines", line=dict(color="black", dash="dash"), name="perfect alignment",
        ))
        st.plotly_chart(fig_pct, use_container_width=True)
        st.caption(
            "Points **above** the diagonal: FIFA percentile > quality percentile → overrated (red).  "
            "Points **below**: quality percentile > FIFA percentile → underrated (green)."
        )

        n_show_pct = st.slider("Top / bottom N", 3, 20, 5, key="ou_pct_n")
        cols_pct   = ["defender_name", "team", "overall_position",
                      fifa_col_ou, metric_col_ou, "quality_pct", "fifa_pct", "gap"]
        c1, c2 = st.columns(2)
        c1.markdown("**Most underrated** (gap most negative)")
        c1.dataframe(pct_df.nsmallest(n_show_pct, "gap")[cols_pct].reset_index(drop=True),
                     use_container_width=True)
        c2.markdown("**Most overrated** (gap most positive)")
        c2.dataframe(pct_df.nlargest(n_show_pct, "gap")[cols_pct].reset_index(drop=True),
                     use_container_width=True)

# ── Tab 10: Volume × Quality ───────────────────────────────────────────────────
with t_vq:
    st.markdown(
        "Single metrics conflate **how much** a defender is involved with **how well**. "
        "Here the two are split:\n\n"
        "- **x = involvement (volume)** — how much defensive action the player is drawn into.\n"
        "- **y = contribution quality** — of that involvement, how much *helped* vs was *at fault*.\n\n"
        "Because `involvement = contribution + fault`, the quality axis is a ratio that "
        "**cancels the sidebar normalization** (per-90 / per-pass all give the same y), so it is a "
        "pure quality signal independent of volume."
    )

    def _find_metric(prefix: str) -> str | None:
        return next((m for m in ACTIVE_METRICS if m.startswith(prefix)), None)

    inv_col = _find_metric("valued_involvement")
    con_col = _find_metric("valued_contribution")
    flt_col = _find_metric("valued_fault")

    if not all([inv_col, con_col, flt_col]):
        st.warning("Selected normalization is missing involvement/contribution/fault columns.")
    else:
        c1, c2, c3 = st.columns(3)
        y_mode = c1.radio(
            "Quality axis (y)",
            ["Ratio  con/fault", "Contribution share  con/(con+fault)", "Net  con − fault"],
            key="vq_ymode",
        )
        fifa_col_vq = c2.selectbox("Colour (FIFA rating)", FIFA_COLS, index=1, key="vq_fifa")
        _vq_pos_opts = ["All"] + sorted(full_df["overall_position"].dropna().unique())
        pos_filt_vq  = c3.selectbox("Filter by position", _vq_pos_opts, index=0, key="vq_pos")

        vq = full_df[[inv_col, con_col, flt_col, fifa_col_vq,
                      "defender_name", "team", "overall_position"]].dropna(subset=[inv_col, con_col, flt_col]).copy()
        if pos_filt_vq != "All":
            vq = vq[vq["overall_position"] == pos_filt_vq]

        denom = vq[con_col] + vq[flt_col]
        if y_mode.startswith("Contribution share"):
            vq["quality"] = np.where(denom > 0, vq[con_col] / denom, np.nan)
            y_label, log_y = "Contribution share  con / (con + fault)", False
        elif y_mode.startswith("Net"):
            vq["quality"] = vq[con_col] - vq[flt_col]
            y_label, log_y = "Net  contribution − fault", False
        else:
            vq["quality"] = vq[con_col] / vq[flt_col]
            y_label, log_y = "Ratio  contribution / fault", False

        # drop edge cases (fault == 0 → inf, any NaN)
        n_before = len(vq)
        vq = vq[np.isfinite(vq["quality"].values)]
        n_dropped = n_before - len(vq)
        if n_dropped:
            st.caption(f"Dropped {n_dropped} player(s) with undefined quality "
                       "(e.g. fault = 0 → ratio is infinite).")
        if len(vq) < 5:
            st.warning("Too few players — loosen filters.")
        else:
            fig_vq = px.scatter(
                vq, x=inv_col, y="quality",
                color=fifa_col_vq, color_continuous_scale="RdYlGn",
                symbol="overall_position" if pos_filt_vq == "All" else None,
                hover_name="defender_name",
                log_y=log_y,
                hover_data={"team": True, "overall_position": True,
                            inv_col: ":.3f", con_col: ":.3f", flt_col: ":.3f",
                            "quality": ":.3f"},
                title=f"Volume × Quality   (x = {inv_col})",
                labels={inv_col: "Involvement (volume)", "quality": y_label},
            )
            xm = vq[inv_col].median()
            ym = vq["quality"].median()
            fig_vq.add_vline(x=xm, line_dash="dot", line_color="grey")
            fig_vq.add_hline(y=ym, line_dash="dot", line_color="grey")
            for label, qx, qy in [
                ("Busy & effective",   0.97, 0.97),
                ("Quiet & effective",  0.03, 0.97),
                ("Busy & error-prone", 0.97, 0.03),
                ("Quiet & error-prone",0.03, 0.03),
            ]:
                fig_vq.add_annotation(
                    x=vq[inv_col].quantile(qx), y=vq["quality"].quantile(qy),
                    text=label, showarrow=False, font=dict(size=10, color="grey"),
                )
            st.plotly_chart(fig_vq, use_container_width=True)
            st.caption(
                "Dotted lines = medians.  Top-right = high volume **and** high quality.  "
                "The y axis isolates quality from volume; colour shows where FIFA agrees."
            )

            # Does the quality axis correlate with FIFA (within position if filtered)?
            v = vq[["quality", inv_col, fifa_col_vq]].dropna()
            if len(v) >= 5:
                r_vq, p_vq = pearsonr(v["quality"].values.astype(float),
                                      v[fifa_col_vq].values.astype(float))
                r_iv, _    = pearsonr(v[inv_col].values.astype(float),
                                      v[fifa_col_vq].values.astype(float))
                st.caption(
                    f"Correlation with **{fifa_col_vq}**:  quality axis r = {r_vq:.3f} (p={p_vq:.3f})  ·  "
                    f"volume axis r = {r_iv:.3f}  — compare which carries more rating signal."
                )

            n_show_vq = st.slider("Top / bottom N (by quality)", 3, 20, 5, key="vq_n")
            cols_vq = ["defender_name", "team", "overall_position",
                       inv_col, con_col, flt_col, "quality", fifa_col_vq]
            c1, c2 = st.columns(2)
            c1.markdown("**Highest quality**")
            c1.dataframe(vq.nlargest(n_show_vq, "quality")[cols_vq].reset_index(drop=True).round(3),
                         use_container_width=True)
            c2.markdown("**Lowest quality**")
            c2.dataframe(vq.nsmallest(n_show_vq, "quality")[cols_vq].reset_index(drop=True).round(3),
                         use_container_width=True)

            # ── Quality vs FIFA rating ────────────────────────────────────────
            st.divider()
            st.markdown(f"**Quality vs {fifa_col_vq}**")
            qr = vq[["quality", fifa_col_vq, "defender_name", "team", "overall_position"]].dropna()
            if len(qr) < 5:
                st.caption("Too few players with a FIFA rating to plot.")
            else:
                xq = qr["quality"].values.astype(float)
                yq = qr[fifa_col_vq].values.astype(float)
                r_qr, p_qr       = pearsonr(xq, yq)
                sl_qr, ic_qr, *_ = linregress(xq, yq)
                fig_qr = px.scatter(
                    qr, x="quality", y=fifa_col_vq,
                    color="overall_position" if pos_filt_vq == "All" else None,
                    symbol="overall_position" if pos_filt_vq == "All" else None,
                    hover_name="defender_name",
                    hover_data={"team": True, "overall_position": True, "quality": ":.3f"},
                    title=f"Quality ({y_label})  vs  {fifa_col_vq}   (r = {r_qr:.3f}, p = {p_qr:.3f})",
                    labels={"quality": y_label},
                )
                _xl = np.linspace(xq.min(), xq.max(), 100)
                fig_qr.add_trace(go.Scatter(
                    x=_xl, y=sl_qr * _xl + ic_qr,
                    mode="lines", line=dict(color="black", dash="dash"), name="regression",
                ))
                st.plotly_chart(fig_qr, use_container_width=True)
                st.caption(
                    "Quality is strongly position-structured (GK high → forward low), so the "
                    "whole-pool line mostly reflects position. Use the **Filter by position** "
                    "selector above to read the relationship within a single role."
                )
