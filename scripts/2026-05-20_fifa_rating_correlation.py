"""
Player-level defensive network metrics vs. FIFA ratings.
Correlations (raw + partial) and ICC(1,1) per metric.

Per-90 metrics are aggregated per player (minutes-weighted mean across matches).
FIFA targets: overall rating, defending rating, def_awareness_rating.
"""
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr
from rapidfuzz import process, fuzz

PER90_FILE     = "scripts/2026-05-20_player_level_per90.csv"
FIFA_FILE      = "scripts/fifa_ratings.csv"
NICKNAME_FILE  = "scripts/nickname_map.csv"

METRICS = [
    "raw_involvement_per90",
    "valued_involvement_per90",
    "raw_fault_per90",
    "valued_fault_per90",
    "raw_contribution_per90",
    "valued_contribution_per90",
    "passes_defended_per90",
]
TARGETS = ["overall rating", "defending rating", "def_awareness_rating"]


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading per-90 data…")
def load_per90():
    return pd.read_csv(PER90_FILE)


@st.cache_data(show_spinner="Loading FIFA ratings…")
def load_fifa():
    df = pd.read_csv(FIFA_FILE)
    df = df[df["comp"].str.contains("World Cup", na=False)].copy()
    df = df.sort_values("overall rating", ascending=False).drop_duplicates("name")
    return df[["name"] + TARGETS]


@st.cache_data(show_spinner="Loading nickname map…")
def load_nickname_map() -> dict:
    """Returns {player_nickname (per90 name): Player (FIFA full name)}."""
    df = pd.read_csv(NICKNAME_FILE).drop_duplicates(subset=["player_nickname"])
    return dict(zip(df["player_nickname"], df["Player"]))


_LIGATURES = str.maketrans({
    "æ": "ae", "Æ": "Ae", "ø": "o", "Ø": "O",
    "å": "a",  "Å": "A",  "ß": "ss", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "Oe",
})

def _normalize(name: str) -> str:
    """Strip accents, expand ligatures, replace hyphens with spaces."""
    name = name.translate(_LIGATURES)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.replace("-", " ")


@st.cache_data(show_spinner="Building name map…")
def build_name_map(per90_names: tuple, fifa_names: tuple, threshold: int = 88) -> dict:
    """
    Returns {per90_name: fifa_name}.

    Matching priority:
      1. Exact match
      2. Normalized exact match  (strips accents/ligatures, hyphen → space)
      3. Nickname map → exact match against FIFA → normalized exact match
      4. Fuzzy fallback on normalized name (last resort)
    """
    fifa_set  = set(fifa_names)
    nick_map  = load_nickname_map()

    # normalized FIFA dict: norm_name → original FIFA name (first = highest rated)
    norm_fifa: dict[str, str] = {}
    for n in fifa_names:
        key = _normalize(n)
        if key not in norm_fifa:
            norm_fifa[key] = n
    norm_fifa_list = list(norm_fifa.keys())

    name_map: dict[str, str] = {}
    for name in per90_names:

        # 1. exact match
        if name in fifa_set:
            name_map[name] = name
            continue

        # 2. normalized exact match (handles accents, ligatures, hyphens)
        norm_name = _normalize(name)
        if norm_name in norm_fifa:
            name_map[name] = norm_fifa[norm_name]
            continue

        # 3. nickname map
        full_name = nick_map.get(name)
        if full_name:
            # 3a. exact match of full name against FIFA
            if full_name in fifa_set:
                name_map[name] = full_name
                continue
            # 3b. normalized exact match (handles Kléper vs Képler)
            norm_full = _normalize(full_name)
            if norm_full in norm_fifa:
                name_map[name] = norm_fifa[norm_full]
                continue

        # 4. fuzzy fallback on normalized per90 name
        result = process.extractOne(norm_name, norm_fifa_list,
                                    scorer=fuzz.WRatio, score_cutoff=threshold)
        if result:
            name_map[name] = norm_fifa[result[0]]

    return name_map


def aggregate_player(per90: pd.DataFrame, starters_only: bool,
                     min_minutes: int, min_matches: int) -> pd.DataFrame:
    df = per90.copy()
    if starters_only:
        df = df[df["starter"] == 1]

    # minutes-weighted mean of per-90 metrics per player
    player_mins = df.groupby("defender_name")["mins_played"].sum().rename("total_mins")
    player_matches = df.groupby("defender_name")["match_id"].nunique().rename("n_matches")

    def wmean(group):
        w = group["mins_played"].values
        return pd.Series(
            {m: np.average(group[m].values, weights=w) for m in METRICS}
        )

    agg = df.groupby("defender_name").apply(wmean, include_groups=False).reset_index()
    agg = agg.merge(player_mins, on="defender_name")
    agg = agg.merge(player_matches, on="defender_name")

    agg = agg[
        (agg["total_mins"] >= min_minutes) &
        (agg["n_matches"] >= min_matches)
    ]
    return agg


def build_merged(per90, starters_only, min_minutes, min_matches):
    agg  = aggregate_player(per90, starters_only, min_minutes, min_matches)
    fifa = load_fifa()

    name_map = build_name_map(
        tuple(agg["defender_name"].unique()),
        tuple(fifa["name"].unique()),
    )

    agg["fifa_name"] = agg["defender_name"].map(name_map)
    merged = agg.dropna(subset=["fifa_name"]).merge(
        fifa.rename(columns={"name": "fifa_name"}),
        on="fifa_name", how="inner",
    )
    return agg, merged


# ── statistics ────────────────────────────────────────────────────────────────

def _resid(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    mask = ~(np.isnan(y) | np.isnan(z))
    r = np.full(len(y), np.nan)
    r[mask] = y[mask] - np.polyval(np.polyfit(z[mask], y[mask], 1), z[mask])
    return r


def corr_table(merged: pd.DataFrame, method: str, partial_var: str | None) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        row = {"metric": m}
        for t in TARGETS:
            sub = merged[[m, t]].copy()
            if partial_var:
                sub = sub.join(merged[[partial_var]])
            sub = sub.dropna()
            if len(sub) < 5:
                row[f"r_{t}"] = np.nan
                row[f"p_{t}"] = np.nan
                continue
            x, y = sub[m].values, sub[t].values
            if partial_var:
                x = _resid(x, sub[partial_var].values)
                y = _resid(y, sub[partial_var].values)
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
            if method == "Pearson":
                r, p = pearsonr(x, y)
            else:
                r, p = spearmanr(x, y)
            row[f"r_{t}"] = round(r, 3)
            row[f"p_{t}"] = round(p, 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def render_corr_table(result: pd.DataFrame):
    r_cols = [f"r_{t}" for t in TARGETS]
    p_cols = [f"p_{t}" for t in TARGETS]
    r_df = result[r_cols].rename(columns=lambda c: c[2:])
    p_df = result[p_cols].rename(columns=lambda c: c[2:])
    display = r_df.applymap(lambda v: f"{v:.3f}" if not np.isnan(v) else "—") + \
              "  (" + p_df.applymap(lambda v: f"{v:.3f}" if not np.isnan(v) else "—") + ")"
    st.dataframe(
        display.style.background_gradient(
            cmap="RdYlGn", gmap=r_df.values, axis=None, vmin=-1, vmax=1
        ),
        use_container_width=True,
    )


def icc_table(per90: pd.DataFrame, starters_only: bool, min_minutes: int, min_matches: int):
    df = per90.copy()
    if starters_only:
        df = df[df["starter"] == 1]

    # apply thresholds at player level
    player_mins = df.groupby("defender_name")["mins_played"].sum()
    player_matches = df.groupby("defender_name")["match_id"].nunique()
    keep = player_mins[
        (player_mins >= min_minutes) &
        (player_matches >= min_matches)
    ].index
    df = df[df["defender_name"].isin(keep)]

    rows = []
    for m in METRICS:
        s = df[["defender_name", m]].dropna()
        if s["defender_name"].nunique() < 2:
            continue
        g = s.groupby("defender_name")[m]
        nt, ng = len(s), g.ngroups
        sz  = g.count()
        mn  = g.mean()
        msb = (sz * (mn - s[m].mean()) ** 2).sum() / (ng - 1)
        msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0  = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        rows.append({
            "metric":    m,
            "ICC":       round(icc, 3),
            "n_players": ng,
            "n_obs":     nt,
            "stability": "stable" if icc > 0.5 else ("moderate" if icc > 0.25 else "low"),
        })

    st.dataframe(
        pd.DataFrame(rows).style.background_gradient(
            cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1
        ),
        use_container_width=True,
    )


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Network Metrics vs FIFA Ratings", layout="wide")
st.title("Defensive Network Metrics vs FIFA Ratings")
st.caption(
    "Per-90 metrics aggregated per player (minutes-weighted mean across matches). "
    "Correlated against FIFA overall / defending / defensive awareness ratings. "
    "ICC(1,1) measures player consistency across matches."
)

per90 = load_per90()

with st.sidebar:
    st.header("Filters")
    starters_only = st.checkbox("Starters only", value=True)
    max_mins = int(per90.groupby("defender_name")["mins_played"].sum().max())
    max_matches = int(per90.groupby("defender_name")["match_id"].nunique().max())
    min_minutes = st.slider(
        "Min total minutes played",
        min_value=0, max_value=max_mins, value=90, step=10,
    )
    min_matches = st.slider(
        "Min matches played",
        min_value=1, max_value=max_matches, value=2, step=1,
    )

agg, merged = build_merged(per90, starters_only, min_minutes, min_matches)

st.caption(
    f"**After threshold:** {len(agg)} players pass filters · "
    f"{len(merged)} players matched with FIFA ratings"
)

unmatched = agg[agg["fifa_name"].isna()]["defender_name"].tolist()
if unmatched:
    with st.expander(f"Players not matched to FIFA ({len(unmatched)}) — genuinely absent from FIFA game"):
        st.write(sorted(unmatched))

tab_corr, tab_icc = st.tabs(["Correlation (Network ↔ FIFA)", "ICC (player consistency)"])

with tab_corr:
    c1, c2 = st.columns(2)
    method = c1.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)
    partial = c2.checkbox("Partial (control total_mins)", value=False)
    partial_var = "total_mins" if partial else None

    if len(merged) < 5:
        st.warning("Too few matched players — loosen the thresholds.")
    else:
        result = corr_table(merged, method, partial_var)
        st.caption("Values: r (p-value). Green = positive correlation.")
        render_corr_table(result)

with tab_icc:
    st.markdown(
        "**ICC(1,1):** >0.5 stable player trait · 0.25–0.5 moderate · <0.25 match/position-driven"
    )
    icc_table(per90, starters_only, min_minutes, min_matches)
