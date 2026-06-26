"""
Player-level defensive network correlation & ICC.
Uses Method 2 edges. Each observation = one defending player in one match.
Total strength = sum of all edge weights in that player's sub-network.

Note: ICC is expected to be low because total strength conflates position
(a striker presses more than a goalkeeper) with true individual trait.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr

EDGE_FILE = "scripts/2026-05-05_player_net_m2_edges.csv"
OUTCOME_FILE = "scripts/2026-04-24_match_level_metrics.csv"
PLAYER_FILE = "scripts/2026-04-29-player_level_metrics.csv"

W = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_fault", "valued_contribution",
]
TARGETS = ["goals_against", "shots_against", "xg_against"]


@st.cache_data(show_spinner="Computing player network strengths…")
def load_data(starters_only: bool):
    edges = pd.read_csv(EDGE_FILE)
    edges["match_team_id"] = edges["match_id"].astype(str) + "_" + edges["defending_team"].astype(str)

    # total strength per player per match = sum of edge weights in their sub-network
    grp = edges.groupby(["match_team_id", "match_id", "defending_team",
                          "defending_team_name", "defender_id", "defender_name"])
    strength = grp[W].sum().reset_index()
    strength.columns = ["match_team_id", "match_id", "defending_team",
                         "defending_team_name", "defender_id", "defender_name"] + W

    # attach starter flag from player-level metrics
    players = pd.read_csv(PLAYER_FILE)[["match_team_id", "defender_id", "starter"]]
    strength = strength.merge(players, on=["match_team_id", "defender_id"], how="left")

    if starters_only:
        strength = strength[strength["starter"] == 1]

    # attach match outcomes
    outcomes = pd.read_csv(OUTCOME_FILE)[
        ["match_team_id", "team_name", "competition_stage",
         "goals_against", "shots_against", "xg_against", "passes_against"]
    ]
    df = strength.merge(outcomes, on="match_team_id", how="left")
    return df


def _resid(y, z):
    mask = ~(np.isnan(y) | np.isnan(z))
    r = np.full(len(y), np.nan)
    r[mask] = y[mask] - np.polyval(np.polyfit(z[mask], y[mask], 1), z[mask])
    return r


def corr_tbl(df, partial: bool):
    rows = []
    for w in W:
        sub = df[[w] + TARGETS + ["passes_against"]].dropna()
        if len(sub) < 5:
            continue
        row = {"metric": w}
        for t in TARGETS:
            if not partial:
                r, p = pearsonr(sub[w], sub[t])
            else:
                a = _resid(sub[w].values, sub["passes_against"].values)
                b = _resid(sub[t].values, sub["passes_against"].values)
                mask = ~(np.isnan(a) | np.isnan(b))
                r, p = pearsonr(a[mask], b[mask])
            row[f"r_{t}"] = round(r, 3)
            row[f"p_{t}"] = round(p, 3)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("metric")
    r_cols = [f"r_{t}" for t in TARGETS]
    p_cols = [f"p_{t}" for t in TARGETS]

    def fmt(df_r, df_p):
        combined = df_r.applymap(lambda v: f"{v: .3f}") + " (" + df_p.applymap(lambda v: f"{v:.3f}") + ")"
        return combined.style.background_gradient(
            cmap="RdYlGn", gmap=df_r.values, axis=None, vmin=-1, vmax=1
        )

    st.dataframe(fmt(result[r_cols].rename(columns=lambda c: c[2:]),
                     result[p_cols].rename(columns=lambda c: c[2:])),
                 use_container_width=True)


def icc_tbl(df):
    rows = []
    for w in W:
        s = df[["defender_id", w]].dropna()
        if s["defender_id"].nunique() < 2:
            continue
        g = s.groupby("defender_id")[w]
        nt, ng = len(s), g.ngroups
        sz, mn = g.count(), g.mean()
        msb = (sz * (mn - s[w].mean()) ** 2).sum() / (ng - 1)
        msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0 = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        rows.append({
            "metric": w,
            "ICC": round(icc, 3),
            "n_players": ng,
            "n_obs": nt,
            "note": "stable trait" if icc > 0.5 else ("moderate" if icc > 0.25 else "match/position-driven"),
        })

    st.dataframe(
        pd.DataFrame(rows).style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
        use_container_width=True,
    )


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Player Network — Correlation & ICC", layout="wide")
st.title("Player-Level Defensive Network — Total Strength")
st.caption(
    "**Method 2 edges.** Each row = one defending player in one match. "
    "Total strength = Σ edge weights in that player's sub-network. "
    "ICC groups by player across matches (expected low: total strength confounds position with trait)."
)

with st.sidebar:
    starters_only = st.checkbox("Starters only", value=True)

df = load_data(starters_only)
st.caption(
    f"{len(df)} player-match observations · "
    f"{df['defender_id'].nunique()} unique players · "
    f"{df['match_id'].nunique()} matches"
)

t_corr, t_icc = st.tabs(["Correlation", "ICC (player consistency)"])

with t_corr:
    st.markdown(
        "**ICC note:** because total strength varies heavily by position, "
        "partial correlations (controlling for `passes_against`) are shown alongside raw."
    )
    for tab, partial in zip(
        st.tabs(["Raw", "Partial (controlling passes_against)"]), [False, True]
    ):
        with
            corr_tbl(df, partial)

with t_icc:
    st.markdown(
        "**ICC(1,1):** >0.5 stable player trait · 0.25–0.5 moderate · <0.25 match/position-driven  \n"
        "Low ICC here is expected — total strength reflects who faces more passes (position), "
        "not just the player's own defensive style."
    )
    stage = st.selectbox(
        "Competition stage", ["All"] + sorted(df["competition_stage"].dropna().unique().tolist())
    )
    dff = df if stage == "All" else df[df["competition_stage"] == stage]
    icc_tbl(dff)
