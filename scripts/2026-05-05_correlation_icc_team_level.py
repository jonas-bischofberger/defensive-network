import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr

outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
squad_size = (pd.read_csv("scripts/2026-05-06_node_level_metrics_with_mins.csv")
              .groupby("match_team_id")["defender_id"].count().rename("n_players"))
edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
              for k in ("average", "min", "product", "sum")}

W = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r",
]
TARGETS = ["goals_against", "shots_against", "xg_against"]
GROUPS = {
    "Total Network Strength": W,
    "Network Density": [c + "_density" for c in W],
    "Gini (player strength inequality)": [c + "_gini" for c in W],
    "Clustering Coefficient (unweighted)": [c + "_cc_unweighted" for c in W],
    "Clustering Coefficient (weighted)": [c + "_cc_weighted" for c in W],
}



def _gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def process(df, thr=1):
    df["match_team_id"] = df["match_id"].astype(str) + "_" + df["defending_team"].astype(str)
    mp = squad_size * (squad_size - 1) / 2
    out = df.groupby("match_team_id")[W].sum()
    out = out.join(pd.DataFrame({
        c + "_density": df[df[c + "_edge_count"] >= thr].groupby("match_team_id").size() / mp
        for c in W
    }))
    extra = {}
    for c in W:
        p1 = df[["match_team_id", "player_1", c]].rename(columns={"player_1": "player"})
        p2 = df[["match_team_id", "player_2", c]].rename(columns={"player_2": "player"})
        ps = pd.concat([p1, p2]).groupby(["match_team_id", "player"])[c].sum()
        extra[c + "_gini"] = ps.groupby("match_team_id").apply(_gini)
        u, w = {}, {}
        for mid, g in df.groupby("match_team_id"):
            e = g[g[c + "_edge_count"] >= thr][["player_1", "player_2", c]]
            if len(e) < 2:
                continue
            G = nx.Graph()
            for _, row in e.iterrows():
                G.add_edge(row["player_1"], row["player_2"], weight=row[c])
            u[mid] = nx.average_clustering(G)
            w[mid] = nx.average_clustering(G, weight="weight")
        extra[c + "_cc_unweighted"] = pd.Series(u)
        extra[c + "_cc_weighted"] = pd.Series(w)
    outcome_cols = ["match_team_id", "team_name", "competition_stage"] + TARGETS + ["passes_against"]
    return out.join(pd.DataFrame(extra)).reset_index().merge(outcomes[outcome_cols], on="match_team_id")


def _resid(y, z):
    mask = ~(np.isnan(y) | np.isnan(z))
    r = np.full(len(y), np.nan)
    r[mask] = y[mask] - np.polyval(np.polyfit(z[mask], y[mask], 1), z[mask])
    return r


def corr_tbl(df, cols, partial=False):
    def _r(m, t):
        if not partial:
            return pearsonr(df[m], df[t])
        s = df[[m, t, "passes_against"]].dropna()
        a, b = _resid(s[m].values, s["passes_against"].values), _resid(s[t].values, s["passes_against"].values)
        mask = ~(np.isnan(a) | np.isnan(b))
        return pearsonr(a[mask], b[mask])
    r = {(m, t): _r(m, t) for m in cols for t in TARGETS}
    to_df = lambda i: pd.Series({k: v[i] for k, v in r.items()}).rename_axis(["metric", "outcome"]).unstack("outcome")
    rdf, pdf = to_df(0), to_df(1)
    disp = rdf.applymap(lambda v: f"{v: .2f}") + " (" + pdf.applymap(lambda v: f"{v:.3f}") + ")"
    st.dataframe(disp.style.background_gradient(cmap="RdYlGn", gmap=rdf.values, axis=None, vmin=-1, vmax=1))


def icc_tbl(df, cols):
    rows = []
    for c in cols:
        s = df[["team_name", c]].dropna()
        if s["team_name"].nunique() < 2:
            continue
        g = s.groupby("team_name")[c]
        nt, ng, sz, mn = len(s), g.ngroups, g.count(), g.mean()
        msb = (sz * (mn - s[c].mean()) ** 2).sum() / (ng - 1)
        msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
        k0 = (nt - (sz ** 2).sum() / nt) / (ng - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        rows.append({"metric": c, "ICC": round(icc, 3), "n_teams": ng, "n_obs": nt,
                     "interpretation": "stable trait" if icc > 0.5 else "match-driven"})
    st.dataframe(pd.DataFrame(rows).style.background_gradient(cmap="RdYlGn", subset=["ICC"], vmin=0, vmax=1),
                 use_container_width=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Defensive Network Analysis")
with st.sidebar:
    method = st.selectbox("Edge weight method", list(edge_dfs))
    thr = st.slider("Edge count threshold (≥)", 1, 20, 1)
df = process(edge_dfs[method], thr)

t_corr, t_icc = st.tabs(["Correlation", "Robustness (ICC)"])

with t_corr:
    for tab, partial in zip(st.tabs(["Raw", "Partial (controlling passes_against)"]), [False, True]):
        with tab:
            for name, cols in GROUPS.items():
                st.subheader(name)
                corr_tbl(df, cols, partial)

with t_icc:
    st.markdown("**ICC(1,1)**: >0.75 stable trait · 0.5–0.75 moderate · <0.5 match-driven")
    stage = st.selectbox("Competition stage", ["All"] + sorted(df["competition_stage"].dropna().unique().tolist()))
    dff = df if stage == "All" else df[df["competition_stage"] == stage]
    if stage != "All":
        st.caption(f"{len(dff)} obs · {dff['team_name'].nunique()} teams")
    for name, cols in GROUPS.items():
        st.subheader(name)
        icc_tbl(dff, cols)
