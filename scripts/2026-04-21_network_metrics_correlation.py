import warnings
import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

EDGE_FILES = {k: f"scripts/2026-04-13_defensive_network_edge({k}).csv"
              for k in ("average", "min", "product", "sum")}
OUTCOME_FILE = "scripts/2026-04-16correlation.csv"
OUTCOME_COLS = ["goals_against_real", "shots_against", "total_xt_against",
                "total_xt_only_positive_against", "total_xt_only_negative_against",
                "total_xt_only_successful_against", "passes_against", "n_tackles"]
METRIC_PAIRS = [
    ("raw_involvement_edge_count",       "raw_involvement"),
    ("raw_fault_edge_count",             "raw_fault"),
    ("raw_contribution_edge_count",      "raw_contribution"),
    ("valued_involvement_edge_count",    "valued_involvement"),
    ("valued_contribution_edge_count",   "valued_contribution"),
    ("valued_fault_edge_count",          "valued_fault"),
    ("raw_responsibility_edge_count",    "raw_responsibility"),
    ("raw_fault_r_edge_count",           "raw_fault_r"),
    ("raw_contribution_r_edge_count",    "raw_contribution_r"),
    ("valued_responsibility_edge_count", "valued_responsibility"),
    ("valued_contribution_r_edge_count", "valued_contribution_r"),
    ("valued_fault_r_edge_count",        "valued_fault_r"),
    ("respon-inv_edge_count",            "respon-inv"),
]


def graph_metrics(G):
    n, e = G.number_of_nodes(), G.number_of_edges()
    if n < 2 or e == 0:
        return {}
    deg = list(dict(G.degree()).values())
    strength = list(dict(G.degree(weight="weight")).values())
    total_w = sum(d["weight"] for *_, d in G.edges(data=True))
    m = {
        "n_nodes": n, "n_edges": e, "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "transitivity": nx.transitivity(G),
        "lcc_ratio": len(max(nx.connected_components(G), key=len)) / n,
        "mean_degree": np.mean(deg), "max_degree": max(deg), "std_degree": np.std(deg),
        "mean_strength": np.mean(strength), "max_strength": max(strength), "std_strength": np.std(strength),
        "total_weight": total_w, "avg_edge_weight": total_w / e,
    }
    for fn, key in [(nx.betweenness_centrality, "betweenness"),
                    (nx.eigenvector_centrality_numpy, "eigenvector"),
                    (nx.pagerank, "pagerank")]:
        try:
            vals = list(fn(G, weight="weight").values())
            m[f"mean_{key}"] = np.mean(vals)
            m[f"max_{key}"] = max(vals)
            if key == "pagerank":
                m["std_pagerank"] = np.std(vals)
        except Exception:
            pass
    return m


@st.cache_data(show_spinner="Computing network metrics…")
def compute_all_metrics(method):
    edges = pd.read_csv(EDGE_FILES[method])
    edges["match_team_id"] = edges["match_id"].astype(str) + "_" + edges["defending_team"].astype(str)
    rows = []
    for mt_id, grp in edges.groupby("match_team_id"):
        row = {"match_team_id": mt_id}
        for cnt_col, val_col in METRIC_PAIRS:
            for tag, col in (("cnt", cnt_col), ("val", val_col)):
                G = nx.Graph()
                for _, r in grp.iterrows():
                    w = r.get(col)
                    if pd.notna(w) and w > 0:
                        G.add_edge(r["player_1"], r["player_2"], weight=float(w))
                row |= {f"{val_col}__{tag}__{k}": v for k, v in graph_metrics(G).items()}
        rows.append(row)
    return pd.DataFrame(rows)


def corr_tables(df, xcols, ycols, covar=None):
    r, p, rp, pp = {}, {}, {}, {}
    for x in xcols:
        for y in ycols:
            sub = df[[x, y]].dropna()
            if len(sub) >= 5:
                rv, pv = pearsonr(sub[x], sub[y])
                r[(x, y)], p[(x, y)] = round(rv, 4), round(pv, 4)
            if covar and y != covar:
                sub2 = df[[x, y, covar]].dropna()
                if len(sub2) >= 6:
                    try:
                        res = pg.partial_corr(data=sub2, x=x, y=y, covar=covar)
                        rp[(x, y)], pp[(x, y)] = round(res["r"].values[0], 4), round(res["p-val"].values[0], 4)
                    except Exception:
                        pass
    to_df = lambda d: pd.Series(d).rename_axis(["metric", "outcome"]).unstack("outcome") if d else pd.DataFrame()
    return to_df(r), to_df(p), to_df(rp), to_df(pp)


def show_r(df, bold_r):
    return df.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-1, vmax=1) \
               .applymap(lambda v: "font-weight:bold" if pd.notna(v) and abs(v) >= bold_r else "")


def show_p(df, sig):
    return df.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=0, vmax=0.2) \
               .applymap(lambda v: "font-weight:bold" if pd.notna(v) and v < sig else "")


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Defensive Network × Outcomes", layout="wide")
st.title("Defensive Network Metrics — Correlation with Outcomes")

outcomes = pd.read_csv(OUTCOME_FILE)
base_names = [val for _, val in METRIC_PAIRS]

with st.sidebar:
    method = st.selectbox("Edge aggregation method", list(EDGE_FILES))
    base_sel = st.selectbox("Base metric (Tab 1)", base_names)
    partial_cv = st.selectbox("Partial-corr control variable", ["(none)"] + OUTCOME_COLS, index=3)
    sig_level = st.slider("α", 0.01, 0.20, 0.05, 0.01)
    bold_r = st.slider("Bold |r| ≥", 0.1, 0.8, 0.3, 0.05)

covar = None if partial_cv == "(none)" else partial_cv
all_m = compute_all_metrics(method)
merged = all_m.merge(outcomes[["match_team_id"] + OUTCOME_COLS], on="match_team_id")
net_cols = [c for c in all_m.columns if c != "match_team_id"]
oc = [c for c in OUTCOME_COLS if c in merged.columns]
st.caption(f"{len(merged)} team-match observations.")

tab1, tab2, tab3 = st.tabs(["Focus: count vs value", "All metrics heatmap", "Significant pairs"])

with tab1:
    cnt_cols = [c for c in net_cols if c.startswith(f"{base_sel}__cnt__")]
    val_cols = [c for c in net_cols if c.startswith(f"{base_sel}__val__")]
    cnt_name = next(c for c, v in METRIC_PAIRS if v == base_sel)

    def show_pair(cols, label, prefix):
        r_df, p_df, rp_df, pp_df = corr_tables(merged, cols, oc, covar)
        if r_df.empty:
            return
        for df in (r_df, p_df, rp_df, pp_df):
            if not df.empty:
                df.index = df.index.str.replace(prefix, "", regex=False)
        st.markdown(f"**{label}**")
        c1, c2 = st.columns(2)
        c1.markdown("Pearson r"); c1.dataframe(show_r(r_df, bold_r), use_container_width=True)
        c2.markdown("p-value");   c2.dataframe(show_p(p_df, sig_level), use_container_width=True)
        if not rp_df.empty and covar:
            c3, c4 = st.columns(2)
            c3.markdown(f"Partial r (ctrl: {covar})"); c3.dataframe(show_r(rp_df, bold_r), use_container_width=True)
            c4.markdown("Partial p");                  c4.dataframe(show_p(pp_df, sig_level), use_container_width=True)

    show_pair(cnt_cols, f"COUNT weight — `{cnt_name}`", f"{base_sel}__cnt__")
    st.divider()
    show_pair(val_cols, f"VALUE weight — `{base_sel}`", f"{base_sel}__val__")

with tab2:
    net_metric = st.selectbox("Network metric", ["mean_strength", "density", "avg_clustering",
                               "mean_degree", "mean_betweenness", "mean_pagerank", "lcc_ratio", "total_weight"])
    for tag, label in (("cnt", "COUNT"), ("val", "VALUE")):
        cols = [f"{v}__{tag}__{net_metric}" for _, v in METRIC_PAIRS if f"{v}__{tag}__{net_metric}" in merged.columns]
        r_df, *_ = corr_tables(merged, cols, oc)
        if not r_df.empty:
            r_df.index = r_df.index.str.replace(f"__{tag}__{net_metric}", "", regex=False)
            st.markdown(f"**{label} weight — `{net_metric}`**")
            st.dataframe(show_r(r_df, bold_r), use_container_width=True)

with tab3:
    r_all, p_all, *_ = corr_tables(merged, net_cols, oc)
    sig_rows = []
    for mc in r_all.index:
        for o in r_all.columns:
            pv = p_all.loc[mc, o]
            if pd.notna(pv) and pv < sig_level:
                pr, pp = np.nan, np.nan
                if covar and o != covar:
                    sub = merged[[mc, o, covar]].dropna()
                    if len(sub) >= 6:
                        try:
                            res = pg.partial_corr(data=sub, x=mc, y=o, covar=covar)
                            pr, pp = round(res["r"].values[0], 4), round(res["p-val"].values[0], 4)
                        except Exception:
                            pass
                parts = mc.split("__")
                sig_rows.append({"base": parts[0], "weight": parts[1] if len(parts) > 1 else "",
                                  "net_metric": parts[2] if len(parts) > 2 else "", "outcome": o,
                                  "pearson_r": r_all.loc[mc, o], "pearson_p": pv,
                                  f"partial_r({covar or '-'})": pr, f"partial_p({covar or '-'})": pp})
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows).sort_values("pearson_p").reset_index(drop=True)
        pr_col, pp_col = f"partial_r({covar or '-'})", f"partial_p({covar or '-'})"
        st.dataframe(sig_df.style
            .background_gradient(subset=["pearson_r", pr_col], cmap="RdYlGn", vmin=-1, vmax=1)
            .background_gradient(subset=["pearson_p", pp_col], cmap="RdYlGn_r", vmin=0, vmax=0.2),
            use_container_width=True)
    else:
        st.info("No significant pairs at the chosen α level.")
