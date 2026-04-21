import pandas as pd
import streamlit as st
import pingouin as pg
from scipy.stats import pearsonr

match_defensive = pd.read_csv("scripts/2026-04-16correlation.csv")

edge_dfs = {
    "average": pd.read_csv("scripts/2026-04-13_defensive_network_edge(average).csv"),
    "min": pd.read_csv("scripts/2026-04-13_defensive_network_edge(min).csv"),
    "product": pd.read_csv("scripts/2026-04-13_defensive_network_edge(product).csv"),
    "sum": pd.read_csv("scripts/2026-04-13_defensive_network_edge(sum).csv"),
}
#
# weight_cols = [
#     'raw_involvement_edge_count', 'raw_involvement', 'raw_fault_edge_count', 'raw_fault', 'raw_contribution_edge_count',
#     'raw_contribution', 'valued_involvement_edge_count', 'valued_involvement', 'valued_contribution_edge_count',
#     'valued_contribution', 'valued_fault_edge_count', 'valued_fault', 'raw_responsibility_edge_count',
#     'raw_responsibility', 'raw_fault_r_edge_count', 'raw_fault_r', 'raw_contribution_r_edge_count',
#     'raw_contribution_r', 'valued_responsibility_edge_count', 'valued_responsibility',
#     'valued_contribution_r_edge_count', 'valued_contribution_r', 'valued_fault_r_edge_count', 'valued_fault_r',
#     'respon-inv_edge_count', 'respon-inv']


weight_cols = ['raw_involvement', 'raw_fault', 'raw_contribution', 'valued_involvement', 'valued_contribution',
                'valued_fault', 'raw_responsibility', 'raw_fault_r', 'raw_contribution_r',  'valued_responsibility',
               'valued_contribution_r', 'valued_fault_r']

target_cols = ['goals_against_real', 'shots_against', "total_xt_against", "total_xt_only_positive_against",
               "total_xt_only_negative_against", "total_xt_only_successful_against", "passes", "passes_against",
               "n_tackles"]
# target_cols = ['goals_against_real', 'shots_against', "total_xt_only_positive_against",
#                "total_xt_only_negative_against", "total_xt_only_successful_against", "passes", "passes_against",
#                "n_tackles"]


def process(df):
    df["match_team_id"] = df["match_id"].astype(str) + "_" + df["defending_team"].astype(str)
    result = df.groupby("match_team_id")[weight_cols].sum().reset_index()
    merged = result.merge(
        match_defensive[['match_team_id'] + target_cols],
        on='match_team_id'
    )
    return merged


# --- Streamlit UI ---
st.title("Defensive Network Correlation Analysis")

method = st.selectbox("Edge weight method", list(edge_dfs.keys()))
merged = process(edge_dfs[method])

r_vals, p_vals = {}, {}
r_vals_partial, p_vals_partial = {}, {}

for col in weight_cols:
    for target in target_cols:
        r, p = pearsonr(merged[col], merged[target])
        r_vals[(col, target)] = r
        p_vals[(col, target)] = p

        if target != 'total_xt_against':
            result = pg.partial_corr(data=merged, x=col, y=target, covar='total_xt_against')
            r_vals_partial[(col, target)] = result['r'].values[0]
            p_vals_partial[(col, target)] = result['p-val'].values[0]

r_df = pd.DataFrame(r_vals, index=['r']).T.unstack()
p_df = pd.DataFrame(p_vals, index=['p']).T.unstack()
r_df_partial = pd.DataFrame(r_vals_partial, index=['r']).T.unstack()
p_df_partial = pd.DataFrame(p_vals_partial, index=['p']).T.unstack()

st.subheader("Pearson Correlation (r)")
st.dataframe(r_df.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-1, vmax=1))
st.subheader("P-value")
st.dataframe(p_df.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=0, vmax=0.1))

st.subheader("Partial Correlation (r) — controlling for total_xt_against")
st.dataframe(r_df_partial.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-1, vmax=1))
st.subheader("Partial P-value")
st.dataframe(p_df_partial.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=0, vmax=0.1))