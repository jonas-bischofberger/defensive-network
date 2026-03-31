import pandas as pd
import streamlit as st
import pandas as pd
from scipy.stats import pearsonr

'''
merge team and match_id 
'''
# # data organization
# df1 = pd.read_csv("team_matchsums_simple.csv")
# df2 = pd.read_csv("contribution_valued.csv")
#
# for df in (df1, df2):
#     df["team_id"] = df["team_id"].astype("string")
#     df["match_info"] = df["match_info"].astype("string").str.strip()
#
#
# df1["team_match_id"] = df1["match_info"] + "_" + df1["team_id"]
# df2["team_match_id"] = df2["match_info"] + "_" + df2["team_id"]
#
# # df1.to_csv("1.csv", index=False)
# df2.to_csv("3.csv", index=False)

'''
correlation
'''
df = pd.read_csv("1.csv")

cols = list(dict.fromkeys([
    "goals_against_real",
    "shots_against",
    "total_inv_weight_f",
    "density_inv_f",
    "total_inv_weight_c",
    "density_inv_c"
]))

df = df[cols]


results = []

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        x, y = df[[cols[i], cols[j]]].dropna().T.values
        r, p = pearsonr(x, y)
        results.append([cols[i], cols[j], r, p])


corr_df = pd.DataFrame(
    results,
    columns=["var_1", "var_2", "pearson_r", "p_value"]
)

print(corr_df.round(4))