import glob
import re
import pandas as pd
import streamlit as st

path = "C:/Users/j.bischofberger/Downloads/sent"

# get all excel fileas

dfs = []
for excel_file in glob.glob(f"{path}/*.csv"):
    st.write(excel_file)
    number = re.findall(r'\d+', excel_file)[-1]

    df_sheet = pd.read_csv(excel_file)
    # assert len(sheets.items()) == 1
    # df_sheet = list(sheets.values())[0]
    df_sheet["radius"] = number

    dfs.append(df_sheet)

df_sheet = pd.concat(dfs, ignore_index=True)

st.write("df_sheet")
st.write(df_sheet)

kpi_cols = [
    "pearson_correlation_only_cbs_def_awareness_directed",
    "pearson_correlation_only_cbs_market_value_directed",
    "Validity Score",
    "match_level_icc",
    "bootstrapped_season_level_icc (corrected by position)",
    "seasonal_autocorrelation (Partial)",
    "Robustness Score",
]

dfs = []
for kpi in kpi_cols:
    df_partial = df_sheet[["radius", kpi, "competitions", "n_competitions", "kpi"]].dropna().rename(columns={kpi: "value"})
    df_partial["meta_metric"] = kpi
    dfs.append(df_partial)

    st.write("df_partial")
    st.write(df_partial)

df_final = pd.concat(dfs, ignore_index=True).set_index(["competitions", "meta_metric", "kpi", "radius"]).sort_index()
df_final = df_final[df_final["n_competitions"] == 1].drop(columns=["n_competitions"])
df_final["radius_5_value"] = df_final.groupby(["competitions", "meta_metric", "kpi"])["value"].transform(lambda x: x.loc[x.index.get_level_values("radius") == "5"].values[0])
df_final["difference_to_radius_5"] = df_final["value"] - df_final["radius_5_value"]
df_final["percentage_difference_to_radius_5"] = df_final["difference_to_radius_5"] / df_final["radius_5_value"]

df_final = df_final[df_final.index.get_level_values("radius") != "5"].reset_index()

st.write("df_final", df_final.shape)
st.write(df_final)

df_pivot = df_final.pivot_table(index=["meta_metric", "kpi"], columns="radius", values=["value", "radius_5_value", "difference_to_radius_5", "percentage_difference_to_radius_5"], aggfunc="mean")
st.write("df_pivot", df_pivot.shape)
st.write(df_pivot)

df_pivot2 = df_final.pivot_table(index="meta_metric", columns="radius", values=["value", "radius_5_value", "difference_to_radius_5", "percentage_difference_to_radius_5"], aggfunc="mean")
st.write("df_pivot2", df_pivot2.shape)
st.write(df_pivot2)


# df = df_partial.groupby("radius").agg(
    #     mean_value=("value", "mean"),
    #     std_value=("value", "std"),
    #     count_value=("value", "count"),
    # ).reset_index()
    # df["sem_value"] = df["std_value"] / df["count_value"] ** 0.5
    # df["ci95_hi"] = df["mean_value"] + 1.96 * df["sem_value"]
    # df["ci95_lo"] = df["mean_value"] - 1.96 * df["sem_value"]
    #
    #
    # st.write(f"{meta_metric}")
    # st.write(df)
