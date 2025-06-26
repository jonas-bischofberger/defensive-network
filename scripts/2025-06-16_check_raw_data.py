import pandas as pd
import streamlit as st

path = st.text_input("Enter the path to the raw data file:", "Y:/w_raw/events/events.csv")
meta_path = st.text_input("Enter the path to the metadata file:", "Y:/w_raw/meta.csv")
df_meta = pd.read_csv(meta_path)
df_meta["match_string"] = df_meta.apply(lambda row: f"{row['competition_name']} {row['season_name']}: {row['match_day']}.ST {row['match_title'].replace('-', ' - ')}", axis=1)

matchid2string = dict(zip(df_meta["match_id"], df_meta["match_string"]))
matchid2matchday = dict(zip(df_meta["match_id"], df_meta["match_day"]))

st.write("df_meta")
st.write(df_meta)

def _process_file(fpath):
    if fpath.endswith(".csv"):
        df = pd.read_csv(fpath)
    elif fpath.endswith(".parquet"):
        df = pd.read_parquet(fpath)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")

    st.write(df)

    df = df[["match_id"]].drop_duplicates(subset="match_id")
    df = df.merge(df_meta[["match_id", "match_string", "match_day"]], on="match_id", how="left")

    return df


import os
if path:
    if os.path.isfile(path) and path.endswith((".csv", ".parquet")):
        df = _process_file(path)
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".csv", ".parquet"))]
        dfs = []
        for f in files:
            st.write(f"Processing file: {f}")
            df = _process_file(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

    for match_day, df_matchday in df.groupby("match_day"):
        with st.expander(f"Match Day {match_day}: {len(df_matchday)} matches"):
            for match_id, match_string in zip(df_matchday["match_id"], df_matchday["match_string"]):
                st.write(f"{match_string} (ID={match_id})")
        st.write("---")
    st.write(f"Total: {len(df)} matches")
    st.write(df)

