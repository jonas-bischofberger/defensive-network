
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from itertools import combinations


import defensive_network.parse.drive
import streamlit as st

files = defensive_network.parse.drive.list_files_in_drive_folder("team_matchsums/10/")
file_names = [file["name"] for file in files]
# st.write(file_names)

all_dfs = []

for file_name in file_names:
    full_path = "team_matchsums/10/" + file_name
    if "fifa-men-s-world-cup-2022" not in file_name:
        continue
    # st.write(full_path)

    df = defensive_network.parse.drive.download_parquet_from_drive(full_path)
    # save as csv, name as the same as the file name
    file_name = file_name.replace(".parquet", ".csv")
    # df.to_csv(file_name, index=False)

    all_dfs.append(df)

# merge together
final_df = pd.concat(all_dfs, ignore_index=True)


# final_df.to_csv("team_matchsums_all_matches.csv", index=False)

print(f"Saved {len(all_dfs)} matches into team_matchsums_all_matches.csv")