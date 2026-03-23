import pandas as pd
from itertools import combinations
import defensive_network.parse.drive

# =========================
# Config
# =========================
FOLDER = "involvement/10/"                 # Drive file
MATCH_FILTER = "fifa-men-s-world-cup-2022"  # only world cup
OUT_CSV = "shared_defensive_team_metrics_all_matches.csv"

# =========================
# Helper: compute per-team metrics for ONE match dataframe
# =========================


def compute_shared_metrics_for_match(df: pd.DataFrame, match_id: str):
    df = df[df["valued_involvement"] != 0].copy()  # filter out non-involvements
    if df.empty:
        return []

    rows = []

    for defending_team in df["defending_team"].dropna().unique():
        df_team = df[df["defending_team"] == defending_team]

        pair_count = {}
        pair_inv = {}

        for p in df_team["involvement_pass_id"].dropna().unique():
            df_pass = df_team[df_team["involvement_pass_id"] == p]

            # defender -> inv
            inv = df_pass.set_index("defender_name")["valued_involvement"].to_dict()

            if len(inv) < 2:
                continue

            for a, b in combinations(sorted(inv), 2):
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
                pair_inv[(a, b)] = pair_inv.get((a, b), 0) + (inv[a] + inv[b])

        # --- nodes ---
        nodes = set()
        for a, b in pair_count.keys():
            nodes.add(a)
            nodes.add(b)

        N = len(nodes)
        denom = N * (N - 1) / 2  # C(N,2)

        total_count_w = float(sum(pair_count.values()))
        total_inv_w = float(sum(pair_inv.values()))

        density_count = (total_count_w / denom) if denom > 0 else 0.0
        density_inv = (total_inv_w / denom) if denom > 0 else 0.0

        rows.append({
            "match_id": match_id,
            "defending_team": defending_team,
            "N_nodes": N,
            "N_edges": len(pair_count),
            "total_count_weight": total_count_w,
            "total_inv_weight": total_inv_w,
            "density_count": density_count,
            "density_inv": density_inv,
        })

    return rows

# =========================
# Main: loop all matches in Drive folder -> compute -> save csv
# =========================
files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)

all_rows = []
for f in files:
    file_name = f["name"]

    if MATCH_FILTER not in file_name:
        continue
    if not file_name.endswith(".parquet"):
        continue

    full_path = FOLDER + file_name
    df_match = defensive_network.parse.drive.download_parquet_from_drive(full_path)

    match_id = file_name.replace(".parquet", "")
    all_rows.extend(compute_shared_metrics_for_match(df_match, match_id))

# save
result_df = pd.DataFrame(all_rows)
# result_df.to_csv(OUT_CSV, index=False)
# print(f"Saved: {OUT_CSV}  | rows={len(result_df)}")
print(result_df.head())


"""
test
"""
# import sys
# import os
# import pandas as pd
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from itertools import combinations
#
#
# import defensive_network.parse.drive
# import streamlit as st
#
# files = defensive_network.parse.drive.list_files_in_drive_folder("involvement/10/")
# file_names = [file["name"] for file in files]
# # st.write(file_names)
#
# for file_name in file_names:
#     full_path = "involvement/10/" + file_name
#     if "fifa-men-s-world-cup-2022" not in file_name:
#         continue
#     # st.write(full_path)
#
#     df = defensive_network.parse.drive.download_parquet_from_drive(full_path)
#     # save as csv, name as the same as the file name
#     file_name = file_name.replace(".parquet", ".csv")
#     df.to_csv(file_name, index=False)
#
#     # st.write(df)
#     break


