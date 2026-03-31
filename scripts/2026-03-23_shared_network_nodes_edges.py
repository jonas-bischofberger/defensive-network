# import pandas as pd
# from itertools import combinations
#
#
# def build_edge_list_for_match(df_match: pd.DataFrame, match_id=None):
#     """
#     Build shared defensive network edge list for one match.
#
#     Each row in the output is one undirected edge (player_1, player_2)
#     for one defending team in one match.
#
#     Edge weights are accumulated across all involvement_pass_id where
#     both defenders jointly participated in defending.
#     """
#
#     # 只保留有防守参与的记录
#     df_match = df_match[df_match["raw_involvement"] != 0].copy()
#
#     if df_match.empty:
#         return pd.DataFrame()
#
#     metric_cols = [
#         "raw_involvement",
#         "raw_fault",
#         "raw_contribution",
#         "valued_involvement",
#         "valued_contribution",
#         "valued_fault",
#     ]
#
#     rows = []
#
#     for defending_team in df_match["defending_team"].dropna().unique():
#         df_team = df_match[df_match["defending_team"] == defending_team].copy()
#
#         edge_dict = {}
#
#         for pass_id, df_pass in df_team.groupby("involvement_pass_id"):
#             # agg_dict = {col: "sum" for col in metric_cols}  # 如果同一个 defender 在同一个 pass 里有多行，先聚合
#             # df_pass_agg = (
#             #     df_pass.groupby("defender_name", as_index=False)
#             #     .agg(agg_dict)
#             # )
#
#             defenders = df_pass["defender_name"].tolist()
#
#             # 少于2人，不形成 shared edge
#             if len(defenders) < 2:
#                 continue
#
#             # 方便后面取值
#             player_metrics = df_pass.set_index("defender_name").to_dict("index")
#
#             for a, b in combinations(sorted(defenders), 2):
#                 key = (match_id, defending_team, a, b)
#
#                 if key not in edge_dict:
#                     edge_dict[key] = {
#                         "match_id": match_id,
#                         "defending_team": defending_team,
#                         "player_1": a,
#                         "player_2": b,
#                         "edge_count": 0,
#                         "raw_involvement": 0.0,
#                         "raw_fault": 0.0,
#                         "raw_contribution": 0.0,
#                         "valued_involvement": 0.0,
#                         "valued_contribution": 0.0,
#                         "valued_fault": 0.0,
#                     }
#
#                 edge_dict[key]["edge_count"] += 1
#
#                 for col in metric_cols:
#                     edge_dict[key][col] += (
#                         player_metrics[a].get(col, 0.0) +
#                         player_metrics[b].get(col, 0.0)
#                     )
#
#         rows.extend(edge_dict.values())
#
#     return pd.DataFrame(rows)
#
#
# # ===== 使用方式 1：单个比赛文件 =====
# df = pd.read_csv("fifa-men-s-world-cup-2022-2-st-poland-saudi-arabia.csv")
#
# # 如果文件里本身有 match_id 列，就自动取；否则手动写一个
# match_id = df["match_id"].iloc[0] if "match_id" in df.columns else "poland_saudi_arabia"
#
# edge_df = build_edge_list_for_match(df, match_id=match_id)
#
# print(edge_df.head())
# print(edge_df.shape)
#
# edge_df.to_csv("shared_defensive_edge_list.csv", index=False)


'''
遍历所有比赛
'''
import sys
import os
import pandas as pd
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import defensive_network.parse.drive


# =========================
# 1. 参数
# =========================
FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"  # only world cup
OUTPUT_FILE = "shared_defensive_edge_list_total222.csv"


# =========================
# 2. 单场比赛 -> edge list
# =========================
def build_edge_list_for_match(df_match: pd.DataFrame, match_id=None, match_name=None):
    # 只保留有防守参与的记录
    df_match = df_match[df_match["raw_involvement"] != 0].copy()

    if df_match.empty:
        return pd.DataFrame()

    metric_cols = [
        "raw_involvement",
        "raw_fault",
        "raw_contribution",
        "valued_involvement",
        "valued_contribution",
        "valued_fault",
    ]

    rows = []

    for defending_team in df_match["defending_team"].dropna().unique():
        df_team = df_match[df_match["defending_team"] == defending_team].copy()
        edge_dict = {}

        for pass_id, df_pass in df_team.groupby("involvement_pass_id"):
            # 如果你非常确定每个 pass_id + defender_name 只有一行，
            # 这里可以不用 groupby；为了稳一点还是保留
            df_pass = (
                df_pass.groupby("defender_name", as_index=False)[metric_cols]
                .sum()
            )

            defenders = df_pass["defender_name"].tolist()

            if len(defenders) < 2:
                continue

            player_metrics = df_pass.set_index("defender_name").to_dict("index")

            for a, b in combinations(sorted(defenders), 2):
                key = (match_id, defending_team, a, b)

                if key not in edge_dict:
                    edge_dict[key] = {
                        "match_id": match_id,
                        "match_name": match_name,
                        "defending_team": defending_team,
                        "player_1": a,
                        "player_2": b,
                        "edge_count": 0,
                        "raw_involvement": 0.0,
                        "raw_fault": 0.0,
                        "raw_contribution": 0.0,
                        "valued_involvement": 0.0,
                        "valued_contribution": 0.0,
                        "valued_fault": 0.0,
                    }

                edge_dict[key]["edge_count"] += 1

                for col in metric_cols:
                    edge_dict[key][col] += (
                        player_metrics[a].get(col, 0.0) +
                        player_metrics[b].get(col, 0.0)
                    )

        rows.extend(edge_dict.values())

    return pd.DataFrame(rows)


# =========================
# 3. 遍历线上所有比赛文件
# =========================
files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)

all_edge_dfs = []

for f in files:
    file_name = f["name"]

    if MATCH_FILTER and MATCH_FILTER not in file_name:
        continue

    if not file_name.endswith(".parquet"):
        continue

    full_path = FOLDER + file_name
    print(f"Processing: {file_name}")

    try:
        df_match = defensive_network.parse.drive.download_parquet_from_drive(full_path)

        match_id = df_match["match_id"].iloc[0]
        match_name = file_name.replace(".parquet", "")
        edge_df = build_edge_list_for_match(df_match, match_id, match_name)

        if not edge_df.empty:
            all_edge_dfs.append(edge_df)

    except Exception as e:
        print(f"Failed on {file_name}: {e}")


# =========================
# 4. 合并并保存
# =========================
if all_edge_dfs:
    final_edge_df = pd.concat(all_edge_dfs, ignore_index=True)
    final_edge_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Final shape:", final_edge_df.shape)
    print(final_edge_df.head())
else:
    print("No valid edge data generated.")

