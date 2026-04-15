import sys
import os
from itertools import combinations
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive


# 1. parameters
FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
ROLE_VALUE_FILE = "FIFA Mens World Cup_10.csv"
OUTPUT_FILE = "2026-04-14_defensive_network_edge(average).csv"
player_info_df = pd.read_csv("2026-04-09_player_average_defensive_positions_all_matches.csv")


metrics = ["raw_involvement", "raw_fault", "raw_contribution", "valued_involvement", "valued_contribution",
           "valued_fault", "raw_responsibility", "raw_fault_r", "raw_contribution_r", "valued_responsibility",
           "valued_contribution_r", "valued_fault_r"]

role_keys = ["role_category_1", "network_receiver_role_category", "defender_role_category"]  # columns to match

role_value_cols = ["raw_responsibility"]

edge_keys = ["match_id", "match_name", "defending_team", "player_1", "player_2"]


# 2. read role value

role_df = pd.read_csv(ROLE_VALUE_FILE)
# keep the chosen columns
role_df = role_df[role_keys + role_value_cols].copy()

# 3. for all matches
files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_match_edge_tables = []
all_match_self_count_tables = []

for metric in metrics:
    player_info_df[f"{metric}_self_inv"] = 0.0

for f in files:
    file_name = f["name"]
    if MATCH_FILTER and MATCH_FILTER not in file_name:
        continue
    if not file_name.endswith(".parquet"):
        continue
    print(f"Processing: {file_name}")
    full_path = FOLDER + file_name
    df_match = defensive_network.parse.drive.download_parquet_from_drive(full_path)

    # # delete passOutcomeType == "B"
    # if "possessionEvents.passOutcomeType" in df_match.columns:
    #     df_match = df_match[df_match["possessionEvents.passOutcomeType"] != "B"].copy()

    match_id = df_match["match_id"].iloc[0]
    match_name = file_name.replace(".parquet", "")

    # 4. merge role-based

    df_match = df_match.merge(role_df, on=role_keys, how="left")

    for col in role_value_cols:
        df_match[col] = df_match[col].fillna(0.0)
    metric_edge_tables = []
    df_match["valued_responsibility"] = df_match["raw_responsibility"] * abs(df_match["pass_xt"])  # valued= raw * xt
    # pass_xt < 0 → contribution
    mask_contribution = df_match["pass_xt"] < 0
    df_match.loc[mask_contribution, "raw_contribution_r"] = df_match.loc[mask_contribution, "raw_responsibility"]
    df_match.loc[mask_contribution, "valued_contribution_r"] = df_match.loc[mask_contribution, "valued_responsibility"]

    # pass_xt > 0 → fault
    mask_fault = df_match["pass_xt"] > 0
    df_match.loc[mask_fault, "raw_fault_r"] = df_match.loc[mask_fault, "raw_responsibility"]
    df_match.loc[mask_fault, "valued_fault_r"] = df_match.loc[mask_fault, "valued_responsibility"]


    # 5. for each metric, build the edge table

    for metric in metrics:
        if metric not in df_match.columns:
            continue

        df_metric = df_match[df_match[metric].fillna(0) != 0].copy()

        rows = []

        for defending_team in df_metric["defending_team"].dropna().unique():
            df_team = df_metric[df_metric["defending_team"] == defending_team].copy()

            edge_dict = {}

            for pass_id, df_pass in df_team.groupby("involvement_pass_id"):
                defenders = sorted(df_pass["defender_name"].tolist())

                # calculate self-inv （only 1 player invovled） and merge to player_info_df
                if len(defenders) < 2:
                    defender = defenders[0]
                    self_inv = df_pass[metric].iloc[0]
                    # print(self_inv)

                    # find the corresponding row in player_info_df
                    mask = ((player_info_df["match_id"] == match_id) &
                            (player_info_df["defending_team"] == defending_team) &
                            (player_info_df["defender_name"] == defender))

                    if mask.any():
                        current = player_info_df.loc[mask, f"{metric}_self_inv"].fillna(0)
                        player_info_df.loc[mask, f"{metric}_self_inv"] = current + self_inv

                else:
                    player_metric = df_pass.set_index("defender_name")[metric].to_dict()

                    for a, b in combinations(defenders, 2):
                        key = (match_id, match_name, defending_team, a, b)

                        if key not in edge_dict:
                            edge_dict[key] = {
                                "match_id": match_id,
                                "match_name": match_name,
                                "defending_team": defending_team,
                                "player_1": a,
                                "player_2": b,
                                f"{metric}_edge_count": 0,
                                metric: 0.0,
                            }

                        edge_dict[key][f"{metric}_edge_count"] += 1
                        val_a = player_metric.get(a, 0.0)
                        val_b = player_metric.get(b, 0.0)
                        # edge_dict[key][metric] += val_a * val_b  # product
                        # edge_dict[key][metric] += min(val_a, val_b)  # min
                        edge_dict[key][metric] += (val_a + val_b) / 2.0  # average of sum
                        # edge_dict[key][metric] += val_a + val_b  # sum
            rows.extend(edge_dict.values())

        metric_edge_df = pd.DataFrame(rows)

        if not metric_edge_df.empty:
            metric_edge_tables.append(metric_edge_df)

    # 6. merge all metrics for the current match

    if metric_edge_tables:
        all_edges = pd.concat(
            [df[edge_keys] for df in metric_edge_tables],
            ignore_index=True
        ).drop_duplicates()

        all_edges = all_edges.sort_values(edge_keys).reset_index(drop=True)

        edge_table = all_edges.copy()

        for metric_df in metric_edge_tables:
            extra_cols = [c for c in metric_df.columns if c not in edge_keys]

            edge_table = edge_table.merge(
                metric_df[edge_keys + extra_cols],
                on=edge_keys,
                how="left"
            )

        all_match_edge_tables.append(edge_table)


# 7. merge all matches
if all_match_edge_tables:
    final_df = pd.concat(all_match_edge_tables, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)
    print("Final shape:", final_df.shape)
    print(final_df.head())
else:
    print("No valid data generated.")


# 8.  player_info_df
player_info_df.to_csv("2026-04-14_player_info_with_self_inv.csv", index=False)
print("Saved: player_info_with_self_inv.csv")

