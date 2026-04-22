import ast
import pandas as pd
import defensive_network.parse.drive

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
# MATCH_FILTER = "fifa-men-s-world-cup-2022-2-st-england-united-states"
player_info_df = pd.read_csv("2026-04-22test.csv")
player_info_df["starter"] = 0  # 默认都是0

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)

for f in files:
    file_name = f["name"]
    print(f"Processing: {file_name}")
    if MATCH_FILTER and MATCH_FILTER not in file_name:
        continue
    if not file_name.endswith(".parquet"):
        continue

    df_match = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + file_name)
    match_id = df_match["match_id"].iloc[0]

    # 提取首发 player_id
    starter_ids = set()
    for col in ["homePlayers", "awayPlayers"]:
        players_list = df_match[col].iloc[0]
        if isinstance(players_list, str):
            players_list = ast.literal_eval(players_list)
        for p in players_list:
            starter_ids.add(p["playerId"])
            print(f"Match {match_id} - Found starter playerId: {p['playerId']}")

    # 打标记
    mask = player_info_df["match_id"] == match_id
    player_info_df.loc[mask, "starter"] = player_info_df.loc[mask, "defender_id"].apply(
        lambda x: 1 if x in starter_ids else 0
    )

# player_info_df.to_csv("2026-04-16_player_info_with_starter.csv", index=False)
player_info_df.to_csv("starter.csv", index=False)
print("Saved")

