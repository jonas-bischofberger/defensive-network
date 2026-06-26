import numpy as np
import pandas as pd
from statsbombpy import sb

pd.set_option("display.max_columns", None)   # 显示所有列
pd.set_option("display.width", None)         # 不限制宽度
pd.set_option("display.max_colwidth", None)  # 列内容不截断


def get_num_from_str(s):
    parts = s.split(":")
    num_str = parts[0]
    num = int(num_str)
    return num


# player_info_df = pd.read_csv('250716network_metrics_final.csv')
# player_info_df["player_match_id"] = player_info_df["Player"].astype(str) + "_" + player_info_df["match_id"].astype(str)
# print(player_info_df.head())

competition = sb.matches(competition_id=43, season_id=106)
player_time_dict = {}
player_team_dict = {}
player_name_to_jersey = {}

for match in competition['match_id']:
    match_lineups = sb.lineups(match_id=match)  # for players' playing time

    # end time
    event_df = sb.events(match_id=match)
    end_time = event_df['minute'].max()

    # # start time
    for team in match_lineups.keys():
        match_player_df = match_lineups[team]
        print(team)

        # match_player_df = match_lineups['Brazil']
        for row in match_player_df.iterrows():  # players' loop
            player = row[1]['player_name']
            player_team_dict[player] = team  # record the players' team
            player_name_to_jersey[player] = row[1]["jersey_number"]  # players' number
            minute_played = row[1]['positions']

            if not minute_played:
                time = 0
            else:
                if minute_played[-1]['to'] is None:  # playing until Final Whistle
                    # print(minute_played[-1]['end_reason'])
                    time = end_time - get_num_from_str(minute_played[0]['from'])
                else:  # substituted
                    time = get_num_from_str(minute_played[-1]['to']) - get_num_from_str(minute_played[0]['from'])
            player_match_id = str(player) + "_" + str(match)

            player_time_dict[player_match_id] = time   # 每场的出场时间
            player_team_dict[player_match_id] = team   # 每场对应球队
            player_name_to_jersey[player_match_id] = row[1]["jersey_number"]

mins_df = pd.DataFrame({
    "player_match_id": list(player_time_dict.keys()),
    "mins_played": list(player_time_dict.values()),
    "team": [player_team_dict[k] for k in player_time_dict.keys()],
    "jersey_number": [player_name_to_jersey[k] for k in player_time_dict.keys()]
})

# split player_match_id back into player_name and match_id
# format is "PlayerName_matchid" where matchid is always a plain integer at the end
mins_df["match_id"] = mins_df["player_match_id"].str.rsplit("_", n=1).str[-1].astype(int)
mins_df["player_name"] = mins_df["player_match_id"].str.rsplit("_", n=1).str[0]

# build match name from competition fixture list (avoids relying on match_id across datasets)
match_names = competition[["match_id", "home_team", "away_team"]].copy()
match_names["match_name"] = match_names["home_team"] + " vs " + match_names["away_team"]

mins_df = mins_df.merge(match_names[["match_id", "match_name"]], on="match_id", how="left")

out = mins_df[["match_name", "team", "player_name", "mins_played"]].sort_values(
    ["match_name", "team", "player_name"]
).reset_index(drop=True)

print(out.head(20))
out.to_csv("player_minutes_per_match.csv", index=False)

# mins_df = pd.read_csv("260121_player_info.csv")
#
# player_info_df = player_info_df.merge(
#     mins_df[["player_match_id", "mins_played", "team", "jersey_number"]],
#     on="player_match_id",
#     how="left")
#
# # print(player_info_df[["Player", "match_id", "player_match_id", "mins_played", "jersey_number"]].head())
#
# # 保存
# player_info_df.to_csv("20260122_test.csv", index=False)
#
# # 可选：检查没匹配上的行
# print("Unmatched mins_played:", player_info_df["mins_played"].isna().sum())
# unmatched = player_info_df[player_info_df["mins_played"].isna()].copy()
# cols = [c for c in ["Player", "match_id", "player_match_id", "team", "opponent", "match_date"] if c in unmatched.columns]
# print(unmatched[cols].to_string(index=False))




