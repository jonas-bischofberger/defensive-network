import pandas as pd
import numpy as np
import defensive_network
import defensive_network.parse.drive


# =========================
# step 1
# =========================
FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
OUTPUT_FILE = "player_average_defensive_positions_all_matches.csv"

METRICS = [
    "raw_involvement",
    "raw_contribution",
    "raw_fault",
    "valued_involvement",
    "valued_contribution",
    "valued_fault",
]


# =========================
# 2. standardize coordinates for first and second half
# =========================
def align_coordinates(df):
    df = df.copy()

    df["x"] = np.where(
        df["section"] == 2,
        -df["defender_x"],
        df["defender_x"]
    )

    df["y"] = np.where(
        df["section"] == 2,
        -df["defender_y"],
        df["defender_y"]
    )

    return df


# =========================
# 3.  home / away
# =========================
def add_home_away_column(df):
    """
    根据原始事件里的球队信息，给 defending_team 补 home_away。
    假设原始表里有：
    - gameEvents.teamId
    - gameEvents.homeTeam   (True/False)
    """
    df = df.copy()

    team_lookup = (
        df[["gameEvents.teamId", "gameEvents.homeTeam"]]
        .dropna(subset=["gameEvents.teamId"])
        .drop_duplicates()
        .rename(columns={
            "gameEvents.teamId": "team_id",
            "gameEvents.homeTeam": "is_home"
        })
    )

    # 如果同一个 team_id 出现多次，保留第一条
    team_lookup = team_lookup.drop_duplicates(subset=["team_id"])

    team_lookup["home_away"] = np.where(
        team_lookup["is_home"] == True,
        "home",
        "away"
    )

    df = df.merge(
        team_lookup[["team_id", "home_away"]],
        left_on="defending_team",
        right_on="team_id",
        how="left"
    )

    df = df.drop(columns=["team_id"])

    return df


# =========================
# 4. player summary
# =========================
def build_player_summary_for_match(df_match):
    required_cols = [
        "match_id",
        "defending_team",
        "defender_id",
        "defender_name",
        "defender_x",
        "defender_y",
        "section",
        "gameEvents.teamId",
        "gameEvents.homeTeam",
    ] + METRICS

    missing_cols = [col for col in required_cols if col not in df_match.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 只保留关键字段完整的行
    df = df_match.dropna(subset=[
        "match_id",
        "defending_team",
        "defender_id",
        "defender_name",
        "defender_x",
        "defender_y",
        "section"
    ]).copy()

    # 统一上下半场方向
    df = align_coordinates(df)

    # 加 home_away
    df = add_home_away_column(df)

    group_cols = [
        "match_id",
        "defending_team",
        "home_away",
        "defender_id",
        "defender_name",
    ]

    # 每个球员一行
    result = (
        df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    # 整体平均位置（不区分 metric，所有防守事件都算）
    overall = (
        df.groupby(group_cols)
        .agg(
            overall_n=("defender_id", "size"),
            overall_avg_x=("x", "mean"),
            overall_avg_y=("y", "mean"),
        )
        .reset_index()
    )

    result = result.merge(overall, on=group_cols, how="left")

    # 各种 metric 单独算：非零即参与
    for metric in METRICS:
        sub = df[df[metric] != 0].copy()

        agg = (
            sub.groupby(group_cols)
            .agg(
                **{
                    f"{metric}_n": (metric, "size"),
                    f"{metric}_avg_x": ("x", "mean"),
                    f"{metric}_avg_y": ("y", "mean"),
                }
            )
            .reset_index()
        )

        result = result.merge(agg, on=group_cols, how="left")

    return result


# =========================
# 5. 遍历所有线上比赛文件
# =========================
def main():
    files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)

    all_player_summaries = []

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

            if df_match.empty:
                print(f"Skipped empty file: {file_name}")
                continue

            player_summary = build_player_summary_for_match(df_match)

            if not player_summary.empty:
                all_player_summaries.append(player_summary)

        except Exception as e:
            print(f"Failed on {file_name}: {e}")

    if not all_player_summaries:
        print("No valid match summaries were created.")
        return

    final_df = pd.concat(all_player_summaries, ignore_index=True)

    # 排序
    sort_cols = ["match_id", "defending_team", "defender_id"]
    final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    # 导出
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Saved to: {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    main()