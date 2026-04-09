import pandas as pd
import numpy as np
import defensive_network
import defensive_network.parse.drive


# 1. settings
FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
OUTPUT_FILE = "2026-04-09_player_average_defensive_positions_all_matches2.csv"

METRICS = ["raw_involvement", "raw_contribution", "raw_fault", "valued_involvement", "valued_contribution",
           "valued_fault"]


# 2. standardize coordinates for first and second half
def align_coordinates(df):
    df = df.copy()
    df["x"] = np.where(df["section"] == 2, -df["defender_x"], df["defender_x"])
    df["y"] = np.where(df["section"] == 1, -df["defender_y"], df["defender_y"])
    # y is a bit strange, if you do same as "x" axis, the defensive position will be flipped
    return df


# 3.  home / away
def add_home_away_column(df):
    df = df.copy()

    team_lookup = (
        df[["gameEvents.teamId", "gameEvents.homeTeam", "gameEvents.teamName"]]
        .dropna(subset=["gameEvents.teamId"])
        .drop_duplicates()
        .rename(columns={
            "gameEvents.teamId": "team_id",
            "gameEvents.homeTeam": "is_home",
            "gameEvents.teamName": "team_name"
        })
    )

    team_lookup = team_lookup.drop_duplicates(subset=["team_id"])

    team_lookup["home_away"] = np.where(
        team_lookup["is_home"] == True,
        "home",
        "away"
    )

    df = df.merge(
        team_lookup[["team_id", "home_away", "team_name"]],
        left_on="defending_team",
        right_on="team_id",
        how="left"
    )

    df = df.drop(columns=["team_id"])

    return df


# 4. player summary
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
        "gameEvents.teamName",
    ] + METRICS

    missing_cols = [col for col in required_cols if col not in df_match.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df_match.dropna(subset=[
        "match_id",
        "defending_team",
        "defender_id",
        "defender_name",
        "defender_x",
        "defender_y",
        "section"
    ]).copy()

    df = align_coordinates(df)  # first or second half

    df = add_home_away_column(df)  # home/away

    group_cols = [
        "match_id",
        "defending_team",
        "team_name",
        "home_away",
        "defender_id",
        "defender_name",
    ]

    # each player one line (one line per match), with their average defensive position and counts for each metric
    result = (
        df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    # overall
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

    # metrics individually
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


# 5. all matches
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

        df_match = defensive_network.parse.drive.download_parquet_from_drive(full_path)
        player_summary = build_player_summary_for_match(df_match)
        all_player_summaries.append(player_summary)

    final_df = pd.concat(all_player_summaries, ignore_index=True)
    final_df = final_df.sort_values(
        ["match_id", "defending_team", "defender_id"]
    ).reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Saved to: {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    main()