import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
MATCH_LEVEL_FILE = "2026-04-24_match_level_metrics.csv"

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)

rows = []
for f in files:
    file_name = f["name"]
    if MATCH_FILTER and MATCH_FILTER not in file_name:
        continue
    if not file_name.endswith(".parquet"):
        continue
    print(f"Processing: {file_name}")

    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + file_name)

    match_id = int(df["match_id"].iloc[0])

    passes_per_team = (
        df.groupby("defending_team")["involvement_pass_id"]
        .nunique()
        .reset_index()
        .rename(columns={"involvement_pass_id": "passes_against"})
    )
    passes_per_team["match_id"] = match_id
    passes_per_team["match_team_id"] = (
        passes_per_team["match_id"].astype(str) + "_" +
        passes_per_team["defending_team"].astype(int).astype(str)
    )
    rows.append(passes_per_team)

passes_df = pd.concat(rows, ignore_index=True)[["match_team_id", "passes_against"]]

outcomes = pd.read_csv(MATCH_LEVEL_FILE)
outcomes = outcomes.merge(passes_df, on="match_team_id", how="left")
outcomes.to_csv(MATCH_LEVEL_FILE, index=False)

print("Done. passes_against added to", MATCH_LEVEL_FILE)
print(outcomes[["match_team_id", "passes_against"]].head(10))
