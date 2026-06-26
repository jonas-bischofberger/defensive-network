import sys, os, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
METRICS = ["raw_involvement", "valued_involvement", "raw_fault", "valued_fault", "raw_contribution",
           "valued_contribution"]
OUTPUT = "scripts/2026-05-01_player_defensive_network_edges.csv"

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_rows = []

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue

    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    df["receiver_name"] = df["receiverPlayerName"].fillna(df["expected_receiver_name"])
    match_id= int(df["match_id"].iloc[0])
    match_name = f["name"].replace(".parquet", "")
    print(f"Processing: {match_name}")

    for (defending_team, defender_name), df_def in df.groupby(["defending_team", "defender_name"]):
        df_inv = df_def[df_def["raw_involvement"].fillna(0) > 0]
        if df_inv.empty:
            continue

        active_metrics = [m for m in METRICS if m in df_inv.columns]
        agg_dict = {
            "n_passes":   ("raw_involvement", "count"),
            "passer_x":   ("x_event", "mean"),
            "passer_y":   ("y_event", "mean"),
            "receiver_x": ("x_target", "mean"),
            "receiver_y": ("y_target", "mean"),
            **{m:              (m, "sum")  for m in active_metrics},
            **{f"{m}_avg":     (m, "mean") for m in active_metrics},
        }

        edges = (df_inv
                 .groupby(["player_name_1", "receiver_name"])
                 .agg(**agg_dict)
                 .reset_index()
                 .rename(columns={"player_name_1": "passer_name"}))

        edges.insert(0, "defender_name",  defender_name)
        edges.insert(0, "defending_team", int(defending_team))
        edges.insert(0, "match_name",     match_name)
        edges.insert(0, "match_id",       match_id)
        all_rows.append(edges)

pd.concat(all_rows, ignore_index=True).to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
