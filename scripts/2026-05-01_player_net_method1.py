import sys, os, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
METRICS = ["raw_involvement", "valued_involvement", "raw_fault", "valued_fault", "raw_contribution",
           "valued_contribution"]
HERE = os.path.dirname(__file__)

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_edges, all_nodes = [], []

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue
    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    match_id, match_name = int(df["match_id"].iloc[0]), f["name"].replace(".parquet", "")
    print(f"Processing: {match_name}")

    for (defending_team, defender_name), df_def in df.groupby(["defending_team", "defender_name"]):
        df_inv = df_def[df_def["raw_involvement"].fillna(0) > 0].copy()
        if df_inv.empty:
            continue

        defender_id = df_inv["defender_id"].iloc[0]
        active = [m for m in METRICS if m in df_inv.columns]
        meta = dict(match_id=match_id, match_name=match_name, defending_team=int(defending_team),
                    defender_name=defender_name, defender_id=defender_id)

        # avg passer position across ALL C+B+D passes this defender was involved in
        passer_pos = df_inv.groupby("player_name_1").agg(
            passer_x=("x_norm", "mean"), passer_y=("y_norm", "mean"),
            passer_id=("player_id_1", "first"))

        # EDGES — C passes: passer → receiver, weighted by involvement
        df_C = df_inv[df_inv["possessionEvents.passOutcomeType"] == "C"]
        if not df_C.empty:
            e = (df_C.groupby(["player_name_1", "player_name_2"])
                 .agg(n_passes=("raw_involvement", "count"),
                      receiver_x=("x_target_norm", "mean"), receiver_y=("y_target_norm", "mean"),
                      receiver_id=("player_id_2", "first"),
                      **{m: (m, "sum") for m in active},
                      **{f"{m}_avg": (m, "mean") for m in active})
                 .reset_index()
                 .rename(columns={"player_name_1": "passer_name", "player_name_2": "receiver_name"})
                 .join(passer_pos[["passer_x", "passer_y", "passer_id"]], on="passer_name")
                 .assign(**meta))
            all_edges.append(e)

        # NODES — B/D passes: passer only, involvement = node size
        df_BD = df_inv[df_inv["possessionEvents.passOutcomeType"].isin(["B", "D"])]
        if not df_BD.empty:
            n = (df_BD.groupby("player_name_1")
                 .agg(n_passes=("raw_involvement", "count"),
                      passer_id=("player_id_1", "first"),
                      **{m: (m, "sum") for m in active},
                      **{f"{m}_avg": (m, "mean") for m in active})
                 .reset_index()
                 .rename(columns={"player_name_1": "passer_name"})
                 .join(passer_pos[["passer_x", "passer_y"]], on="passer_name")
                 .assign(**meta))
            all_nodes.append(n)

for data, name in [(all_edges, "m1_edges"), (all_nodes, "m1_nodes")]:
    pd.concat(data, ignore_index=True).to_csv(os.path.join(HERE, f"2026-05-01_player_net_{name}.csv"), index=False)
print("Done")
