import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER       = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
METRICS      = ["raw_involvement", "valued_involvement", "raw_fault", "valued_fault",
                "raw_contribution", "valued_contribution"]
HERE         = os.path.dirname(__file__)

# ── Reference data ─────────────────────────────────────────────────────────────
meta = pd.read_csv(os.path.join(HERE, "meta_worldcup.csv"))
match_id_2_name = dict(zip(meta["match_id"],
                           meta["home_team_name"] + " vs " + meta["guest_team_name"]))
team_name_lookup = {}
for _, r in meta.iterrows():
    team_name_lookup[int(r["home_team_id"])] = r["home_team_name"]
    team_name_lookup[int(r["guest_team_id"])] = r["guest_team_name"]

# starter: (match_id, defender_id) → 0/1
pl = pd.read_csv(os.path.join(HERE, "2026-04-29-player_level_metrics.csv"))
starter_lookup = pl.set_index(["match_id", "defender_id"])["starter"].to_dict()

# ── Main loop ──────────────────────────────────────────────────────────────────
files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_nodes, all_edges = [], []

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue

    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    match_id   = int(df["match_id"].iloc[0])
    match_name = match_id_2_name.get(match_id, f["name"].replace(".parquet", ""))
    print(f"Processing: {match_name}")

    # Unify receiver columns across all pass types
    # C: actual receiver; B/D: expected receiver
    df["receiver_name"] = df["player_name_2"]
    df["receiver_id"]   = df["player_id_2"]
    df["receiver_x"]    = df["x_target_norm"]
    df["receiver_y"]    = df["y_target_norm"]

    is_bd = df["possessionEvents.passOutcomeType"].isin(["B", "D"])
    df.loc[is_bd, "receiver_name"] = df.loc[is_bd, "expected_receiver_name"]
    df.loc[is_bd, "receiver_id"]   = df.loc[is_bd, "expected_receiver"]

    for (defending_team, defender_name), df_def in df.groupby(["defending_team", "defender_name"]):
        df_inv = df_def[df_def["raw_involvement"].fillna(0) > 0].copy()
        if df_inv.empty:
            continue

        defending_team      = int(defending_team)
        defender_id         = int(df_inv["defender_id"].iloc[0])
        defending_team_name = team_name_lookup.get(defending_team, str(defending_team))
        active              = [m for m in METRICS if m in df_inv.columns]

        meta_dict = dict(
            match_id=match_id, match_name=match_name,
            defending_team=defending_team, defending_team_name=defending_team_name,
            defender_id=defender_id, defender_name=defender_name,
        )

        # ── Player positions ─────────────────────────────────────────────────
        # As passer (C + B + D): x_norm / y_norm
        passer_pos = (df_inv[["player_name_1", "player_id_1", "x_norm", "y_norm"]]
                      .rename(columns={"player_name_1": "player_name",
                                       "player_id_1":   "player_id",
                                       "x_norm": "x", "y_norm": "y"}))

        # As receiver (C + B/D expected): x_target_norm / y_target_norm
        recv_pos = (df_inv[["receiver_name", "receiver_id", "receiver_x", "receiver_y"]]
                    .dropna(subset=["receiver_name"])
                    .rename(columns={"receiver_name": "player_name",
                                     "receiver_id":   "player_id",
                                     "receiver_x":    "x",
                                     "receiver_y":    "y"}))

        pos_all = pd.concat([passer_pos, recv_pos], ignore_index=True).dropna(subset=["x", "y"])
        pos_agg = (pos_all.groupby("player_name")
                   .agg(player_x=("x", "mean"),
                        player_y=("y", "mean"),
                        player_id=("player_id", "first"))
                   .reset_index())

        # ── Node table ───────────────────────────────────────────────────────
        nodes = pos_agg.copy()
        nodes["starter"] = nodes["player_id"].apply(
            lambda pid: starter_lookup.get((match_id, int(pid)), np.nan)
            if pd.notna(pid) else np.nan
        )
        for col, val in reversed(list(meta_dict.items())):
            nodes.insert(0, col, val)
        all_nodes.append(nodes)

        # ── Edge table (all pass types) ──────────────────────────────────────
        df_with_recv = df_inv.dropna(subset=["receiver_name"]).copy()
        if df_with_recv.empty:
            continue

        agg_dict = {"n_passes": ("raw_involvement", "count")}
        for m in active:
            agg_dict[m]          = (m, "sum")
            agg_dict[f"{m}_avg"] = (m, "mean")

        edges = (df_with_recv
                 .groupby(["player_name_1", "player_id_1", "receiver_name", "receiver_id"])
                 .agg(**agg_dict)
                 .reset_index()
                 .rename(columns={"player_name_1": "passer_name", "player_id_1": "passer_id",
                                  "receiver_name": "receiver_name", "receiver_id": "receiver_id"}))
        for col, val in reversed(list(meta_dict.items())):
            edges.insert(0, col, val)
        all_edges.append(edges)

pd.concat(all_nodes, ignore_index=True).to_csv(
    os.path.join(HERE, "2026-05-05_player_net_m2_nodes.csv"), index=False)
pd.concat(all_edges, ignore_index=True).to_csv(
    os.path.join(HERE, "2026-05-05_player_net_m2_edges.csv"), index=False)
print("Done")
