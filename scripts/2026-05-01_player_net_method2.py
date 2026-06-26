import sys, os, re, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER       = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
METRICS      = ["raw_involvement", "valued_involvement", "raw_fault", "valued_fault",
                "raw_contribution", "valued_contribution"]
HERE         = os.path.dirname(__file__)


def lookup_xy(players_str, player_id):
    m = re.search(rf"'playerId': {int(player_id)}[^}}]*?'x': ([-\d.]+)[^}}]*?'y': ([-\d.]+)", str(players_str))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_edges = []

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue
    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    match_id, match_name = int(df["match_id"].iloc[0]), f["name"].replace(".parquet", "")
    print(f"Processing: {match_name}")

    # C passes: use actual receiver and normalised coords
    df["receiver_name"] = df["player_name_2"]
    df["receiver_id"]   = df["player_id_2"]
    df["receiver_x"]    = df["x_target_norm"]
    df["receiver_y"]    = df["y_target_norm"]

    # B/D passes: use expected receiver, look up coords from tracking + normalise
    bd = df["possessionEvents.passOutcomeType"].isin(["B", "D"]) & df["expected_receiver"].notna()
    df.loc[bd, ["receiver_name", "receiver_id"]] = df.loc[bd, ["expected_receiver_name", "expected_receiver"]].values
    bd_coords = df[bd].apply(
        lambda r: lookup_xy(r["homePlayers"] if r["gameEvents.homeTeam"] else r["awayPlayers"],
                            r["expected_receiver"]),
        axis=1, result_type="expand").rename(columns={0: "receiver_x", 1: "receiver_y"})
    bd_coords[["receiver_x", "receiver_y"]] *= df.loc[bd, "section"].map({1: 1, 2: -1}).values[:, None]
    df.loc[bd, ["receiver_x", "receiver_y"]] = bd_coords.values

    for (defending_team, defender_name), df_def in df.groupby(["defending_team", "defender_name"]):
        df_inv = df_def[(df_def["raw_involvement"].fillna(0) > 0) & df_def["receiver_name"].notna()].copy()
        if df_inv.empty:
            continue

        active = [m for m in METRICS if m in df_inv.columns]
        meta   = dict(match_id=match_id, match_name=match_name, defending_team=int(defending_team),
                      defender_name=defender_name, defender_id=df_inv["defender_id"].iloc[0])

        passer_pos = df_inv.groupby("player_name_1").agg(
            passer_x=("x_norm", "mean"), passer_y=("y_norm", "mean"),
            passer_id=("player_id_1", "first"))

        e = (df_inv.groupby(["player_name_1", "receiver_name"])
             .agg(n_passes=("raw_involvement", "count"),
                  pass_type=("possessionEvents.passOutcomeType", lambda x: "/".join(x.unique())),
                  receiver_x=("receiver_x", "mean"), receiver_y=("receiver_y", "mean"),
                  receiver_id=("receiver_id", "first"),
                  **{m: (m, "sum") for m in active},
                  **{f"{m}_avg": (m, "mean") for m in active})
             .reset_index()
             .rename(columns={"player_name_1": "passer_name"})
             .join(passer_pos[["passer_x", "passer_y", "passer_id"]], on="passer_name")
             .assign(**meta))
        all_edges.append(e)

pd.concat(all_edges, ignore_index=True).to_csv(os.path.join(HERE, "2026-05-01_player_net_m2_edges.csv"), index=False)
print("Done")
