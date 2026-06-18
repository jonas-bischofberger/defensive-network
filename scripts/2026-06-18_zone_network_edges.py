"""
Zone-filtered shared defensive network edges.

For each edge-weight method and each pitch zone (thirds scheme, same definitions
as 2026-06-08_team_zone_metrics.py) compute player-pair edges and save:

    scripts/2026-06-18_zone_network_edge(average).csv
    scripts/2026-06-18_zone_network_edge(min).csv
    scripts/2026-06-18_zone_network_edge(product).csv
    scripts/2026-06-18_zone_network_edge(sum).csv

Each file has the same schema as the full-match edge files plus a `zone` column.

Zone definitions — ball position from the *defending* team's perspective:
    x_def = -x_norm   (positive = ball deep in opponent's half = high press)
    own        : x_def < -17.5
    mid        : -17.5 <= x_def < 17.5
    high_press : x_def >= 17.5
"""
import sys, os
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
HERE = os.path.dirname(__file__)

METRICS = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
]

ZONES = [
    ("own",        -np.inf, -17.5),
    ("mid",        -17.5,   17.5),
    ("high_press",  17.5,   np.inf),
]

METHODS = {
    "average": lambda va, vb: (va + vb) / 2.0,
    "min":     lambda va, vb: min(va, vb),
    "product": lambda va, vb: va * vb,
    "sum":     lambda va, vb: va + vb,
}

EDGE_KEYS = ["match_id", "match_name", "defending_team", "zone", "player_1", "player_2"]

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
parquet_files = [
    f for f in files
    if MATCH_FILTER in f["name"] and f["name"].endswith(".parquet")
]
print(f"Found {len(parquet_files)} parquet files")

for method_name, agg_fn in METHODS.items():
    print(f"\n=== Method: {method_name} ===")
    all_edge_tables = []

    for f in parquet_files:
        print(f"  {f['name']}")
        df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
        df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
        df = df.dropna(subset=["x_norm", "defending_team", "involvement_pass_id"])
        if df.empty:
            continue

        match_id = int(df["match_id"].iloc[0])
        match_name = f["name"].replace(".parquet", "")
        # x_norm is normalized for regulation (sections 1 & 2 share one frame) but
        # extra time (sections 3 & 4) is mirrored, so re-align ET before deriving
        # x_def. NB: this is a *different* convention than defender_x, which flips
        # every period — so here we negate only sections 3 & 4, not even sections.
        x_norm_aligned = np.where(df["section"].fillna(1).astype(int) >= 3, -df["x_norm"], df["x_norm"])
        df["x_def"] = -x_norm_aligned

        metric_edge_tables = []

        for metric in METRICS:
            if metric not in df.columns:
                continue
            df_metric = df[df[metric].fillna(0) != 0].copy()
            rows = []

            for zone_name, lo, hi in ZONES:
                df_zone = df_metric[(df_metric["x_def"] >= lo) & (df_metric["x_def"] < hi)]

                for defending_team in df_zone["defending_team"].dropna().unique():
                    dt = int(defending_team)
                    df_team = df_zone[df_zone["defending_team"] == defending_team]
                    edge_dict = {}

                    for pass_id, df_pass in df_team.groupby("involvement_pass_id"):
                        defenders = sorted(df_pass["defender_name"].tolist())
                        if len(defenders) < 2:
                            continue
                        player_vals = df_pass.set_index("defender_name")[metric].to_dict()

                        for a, b in combinations(defenders, 2):
                            key = (match_id, match_name, dt, zone_name, a, b)
                            if key not in edge_dict:
                                edge_dict[key] = {
                                    "match_id": match_id,
                                    "match_name": match_name,
                                    "defending_team": dt,
                                    "zone": zone_name,
                                    "player_1": a,
                                    "player_2": b,
                                    f"{metric}_edge_count": 0,
                                    metric: 0.0,
                                }
                            edge_dict[key][f"{metric}_edge_count"] += 1
                            val_a = player_vals.get(a, 0.0)
                            val_b = player_vals.get(b, 0.0)
                            edge_dict[key][metric] += agg_fn(val_a, val_b)

                    rows.extend(edge_dict.values())

            if rows:
                metric_edge_tables.append(pd.DataFrame(rows))

        if not metric_edge_tables:
            continue

        all_edges = (
            pd.concat([t[EDGE_KEYS] for t in metric_edge_tables], ignore_index=True)
            .drop_duplicates()
            .sort_values(EDGE_KEYS)
            .reset_index(drop=True)
        )
        edge_table = all_edges.copy()
        for mdf in metric_edge_tables:
            extra_cols = [c for c in mdf.columns if c not in EDGE_KEYS]
            edge_table = edge_table.merge(mdf[EDGE_KEYS + extra_cols], on=EDGE_KEYS, how="left")
        all_edge_tables.append(edge_table)

    if all_edge_tables:
        final_df = pd.concat(all_edge_tables, ignore_index=True)
        out = os.path.join(HERE, f"2026-06-18_zone_network_edge({method_name}).csv")
        final_df.to_csv(out, index=False)
        print(f"  -> Saved {out} ({len(final_df):,} rows)")

# ── Zone-specific player positions ────────────────────────────────────────────
# Compute per-zone average defender (x, y) for each metric, mirroring how
# 2026-03-30_player_position.py builds the full-match player_df.
# Only passes that fall within each zone are included in the averages.

print("\n=== Zone positions ===")
POS_KEYS = ["match_id", "defending_team", "defender_name", "zone"]
pos_tables = []

for f in parquet_files:
    print(f"  {f['name']}")
    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    df = df.dropna(subset=["x_norm", "defending_team", "defender_name", "defender_x", "defender_y", "section"])
    if df.empty:
        continue

    match_id = int(df["match_id"].iloc[0])
    df["match_id"] = match_id
    df["defending_team"] = df["defending_team"].astype(int)
    # see note above: x_norm is aligned for regulation but mirrored in extra time
    x_norm_aligned = np.where(df["section"].fillna(1).astype(int) >= 3, -df["x_norm"], df["x_norm"])
    df["x_def"] = -x_norm_aligned

    # align coordinates across halves (mirrors 2026-03-30_player_position.py)
    # odd sections (1, 3) share one orientation, even sections (2, 4) the other;
    # flip x on even sections and y on odd sections so extra time aligns too.
    df["x"] = np.where(df["section"] % 2 == 0, -df["defender_x"], df["defender_x"])
    df["y"] = np.where(df["section"] % 2 == 1, -df["defender_y"], df["defender_y"])

    for zone_name, lo, hi in ZONES:
        df_zone = df[(df["x_def"] >= lo) & (df["x_def"] < hi)].copy()
        if df_zone.empty:
            continue

        group_cols = ["match_id", "defending_team", "defender_name"]

        overall = (
            df_zone.groupby(group_cols)
            .agg(overall_avg_x=("x", "mean"), overall_avg_y=("y", "mean"))
            .reset_index()
        )
        overall["zone"] = zone_name

        for metric in METRICS:
            if metric not in df_zone.columns:
                continue
            sub = df_zone[df_zone[metric].fillna(0) != 0]
            if sub.empty:
                continue
            agg = (
                sub.groupby(group_cols)
                .agg(**{f"{metric}_avg_x": ("x", "mean"), f"{metric}_avg_y": ("y", "mean")})
                .reset_index()
            )
            overall = overall.merge(agg, on=group_cols, how="left")

        pos_tables.append(overall)

if pos_tables:
    zone_pos_df = pd.concat(pos_tables, ignore_index=True)
    out = os.path.join(HERE, "2026-06-18_zone_network_positions.csv")
    zone_pos_df.to_csv(out, index=False)
    print(f"  -> Saved {out} ({len(zone_pos_df):,} rows)")
