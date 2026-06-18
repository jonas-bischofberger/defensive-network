"""
Targeted regeneration of the zone-network CSVs for the extra-time matches only,
after fixing the x_norm extra-time mirroring (zone classification) and the
defender_x/y section alignment.

Only matches that reached extra time (sections 3/4) are affected; regulation
rows and non-ET matches were already correct. We recompute the ET matches with
the fixed logic (identical to 2026-06-18_zone_network_edges.py) and replace just
those match_ids' rows in each zone CSV.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

import defensive_network.parse.drive as drive

HERE = os.path.dirname(__file__)
FOLDER = "involvement/10/"

# the three matches that carry extra-time data in this dataset
ET_FILES = [
    "fifa-men-s-world-cup-2022-4-st-japan-croatia.parquet",
    "fifa-men-s-world-cup-2022-4-st-morocco-spain.parquet",
    "fifa-men-s-world-cup-2022-8-st-argentina-france.parquet",
]

METRICS = ["raw_involvement", "raw_fault", "raw_contribution",
           "valued_involvement", "valued_contribution", "valued_fault"]
ZONES = [("own", -np.inf, -17.5), ("mid", -17.5, 17.5), ("high_press", 17.5, np.inf)]
METHODS = {
    "average": lambda va, vb: (va + vb) / 2.0,
    "min":     lambda va, vb: min(va, vb),
    "product": lambda va, vb: va * vb,
    "sum":     lambda va, vb: va + vb,
}
EDGE_KEYS = ["match_id", "match_name", "defending_team", "zone", "player_1", "player_2"]
POS_GROUP = ["match_id", "defending_team", "defender_name"]


def aligned_x_def(df):
    # x_norm: aligned for regulation, mirrored in ET -> negate sections 3/4
    xn = np.where(df["section"].fillna(1).astype(int) >= 3, -df["x_norm"], df["x_norm"])
    return -xn


def regen_edges(dfs):
    out = {m: [] for m in METHODS}
    for name, df0 in dfs.items():
        match_name = name.replace(".parquet", "")
        df = df0[df0["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
        df = df.dropna(subset=["x_norm", "defending_team", "involvement_pass_id"])
        if df.empty:
            continue
        match_id = int(df["match_id"].iloc[0])
        df["defending_team"] = df["defending_team"].astype(int)
        df["x_def"] = aligned_x_def(df)
        for method_name, agg_fn in METHODS.items():
            metric_tables = []
            for metric in METRICS:
                if metric not in df.columns:
                    continue
                dm = df[df[metric].fillna(0) != 0].copy()
                rows = []
                for zone_name, lo, hi in ZONES:
                    dz = dm[(dm["x_def"] >= lo) & (dm["x_def"] < hi)]
                    for dt in dz["defending_team"].dropna().unique():
                        dt = int(dt)
                        dteam = dz[dz["defending_team"] == dt]
                        ed = {}
                        for _, dp in dteam.groupby("involvement_pass_id"):
                            defs = sorted(dp["defender_name"].tolist())
                            if len(defs) < 2:
                                continue
                            vals = dp.set_index("defender_name")[metric].to_dict()
                            for a, b in combinations(defs, 2):
                                key = (match_id, match_name, dt, zone_name, a, b)
                                if key not in ed:
                                    ed[key] = {"match_id": match_id, "match_name": match_name,
                                               "defending_team": dt, "zone": zone_name,
                                               "player_1": a, "player_2": b,
                                               f"{metric}_edge_count": 0, metric: 0.0}
                                ed[key][f"{metric}_edge_count"] += 1
                                ed[key][metric] += agg_fn(vals.get(a, 0.0), vals.get(b, 0.0))
                        rows.extend(ed.values())
                if rows:
                    metric_tables.append(pd.DataFrame(rows))
            if not metric_tables:
                continue
            base = (pd.concat([t[EDGE_KEYS] for t in metric_tables], ignore_index=True)
                    .drop_duplicates().sort_values(EDGE_KEYS).reset_index(drop=True))
            for mdf in metric_tables:
                extra = [c for c in mdf.columns if c not in EDGE_KEYS]
                base = base.merge(mdf[EDGE_KEYS + extra], on=EDGE_KEYS, how="left")
            out[method_name].append(base)
    return {m: pd.concat(v, ignore_index=True) for m, v in out.items() if v}


def regen_positions(dfs):
    tables = []
    for name, df0 in dfs.items():
        df = df0[df0["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
        df = df.dropna(subset=["x_norm", "defending_team", "defender_name", "defender_x", "defender_y", "section"])
        if df.empty:
            continue
        match_id = int(df["match_id"].iloc[0])
        df["match_id"] = match_id
        df["defending_team"] = df["defending_team"].astype(int)
        df["x_def"] = aligned_x_def(df)
        # defender position: flips every period -> flip x on even, y on odd sections
        df["x"] = np.where(df["section"] % 2 == 0, -df["defender_x"], df["defender_x"])
        df["y"] = np.where(df["section"] % 2 == 1, -df["defender_y"], df["defender_y"])
        for zone_name, lo, hi in ZONES:
            dz = df[(df["x_def"] >= lo) & (df["x_def"] < hi)].copy()
            if dz.empty:
                continue
            overall = (dz.groupby(POS_GROUP)
                       .agg(overall_avg_x=("x", "mean"), overall_avg_y=("y", "mean")).reset_index())
            overall["zone"] = zone_name
            for metric in METRICS:
                if metric not in dz.columns:
                    continue
                sub = dz[dz[metric].fillna(0) != 0]
                if sub.empty:
                    continue
                agg = (sub.groupby(POS_GROUP)
                       .agg(**{f"{metric}_avg_x": ("x", "mean"), f"{metric}_avg_y": ("y", "mean")}).reset_index())
                overall = overall.merge(agg, on=POS_GROUP, how="left")
            tables.append(overall)
    return pd.concat(tables, ignore_index=True)


def patch_csv(path, new_rows, et_ids):
    df = pd.read_csv(path)
    bak = path + ".bak"
    if not os.path.exists(bak):
        df.to_csv(bak, index=False)
    kept = df[~df["match_id"].isin(et_ids)]
    new_rows = new_rows.reindex(columns=df.columns)
    out = pd.concat([kept, new_rows], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"  {os.path.basename(path)}: {len(df)} -> {len(out)} rows "
          f"(replaced {int(df['match_id'].isin(et_ids).sum())} ET rows with {len(new_rows)})", flush=True)


def main():
    print("Downloading ET matches...", flush=True)
    dfs = {f: drive.download_parquet_from_drive(FOLDER + f) for f in ET_FILES}
    et_ids = sorted({int(d["match_id"].iloc[0]) for d in dfs.values()})
    print("ET match_ids:", et_ids, flush=True)

    print("Regenerating edges...", flush=True)
    edges = regen_edges(dfs)
    for method, rows in edges.items():
        patch_csv(os.path.join(HERE, f"2026-06-18_zone_network_edge({method}).csv"), rows, et_ids)

    print("Regenerating positions...", flush=True)
    pos = regen_positions(dfs)
    patch_csv(os.path.join(HERE, "2026-06-18_zone_network_positions.csv"), pos, et_ids)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
