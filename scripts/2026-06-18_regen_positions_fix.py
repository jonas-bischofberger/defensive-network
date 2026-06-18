"""
Regenerate corrected player positions after the section-coordinate fix
(align_coordinates now flips x on even sections and y on odd sections, so
extra-time sections 3 & 4 are aligned too).

Recomputes full-match positions for every WC 2022 match using the fixed
build_player_summary_for_match(), then patches ONLY the averaged-coordinate
columns (overall_avg_x/y and <metric>_avg_x/y) into the consumed node-level
CSVs, keyed on (match_id, defending_team, defender_id). Counts and all other
columns (mins_played, responsibility, game state) are left untouched, since
the coordinate fix changes only where averaged positions land, not which
events exist. Non-ET matches are unaffected and serve as a built-in check.
"""
import importlib.util
import os

import pandas as pd

import defensive_network.parse.drive as drive

HERE = os.path.dirname(__file__)
FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"

# Load the fixed builder from the hyphenated script file.
_spec = importlib.util.spec_from_file_location(
    "player_position", os.path.join(HERE, "2026-03-30_player_position.py")
)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

CONSUMED = [
    os.path.join(HERE, "2026-05-06_node_level_metrics_with_mins.csv"),
    os.path.join(HERE, "2026-06-07_node_level_metrics_with_gs.csv"),
]

KEYS = ["match_id", "defending_team", "defender_id"]
POS_COLS = ["overall_avg_x", "overall_avg_y"] + [
    f"{m}_{ax}" for m in pp.METRICS for ax in ("avg_x", "avg_y")
]


def _round_of(name):
    return name.split("fifa-men-s-world-cup-2022-")[1].split("-st-")[0]


def build_corrected_positions():
    # The coordinate bug only changes positions for matches that reach extra
    # time (sections 3/4). Group-stage (rounds 1-3) and league matches never do,
    # so only knockout rounds are candidates. We still confirm sections 3/4 are
    # actually present per match rather than assuming which games went to ET.
    files = drive.list_files_in_drive_folder(FOLDER)
    candidates = [
        f["name"] for f in files
        if MATCH_FILTER in f["name"] and f["name"].endswith(".parquet")
        and _round_of(f["name"]) not in ("1", "2", "3")
    ]
    parts = []
    for name in sorted(candidates):
        dfm = drive.download_parquet_from_drive(FOLDER + name)
        sections = set(pd.to_numeric(dfm["section"], errors="coerce").dropna().astype(int))
        has_et = bool(sections & {3, 4})
        print(f"  {name}: sections={sorted(sections)} -> {'EXTRA TIME, rebuilding' if has_et else 'skip'}", flush=True)
        if has_et:
            parts.append(pp.build_player_summary_for_match(dfm))
    out = pd.concat(parts, ignore_index=True)
    for k in KEYS:
        out[k] = pd.to_numeric(out[k], errors="coerce")
    return out


def patch_file(path, corrected):
    if not os.path.exists(path):
        print(f"  SKIP (missing): {path}", flush=True)
        return
    df = pd.read_csv(path)
    orig = df.copy()
    for k in KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    pos = corrected[KEYS + POS_COLS].drop_duplicates(KEYS)
    merged = df.merge(pos, on=KEYS, how="left", suffixes=("", "_new"))
    for c in POS_COLS:
        new = c + "_new"
        if new in merged.columns:
            merged[c] = merged[new].combine_first(merged[c])
    merged = merged[df.columns]

    # report changed rows
    cmp_cols = [c for c in POS_COLS if c in orig.columns]
    a = orig[KEYS + cmp_cols].set_index(KEYS).sort_index()
    b = merged.set_index(KEYS)[cmp_cols].sort_index()
    a, b = a.align(b, join="inner")
    changed = (~((a - b).abs() < 1e-6) & ~(a.isna() & b.isna())).any(axis=1)
    n_changed = int(changed.sum())
    changed_matches = sorted(a.index[changed].get_level_values("match_id").unique().tolist())

    bak = path + ".bak"
    if not os.path.exists(bak):
        orig.to_csv(bak, index=False)
        print(f"  backup -> {bak}", flush=True)
    merged.to_csv(path, index=False)
    print(f"  wrote {path}: {n_changed} rows changed across matches {changed_matches}", flush=True)


def main():
    print("Building corrected positions for all WC matches...", flush=True)
    corrected = build_corrected_positions()
    corrected.to_csv(os.path.join(HERE, "2026-06-18_player_positions_fixed.csv"), index=False)
    print(f"Corrected positions: {len(corrected)} rows, {corrected['match_id'].nunique()} matches", flush=True)
    for path in CONSUMED:
        print(f"Patching {os.path.basename(path)}", flush=True)
        patch_file(path, corrected)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
