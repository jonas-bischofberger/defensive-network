"""
Build per-(match, defender) "passes against while on field".

Motivation
----------
The existing `passes_against` (from match_level_metrics) is a TEAM-LEVEL,
full-match opponent pass count: every player on a team in a given match
shares the same value, regardless of how long they were on the pitch.
A substitute who played 20 minutes is divided by the same denominator as a
90-minute starter — so `*_per_pass_against` understates subs' rates.

This script computes a player-specific opportunity denominator:
for each defender, the number of opponent passes that happened *while that
player was actually on the pitch*.

Method
------
The raw PFF involvement data (involvement/10/) is exploded to one row per
(pass × defending player): for every opponent pass, the model emits exactly
one row for each of the 11 defending players on the pitch at that moment.
So the rows for a given `involvement_pass_id` ARE the on-field defending XI
for that pass.

Therefore the number of opponent passes a defender faced while on the pitch
is simply the number of distinct `involvement_pass_id` values they appear in.
Substitutions are handled automatically: a player only has rows for passes
that happened while they were on the field (a sub who comes on at 70' only
appears from then on; a player subbed off stops appearing).

Pass universe = passOutcomeType in {C, B, D} (complete / blocked / defended),
identical to the m2 edge builder, so this denominator is consistent with the
involvement numerators.

Output
------
scripts/2026-06-18_passes_against_onfield.csv
  columns: match_id, defender_id, defender_name, passes_against_on_field
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER       = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
HERE         = os.path.dirname(__file__)
OUTPUT_FILE  = os.path.join(HERE, "2026-06-18_passes_against_onfield.csv")

files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
all_rows = []

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue

    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])

    # Same pass universe as the m2 edge builder: complete / blocked / defended.
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    if df.empty:
        continue

    match_id = int(df["match_id"].iloc[0])

    # passes_against_on_field = # distinct passes this defender appears in
    # (each pass = the 11 defending players on the pitch at that moment).
    cnt = (df.groupby(["defender_id", "defender_name"])["involvement_pass_id"]
             .nunique()
             .reset_index(name="passes_against_on_field"))
    cnt.insert(0, "match_id", match_id)
    all_rows.append(cnt)

    match_name = f["name"].replace(".parquet", "")
    print(f"Processed {match_name}: {len(cnt)} players, "
          f"{df['involvement_pass_id'].nunique()} unique opponent passes")

out = pd.concat(all_rows, ignore_index=True)
out["defender_id"] = out["defender_id"].astype(int)
out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(out)} rows -> {OUTPUT_FILE}")
print(f"  matches: {out['match_id'].nunique()}  players: {out['defender_id'].nunique()}")
print(f"  passes_against_on_field range: "
      f"{out['passes_against_on_field'].min()} - {out['passes_against_on_field'].max()}")
