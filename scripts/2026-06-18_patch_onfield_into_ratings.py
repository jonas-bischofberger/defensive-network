"""
Patch the on-field opportunity-adjusted columns straight into the existing
app dataset, WITHOUT re-running FIFA name matching.

Why this exists
---------------
The canonical way to add the new normalization is to re-run:
  2026-05-23_build_enriched_player_dataset.py   (adds the columns)
  2026-05-25_match_fifa_ratings.py              (re-attaches FIFA ratings)

Re-running the second step reproduces the *identical* name match (it keys off
defender_name / country, which we don't touch), so it is safe. But to avoid
touching the matching step at all, this script simply merges
`passes_against_on_field` into the file the app already reads and computes the
new per-100 columns from the raw metric columns that are already there.

Idempotent: re-running just overwrites the same columns.

Run AFTER 2026-06-18_passes_against_onfield.py has produced its CSV.
"""
import pandas as pd

HERE          = "/Users/runqingma/学习/Projectsss/PHD/defensive-network/scripts/"
RATINGS_FILE  = HERE + "2026-05-25_player_level_enriched_with_ratings.csv"
ONFIELD_FILE  = HERE + "2026-06-18_passes_against_onfield.csv"

METRICS = [
    "raw_involvement", "valued_involvement",
    "raw_fault",       "valued_fault",
    "raw_contribution","valued_contribution",
]

df      = pd.read_csv(RATINGS_FILE)
onfield = pd.read_csv(ONFIELD_FILE)[["match_id", "defender_id", "passes_against_on_field"]]

# Drop any prior patch columns so re-running is clean.
patch_cols = (["passes_against_on_field"]
              + [f"{m}_per_pass_against_onfield" for m in METRICS]
              + ["passes_defended_per_pass_against_onfield"])
df = df.drop(columns=[c for c in patch_cols if c in df.columns])

df = df.merge(onfield, on=["match_id", "defender_id"], how="left")
print(f"passes_against_on_field joined: "
      f"{df['passes_against_on_field'].notna().sum()} / {len(df)}")

for m in METRICS:
    df[f"{m}_per_pass_against_onfield"] = df[m] / df["passes_against_on_field"] * 100
df["passes_defended_per_pass_against_onfield"] = (
    df["passes_defended"] / df["passes_against_on_field"] * 100
)

df.to_csv(RATINGS_FILE, index=False)
print(f"Patched {len(df)} rows -> {RATINGS_FILE}")
print(df[["defender_name", "mins_played",
          "passes_against", "passes_against_on_field",
          "raw_involvement_per_pass_against",
          "raw_involvement_per_pass_against_onfield"]].head(8).round(3).to_string())
