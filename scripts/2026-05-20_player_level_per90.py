"""
Build a per-90 player-level dataset from Method 2 edges + playing minutes.

Output: scripts/2026-05-20_player_level_per90.csv
  One row per (match_id, defender_id). All players retained (no mins filter
  here — apply a threshold in downstream analysis as needed).

Columns:
  match_id / match_name / defending_team / defending_team_name /
  defender_id / defender_name / match_team_id / competition_stage /
  mins_played / starter
  raw_involvement, valued_involvement,
  raw_fault,       valued_fault,
  raw_contribution,valued_contribution,
  passes_defended          -- count of opposing passes this defender was
                              involved in defending (NOT the defender's
                              own passes)
  *_per90                  -- each metric normalised to per-90 minutes
"""
import pandas as pd

HERE = "scripts/"

EDGE_FILE   = HERE + "2026-05-05_player_net_m2_edges.csv"
MINS_FILE   = HERE + "2026-05-06_player_minutes.csv"
PLAYER_FILE = HERE + "2026-04-29-player_level_metrics.csv"
OUTPUT_FILE = HERE + "2026-05-20_player_level_per90.csv"

METRICS = [
    "raw_involvement", "valued_involvement",
    "raw_fault",       "valued_fault",
    "raw_contribution","valued_contribution",
]

# ── 1. Load & aggregate M2 edges per defender per match ──────────────────────
edges = pd.read_csv(EDGE_FILE)

group_cols = ["match_id", "match_name", "defending_team",
              "defending_team_name", "defender_id", "defender_name"]

agg = (edges.groupby(group_cols)[METRICS + ["n_passes"]]
       .sum()
       .reset_index()
       .rename(columns={"n_passes": "passes_defended"}))

# ── 2. Join playing minutes ───────────────────────────────────────────────────
mins = pd.read_csv(MINS_FILE)[
    ["match_id", "defending_team", "defender_id", "mins_played"]
]
df = agg.merge(mins, on=["match_id", "defending_team", "defender_id"], how="left")

# ── 3. Join starter flag & competition stage ──────────────────────────────────
player = pd.read_csv(PLAYER_FILE)[
    ["match_id", "defending_team", "defender_id", "starter"]
].drop_duplicates()
df = df.merge(player, on=["match_id", "defending_team", "defender_id"], how="left")

df["match_team_id"] = df["match_id"].astype(str) + "_" + df["defending_team"].astype(str)

# ── 4. Per-90 normalisation (all players kept; filter downstream if needed) ───
for m in METRICS + ["passes_defended"]:
    df[f"{m}_per90"] = df[m] / df["mins_played"] * 90

# ── 5. Save ───────────────────────────────────────────────────────────────────
col_order = (group_cols + ["match_team_id", "mins_played", "starter"]
             + METRICS + ["passes_defended"]
             + [f"{m}_per90" for m in METRICS] + ["passes_defended_per90"])
df = df[col_order]

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} rows -> {OUTPUT_FILE}")
print(f"  unique players : {df['defender_id'].nunique()}")
print(f"  unique matches : {df['match_id'].nunique()}")
print(f"  mins range     : {df['mins_played'].min():.0f} - {df['mins_played'].max():.0f}")
print()
print("Per-90 metric summary (all players):")
per90_cols = [f"{m}_per90" for m in METRICS]
print(df[per90_cols].describe().round(3).to_string())
