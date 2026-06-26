"""
Build enriched player-level dataset.

Adds to 2026-05-20_player_level_per90.csv:
  1. Player position  (overall_position, most_common_position)
     -- matched via exact -> fuzzy -> nickname map
  2. passes_against   (team-level opponent passes, from match_level_metrics)
  3. Three normalisations for each metric:
       _per90       : metric / mins_played * 90          (already present, kept)
       _per_pass_against : metric / passes_against * 100 (per 100 opp passes faced by team)
       _per_pass_defended: metric / passes_defended * 100 (per 100 passes this player defended)

Output: scripts/2026-05-23_player_level_enriched.csv
"""
import unicodedata
import pandas as pd
from rapidfuzz import fuzz, process

PER90_FILE    = "scripts/2026-05-20_player_level_per90.csv"
POSITION_FILE = "scripts/player_position2022.csv"
MATCHLEVEL_FILE = "scripts/2026-04-24_match_level_metrics.csv"
NICKNAME_FILE = "scripts/nickname_map.csv"
ONFIELD_FILE  = "scripts/2026-06-18_passes_against_onfield.csv"
OUTPUT_FILE   = "scripts/2026-05-23_player_level_enriched.csv"

FUZZY_THRESHOLD = 85

METRICS = [
    "raw_involvement", "valued_involvement",
    "raw_fault",       "valued_fault",
    "raw_contribution","valued_contribution",
]

# ── Name normalisation (same as existing scripts) ─────────────────────────────
_LIGATURES = str.maketrans({
    "æ": "ae", "Æ": "Ae", "ø": "o",  "Ø": "O",
    "å": "a",  "Å": "A",  "ß": "ss", "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "Oe",
})

def normalise(name: str) -> str:
    name = str(name).translate(_LIGATURES)
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def best_fuzzy(name: str, candidates: list[str], threshold: int) -> str | None:
    norm_name = normalise(name)
    norm_cands = [normalise(c) for c in candidates]
    result = process.extractOne(
        norm_name, norm_cands, scorer=fuzz.WRatio, score_cutoff=threshold
    )
    if result is None:
        return None
    return candidates[norm_cands.index(result[0])]


# ── Load data ─────────────────────────────────────────────────────────────────
per90 = pd.read_csv(PER90_FILE)
pos   = pd.read_csv(POSITION_FILE).dropna(subset=["player"])
ml    = pd.read_csv(MATCHLEVEL_FILE)[["match_team_id", "passes_against"]]
nicks = pd.read_csv(NICKNAME_FILE).drop_duplicates(subset=["player_nickname"])
# nickname_map: short/nickname -> full name (as in position file)
nick_map = dict(zip(nicks["player_nickname"], nicks["Player"]))

# ── Step 1: join passes_against ───────────────────────────────────────────────
per90 = per90.merge(ml, on="match_team_id", how="left")
print(f"passes_against joined: {per90['passes_against'].notna().sum()} / {len(per90)}")

# ── Step 1b: join on-field passes_against (per-player, opportunity-adjusted) ───
# Opponent passes that happened while THIS player was on the pitch (see
# 2026-06-18_passes_against_onfield.py). Keyed on (match_id, defender_id).
onfield = pd.read_csv(ONFIELD_FILE)[["match_id", "defender_id", "passes_against_on_field"]]
per90 = per90.merge(onfield, on=["match_id", "defender_id"], how="left")
print(f"passes_against_on_field joined: "
      f"{per90['passes_against_on_field'].notna().sum()} / {len(per90)}")

# ── Step 2: match defender_name -> position file player name ──────────────────
pos_lookup = pos.drop_duplicates(subset=["player"]).set_index("player")[
    ["overall_position", "most_common_position"]
]
pos_names  = pos_lookup.index.tolist()
norm_pos   = {normalise(n): n for n in pos_names}   # normalised -> original

def match_to_position(short_name: str) -> str | None:
    # pass 1: exact
    if short_name in pos_lookup.index:
        return short_name
    # pass 2: normalised exact
    nk = normalise(short_name)
    if nk in norm_pos:
        return norm_pos[nk]
    # pass 3: nickname -> full name -> exact / normalised exact
    full = nick_map.get(short_name)
    if full:
        if full in pos_lookup.index:
            return full
        nf = normalise(full)
        if nf in norm_pos:
            return norm_pos[nf]
    # pass 4: fuzzy on short name
    hit = best_fuzzy(short_name, pos_names, FUZZY_THRESHOLD)
    if hit:
        return hit
    # pass 5: fuzzy on nickname-expanded full name
    if full:
        hit = best_fuzzy(full, pos_names, FUZZY_THRESHOLD)
        if hit:
            return hit
    return None

unique_defenders = per90["defender_name"].unique()
name_map: dict[str, str | None] = {n: match_to_position(n) for n in unique_defenders}

matched   = sum(v is not None for v in name_map.values())
unmatched = [k for k, v in name_map.items() if v is None]
print(f"Position matched: {matched} / {len(unique_defenders)}")
if unmatched:
    print(f"Unmatched ({len(unmatched)}):")
    for n in sorted(unmatched):
        print(f"  {n!r}")

per90["_pos_key"] = per90["defender_name"].map(name_map)
per90 = per90.merge(
    pos_lookup.reset_index().rename(columns={"player": "_pos_key"}),
    on="_pos_key", how="left"
).drop(columns=["_pos_key"])

# ── Step 3: three normalisations ──────────────────────────────────────────────
for m in METRICS:
    # per90 already exists; recompute cleanly in case
    per90[f"{m}_per90"] = per90[m] / per90["mins_played"] * 90

    # per 100 opponent passes (team-level denominator)
    per90[f"{m}_per_pass_against"] = per90[m] / per90["passes_against"] * 100

    # per 100 passes this player was involved in defending (player-level denominator)
    per90[f"{m}_per_pass_defended"] = per90[m] / per90["passes_defended"] * 100

    # per 100 opponent passes faced WHILE ON FIELD (per-player denominator)
    per90[f"{m}_per_pass_against_onfield"] = per90[m] / per90["passes_against_on_field"] * 100

# passes_defended normalisation too
per90["passes_defended_per90"]         = per90["passes_defended"] / per90["mins_played"] * 90
per90["passes_defended_per_pass_against"] = per90["passes_defended"] / per90["passes_against"] * 100
per90["passes_defended_per_pass_against_onfield"] = (
    per90["passes_defended"] / per90["passes_against_on_field"] * 100
)

# ── Save ──────────────────────────────────────────────────────────────────────
col_order = [
    "match_id", "match_name", "defending_team", "defending_team_name",
    "defender_id", "defender_name", "match_team_id",
    "overall_position", "most_common_position",
    "mins_played", "starter",
    "passes_against", "passes_against_on_field", "passes_defended",
] + METRICS + [
    f"{m}_per90"                      for m in METRICS
] + [
    f"{m}_per_pass_against"           for m in METRICS
] + [
    f"{m}_per_pass_defended"          for m in METRICS
] + [
    f"{m}_per_pass_against_onfield"   for m in METRICS
] + [
    "passes_defended_per90",
    "passes_defended_per_pass_against",
    "passes_defended_per_pass_against_onfield",
]

per90 = per90[col_order]
per90.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(per90)} rows -> {OUTPUT_FILE}")
print(f"  unique players : {per90['defender_name'].nunique()}")
print(f"  unique matches : {per90['match_id'].nunique()}")
print(f"  position coverage: {per90['overall_position'].notna().sum()} / {len(per90)} rows")
print()
print("overall_position distribution:")
print(per90["overall_position"].value_counts())
print()
print("Sample per-pass-against metrics:")
print(per90[["defender_name", "overall_position",
             "raw_involvement", "raw_involvement_per90",
             "raw_involvement_per_pass_against",
             "raw_involvement_per_pass_defended"]].head(5).round(4).to_string())
