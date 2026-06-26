"""
Match FIFA ratings to 2026-05-23_player_level_enriched.csv.

Matching cascade (per player, country-filtered for speed):
  0. Manual hard-coded overrides (exact FIFA name known)
  1. Exact match  (country-filtered FIFA)
  2. Case-insensitive  (country-filtered)
  3. Accent + hyphen normalised  (country-filtered)
  4. nickname_map lookup → full-FIFA search  (cross-country: Ghana→Belgium etc.)
  5. token_sort_ratio ≥ 85  (country-filtered, accent+hyphen normalised)
  6. token_set_ratio  ≥ 85  (country-filtered, for mononym subsets)
"""

import unicodedata
import pandas as pd
from rapidfuzz import fuzz

# ── load data ────────────────────────────────────────────────────────────────

ENRICHED_PATH = (
    "/Users/runqingma/学习/Projectsss/PHD/defensive-network/scripts/"
    "2026-05-23_player_level_enriched.csv"
)
FIFA_PATH = (
    "/Users/runqingma/学习/Projectsss/PHD/defensive-network/scripts/fifa_ratings.csv"
)
NICK_PATH = (
    "/Users/runqingma/学习/Projectsss/PHD/defensive-network/scripts/nickname_map.csv"
)

enriched = pd.read_csv(ENRICHED_PATH)
fifa_all = pd.read_csv(FIFA_PATH)
nick     = pd.read_csv(NICK_PATH)

# ── constants ────────────────────────────────────────────────────────────────

WC_COUNTRIES = [
    "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
    "Costa Rica", "Croatia", "Denmark", "Ecuador", "England", "France",
    "Germany", "Ghana", "Iran", "Japan", "Mexico", "Morocco", "Netherlands",
    "Poland", "Portugal", "Qatar", "Saudi Arabia", "Senegal", "Serbia",
    "South Korea", "Spain", "Switzerland", "Tunisia", "United States",
    "Uruguay", "Wales",
]

# Map enriched country name → FIFA country name (where they differ)
COUNTRY_MAP = {
    "South Korea": "Korea Republic",
}

# Manual overrides: (normalize(enriched name), enriched country) → FIFA full name
# Used for players whose names cannot be resolved algorithmically
MANUAL = {
    # Portugal – Otávio listed under Brazil in FIFA FUT
    ("otavio",             "Portugal"): "Otávio Edmilson da Silva Monteiro",
    # Morocco – Munir El Haddadi listed under Spain in FIFA FUT
    ("munir",              "Morocco"):  "Munir El Haddadi",
    # Morocco goalkeeper – Yassine Bounou is in FIFA Morocco
    ("bono",               "Morocco"):  "Yassine Bounou",
    # Portugal – spelling difference Kléper/Képler
    ("pepe",               "Portugal"): "Képler Laveran Lima Ferreira",
    # Portugal – abbreviated FIFA name
    ("cristiano ronaldo",  "Portugal"): "C. Ronaldo dos Santos Aveiro",
    # Cameroon – common name vs FIFA name
    ("pierre kunde",       "Cameroon"): "Kunde Malong",
    # Spain – mononym/abbreviated
    ("dani olmo",          "Spain"):    "Daniel Olmo Carvajal",
    ("alex balde",         "Spain"):    "Alejandro Balde Martínez",
    ("koke",               "Spain"):    "Jorge Resurrección",
    # Brazil – nickmap has wrong middle-name spelling vs FIFA entry
    ("fred",               "Brazil"):   "Frederico de Paula Santos",
    # South Korea – nickmap stores "Kang-In Lee" but FIFA has "Kangin Lee" (no space)
    ("kang in lee",        "South Korea"): "Kangin Lee",
}

FUZZY_THRESHOLD_SORT = 85   # token_sort_ratio – handles word-order / hyphens
FUZZY_THRESHOLD_SET  = 85   # token_set_ratio  – handles mononyms / subsets
# Threshold used when fuzzy-matching nickmap full-name against country pool
# (catches cases where nickmap full name ≈ but ≠ FIFA full name, e.g. Casemiro)
NICK_FUZZY_SET_THRESHOLD = 90

# Players to force-skip (genuinely not in FIFA; mononym/common name causes false positives)
SKIP = {
    ("pedro", "Brazil"),   # Pedro Guilherme not in FIFA; "pedro" matches many Brazilian names
}

# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Lowercase, strip diacritics, replace hyphens with space."""
    s = str(s).replace("-", " ")
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return s.lower().strip()


def build_pool(country: str) -> pd.DataFrame:
    """Return deduplicated FIFA rows for one country (highest rating wins)."""
    fifa_country = COUNTRY_MAP.get(country, country)
    pool = fifa_all[fifa_all["country"] == fifa_country].copy()
    pool = (
        pool.sort_values("overall rating", ascending=False)
        .drop_duplicates(subset=["name"])
        .reset_index(drop=True)
    )
    pool["_norm"] = pool["name"].map(normalize)
    return pool


# Pre-build pools per country for speed
pools: dict[str, pd.DataFrame] = {}
for c in WC_COUNTRIES:
    pools[c] = build_pool(c)

# Deduplicated FIFA (for cross-country nickname lookups)
fifa_dedup = (
    fifa_all.sort_values("overall rating", ascending=False)
    .drop_duplicates(subset=["name"])
    .reset_index(drop=True)
)
fifa_dedup["_norm"] = fifa_dedup["name"].map(normalize)

# Deduplicate nickname_map: one canonical FIFA name per nickname
nick_dedup = (
    nick.drop_duplicates(subset=["player_nickname"])
    .reset_index(drop=True)
)
nick_dedup["_nick_norm"] = nick_dedup["player_nickname"].map(normalize)
nick_dedup["_full_norm"] = nick_dedup["Player"].map(normalize)

# ── per-player matching ──────────────────────────────────────────────────────

players = (
    enriched.drop_duplicates("defender_id")[
        ["defender_id", "defender_name", "defending_team_name"]
    ]
    .copy()
    .reset_index(drop=True)
)

RATING_COLS = ["overall rating", "defending rating", "def_awareness_rating", "interceptions"]

results = []

for _, row in players.iterrows():
    raw_name = row["defender_name"]
    country  = row["defending_team_name"]
    pool     = pools[country]
    pnorm    = normalize(raw_name)

    matched_row = None
    method      = None

    # ── 0a. Force-skip list ──────────────────────────────────────────────────
    if (pnorm, country) in SKIP:
        results.append({
            "defender_id": row["defender_id"], "defender_name": raw_name,
            "defending_team_name": country, "fifa_name": None,
            "fifa_country": None, "match_method": "NO MATCH",
            "overall_rating": None, "defending_rating": None,
            "def_awareness_rating": None, "interceptions_rating": None,
        })
        continue

    # ── 0b. Manual hard-coded override ──────────────────────────────────────
    manual_key = (pnorm, country)
    if manual_key in MANUAL:
        fifa_name = MANUAL[manual_key]
        hit = fifa_dedup[fifa_dedup["_norm"] == normalize(fifa_name)]
        if not hit.empty:
            matched_row = hit.iloc[0]
            method = "manual"

    # ── 1. Exact match ───────────────────────────────────────────────────────
    if matched_row is None:
        hit = pool[pool["name"] == raw_name]
        if not hit.empty:
            matched_row = hit.iloc[0]
            method = "exact"

    # ── 2. Case-insensitive ──────────────────────────────────────────────────
    if matched_row is None:
        hit = pool[pool["name"].str.lower() == raw_name.lower()]
        if not hit.empty:
            matched_row = hit.iloc[0]
            method = "case-insensitive"

    # ── 3. Accent + hyphen normalised ────────────────────────────────────────
    if matched_row is None:
        hit = pool[pool["_norm"] == pnorm]
        if not hit.empty:
            matched_row = hit.iloc[0]
            method = "accent-normalised"

    # ── 4. nickname_map → full-FIFA search ──────────────────────────────────
    if matched_row is None:
        nk = nick_dedup[nick_dedup["_nick_norm"] == pnorm]
        if not nk.empty:
            full_norm = nk.iloc[0]["_full_norm"]
            # 4a. exact normalized match across full FIFA (handles cross-country)
            hit = fifa_dedup[fifa_dedup["_norm"] == full_norm]
            if not hit.empty:
                matched_row = hit.iloc[0]
                method = "nickname_map"
            else:
                # 4b. token_set fallback within country pool only
                # (catches "Carlos Henrique Casimiro" ≈ "Carlos Henrique Venancio Casimiro")
                # Guard: first token of nickmap full_norm must appear in the FIFA match
                # (prevents "Jhegson Sebastián Méndez..." matching "Sebastían Méndez")
                if not pool.empty:
                    pool_norms = pool["_norm"].tolist()
                    scores = [(n, fuzz.token_set_ratio(full_norm, n)) for n in pool_norms]
                    best = max(scores, key=lambda x: x[1])
                    first_tok = full_norm.split()[0]
                    if best[1] >= NICK_FUZZY_SET_THRESHOLD and first_tok in best[0].split():
                        idx = pool_norms.index(best[0])
                        matched_row = pool.iloc[idx]
                        method = f"nickname_map~({best[1]:.0f})"

    # ── 5. Fuzzy token_sort_ratio (country pool) ─────────────────────────────
    if matched_row is None and not pool.empty:
        pool_norms = pool["_norm"].tolist()
        scores = [(n, fuzz.token_sort_ratio(pnorm, n)) for n in pool_norms]
        best_score = max(scores, key=lambda x: x[1])
        if best_score[1] >= FUZZY_THRESHOLD_SORT:
            idx = pool_norms.index(best_score[0])
            matched_row = pool.iloc[idx]
            method = f"fuzzy-sort({best_score[1]:.0f})"

    # ── 6. Fuzzy token_set_ratio (country pool, for mononyms) ────────────────
    if matched_row is None and not pool.empty:
        pool_norms = pool["_norm"].tolist()
        scores = [(n, fuzz.token_set_ratio(pnorm, n)) for n in pool_norms]
        best_score = max(scores, key=lambda x: x[1])
        if best_score[1] >= FUZZY_THRESHOLD_SET:
            idx = pool_norms.index(best_score[0])
            matched_row = pool.iloc[idx]
            method = f"fuzzy-set({best_score[1]:.0f})"

    # ── record ───────────────────────────────────────────────────────────────
    if matched_row is not None:
        results.append({
            "defender_id":          row["defender_id"],
            "defender_name":        raw_name,
            "defending_team_name":  country,
            "fifa_name":            matched_row["name"],
            "fifa_country":         matched_row["country"],
            "match_method":         method,
            "overall_rating":       matched_row["overall rating"],
            "defending_rating":     matched_row["defending rating"],
            "def_awareness_rating": matched_row["def_awareness_rating"],
            "interceptions_rating": matched_row["interceptions"],
        })
    else:
        results.append({
            "defender_id":          row["defender_id"],
            "defender_name":        raw_name,
            "defending_team_name":  country,
            "fifa_name":            None,
            "fifa_country":         None,
            "match_method":         "NO MATCH",
            "overall_rating":       None,
            "defending_rating":     None,
            "def_awareness_rating": None,
            "interceptions_rating": None,
        })

match_df = pd.DataFrame(results)

# ── reports ──────────────────────────────────────────────────────────────────

total    = len(match_df)
matched  = match_df[match_df["match_method"] != "NO MATCH"]
no_match = match_df[match_df["match_method"] == "NO MATCH"]

print("=" * 70)
print("OVERALL MATCH SUMMARY")
print("=" * 70)
print(f"Total players :  {total}")
print(f"Matched       :  {len(matched)}  ({len(matched)/total*100:.1f}%)")
print(f"No match      :  {len(no_match)}  ({len(no_match)/total*100:.1f}%)")
print()
print("By method:")
for m, cnt in match_df["match_method"].value_counts().items():
    print(f"  {m:<28} {cnt:>4}  ({cnt/total*100:.1f}%)")

# Per-country table
print()
print("=" * 70)
print("BY COUNTRY")
print("=" * 70)

def country_row(g):
    total_c   = len(g)
    matched_c = (g["match_method"] != "NO MATCH").sum()
    return pd.Series({
        "total":     total_c,
        "matched":   matched_c,
        "unmatched": total_c - matched_c,
        "pct":       f"{matched_c/total_c*100:.0f}%",
    })

country_stats = (
    match_df.groupby("defending_team_name", group_keys=False)
    .apply(country_row, include_groups=False)
    .reset_index()
    .sort_values("unmatched", ascending=False)
)
print(country_stats.to_string(index=False))

# Fuzzy matches to verify
print()
print("=" * 70)
print("FUZZY MATCHES — PLEASE VERIFY")
print("=" * 70)
fuzzy = match_df[match_df["match_method"].str.startswith("fuzzy", na=False)]
for _, r in fuzzy.sort_values("defending_team_name").iterrows():
    flag = "  " if r["fifa_country"] == COUNTRY_MAP.get(r["defending_team_name"], r["defending_team_name"]) else "⚠ cross-country"
    print(f"  [{r['defending_team_name']:<15}]  '{r['defender_name']}'  →  '{r['fifa_name']}'  ({r['match_method']}) {flag}")

# Nickname-map matches (cross-country worth noting)
print()
print("=" * 70)
print("NICKNAME-MAP MATCHES (cross-country flagged)")
print("=" * 70)
nk_matches = match_df[match_df["match_method"].str.startswith("nickname_map", na=False)]
for _, r in nk_matches.sort_values("defending_team_name").iterrows():
    expected_fifa_country = COUNTRY_MAP.get(r["defending_team_name"], r["defending_team_name"])
    cross = " ⚠ cross-country" if r["fifa_country"] != expected_fifa_country else ""
    print(f"  [{r['defending_team_name']:<15}]  '{r['defender_name']}'  →  '{r['fifa_name']}' ({r['fifa_country']}){cross}")

# Unmatched
print()
print("=" * 70)
print("UNMATCHED PLAYERS")
print("=" * 70)
for _, r in no_match.sort_values("defending_team_name").iterrows():
    print(f"  [{r['defending_team_name']:<15}]  {r['defender_name']}")

# ── merge into enriched and write output ─────────────────────────────────────

RATING_MAP = (
    match_df[["defender_id", "overall_rating", "defending_rating",
              "def_awareness_rating", "interceptions_rating"]]
    .drop_duplicates("defender_id")
)

out = enriched.merge(RATING_MAP, on="defender_id", how="left")

OUT_PATH = (
    "/Users/runqingma/学习/Projectsss/PHD/defensive-network/scripts/"
    "2026-05-25_player_level_enriched_with_ratings.csv"
)
out.to_csv(OUT_PATH, index=False)

n_rows_with = out["overall_rating"].notna().sum()
print()
print("=" * 70)
print("OUTPUT WRITTEN")
print("=" * 70)
print(f"  File   : {OUT_PATH}")
print(f"  Rows   : {len(out)}")
print(f"  Rows with FIFA rating : {n_rows_with} / {len(out)}"
      f"  ({n_rows_with/len(out)*100:.1f}%)")
print(f"  New columns: overall_rating, defending_rating,"
      f" def_awareness_rating, interceptions_rating")
