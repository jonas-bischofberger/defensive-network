"""
Merge player_minutes_per_match.csv into 2026-04-29-player_level_metrics.csv.

Join key: (match_name, team_name, player_name)
  - match_name is constructed from metrics' home_away column (identical in both datasets)
  - Croatia vs Morocco duplicate is resolved naturally (different home/away assignment)
  - Player names differ between lineup API (full names) and events (short names):
      "Cody Gakpo"       ↔  "Cody Mathès Gakpo"          (extra middle name)
      "Matthijs De Ligt" ↔  "Matthijs de Ligt"            (capitalisation)
      "Casemiro"         ↔  "Carlos Henrique Casimiro"    (nickname)
    Pass 2: fuzzy token_set_ratio for extra-middle-name / capitalisation cases
    Pass 3: manual nickname → full_name map for cases with no shared tokens
"""
import unicodedata
import pandas as pd
from rapidfuzz import fuzz

MINUTES_FILE  = "scripts/player_minutes_per_match.csv"
METRICS_FILE  = "scripts/2026-04-29-player_level_metrics.csv"
OUTPUT_FILE   = "scripts/2026-05-06_player_level_metrics_with_mins.csv"
FUZZY_THRESHOLD = 80

# manual map: short/nickname (as it appears in defender_name) → full name (as in player_minutes)
NICKNAME_MAP = {
    # Brazil
    "Casemiro":    "Carlos Henrique Casimiro",
    "Raphinha":    "Raphael Dias Belloli",
    "Marquinhos":  "Marcos Aoás Corrêa",
    "Lucas Paquetá": "Lucas Tolentino Coelho de Lima",
    "Fabinho":     "Fábio Henrique Tavares",
    "Dani Alves":  "Daniel Alves da Silva",
    "Fred":        "Frederico Rodrigues Santos",
    # Spain
    "Pedri":       "Pedro González López",
    "Koke":        "Jorge Resurrección Merodio",
    "Nico Williams": "Nicholas Williams Arthuer",
    "Ansu Fati":   "Anssumane Fati",
    "Alex Balde":  "Alejandro Balde Martínez",
    "Dani Olmo":   "Daniel Olmo Carvajal",
    "Rodri":       "Rodrigo Hernández Cascante",
    "Gavi":        "Pablo Martín Páez Gavira",
    # Morocco
    "Bono":                "Yassine Bounou",
    "Yahia Attiat-Allah":  "Yahia Attiyat allah",
    # Portugal
    "Vitinha": "Vitor Machado Ferreira",
    "Pepe":    "Kléper Laveran Lima Ferreira",
    # Uruguay (second entry handles encoding corruption in source file)
    "Maxi Gomez":          "Maximiliano Gómez González",
    "Mathías Olivera":     "Mathías Olivera Miramontes",
    "Mathí­as Olivera": "Mathías Olivera Miramontes",  # corrupted encoding variant
    # Argentina
    "Papu Gómez": "Alejandro Darío Gómez",
    # Saudi Arabia
    "Mohammed Al-Owais":  "Mohammed Khalil Al Owais",
    "Saud Abdulhamid":    "Saud Abdullah Abdul Hamid",
    "Sultan Al-Ghannam":  "Sultan Abdullah Salim Al Ghannam",
    "Hassan Tambakti":    "Hassan Mohammed Al-Tambakti",
    "Salem Al-Dawsari":   "Salem Mohammed Al Dawsari",
    "Salman Al-Faraj":    "Salman Mohammed Al Faraj",
    "Nawaf Al-Abed":      "Nawaf Shaker Al Abid",
    "Sami Al-Najei":      "Sami Khalil Al Naji",
    "Firas Al-Buraikan":  "Firas Tariq Nasser Al Albirakan",
    "Saleh Al-Shehri":    "Saleh Khalid Al Shehri",
    "Yasser Al-Shahrani": "Yasir Gharsan Al Shahrani",
    # Qatar
    "Mohammed Waad":   "Mohammed Waed Abdulwahhab Al Bayati",
    "Hasan Al-Haydos": "Hassan Khalid Al Heidos",
    "Saad Al-Sheeb":   "Saad Abdullah Al Sheeb",
    "Bassam Al-Rawi":  "Bassam Hisham Al Rawi",
    "Musab Khoder":    "Mosaab Khoder Jibril",
    "Ali Asad":        "Ali Assadalla Thaimn Qambar",
    "Ahmed Alaaeldin": "Ahmed Alaa Eldin Abdelmotaal",
    # United States
    "Matt Turner":   "Matthew Charles Turner",
    "Kellyn Acosta": "Kellyn Kai Perry-Acosta",
    # Canada
    "Mark-Anthony Kaye": "Mark Anthony Kaye",
    # Tunisia
    "Ghaylen Chaaleli": "Ghilane Chalali",
    # Iran
    "Hossein Kanaani": "Mohammad Hossein Kanani Zadegan",
    # Cameroon
    "Samuel Oum Gouet": "Samuel Yves Oum Gwet",
    # France
    "Dayot Upamecano": "Dayotchanculle Upamecano",
    # Switzerland
    "Breel Embolo": "Breel-Donald Embolo",
    # Denmark (æ does not strip via NFD; mapping handles it explicitly)
    "Simon Kjaer": "Simon Thorup Kjær",
}


def strip_accents(s: str) -> str:
    # replace ligatures/special Scandinavian letters before NFD stripping
    s = s.replace("æ", "ae").replace("Æ", "AE").replace("ø", "o").replace("Ø", "O")
    s = s.replace("å", "a").replace("Å", "A")
    # remove soft hyphens and other invisible chars
    s = "".join(c for c in s if unicodedata.category(c) not in ("Cf",))
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalise(s: str) -> str:
    return strip_accents(s).lower().strip()


def best_fuzzy_match(name: str, candidates: list[str], threshold: int) -> str | None:
    norm_name = normalise(name)
    best_score, best_cand = 0, None
    for cand in candidates:
        score = fuzz.token_set_ratio(norm_name, normalise(cand))
        if score > best_score:
            best_score, best_cand = score, cand
    return best_cand if best_score >= threshold else None


# ── load ─────────────────────────────────────────────────────────────────────
minutes = pd.read_csv(MINUTES_FILE)                        # match_name, team, player_name, mins_played
metrics = pd.read_csv(METRICS_FILE)                        # match_id, team_name, home_away, defender_name, ...

# ── build match_name for metrics rows ────────────────────────────────────────
match_teams = metrics[["match_id", "team_name", "home_away"]].drop_duplicates()
home = match_teams[match_teams["home_away"] == "home"].set_index("match_id")["team_name"]
away = match_teams[match_teams["home_away"] == "away"].set_index("match_id")["team_name"]
match_name_map = (home.rename("home").to_frame()
                  .join(away.rename("away"))
                  .assign(match_name=lambda d: d["home"] + " vs " + d["away"]))
metrics = metrics.merge(match_name_map[["match_name"]], left_on="match_id", right_index=True, how="left")

# ── pass 1: exact join ────────────────────────────────────────────────────────
minutes_r = minutes.rename(columns={"team": "team_name", "player_name": "defender_name"})
merged = metrics.merge(
    minutes_r[["match_name", "team_name", "defender_name", "mins_played"]],
    on=["match_name", "team_name", "defender_name"],
    how="left",
)
print(f"Pass 1 (exact):  matched {merged['mins_played'].notna().sum()} / {len(merged)}")

# ── pass 2: fuzzy join for still-unmatched rows ───────────────────────────────
# build lookup: (match_name, team) → list of full names in minutes
lookup: dict[tuple, list[str]] = (
    minutes.groupby(["match_name", "team"])["player_name"]
    .apply(list)
    .to_dict()
)

fuzzy_map: dict[tuple, str | None] = {}   # (match_name, team_name, short_name) → full_name or None

unmatched_mask = merged["mins_played"].isna()
for _, row in merged[unmatched_mask][["match_name", "team_name", "defender_name"]].drop_duplicates().iterrows():
    key = (row["match_name"], row["team_name"], row["defender_name"])
    candidates = lookup.get((row["match_name"], row["team_name"]), [])
    fuzzy_map[key] = best_fuzzy_match(row["defender_name"], candidates, FUZZY_THRESHOLD)

# build a minutes lookup by (match_name, team, full_name) → mins_played
mins_lookup = minutes.set_index(["match_name", "team", "player_name"])["mins_played"].to_dict()

def fill_fuzzy(row):
    if pd.notna(row["mins_played"]):
        return row["mins_played"]
    key = (row["match_name"], row["team_name"], row["defender_name"])
    full_name = fuzzy_map.get(key)
    if full_name is None:
        return float("nan")
    return mins_lookup.get((row["match_name"], row["team_name"], full_name), float("nan"))

merged["mins_played"] = merged.apply(fill_fuzzy, axis=1)
print(f"Pass 2 (fuzzy):  matched {merged['mins_played'].notna().sum()} / {len(merged)}")

# ── pass 3: manual nickname map ───────────────────────────────────────────────
def fill_manual(row):
    if pd.notna(row["mins_played"]):
        return row["mins_played"]
    full_name = NICKNAME_MAP.get(row["defender_name"])
    if full_name is None:
        return float("nan")
    return mins_lookup.get((row["match_name"], row["team_name"], full_name), float("nan"))

merged["mins_played"] = merged.apply(fill_manual, axis=1)
print(f"Pass 3 (manual): matched {merged['mins_played'].notna().sum()} / {len(merged)}")

# ── diagnostics ──────────────────────────────────────────────────────────────
still_unmatched = merged[merged["mins_played"].isna()][
    ["match_name", "team_name", "defender_name"]
].drop_duplicates()

if len(still_unmatched):
    print(f"\nStill unmatched ({len(still_unmatched)} unique players):")
    print(still_unmatched.to_string(index=False))
else:
    print("\nAll players matched!")

# show a sample of fuzzy matches made
fuzzy_made = {k: v for k, v in fuzzy_map.items() if v is not None}
if fuzzy_made:
    print(f"\nSample fuzzy matches (first 15):")
    for (mn, team, short), full in list(fuzzy_made.items())[:15]:
        print(f"  {short!r:35s} → {full!r}")

# ── save ─────────────────────────────────────────────────────────────────────
out = merged.drop(columns=["match_name"])   # match_name was a helper column
out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
