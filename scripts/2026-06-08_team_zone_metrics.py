"""
Team-level defensive metrics split by PITCH ZONE of the ball (pass origin).

Idea
----
A defensive action is not neutral on its own — *where* it happens matters:
  - High press  : opponent pinned in their own half  -> proactive, high reward
  - Forced block: ball near the defending team's goal -> reactive, high risk

Zoning is based on the ball position when the pass is made (`x_norm`, the
passer's x). `x_norm = x_event * playing_direction` is already normalised so the
team *in possession* always attacks toward +x, which automatically cancels both
the home/away orientation AND the first/second-half flip. We flip once more to
the DEFENDING team's perspective:

    x_def = -x_norm          # larger = closer to opponent goal = higher press

Pitch is impect coordinates, x in [-52.5, +52.5] (centre = 0, length 105).

Three zoning schemes are all computed (selectable later in the app):
  - thirds      : own < -17.5 | mid [-17.5,17.5] | high_press > 17.5   (pitch thirds)
  - scheme_4060 : own < -10.5 | mid [-10.5,10.5] | high_press > 10.5   (40/60 on 0-100)
  - half        : own < 0 | high_press >= 0                            (own vs opp half)

Output (long format): one row per (match_id, defending_team, scheme, zone).

Passes: all scored pass outcomes (C/B/D) are included, i.e. BOTH successful (C)
and unsuccessful (B/D) passes, so contribution (mostly from stopped passes) and
fault (mostly from completed passes) are both captured. `success` = outcome 'C'.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive

FOLDER       = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
HERE         = os.path.dirname(__file__)
OUTPUT_FILE  = os.path.join(HERE, "2026-06-08_team_zone_metrics.csv")

METRICS = ["raw_involvement", "raw_contribution", "raw_fault",
           "valued_involvement", "valued_contribution", "valued_fault"]

# zone schemes: list of (low, high, label); intervals are [low, high)
SCHEMES = {
    "thirds":      [(-np.inf, -17.5, "own"), (-17.5, 17.5, "mid"), (17.5, np.inf, "high_press")],
    "scheme_4060": [(-np.inf, -10.5, "own"), (-10.5, 10.5, "mid"), (10.5, np.inf, "high_press")],
    "half":        [(-np.inf,   0.0, "own"),                       (0.0,  np.inf, "high_press")],
}

# ── Reference data ─────────────────────────────────────────────────────────────
meta = pd.read_csv(os.path.join(HERE, "meta_worldcup.csv"))
team_name_lookup = {}
for _, r in meta.iterrows():
    team_name_lookup[int(r["home_team_id"])]  = r["home_team_name"]
    team_name_lookup[int(r["guest_team_id"])] = r["guest_team_name"]


def assign_zone(x_def, scheme):
    bins   = [b[0] for b in SCHEMES[scheme]] + [SCHEMES[scheme][-1][1]]
    labels = [b[2] for b in SCHEMES[scheme]]
    return pd.cut(x_def, bins=bins, labels=labels, right=False)


# ── Main loop ──────────────────────────────────────────────────────────────────
files   = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
records = []
sanity  = []   # collect x_def distribution per (match, defending_team) for a symmetry check

for f in files:
    if MATCH_FILTER not in f["name"] or not f["name"].endswith(".parquet"):
        continue

    df = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + f["name"])
    # All scored passes -> includes successful (C) AND unsuccessful (B/D)
    df = df[df["possessionEvents.passOutcomeType"].isin(["C", "B", "D"])].copy()
    df = df.dropna(subset=["x_norm", "defending_team"])
    if df.empty:
        continue

    match_id = int(df["match_id"].iloc[0])
    print(f"Processing match {match_id}: {f['name']}")

    df["defending_team"] = df["defending_team"].astype(int)
    df["x_def"]   = -df["x_norm"]                                       # defending-team perspective
    df["success"] = (df["possessionEvents.passOutcomeType"] == "C").astype(int)
    active        = [m for m in METRICS if m in df.columns]

    # Pass identifier so a single pass counts once even though it generates one
    # row per involved defender. Prefer the model's pass id, fall back to frame.
    pass_id_col = next((c for c in ("involvement_pass_id", "frame_id") if c in df.columns), None)
    if pass_id_col is None:
        raise ValueError("No pass id column (involvement_pass_id / frame_id) — cannot "
                         "dedupe passes; got: " + ", ".join(df.columns[:30]))

    # sanity: mean x_def per team should be near 0 and symmetric across the two teams
    for tid, sub in df.groupby("defending_team"):
        sanity.append(dict(match_id=match_id, defending_team=int(tid),
                           team=team_name_lookup.get(int(tid), str(tid)),
                           n=len(sub), x_def_mean=round(sub["x_def"].mean(), 2),
                           x_def_min=round(sub["x_def"].min(), 1),
                           x_def_max=round(sub["x_def"].max(), 1)))

    for scheme in SCHEMES:
        df["zone"] = assign_zone(df["x_def"], scheme)
        for (tid, zone), g in df.groupby(["defending_team", "zone"], observed=True):
            g_inv  = g[g["raw_involvement"].fillna(0) > 0]              # defended passes
            uniq   = g.drop_duplicates(pass_id_col)                     # one row per pass
            rec = dict(
                match_id=match_id,
                defending_team=int(tid),
                defending_team_name=team_name_lookup.get(int(tid), str(tid)),
                match_team_id=f"{match_id}_{int(tid)}",
                scheme=scheme,
                zone=str(zone),
                n_pass_rows=len(g),                                     # defender-action rows
                n_passes=int(g[pass_id_col].nunique()),                 # unique passes in zone
                n_actions=int(g_inv[pass_id_col].nunique()),            # unique DEFENDED passes
                n_success=int(uniq["success"].sum()),                  # unique successful passes
            )
            for m in active:
                rec[f"{m}_sum"]   = float(g[m].fillna(0).sum())                       # load
                rec[f"{m}_npass"] = int(g.loc[g[m].fillna(0) != 0, pass_id_col].nunique())  # unique passes with m>0
            records.append(rec)

out = pd.DataFrame(records).sort_values(
    ["match_id", "defending_team", "scheme", "zone"]).reset_index(drop=True)
out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(out)} rows to {OUTPUT_FILE}")

# ── Sanity check: are the two teams symmetric? ─────────────────────────────────
san = pd.DataFrame(sanity)
print("\n[sanity] x_def mean per team (should hover near 0, both teams comparable):")
print(san.groupby("team")["x_def_mean"].mean().round(2).to_string())
print(f"[sanity] global x_def range: {san['x_def_min'].min()} .. {san['x_def_max'].max()} "
      f"(expected ~ -52.5 .. +52.5)")
