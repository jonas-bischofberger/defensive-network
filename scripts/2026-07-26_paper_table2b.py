"""Companion to Table 2: the SAME ICC family, but outcome correlations instead of ICC.

Table 2  (tab:net_iccs)      -- between-match stability   : can you recognise the team?
Table 2b (tab:net_outcomes)  -- outcome correlations      : does it predict conceding?

Identical 23-cell metric family (asserted against the dashboard's ICC_FAMILY_COLS):

    Total network strength            x 6 network variants
    Gini coefficient                  x 6 network variants
    Freeman centralization (weighted) x 6 network variants
    Block geometry (positional)       x 5 metrics

Each metric is correlated with all three conceded outcomes (goals / shots / xG),
twice: raw, and partial on `passes_against`. The partial column is the one that
carries the argument -- network volume, shots conceded and xG conceded all scale with
how long the opponent keeps the ball, so a raw r largely measures dominance rather
than defensive quality.

    23 metrics x 3 outcomes x {raw, partial} = 138 cells

Correction is Westfall-Young step-down min-p on |r| with B shared permutations of the
row order (one permutation applied to all 138 cells at once), the same estimator as
the dashboard's Correlation tab. The shared permutation is what makes this affordable:
the family is dependent on three axes simultaneously -- the six weight variants of a
metric (r ~ 0.9), the three outcomes (shots/xG r ~ 0.8), and the raw vs partial
version of one cell -- so a Bonferroni/Sidak price of 138 independent tests would be
absurd. Complete cases over all 23 metrics + 3 outcomes + the control.

Sign convention: outcomes are all *conceded*, so negative r = defensively good.
"""
import warnings, importlib.util, sys, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location('tla', 'scripts/2026-05-28_team_level_analysis.py')
m = importlib.util.module_from_spec(spec); sys.modules['tla'] = m; spec.loader.exec_module(m)

B    = 10000
SEED = 20260726
THR  = 1
CTRL = "passes_against"
print(f"method={m.method!r}  B={B}\n", flush=True)

df = m.process(m.edge_dfs[m.method], THR)
print(f"process(): {len(df)} match-teams, {df.team_name.nunique()} teams", flush=True)

# ── the family (must be byte-identical to Table 2's) ──────────────────────────
GINI     = [c + "_gini" for c in m.WEIGHT_COLS]
FREEMAN  = [c + "_centralization_weighted" for c in m.WEIGHT_COLS]
STRENGTH = list(m.WEIGHT_COLS)
GEOM     = [c for c in m.GEOM_COLS if c in df.columns]
FAMILY   = STRENGTH + GINI + FREEMAN + GEOM
assert FAMILY == [c for c in m.ICC_FAMILY_COLS if c in df.columns], \
    "family drifted from the dashboard's ICC_FAMILY_COLS"
print(f"family: {len(FAMILY)} metrics x {len(m.OUTCOME_COLS)} outcomes x 2 = "
      f"{len(FAMILY) * len(m.OUTCOME_COLS) * 2} cells", flush=True)

wy = m.corr_wy_family(df, FAMILY, nperm=B, seed=SEED, control=CTRL)
if wy is None:
    sys.exit("corr_wy_family returned None -- not enough complete data")
long = m.compute_corr_rows(df, FAMILY, control=CTRL)
pos  = {c: i for i, c in enumerate(wy["cells"])}
long["p_perm"] = [round(float(wy["p_perm"][pos[k]]), 4)
                  if (k := (r.metric, r.outcome, r.kind)) in pos else np.nan
                  for r in long.itertuples()]
long["p_WY"]   = [round(float(wy["p_WY"][pos[k]]), 4)
                  if (k := (r.metric, r.outcome, r.kind)) in pos else np.nan
                  for r in long.itertuples()]
assert len(long) == len(FAMILY) * len(m.OUTCOME_COLS) * 2, "not every cell was computed"

WEIGHT_LABEL = {
    "raw_involvement":     "Raw involvement",
    "raw_fault":           "Raw fault",
    "raw_contribution":    "Raw contribution",
    "valued_involvement":  "Valued involvement",
    "valued_fault":        "Valued fault",
    "valued_contribution": "Valued contribution",
}
BLOCKS = [
    ("Total network strength",
     [(WEIGHT_LABEL[w], w) for w in m.WEIGHT_COLS]),
    ("Gini coefficient (player strength inequality)",
     [(WEIGHT_LABEL[w], w + "_gini") for w in m.WEIGHT_COLS]),
    ("Freeman centralization (weighted)",
     [(WEIGHT_LABEL[w], w + "_centralization_weighted") for w in m.WEIGHT_COLS]),
    ("Block geometry (positional, network-free)",
     [(m.GEOM_LABEL[c], c) for c in GEOM]),
]

# GEOM_LABEL is the dashboard's unicode display string; the paper needs math mode.
GEOM_TEX = {
    "block_x_spread":    r"Block depth (SD of mean $x$)",
    "block_y_spread":    r"Block width (SD of mean $y$)",
    "block_area":        r"Block area (depth $\times$ width)",
    "block_aspect":      r"Block aspect (depth / width)",
    "block_line_height": r"Line height (mean $x$, oriented)",
}
TEX_LABEL = {**{w: WEIGHT_LABEL[w] for w in m.WEIGHT_COLS},
             **{w + "_gini": WEIGHT_LABEL[w] for w in m.WEIGHT_COLS},
             **{w + "_centralization_weighted": WEIGHT_LABEL[w] for w in m.WEIGHT_COLS},
             **GEOM_TEX}
OUT_TEX = {"goals_against": "Goals conceded", "shots_against": "Shots conceded",
           "xg_against": "xG conceded"}

piv_r = long.set_index(["metric", "outcome", "kind"])["r"]
piv_q = long.set_index(["metric", "outcome", "kind"])["p_WY"]

# ── console ───────────────────────────────────────────────────────────────────
pd.set_option("display.width", 240)
wide = long.pivot_table(index="metric", columns=["outcome", "kind"],
                        values=["r", "p_WY"]).reindex(FAMILY)
print("\n" + "=" * 110)
print(f"TABLE 2b  tab:net_outcomes   ({len(FAMILY)} metrics x {len(m.OUTCOME_COLS)} outcomes "
      f"x raw/partial, WY over {len(wy['cells'])} cells)")
print("=" * 110)
print(wide.round(3).to_string())
print(f"\nn={wy['n_obs']} match-teams (complete cases over all {len(FAMILY)} metrics), "
      f"{wy['n_teams']} teams · control={CTRL}")
print(f"alpha_FWER={wy['alpha_fwer']:.4f}   m_eff={wy['m_eff']:.1f} of {len(wy['cells'])}   "
      f"#p<.05={int((long.p < .05).sum())}/{len(long)}   "
      f"#p_WY<.05={int((long.p_WY < .05).sum())}/{len(long)}")

print("\n--- whole family, sorted by p_WY (top 25) ---")
print(long.sort_values("p_WY")[["metric", "outcome", "kind", "r", "p", "p_perm", "p_WY"]]
          .head(25).to_string(index=False))
print("\n--- survivors (p_WY < .05) ---")
surv = long[long.p_WY < .05].sort_values("p_WY")
print(surv[["metric", "outcome", "kind", "r", "p", "p_WY"]].to_string(index=False)
      if len(surv) else "  none")
print(f"\n--- by kind: #p_WY<.05  raw {int((surv.kind == 'raw').sum())} / "
      f"partial {int((surv.kind == 'partial').sum())} ---")


# ── LaTeX ─────────────────────────────────────────────────────────────────────
def _r_tex(v):
    if pd.isna(v):
        return "--"
    return f"{v:.2f}".replace("-", "$-$")


def _stars(q):
    if pd.isna(q):
        return ""
    return "$^{***}$" if q < 0.001 else "$^{**}$" if q < 0.01 else "$^{*}$" if q < 0.05 else ""


L = []
L.append(r"\begin{table}[ht]")
L.append(r"\centering")
L.append(r"\caption{Association between network volume, network topology and")
L.append(r"defensive-block geometry and what a team concedes. Pearson correlations at")
L.append(rf"match-team level ($n={wy['n_obs']}$, {wy['n_teams']} teams, minimum edge")
L.append(r"aggregation); \textit{raw} and \textit{partial} (both variables residualised on")
L.append(r"the opponent's pass volume). All three outcomes are conceded, so a negative")
L.append(r"coefficient is defensively favourable. The same metric family as")
L.append(r"Table~\ref{tab:net_iccs}: what is stable between matches and what predicts")
L.append(r"conceding are two different questions, and controlling for exposure removes")
L.append(r"most of the raw association.}")
L.append(r"\label{tab:net_outcomes}")
L.append(r"\begin{tabular}{l" + "rr" * len(m.OUTCOME_COLS) + "}")
L.append(r"\toprule")
L.append(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{OUT_TEX[o]}}}}}"
                            for o in m.OUTCOME_COLS) + r" \\")
_start = 2
_cmids = []
for _ in m.OUTCOME_COLS:
    _cmids.append(rf"\cmidrule(lr){{{_start}-{_start + 1}}}")
    _start += 2
L.append("".join(_cmids))
L.append(r"\textbf{Metric} & " +
         " & ".join(r"\textit{raw} & \textit{partial}" for _ in m.OUTCOME_COLS) + r" \\")
for block, items in BLOCKS:
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{1 + 2 * len(m.OUTCOME_COLS)}}}{{l}}{{\textit{{{block}}}}} \\")
    for _lab, c in items:
        cells = []
        for o in m.OUTCOME_COLS:
            for kind in ("raw", "partial"):
                key = (c, o, kind)
                if key in piv_r.index:
                    cells.append(_r_tex(piv_r[key]) + _stars(piv_q[key]))
                else:
                    cells.append("--")
        L.append(f"\\quad {TEX_LABEL[c]} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append("")
# NB: no `tablenotes` -- that environment only exists inside a threeparttable.
L.append(r"\vspace{0.9ex}")
L.append(r"\begin{minipage}{0.96\linewidth}")
L.append(r"\footnotesize")
L.append(r"\textit{Note:} Pearson $r$ between the metric and the outcome conceded in the")
L.append(r"same match. \textit{raw} = unadjusted; \textit{partial} = both variables")
L.append(r"residualised on the opponent's total pass count ($\mathrm{df}=n-3$), which")
L.append(r"proxies exposure: network volume, shots conceded and xG conceded all grow with")
L.append(r"how long the opponent has the ball, so the raw coefficient largely reflects")
L.append(r"territorial dominance rather than defensive quality.")
L.append(r"Stars are Westfall--Young step-down min-P family-wise $p$-values from")
L.append(rf"{B:,} permutations of the row order, family-wise over all {len(wy['cells'])}")
L.append(rf"cells of the table ({len(FAMILY)} metrics $\times$ {len(m.OUTCOME_COLS)} outcomes")
L.append(r"$\times$ raw/partial) -- every cell that enters the correction is reported. One")
L.append(r"permutation is applied to every cell simultaneously, so the correction absorbs")
L.append(r"the dependence between the six network variants ($r\approx0.9$), between the")
L.append(r"three outcomes, and between the raw and partial version of a cell, rather than")
L.append(rf"treating them as independent (effective family size")
L.append(rf"$m_{{\mathrm{{eff}}}}={wy['m_eff']:.1f}$ of {len(wy['cells'])}).")
L.append(r"\par\smallskip")
L.append(r"$^{*}p_{\mathrm{WY}}<0.05$, $^{**}p_{\mathrm{WY}}<0.01$, "
         r"$^{***}p_{\mathrm{WY}}<0.001$.")
L.append(r"\end{minipage}")
L.append(r"\end{table}")
tex = "\n".join(L) + "\n"

out_tex = "scripts/2026-07-26_paper_table2b.tex"
out_csv = "scripts/2026-07-26_table2b_wy.csv"
with open(out_tex, "w") as f:
    f.write(tex)
long.to_csv(out_csv, index=False)
print("\n" + "=" * 110)
print(tex)
print(f"wrote {out_tex} and {out_csv}")
