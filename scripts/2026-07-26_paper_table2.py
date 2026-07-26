"""Regenerate the paper's Table 2 (tab:net_iccs) with Westfall-Young correction.

Replaces the manuscript's 3-cell BH table (Gini / Freeman / assortativity on one
arbitrarily chosen network variant) with the ICC sweep the dashboard tests:

    Total network strength            x 6 network variants
    Gini coefficient                  x 6 network variants
    Freeman centralization (weighted) x 6 network variants
    Block geometry (positional)       x 5 metrics
    ---------------------------------------------------
                                        23 cells

Every cell of the family is reported in the table and every reported cell enters the
correction -- the family is exactly the dashboard's ICC_FAMILY_COLS, asserted below.
(The block depth/width range metrics and the lateral-centre negative control were
removed from the dashboard itself on 2026-07-26, so there is nothing left to exclude
here: ranges duplicated the spread metrics at r > 0.9 and lateral centre existed only
to demonstrate a null.)

Correction is Westfall-Young step-down min-p on the ANOVA F = MSB/MSW, with B shared
permutations of the team-label vector (one permutation applied to all 23 cells at
once), so the family's real dependence -- the six weight variants of a metric
correlate at r ~ 0.9 -- is absorbed instead of being paid for as if independent.
Same estimator as the dashboard's "Westfall-Young step-down min-p (whole sweep)"
option, at the B used for Table 3.

Complete cases over all 23 columns, so one permutation is meaningful everywhere.
"""
import warnings, importlib.util, sys, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location('tla', 'scripts/2026-05-28_team_level_analysis.py')
m = importlib.util.module_from_spec(spec); sys.modules['tla'] = m; spec.loader.exec_module(m)

B    = 10000
SEED = 20260726
THR  = 1
print(f"method={m.method!r}  B={B}\n", flush=True)

df = m.process(m.edge_dfs[m.method], THR)
print(f"process(): {len(df)} match-teams, {df.team_name.nunique()} teams", flush=True)

# ── the family ────────────────────────────────────────────────────────────────
GINI     = [c + "_gini" for c in m.WEIGHT_COLS]
FREEMAN  = [c + "_centralization_weighted" for c in m.WEIGHT_COLS]
STRENGTH = list(m.WEIGHT_COLS)
GEOM     = [c for c in m.GEOM_COLS if c in df.columns]
FAMILY   = STRENGTH + GINI + FREEMAN + GEOM
assert FAMILY == [c for c in m.ICC_FAMILY_COLS if c in df.columns], \
    "family drifted from the dashboard's ICC_FAMILY_COLS"
print(f"family: {len(FAMILY)} cells "
      f"({len(STRENGTH)} strength + {len(GINI)} Gini + {len(FREEMAN)} Freeman + {len(GEOM)} geometry)",
      flush=True)

wy = m.icc_wy_family(df, FAMILY, nperm=B, seed=SEED)
if wy is None:
    sys.exit("icc_wy_family returned None -- not enough replicated data")
icc = {r["metric"]: r for r in m.compute_icc_rows(df, FAMILY)}
pos = {c: i for i, c in enumerate(wy["cells"])}

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

rows = []
for block, items in BLOCKS:
    for lab, c in items:
        r = icc.get(c, {})
        i = pos.get(c)
        rows.append(dict(
            block=block, label=lab, metric=c,
            ICC=r.get("ICC", np.nan), F=r.get("F", np.nan), p=r.get("p", np.nan),
            p_perm=round(float(wy["p_perm"][i]), 4) if i is not None else np.nan,
            p_WY=round(float(wy["p_WY"][i]), 4) if i is not None else np.nan,
        ))
T = pd.DataFrame(rows)

pd.set_option("display.width", 220)
print("\n" + "=" * 100)
print(f"TABLE 2  tab:net_iccs   (strength + Gini + Freeman x 6 variants + {len(GEOM)} geometry, "
      f"WY over {len(FAMILY)} cells)")
print("=" * 100)
print(T.drop(columns="block").to_string(index=False))
print(f"\nn={wy['n_obs']} match-teams (complete cases over all {len(wy['cells'])} cells), "
      f"{wy['n_teams']} teams")
print(f"alpha_FWER={wy['alpha_fwer']:.4f}   m_eff={wy['m_eff']:.1f} of {len(wy['cells'])}   "
      f"#p<.05={int((T.p < .05).sum())}/{len(T)}   "
      f"#p_WY<.05={int((T.p_WY < .05).sum())}/{len(T)}")
assert len(T) == len(FAMILY), "table no longer reports every cell of the family"

print("\n--- whole family, sorted by p_WY ---")
ALL = T.copy()
print(ALL[["block", "label", "ICC", "F", "p", "p_perm", "p_WY"]]
      .sort_values("p_WY").to_string(index=False))

# ── LaTeX ─────────────────────────────────────────────────────────────────────
def _tex(v, fmt="{:.3f}", dollar=True):
    if pd.isna(v):
        return "--"
    s = fmt.format(v).replace("-", "$-$") if dollar else fmt.format(v)
    return s


def _p(v):
    """4-dp p, but never a misleading '0.0000'."""
    if pd.isna(v):
        return "--"
    return r"$<$0.0001" if v < 0.0001 else f"{v:.4f}"


def _stars(q):
    return "$^{***}$" if q < 0.001 else "$^{**}$" if q < 0.01 else "$^{*}$" if q < 0.05 else ""


L = []
L.append(r"\begin{table}[ht]")
L.append(r"\centering")
L.append(r"\caption{Between-match stability of network volume, network topology and")
L.append(r"defensive-block geometry. ICC(1,1) with team identity as the grouping factor,")
L.append(rf"all six network variants ($n={wy['n_obs']}$ match-teams, {wy['n_teams']} teams, minimum")
L.append(r"edge aggregation). Total network strength is reproducible in five of its six")
L.append(r"variants and line height and block depth are reproducible, whereas no")
L.append(r"topology metric (Gini, Freeman centralization) survives family-wise correction:")
L.append(r"what recurs about a team is how much it co-defends and where, not how that")
L.append(r"volume is distributed over the network.}")
L.append(r"\label{tab:net_iccs}")
L.append(r"\begin{tabular}{lrrrr}")
L.append(r"\toprule")
L.append(r"\textbf{Metric} & \textbf{ICC} & \textbf{F} & \textbf{p} & "
         r"\textbf{$p_{\mathrm{WY}}$} \\")
for block, items in BLOCKS:
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{block}}}}} \\")
    for _lab, c in items:
        r = T[T.metric == c].iloc[0]
        L.append(f"\\quad {TEX_LABEL[c]} & {_tex(r.ICC)} & {_tex(r.F, '{:.2f}')} & "
                 f"{_p(r.p)} & {_p(r.p_WY)}{_stars(r.p_WY)} \\\\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append("")
# NB: no `tablenotes` -- that environment only exists inside a threeparttable.
# A minipage needs no package at all and matches Table 1's note style.
L.append(r"\vspace{0.9ex}")
L.append(r"\begin{minipage}{0.96\linewidth}")
L.append(r"\footnotesize")
L.append(r"\textit{Note:} ICC = intraclass correlation coefficient ICC(1,1);")
L.append(r"F = ANOVA $F=\mathrm{MSB}/\mathrm{MSW}$ testing $H_0$: ICC $=0$;")
L.append(r"p = uncorrected parametric p-value; $p_{\mathrm{WY}}$ = Westfall--Young step-down")
L.append(rf"min-P p-value from {B:,} permutations of the team-label vector, family-wise over")
L.append(rf"all {len(wy['cells'])} cells of the table -- every cell that enters the correction is")
L.append(r"reported. One permutation is applied to every cell simultaneously, so the")
L.append(r"correction absorbs the dependence between the six network variants")
L.append(rf"($r\approx0.9$) rather than treating the cells as independent (effective family")
L.append(rf"size $m_{{\mathrm{{eff}}}}={wy['m_eff']:.1f}$ of {len(wy['cells'])}).")
L.append(r"Block geometry is computed from the starting XI's mean defensive-action positions")
L.append(r"(outfield players only, oriented so each team defends the $-x$ end).")
L.append(r"\par\smallskip")
L.append(r"$^{*}p_{\mathrm{WY}}<0.05$, $^{**}p_{\mathrm{WY}}<0.01$, "
         r"$^{***}p_{\mathrm{WY}}<0.001$.")
L.append(r"\end{minipage}")
L.append(r"\end{table}")
tex = "\n".join(L) + "\n"

out_tex = "scripts/2026-07-26_paper_table2.tex"
out_csv = "scripts/2026-07-26_table2_wy.csv"
with open(out_tex, "w") as f:
    f.write(tex)
ALL.to_csv(out_csv, index=False)
print("\n" + "=" * 100)
print(tex)
print(f"wrote {out_tex} and {out_csv}")
