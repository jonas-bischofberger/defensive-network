"""Regenerate the paper's tables 1, 2 and 4 from the current dashboard code.

Table 1  tab:team_level_table   total network strength x outcomes (raw+partial) + ICC
Table 2  tab:net_iccs           Gini / Freeman(weighted) ICC -- SUPERSEDED by
                                2026-07-26_paper_table2.py (23-cell WY family); the
                                manuscript's assortativity and unweighted-centralization
                                cells no longer exist
Table 4  tab:style-outcome-corr zone involvement share x outcomes, raw + FIFA-controlled

All at the sidebar default edge method (min) so the whole paper uses ONE network.
"""
import warnings, importlib.util, sys, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, pandas as pd
from scipy.stats import pearsonr, t as t_dist, f as f_dist, mannwhitneyu

spec = importlib.util.spec_from_file_location('tla', 'scripts/2026-05-28_team_level_analysis.py')
m = importlib.util.module_from_spec(spec); sys.modules['tla'] = m; spec.loader.exec_module(m)

pd.set_option("display.width", 240)
print(f"### edge method = {m.method!r}   (sidebar default)\n")

THR = 1
df = m.process(m.edge_dfs[m.method], THR)
print(f"process(): {len(df)} match-teams, {df.team_name.nunique()} teams\n")


def stars(p, lv=(0.001, 0.01, 0.05)):
    return "***" if p < lv[0] else "**" if p < lv[1] else "*" if p < lv[2] else ""


def partial_r(x, y, z):
    """Partial Pearson r of x,y given z, with the CORRECT df = n-3."""
    s = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    n = len(s)
    a = s.x - np.polyval(np.polyfit(s.z, s.x, 1), s.z)
    b = s.y - np.polyval(np.polyfit(s.z, s.y, 1), s.z)
    r = float(np.corrcoef(a, b)[0, 1])
    dfree = n - 3
    tt = r * np.sqrt(dfree / max(1 - r * r, 1e-300))
    return r, float(2 * t_dist.sf(abs(tt), dfree)), n


def icc11(d, col):
    s = d[["team_name", col]].dropna()
    g = s.groupby("team_name")[col]
    nt, ng, sz, mn = len(s), g.ngroups, g.count(), g.mean()
    msb = (sz * (mn - s[col].mean()) ** 2).sum() / (ng - 1)
    msw = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (nt - ng)
    k0 = (nt - (sz ** 2).sum() / nt) / (ng - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    F = msb / msw
    return icc, F, float(f_dist.sf(F, ng - 1, nt - ng)), nt, ng


# ══ TABLE 1 ═══════════════════════════════════════════════════════════════════
print("=" * 100)
print("TABLE 1  tab:team_level_table   (total network strength)")
print("=" * 100)
rows = []
for w in m.WEIGHT_COLS:
    r = {"metric": w}
    for oc in m.OUTCOME_COLS:
        s = df[[w, oc]].dropna()
        rr, pp = pearsonr(s[w], s[oc])
        r[f"{oc}|raw"] = f"{rr:.2f}{stars(pp)}"
        r[f"{oc}|raw_p"] = round(pp, 4)
        pr, pv, nn = partial_r(df[w], df[oc], df["passes_against"])
        r[f"{oc}|par"] = f"{pr:.2f}{stars(pv)}"
        r[f"{oc}|par_p"] = round(pv, 4)
    i, F, p, nt, ng = icc11(df, w)
    r["ICC"] = round(i, 3); r["ICC_F"] = round(F, 2); r["ICC_p"] = round(p, 4)
    rows.append(r)
T1 = pd.DataFrame(rows).set_index("metric")
T1["ICC_q"] = np.round(m._bh_qvalues(T1["ICC_p"].values), 4)
T1["ICC_str"] = [f"{v:.2f}{stars(q)}" for v, q in zip(T1.ICC, T1.ICC_q)]
print(T1[[c for c in T1.columns if not c.endswith("_p")]].to_string())
print("\np-values:")
print(T1[[c for c in T1.columns if c.endswith("_p")]].to_string())
print("\nBH-q over all 36 correlation cells of table 1:")
allp = [T1.loc[w, f"{oc}|{k}_p"] for w in m.WEIGHT_COLS for oc in m.OUTCOME_COLS for k in ("raw", "par")]
q36 = m._bh_qvalues(allp)
print(pd.DataFrame({"cell": [f"{w} {oc} {k}" for w in m.WEIGHT_COLS for oc in m.OUTCOME_COLS
                             for k in ("raw", "par")], "p": np.round(allp, 4),
                    "q": np.round(q36, 4)}).to_string(index=False))

# ══ TABLE 2 ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TABLE 2  tab:net_iccs   (topology ICC)")
print("=" * 100)
# SUPERSEDED by scripts/2026-07-26_paper_table2.py (23-cell family, Westfall-Young).
# Two of the manuscript's three cells no longer exist: unweighted Freeman
# centralization and degree assortativity were deleted from the analysis on
# 2026-07-26 with the rest of the topology-only families, so the paper's 3-cell BH
# table cannot be reproduced -- only its Gini and weighted-Freeman rows can.
TOPO = {"Gini coefficient": "_gini",
        "Freeman centralization (weighted)": "_centralization_weighted"}
rows = []
for w in m.WEIGHT_COLS:
    for lab, suf in TOPO.items():
        c = w + suf
        if c not in df.columns:
            continue
        i, F, p, nt, ng = icc11(df, c)
        rows.append(dict(weight=w, metric=lab, ICC=round(i, 3), F=round(F, 2),
                         p=round(p, 4), n=nt, teams=ng))
T2 = pd.DataFrame(rows)
sub = T2[T2.weight == "valued_involvement"].copy()
sub["q(cell)"] = np.round(m._bh_qvalues(sub["p"].values), 4)
sub["sig"] = [stars(q) for q in sub["q(cell)"]]
print(f"--- the {len(sub)} surviving cells of the paper's 3-cell version, "
      f"valued_involvement (assortativity and unweighted centralization are gone) ---")
print(sub.to_string(index=False))
print(f"\n--- all weights x all surviving topology metrics "
      f"(BH over the whole {len(T2)}-cell block) ---")
T2["q(all)"] = np.round(m._bh_qvalues(T2["p"].values), 4)
T2["sig"] = [stars(q) for q in T2["q(all)"]]
print(T2.sort_values("p").to_string(index=False))

# correlations of the topology metrics with outcomes (paper claims all p>0.05)
print("\n--- topology x outcome correlations (paper: 'all p>0.05') ---")
rows = []
for w in m.WEIGHT_COLS:
    for lab, suf in TOPO.items():
        c = w + suf
        if c not in df.columns:
            continue
        for oc in m.OUTCOME_COLS:
            s = df[[c, oc]].dropna()
            rr, pp = pearsonr(s[c], s[oc])
            rows.append(dict(metric=c, outcome=oc, r=round(rr, 3), p=round(pp, 4), n=len(s)))
TC = pd.DataFrame(rows)
TC["q"] = np.round(m._bh_qvalues(TC["p"].values), 4)
print(TC[TC.p < 0.05].sort_values("p").to_string(index=False))
print(f"({(TC.p < 0.05).sum()} of {len(TC)} raw p<0.05; {(TC.q < 0.05).sum()} survive BH)")

# ══ eta^2 numbers quoted in section 5.2 ═══════════════════════════════════════
print("\n" + "=" * 100)
print("SECTION 5.2 numbers (eta^2 axis selection)")
print("=" * 100)
E = m.eta_sq_tbl(df, nperm=2000)
print(E.head(15).to_string(index=False))
a, b = "valued_involvement", "valued_involvement_centralization_weighted"
s = df[[a, b]].dropna()
print(f"\ncorr(total strength, Freeman cent_w) [{a}]: r={pearsonr(s[a], s[b])[0]:.3f} "
      f"p={pearsonr(s[a], s[b])[1]:.3f}  n={len(s)}")
s = df[[a, "block_line_height"]].dropna()
print(f"corr(total strength, line height): r={pearsonr(s[a], s['block_line_height'])[0]:.3f} "
      f"p={pearsonr(s[a], s['block_line_height'])[1]:.4f}  n={len(s)}")
s = df[[a, "block_x_spread"]].dropna()
print(f"corr(total strength, block depth): r={pearsonr(s[a], s['block_x_spread'])[0]:.3f} "
      f"p={pearsonr(s[a], s['block_x_spread'])[1]:.4f}  n={len(s)}")

# ══ TABLE 4 ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TABLE 4  tab:style-outcome-corr   (zone share x outcome, team level)")
print("=" * 100)
for kind in ("raw", "valued"):
    ts = m.build_team_style(m.zone_raw, "thirds", m.outcomes, kind=kind)
    ts["fifa_rating"] = ts.index.map(m.fifa_team_rating)
    rows = []
    for pre, ml in [("", "Involvement"), ("con_", "Contribution"), ("fault_", "Fault")]:
        for z, zl in [("high_press", "High press"), ("mid", "Mid"), ("own", "Own third")]:
            key = f"{pre}{z}_share"
            r = {"metric": ml, "zone": zl}
            for oc in m.OUTCOME_COLS:
                s = ts[[key, oc]].dropna()
                rr, pp = pearsonr(s[key], s[oc])
                r[f"raw|{oc[:5]}"] = rr; r[f"raw|{oc[:5]}_p"] = pp
                pr, pv, _ = partial_r(ts[key], ts[oc], ts["fifa_rating"])
                r[f"fifa|{oc[:5]}"] = pr; r[f"fifa|{oc[:5]}_p"] = pv
            rows.append(r)
    T4 = pd.DataFrame(rows)
    pcols = [c for c in T4.columns if c.endswith("_p")]
    flat = T4[pcols].values.flatten()
    qs = m._bh_qvalues(flat).reshape(T4[pcols].shape)
    disp = T4[["metric", "zone"]].copy()
    for j, pc in enumerate(pcols):
        base = pc[:-2]
        disp[base] = [f"{v:+.2f}{stars(q, (0.001, 0.01, 0.05))}"
                      for v, q in zip(T4[base], qs[:, j])]
    print(f"\n--- kind={kind}  (n={len(ts)} teams; stars = BH-FDR q over all "
          f"{flat.size} cells of the table) ---")
    print(disp.to_string(index=False))
    if kind == "raw":
        rawp = T4[["metric", "zone"]].copy()
        for pc in pcols:
            rawp[pc] = np.round(T4[pc], 4)
        print("\nuncorrected p:")
        print(rawp.to_string(index=False))
        print("\nBH q:")
        qd = T4[["metric", "zone"]].copy()
        for j, pc in enumerate(pcols):
            qd[pc[:-2]] = np.round(qs[:, j], 4)
        print(qd.to_string(index=False))

# knockout Mann-Whitney claim
ts = m.build_team_style(m.zone_raw, "thirds", m.outcomes, kind="raw")
print("\n--- knockout vs group-only, Mann-Whitney (paper: all p>0.05) ---")
for pre, ml in [("", "inv"), ("con_", "con"), ("fault_", "fault")]:
    for z in ["high_press", "mid", "own"]:
        k = f"{pre}{z}_share"
        a_ = ts[ts.reached_knockout][k].dropna(); b_ = ts[~ts.reached_knockout][k].dropna()
        print(f"  {ml:6s} {z:11s} p={mannwhitneyu(a_, b_)[1]:.3f}")
