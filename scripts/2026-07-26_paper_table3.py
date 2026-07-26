"""Regenerate the paper's Table 3 (tab:zonal_identity) AFTER the orientation fix.

Self-similarity estimator only (within- vs between-team cosine of the role-pair
co-defending pattern), 6 edge weights x 3 defensive phases = 18 cells, Westfall-Young
step-down min-p with B=10,000 shared team-label permutations -- exactly the family
`pressing_wy_family` uses for its `self` block, just without the two other estimators
(which the paper does not report) so B can be pushed to 10k.

Everything is computed on the match-teams present in ALL 18 cells, so one permutation
is meaningful everywhere and within/between/delta are mutually consistent.
"""
import warnings, importlib.util, sys, logging, time
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location('tla', 'scripts/2026-05-28_team_level_analysis.py')
m = importlib.util.module_from_spec(spec); sys.modules['tla'] = m; spec.loader.exec_module(m)

B = 10000
SEED = 20260710
METHOD = m.method
print(f"method={METHOD}  B={B}\n", flush=True)

cells = [(w, z) for w in m.WEIGHT_COLS for z in m.PRESS_FAMILY_ZONES]
pats = {c: m.pressing_role_patterns(METHOD, c[0], c[1]) for c in cells}
idx = None
for c in cells:
    s = pd.Index(pats[c][1])
    idx = s if idx is None else idx.intersection(s)
idx = list(idx)
n = len(idx)
teams = np.array([m._TEAM_BY_ID.get(int(x.split("_")[1]), x.split("_")[1]) for x in idx])
codes, uniq = pd.factorize(teams)
print(f"common match-teams: {n}   teams: {len(uniq)}", flush=True)

U = {c: m._unit_rows(pd.DataFrame(pats[c][0], index=pats[c][1],
                                  columns=pats[c][3]).loc[idx].values) for c in cells}

iu = np.triu_indices(n, 1)
npairs = len(iu[0])
CV = np.array([(U[c] @ U[c].T)[iu] for c in cells])          # (18, npairs)
tot = CV.sum(1)
same = (codes[iu[0]] == codes[iu[1]]).astype(np.float64)
k_obs = same.sum()
within_obs = CV @ same / k_obs
between_obs = (tot - CV @ same) / (npairs - k_obs)
obs = within_obs - between_obs
print(f"pairs: {npairs}  within-team pairs: {int(k_obs)}\n", flush=True)

rng = np.random.default_rng(SEED)
null = np.empty((B, len(cells)))
t0 = time.time()
CHUNK = 500
for s0 in range(0, B, CHUNK):
    s1 = min(s0 + CHUNK, B)
    M = np.empty((s1 - s0, npairs), dtype=np.float64)
    for b in range(s0, s1):
        pc = codes[rng.permutation(n)]
        M[b - s0] = pc[iu[0]] == pc[iu[1]]
    k = M.sum(1)                                             # constant = k_obs
    S = M @ CV.T                                             # (chunk, 18)
    null[s0:s1] = S / k[:, None] - (tot[None, :] - S) / (npairs - k)[:, None]
    print(f"  {s1}/{B}  ({time.time()-t0:.0f}s)", flush=True)

p_raw, p_wy, alpha, meff = m._wy_stepdown(obs, null)

T = pd.DataFrame({
    "weight": [c[0] for c in cells],
    "phase": [c[1] for c in cells],
    "within": np.round(within_obs, 3),
    "between": np.round(between_obs, 3),
    "delta": np.round(obs, 3),
    "p": np.round(p_raw, 4),
    "p_WY": np.round(p_wy, 4),
})
pd.set_option("display.width", 200)
print("\n" + "=" * 90)
print("TABLE 3  tab:zonal_identity  (POST orientation fix)")
print("=" * 90)
print(T.to_string(index=False))
print(f"\nalpha_FWER={alpha:.4f}   m_eff={meff:.1f} of {len(cells)}   "
      f"#p<.05={int((T.p<.05).sum())}   #p_WY<.05={int((T.p_WY<.05).sum())}")
print("\nsorted by p_WY:")
print(T.sort_values("p_WY").to_string(index=False))
T.to_csv("scripts/2026-07-26_table3_postfix.csv", index=False)
