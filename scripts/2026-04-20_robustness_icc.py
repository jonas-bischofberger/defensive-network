"""
Robustness analysis of defensive network metrics at the team level.

Uses FIFA Men's World Cup 2022 data.  Each team plays 3-7 matches, giving
repeated observations of the same team's defensive network.

Three complementary measures of stability are reported per metric:

  ICC(1,1)  — one-way random-effects intraclass correlation coefficient.
              Teams = subjects, matches = replicate measurements.
              High ICC means the metric reliably distinguishes teams across
              matches (between-team variance >> within-team variance).
              Reference thresholds (Koo & Mae 2016):
                < 0.50  poor  |  0.50–0.75  moderate  |  0.75–0.90  good  |  > 0.90  excellent

  CV        — within-team coefficient of variation (std / |mean|), averaged
              over teams.  Complements ICC by showing absolute within-team
              fluctuation irrespective of how teams differ from each other.

  Mixed-LME — ICC re-derived from a random-intercept linear mixed model
              (team as random effect).  Equivalent to ICC(1,1) but relies on
              REML estimation rather than SS decomposition, which handles
              unbalanced designs more precisely.
"""

import sys
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive
import defensive_network.utility.general

FOLDER = "involvement/10/"
MATCH_FILTER = "fifa-men-s-world-cup-2022"
METRICS = ["valued_involvement", "valued_fault", "valued_contribution"]
OUT_DIR = os.path.dirname(__file__)
OUT_CSV = os.path.join(OUT_DIR, "2026-04-20_robustness_icc.csv")
OUT_PLOT_ICC = os.path.join(OUT_DIR, "2026-04-20_robustness_icc.png")
OUT_PLOT_CV = os.path.join(OUT_DIR, "2026-04-20_robustness_cv.png")
OUT_PLOT_PROFILES = os.path.join(OUT_DIR, "2026-04-20_robustness_profiles.png")


# ---------------------------------------------------------------------------
# Data loading  (mirrors 2026-04-19_world_cup_outcome_correlation.py)
# ---------------------------------------------------------------------------

def _compute_team_network_metrics(df_team: pd.DataFrame) -> dict:
    result = {}
    for metric in METRICS:
        if metric not in df_team.columns:
            continue
        df_m = df_team[df_team[metric].fillna(0) != 0].copy()
        if df_m.empty:
            continue
        pair_count: dict = {}
        pair_weight: dict = {}
        for _pass_id, df_pass in df_m.groupby("involvement_pass_id"):
            inv = df_pass.set_index("defender_name")[metric].to_dict()
            defenders = sorted(inv.keys())
            if len(defenders) < 2:
                continue
            for a, b in combinations(defenders, 2):
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
                pair_weight[(a, b)] = pair_weight.get((a, b), 0) + (inv[a] + inv[b]) / 2
        nodes: set = set()
        for a, b in pair_count:
            nodes.add(a)
            nodes.add(b)
        N = len(nodes)
        if N < 2:
            continue
        max_edges = N * (N - 1) / 2
        result[f"{metric}_n_players"] = N
        result[f"{metric}_n_edges"] = len(pair_count)
        result[f"{metric}_edge_density"] = len(pair_count) / max_edges
        result[f"{metric}_coactivity"] = float(sum(pair_count.values()))
        result[f"{metric}_total_weight"] = float(sum(pair_weight.values()))
        result[f"{metric}_weight_density"] = float(sum(pair_weight.values())) / max_edges
    return result


def load_data() -> pd.DataFrame:
    print("Listing Drive files...")
    files = defensive_network.parse.drive.list_files_in_drive_folder(FOLDER)
    rows = []
    target_files = [f for f in files if MATCH_FILTER in f["name"] and f["name"].endswith(".parquet")]
    for f in defensive_network.utility.general.progress_bar(target_files, desc="Computing network metrics"):
        file_name = f["name"]
        match_name = file_name.replace(".parquet", "")
        df_match = defensive_network.parse.drive.download_parquet_from_drive(FOLDER + file_name)
        match_id = df_match["match_id"].iloc[0]
        for defending_team, df_team in df_match.groupby("defending_team"):
            row = {"match_id": match_id, "match_name": match_name, "defending_team": defending_team}
            row.update(_compute_team_network_metrics(df_team))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical methods
# ---------------------------------------------------------------------------

def icc_oneway(data: pd.DataFrame, subject_col: str, measure_col: str) -> dict:
    """
    ICC(1,1): one-way random-effects, single measures.

    Returns a dict with: icc, ci_lower, ci_upper, p_value, n_subjects,
    mean_k (harmonic mean of measurements per subject), ms_between, ms_within.
    """
    groups = (
        data.groupby(subject_col)[measure_col]
        .apply(list)
        .apply(lambda v: [x for x in v if not np.isnan(x)])
    )
    groups = groups[groups.map(len) >= 2]

    nan_result = {k: np.nan for k in
                  ["icc", "ci_lower", "ci_upper", "p_value", "n_subjects", "mean_k",
                   "ms_between", "ms_within", "f_stat"]}

    if len(groups) < 2:
        return nan_result

    n = len(groups)
    all_vals = [x for vals in groups for x in vals]
    grand_mean = np.mean(all_vals)

    ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in groups)
    ss_within = sum(sum((x - np.mean(vals)) ** 2 for x in vals) for vals in groups)
    df_between = n - 1
    df_within = sum(len(vals) - 1 for vals in groups)

    if df_between == 0 or df_within == 0:
        return nan_result

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # harmonic mean of k per subject
    ks = [len(v) for v in groups]
    k0 = len(ks) / sum(1.0 / k for k in ks)

    denom = ms_between + (k0 - 1) * ms_within
    if denom == 0:
        return nan_result

    icc = (ms_between - ms_within) / denom
    f_stat = ms_between / ms_within if ms_within > 0 else np.inf
    p_value = 1 - scipy.stats.f.cdf(f_stat, df_between, df_within)

    # 95 % CI via F-distribution (Shrout & Fleiss 1979, eq. (7))
    alpha = 0.05
    fu = scipy.stats.f.ppf(1 - alpha / 2, df_between, df_within)
    fl = scipy.stats.f.ppf(1 - alpha / 2, df_within, df_between)
    fl = max(fl, 1e-9)

    f_low = f_stat / fu
    f_high = f_stat * fl

    ci_lower = (f_low - 1) / (f_low + k0 - 1)
    ci_upper = (f_high - 1) / (f_high + k0 - 1)

    return dict(icc=icc, ci_lower=ci_lower, ci_upper=ci_upper, p_value=p_value,
                n_subjects=n, mean_k=k0, ms_between=ms_between, ms_within=ms_within,
                f_stat=f_stat)


def icc_lme(data: pd.DataFrame, subject_col: str, measure_col: str) -> float:
    """
    ICC from random-intercept LME: sigma_b^2 / (sigma_b^2 + sigma_e^2).
    Falls back to nan on error.
    """
    try:
        import statsmodels.formula.api as smf
        df = data[[subject_col, measure_col]].dropna().copy()
        df = df.rename(columns={subject_col: "subject", measure_col: "y"})
        # need >= 2 groups with >= 2 obs
        counts = df.groupby("subject")["y"].count()
        df = df[df["subject"].isin(counts[counts >= 2].index)]
        if df["subject"].nunique() < 2:
            return np.nan
        md = smf.mixedlm("y ~ 1", df, groups=df["subject"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = md.fit(reml=True, method="lbfgs")
        sigma_b2 = float(res.cov_re.iloc[0, 0])
        sigma_e2 = float(res.scale)
        if sigma_b2 + sigma_e2 == 0:
            return np.nan
        return sigma_b2 / (sigma_b2 + sigma_e2)
    except Exception:
        return np.nan


def within_team_cv(data: pd.DataFrame, subject_col: str, measure_col: str) -> pd.Series:
    """CV (std/|mean|) per team; returns Series indexed by team."""
    def _cv(x):
        m = x.mean()
        return np.nan if m == 0 or np.isnan(m) else x.std(ddof=1) / abs(m)
    return data.groupby(subject_col)[measure_col].apply(_cv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- load / map IDs -------------------------------------------------------
    print("Loading meta...")
    df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=False)
    df_meta = df_meta[df_meta["competition_name"] == "FIFA Men's World Cup"].copy()
    df_meta["home_team_id"] = df_meta["home_team_id"].astype(str).str.replace(".0", "", regex=False)
    df_meta["guest_team_id"] = df_meta["guest_team_id"].astype(str).str.replace(".0", "", regex=False)
    teamid2name = {
        **df_meta.set_index("home_team_id")["home_team_name"].to_dict(),
        **df_meta.set_index("guest_team_id")["guest_team_name"].to_dict(),
    }

    df = load_data()

    df_meta["match_id_key"] = df_meta["match_id"].astype(str).str.replace(".0", "", regex=False)
    matchid2string = df_meta.set_index("match_id_key")["match_string"].to_dict()
    df["match_id_key"] = df["match_id"].astype(str).str.replace(".0", "", regex=False)
    df["match_string"] = df["match_id_key"].map(matchid2string)
    df["team_name"] = df["defending_team"].astype(str).str.replace(".0", "", regex=False).map(teamid2name)
    df = df.dropna(subset=["team_name", "match_string"])

    print(f"\nDataset: {len(df)} team-match rows | "
          f"{df['team_name'].nunique()} teams | "
          f"{df['match_string'].nunique()} matches")

    network_cols = [c for c in df.columns if any(c.startswith(m + "_") for m in METRICS)]
    print(f"Analysing {len(network_cols)} metrics: {network_cols}\n")

    # --- per-metric analysis --------------------------------------------------
    rows = []
    for col in network_cols:
        valid = df[["team_name", "match_string", col]].dropna()
        counts = valid.groupby("team_name")[col].count()
        valid = valid[valid["team_name"].isin(counts[counts >= 2].index)]

        icc_res = icc_oneway(valid, "team_name", col)
        icc_lme_val = icc_lme(valid, "team_name", col)

        cv_series = within_team_cv(valid, "team_name", col)
        mean_cv = cv_series.mean()
        median_cv = cv_series.median()

        row = {
            "metric": col,
            "metric_group": next((m for m in METRICS if col.startswith(m + "_")), ""),
            "icc_oneway": round(icc_res["icc"], 3) if not np.isnan(icc_res["icc"]) else np.nan,
            "icc_ci_lower": round(icc_res["ci_lower"], 3) if not np.isnan(icc_res["ci_lower"]) else np.nan,
            "icc_ci_upper": round(icc_res["ci_upper"], 3) if not np.isnan(icc_res["ci_upper"]) else np.nan,
            "icc_p_value": round(icc_res["p_value"], 4) if not np.isnan(icc_res["p_value"]) else np.nan,
            "icc_lme": round(icc_lme_val, 3) if not np.isnan(icc_lme_val) else np.nan,
            "n_teams": int(icc_res["n_subjects"]) if not np.isnan(icc_res["n_subjects"]) else np.nan,
            "mean_matches_per_team": round(icc_res["mean_k"], 2) if not np.isnan(icc_res["mean_k"]) else np.nan,
            "f_stat": round(icc_res["f_stat"], 2) if not np.isnan(icc_res["f_stat"]) else np.nan,
            "mean_cv": round(mean_cv, 3) if not np.isnan(mean_cv) else np.nan,
            "median_cv": round(median_cv, 3) if not np.isnan(median_cv) else np.nan,
        }
        rows.append(row)
        print(f"  {col:45s}  ICC={row['icc_oneway']:.3f} [{row['icc_ci_lower']:.3f}, {row['icc_ci_upper']:.3f}]"
              f"  p={row['icc_p_value']:.4f}  LME-ICC={row['icc_lme']:.3f}  CV={row['mean_cv']:.3f}")

    df_res = pd.DataFrame(rows).sort_values("icc_oneway", ascending=False)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved: {OUT_CSV}")

    # -----------------------------------------------------------------------
    # Plot 1 – Forest plot of ICC(1,1) with 95 % CI, grouped by metric type
    # -----------------------------------------------------------------------
    def _icc_color(v):
        if np.isnan(v):
            return "#9E9E9E"
        if v >= 0.75:
            return "#1565C0"
        if v >= 0.50:
            return "#F57C00"
        return "#C62828"

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 6), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, mg in zip(axes, METRICS):
        sub = df_res[df_res["metric_group"] == mg].copy()
        sub["short"] = sub["metric"].str.replace(f"{mg}_", "", regex=False)
        sub = sub.sort_values("icc_oneway")

        y = np.arange(len(sub))
        colors = [_icc_color(v) for v in sub["icc_oneway"]]
        xerr_low = np.clip(sub["icc_oneway"] - sub["icc_ci_lower"], 0, None)
        xerr_high = np.clip(sub["icc_ci_upper"] - sub["icc_oneway"], 0, None)

        ax.barh(y, sub["icc_oneway"], color=colors, alpha=0.85,
                xerr=[xerr_low, xerr_high], capsize=4,
                error_kw={"ecolor": "#555", "linewidth": 1.2})

        # LME ICC as small black diamond overlay
        ax.scatter(sub["icc_lme"].values, y, marker="D", color="black",
                   s=30, zorder=5, label="LME-ICC")

        ax.axvline(0.75, color="#1565C0", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(0.50, color="#F57C00", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-0.15, 1.1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["short"], fontsize=9)
        ax.set_xlabel("ICC(1,1)", fontsize=10)
        ax.set_title(mg.replace("_", " ").title(), fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.legend(loc="lower right", fontsize=8)

    # shared legend for thresholds
    patches = [
        mpatches.Patch(color="#1565C0", alpha=0.85, label="Good/Excellent ≥0.75"),
        mpatches.Patch(color="#F57C00", alpha=0.85, label="Moderate 0.50–0.75"),
        mpatches.Patch(color="#C62828", alpha=0.85, label="Poor <0.50"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "ICC(1,1) with 95 % CI — Defensive Network Metric Robustness\n"
        "FIFA Men's World Cup 2022 | team level across matches | ◆ = LME-ICC",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT_PLOT_ICC, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ICC plot saved: {OUT_PLOT_ICC}")

    # -----------------------------------------------------------------------
    # Plot 2 – Within-team CV (mean over teams) for all metrics
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(df_res) * 0.4)))
    df_cv = df_res.sort_values("mean_cv")
    colors_cv = ["#388E3C" if v <= 0.20 else "#F57C00" if v <= 0.40 else "#C62828"
                 for v in df_cv["mean_cv"]]
    short_labels = df_cv["metric"].str.replace(
        "|".join(METRICS), "", regex=True
    ).str.strip("_")
    ax2.barh(short_labels, df_cv["mean_cv"], color=colors_cv, alpha=0.85)
    # median CV as dots
    ax2.scatter(df_cv["median_cv"].values, np.arange(len(df_cv)),
                marker="|", color="black", s=80, zorder=5, label="Median CV")
    ax2.axvline(0.20, color="#388E3C", linestyle="--", linewidth=1, alpha=0.7, label="CV=0.20")
    ax2.axvline(0.40, color="#F57C00", linestyle="--", linewidth=1, alpha=0.7, label="CV=0.40")
    ax2.set_xlabel("Mean within-team CV  (std / |mean|)", fontsize=10)
    ax2.set_title(
        "Within-team Coefficient of Variation\n"
        "(averaged over teams; | = median CV)",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PLOT_CV, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CV plot saved: {OUT_PLOT_CV}")

    # -----------------------------------------------------------------------
    # Plot 3 – Per-team profiles: mean ± std across matches, top ICC metric
    # -----------------------------------------------------------------------
    best_metric = df_res.dropna(subset=["icc_oneway"]).iloc[0]["metric"]
    print(f"\nPer-team profile plot for best-ICC metric: {best_metric}")

    valid = df[["team_name", "match_string", best_metric]].dropna()
    team_stats = valid.groupby("team_name")[best_metric].agg(["mean", "std", "count"])
    team_stats = team_stats[team_stats["count"] >= 2].sort_values("mean", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(14, max(5, len(team_stats) * 0.35)))
    y3 = np.arange(len(team_stats))
    ax3.barh(y3, team_stats["mean"], xerr=team_stats["std"], color="#1976D2", alpha=0.75,
             capsize=3, error_kw={"ecolor": "#555", "linewidth": 1})
    # individual match dots
    for i, (team, _) in enumerate(team_stats.iterrows()):
        vals = valid[valid["team_name"] == team][best_metric].values
        ax3.scatter(vals, [i] * len(vals), color="black", s=15, alpha=0.6, zorder=5)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(team_stats.index, fontsize=8)
    ax3.set_xlabel(best_metric, fontsize=10)
    ax3.set_title(
        f"Team profiles — {best_metric}\n"
        f"Bars = mean ± SD across matches; dots = individual matches\n"
        f"ICC(1,1) = {df_res[df_res['metric'] == best_metric]['icc_oneway'].iloc[0]:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PLOT_PROFILES, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Profiles plot saved: {OUT_PLOT_PROFILES}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(df_res[["metric", "icc_oneway", "icc_ci_lower", "icc_ci_upper",
                  "icc_p_value", "icc_lme", "mean_cv"]].to_string(index=False))


if __name__ == "__main__":
    main()
