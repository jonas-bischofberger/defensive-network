import functools

import sklearn.utils

import pandas as pd
import numpy as np
import streamlit as st


def aggregate_matchsums(df_matchsums, group_cols, minute_col="minutes"):
    """
    # >>> df = pd.DataFrame({"player_id": ["a", "b", "c"], "metric_1": [0.1, 0.2, 0.3], "metric_2": [0.4, 0.5, 0.6], "minutes": [90, 90, 90]})
    # >>> aggregate_matchsums(df, ["player_id"])
    #            metric_1  metric_2  minutes
    # player_id
    # a               0.1       0.4       90
    # b               0.2       0.5       90
    # c               0.3       0.6       90
    """
    numeric_cols = [col for col in df_matchsums.select_dtypes(include=['number']).columns.tolist() if col != minute_col]
    dfg = df_matchsums.groupby(group_cols)[numeric_cols + [minute_col]].sum()
    dfg_total = dfg[numeric_cols].div(dfg[minute_col], axis=0) * 90 # normalize by minutes
    dfg_total[minute_col] = dfg[minute_col]
    return dfg_total


def compute_bv(df_matchsums, player_col, agg_func, agg_metric_cols, n_bootstrap=1000):
    from sklearn.utils import resample
    import numpy as np
    import pandas as pd

    player_ids = df_matchsums[player_col].unique()
    # Initialize: {metric: {player_id: [boot_samples]}}
    metric_boot_samples = {metric: {pid: [] for pid in player_ids} for metric in agg_metric_cols}

    for _ in range(n_bootstrap):
        resampled_dfs = []

        for pid in player_ids:
            df_player = df_matchsums[df_matchsums[player_col] == pid]
            if len(df_player) == 0:
                continue
            df_resampled = resample(df_player, replace=True)
            resampled_dfs.append(df_resampled)

        df_boot = pd.concat(resampled_dfs, axis=0)
        df_agg = agg_func(df_boot)

        for pid in player_ids:
            if pid not in df_agg.index:
                continue
            for metric in agg_metric_cols:
                metric_boot_samples[metric][pid].append(df_agg.loc[pid, metric])

    # Compute BV = average variance of each player's metric across resamples
    bv_dict = {}
    for metric in agg_metric_cols:
        player_vars = []
        for pid in player_ids:
            values = metric_boot_samples[metric][pid]
            if len(values) > 1:
                player_vars.append(np.var(values, ddof=1))
        bv_dict[metric] = np.mean(player_vars) if player_vars else np.nan

    return pd.Series(bv_dict)


def discrimination(df_matchsums, player_col, agg_metric_cols, agg_func):
    # Dsm = 1 − Esm[Vspm[X]] / Vsm[X]

    def bootstrap_matchsum_aggregation(df_matchsums, agg_func, n_bootstrap=1000):
        dfgs = []
        for i in range(n_bootstrap):
            df_resampled_matchsums = sklearn.utils.resample(df_matchsums, replace=True)
            df_resampled_agg = agg_func(df_resampled_matchsums)
            dfgs.append(df_resampled_agg.reset_index())
        return pd.concat(dfgs, axis=0).reset_index(drop=True)

    df = bootstrap_matchsum_aggregation(df_matchsums, agg_func)
    # dfg_bv = df.groupby(player_col).agg({
    #     col: "var" for col in agg_metric_cols
    # }).mean()
    dfg_bv = compute_bv(df_matchsums, player_col, agg_func, agg_metric_cols)

    st.write("dfg_bv")

    st.write(dfg_bv)

    def bootstraped_variances(df_agg, n_bootstrap=1000):
        dfgs = []
        for i in range(n_bootstrap):
            df_resampled_agg = sklearn.utils.resample(df_agg, replace=True)
            df_resampled_agg_var = df_resampled_agg[agg_metric_cols].var()
            dfgs.append(df_resampled_agg_var.reset_index())
        return pd.concat(dfgs, axis=0).groupby("index").mean().T

    df_agg = agg_func(df_matchsums)
    # dfg_sv = df_agg[agg_metric_cols].var()
    dfg_sv = df_agg[agg_metric_cols].var(ddof=1)
    st.write("dfg_sv")
    st.write(dfg_sv)
    dfg_sv2 = bootstraped_variances(df_agg)
    st.write("dfg_sv2")
    st.write(dfg_sv2)

    discrimination = 1 - (dfg_bv / dfg_sv2)
    st.write("Discrimination")
    st.write(discrimination)
    st.write(1 - (dfg_bv / dfg_sv))
    print(discrimination)


def bayesian():
    import pandas as pd

    # Simulate some toy match-level data
    df = pd.DataFrame({
        "player_id": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
        "metric": [0.1 + 0.01 * i for i in range(10)] +  # Player A: stable
                  [0.2 + 0.05 * i for i in range(10)] +  # Player B: variable
                  [0.3 + 0.00 * i for i in range(10)],  # Player C: constant
        "minutes": [90] * 30
    })
    st.write(df)

    import pymc as pm
    import numpy as np
    import arviz as az

    # Encode players as integer indices
    df["player_idx"] = pd.Categorical(df["player_id"]).codes
    player_idx = df["player_idx"].values
    y = df["metric"].values
    n_players = df["player_idx"].nunique()

    st.write(df)

    # y_ij = μ+offset_i + ϵ_ij
    with pm.Model() as model:
        # Priors
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        sigma_between = pm.HalfNormal("sigma_between", sigma=1.0)
        sigma_within = pm.HalfNormal("sigma_within", sigma=1.0)
        # vis
        st.write("mu", mu)
        st.write("sigma_between", sigma_between)
        st.write("sigma_within", sigma_within)

        # Random intercept per player
        player_offset = pm.Normal("player_offset", mu=0.0, sigma=sigma_between, shape=n_players)
        player_mean = mu + player_offset[player_idx]
        st.write("player_mean", player_mean)
        st.write("player_offset", player_offset)

        # Observation model
        y_obs = pm.Normal("y_obs", mu=player_mean, sigma=sigma_within, observed=y)
        st.write("y_obs", y_obs)

        # Derived: Discrimination
        discrimination = pm.Deterministic("discrimination", 1 - sigma_within**2 / (sigma_within**2 + sigma_between**2))
        st.write("discrimination", discrimination)

        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)
        st.write("Trace")
        st.write(trace)

        az.summary(trace, var_names=["sigma_between", "sigma_within", "discrimination"])
        az.plot_posterior(trace, var_names=["discrimination"])


def simplest_discrimination(df, player_col, metric_col):
    df_clean = df[[player_col, metric_col]].dropna()

    var_total = df_clean[metric_col].var(ddof=1)

    var_within = (
        df_clean
        .groupby(player_col)[metric_col]
        .var(ddof=1)
        .dropna()
        .mean()
    )
    var_between = (
        df_clean
        .groupby(player_col)[metric_col]
        .mean()
        .var(ddof=1)
    )

    stability = 1 - (var_within / var_total)
    discrimination = var_between / var_total
    return stability, discrimination


@st.cache_resource
def _get_csv(fpath):
    return pd.read_csv(fpath)

@st.cache_resource
def _get_parquet(fpath):
    return pd.read_parquet(fpath)


if __name__ == '__main__':
    df = pd.DataFrame({
        "player_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
        "season": ["2021", "2022", "2023"] * 4,
        "metric_1": [0.1, 0.11, 0.09, 0.2, 0.21, 0.19, 0.3, 0.31, 0.29, 0.4, 0.41, 0.39],  # high stabiliy, high discrimination
        "metric_1b": [0.1, 0.1001, 0.0999, 0.2, 0.2001, 0.1999, 0.3, 0.3001, 0.2999, 0.4, 0.4001, 0.3999],  # extremely high stabiliy, high discrimination
        "metric_2": [0.1, 0.5, 0.9, 0.099, 0.499, 0.899] * 2,  # low stability, low discrimination
        "metric_3": [10, 50, 90, 9.9, 49.9, 89.9] * 2,  # same as metric_2 but scaled
        "minutes": [90] * 12,
    })
    df = _get_parquet("s3://vfb-datalake/tracking_general_processing/skillcorner/v3/preprocessed/physical.parquet")
    st.write("df")
    st.write(df.head())

    st.write("Simplest discrimination")
    for metric in ["metric_1", "metric_1b", "metric_2", "metric_3"]:
        disc, stability = simplest_discrimination(df, "player_id", metric)
        st.write(f"Simplest Discrimination for {metric}: {disc}, Stability: {stability}")

    agg_by_player = functools.partial(aggregate_matchsums, group_cols=["player_id"], minute_col="minutes")
    discrimination(df, "player_id", ["metric_1", "metric_1b", "metric_2", "metric_3"], agg_func=agg_by_player)

    st.stop()
