import pandas as pd
import pingouin as pg


def partial_correlation_matrix(df, covariate_cols, y_col, methods=["pearson", "spearman"], exclude_covariate_cols=True):
    """
    >>> df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [3, -3, -4, 3], "c": [1, -2, -15, 10], "d": [1, 2, 3, 5], "e": [0, 0, 0, 0]})
    >>> partial_correlation_matrix(df, ["b"], "c")
      variable    method  n         r        CI95%     p-val
    0        a   pearson  4  0.203310  [-1.0, 1.0]  0.869660
    1        d   pearson  4  0.233097  [-1.0, 1.0]  0.850228
    2        e   pearson  4       NaN          NaN       NaN
    3        a  spearman  4  0.301681  [-1.0, 1.0]  0.804905
    4        d  spearman  4  0.301681  [-1.0, 1.0]  0.804905
    5        e  spearman  4       NaN          NaN       NaN
    """
    dfs = []
    for method in methods:
        for col in df.columns:
            if col == y_col:
                continue
            if col in covariate_cols and exclude_covariate_cols:
                continue
            df_pcorr = pg.partial_corr(data=df, method=method, x=col, y=y_col, x_covar=covariate_cols)
            df_pcorr["variable"] = col
            dfs.append(df_pcorr)
    df_pcorr = pd.concat(dfs).reset_index(names="method")
    df_pcorr = df_pcorr[["variable"] + [col for col in df_pcorr.columns if col != "variable"]]
    return df_pcorr
