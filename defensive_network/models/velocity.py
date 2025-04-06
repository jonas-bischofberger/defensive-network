import pandas as pd
import numpy as np


def add_velocity(df_tracking, time_col="datetime_tracking", player_col="player_id", x_col="x_tracking", y_col="y_tracking", new_vx_col="vx", new_vy_col="vy", new_v_col="v"):
    """
    >>> df_tracking = pd.DataFrame
    """
    df_tracking["datetime_tracking"] = pd.to_datetime(df_tracking["datetime_tracking"])
    df_tracking = df_tracking.sort_values(time_col)
    groups = []
    for player, df_tracking_player in df_tracking.groupby(player_col):
        df_tracking_player = df_tracking_player.sort_values(time_col)
        df_tracking_player[new_vx_col] = df_tracking_player[x_col].diff() / df_tracking_player[time_col].diff().dt.total_seconds()
        df_tracking_player[new_vy_col] = df_tracking_player[y_col].diff() / df_tracking_player[time_col].diff().dt.total_seconds()
        groups.append(df_tracking_player)
    df = pd.concat(groups)
    if new_v_col is not None:
        df[new_v_col] = np.sqrt(df[new_vx_col] ** 2 + df[new_vy_col] ** 2)
    return df
