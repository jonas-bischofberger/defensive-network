import pandas as pd
import numpy as np


def add_velocity(df_tracking, time_col="datetime_tracking", player_col="player_id", x_col="x_tracking", y_col="y_tracking", new_vx_col="vx", new_vy_col="vy", new_v_col="v"):
    """
    >>> import defensive_network.utility.dataframes
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df_tracking = pd.DataFrame({"datetime_tracking": [0, 1e9, 2e9] * 2, "player_id": [0, 0, 0, 1, 1, 1], "x_tracking": [1, 2, 4] * 2, "y_tracking": [0, 0, 0] * 2})
    >>> add_velocity(df_tracking)
        datetime_tracking  player_id  x_tracking  y_tracking   vx   vy    v
    0 1970-01-01 00:00:00          0           1           0  1.0  0.0  1.0
    1 1970-01-01 00:00:01          0           2           0  1.0  0.0  1.0
    2 1970-01-01 00:00:02          0           4           0  2.0  0.0  2.0
    3 1970-01-01 00:00:00          1           1           0  1.0  0.0  1.0
    4 1970-01-01 00:00:01          1           2           0  1.0  0.0  1.0
    5 1970-01-01 00:00:02          1           4           0  2.0  0.0  2.0
    """
    df_tracking[time_col] = pd.to_datetime(df_tracking[time_col])
    df_tracking = df_tracking.sort_values(time_col)
    groups = []
    for player, df_tracking_player in df_tracking.groupby(player_col):
        df_tracking_player = df_tracking_player.sort_values(time_col)
        df_tracking_player[new_vx_col] = df_tracking_player[x_col].diff() / df_tracking_player[time_col].diff().dt.total_seconds()
        df_tracking_player[new_vy_col] = df_tracking_player[y_col].diff() / df_tracking_player[time_col].diff().dt.total_seconds()
        if len(df_tracking_player) > 1:
            # df_tracking_player[new_vx_col].iloc[0] = df_tracking_player[new_vx_col].iloc[1]
            # df_tracking_player[new_vy_col].iloc[0] = df_tracking_player[new_vy_col].iloc[1]
            df_tracking_player.loc[df_tracking_player.index[0], new_vx_col] = df_tracking_player.iloc[1][new_vx_col]
            df_tracking_player.loc[df_tracking_player.index[0], new_vy_col] = df_tracking_player.iloc[1][new_vy_col]

        groups.append(df_tracking_player)

    df = pd.concat(groups)
    if new_v_col is not None:
        df[new_v_col] = np.sqrt(df[new_vx_col] ** 2 + df[new_vy_col] ** 2)
    return df
