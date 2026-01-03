import pandas as pd
import numpy as np


def add_velocity_old(df_tracking, time_col="datetime_tracking", player_col="player_id", x_col="x_tracking", y_col="y_tracking", new_vx_col="vx", new_vy_col="vy", new_v_col="v"):
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


def add_velocity(
    df_tracking,
    time_col="datetime_tracking",
    player_col="player_id",
    x_col="x_tracking",
    y_col="y_tracking",
    new_vx_col="vx",
    new_vy_col="vy",
    new_v_col="v",
):
    """
    Compute per-player velocities (vx, vy) and speed (v) from tracking positions.
    Robust to duplicate timestamps/frames per player: velocities are computed on the
    unique-time series, then mapped back to all duplicate rows at that time.

    >>> import pandas as pd
    >>> df_tracking = pd.DataFrame({
    ...     "datetime_tracking": [0, 1e9, 2e9] * 2,
    ...     "player_id": [0, 0, 0, 1, 1, 1],
    ...     "x_tracking": [1, 2, 4] * 2,
    ...     "y_tracking": [0, 0, 0] * 2
    ... })
    >>> add_velocity(df_tracking)
        datetime_tracking  player_id  x_tracking  y_tracking   vx   vy    v
    0 1970-01-01 00:00:00          0           1           0  1.0  0.0  1.0
    1 1970-01-01 00:00:01          0           2           0  1.0  0.0  1.0
    2 1970-01-01 00:00:02          0           4           0  2.0  0.0  2.0
    3 1970-01-01 00:00:00          1           1           0  1.0  0.0  1.0
    4 1970-01-01 00:00:01          1           2           0  1.0  0.0  1.0
    5 1970-01-01 00:00:02          1           4           0  2.0  0.0  2.0
    """
    # Work on a copy to avoid surprising in-place changes
    df_tracking = df_tracking.copy()

    # Ensure datetime
    df_tracking[time_col] = pd.to_datetime(df_tracking[time_col])

    # Match original behavior: sort globally by time, then process per player
    df_tracking = df_tracking.sort_values(time_col)

    groups = []
    for player, df_p in df_tracking.groupby(player_col):
        df_p = df_p.sort_values(time_col).copy()

        # Build a unique-time series for this player (keep last row at each timestamp)
        uniq = df_p.drop_duplicates(subset=[time_col], keep="last").sort_values(time_col).copy()

        # Compute velocity on unique-time series
        dt = uniq[time_col].diff().dt.total_seconds().replace(0, np.nan)
        uniq[new_vx_col] = uniq[x_col].diff() / dt
        uniq[new_vy_col] = uniq[y_col].diff() / dt

        # Fill first velocity with the next available (same intent as your original code)
        if len(uniq) > 1:
            uniq.iloc[0, uniq.columns.get_loc(new_vx_col)] = uniq.iloc[1][new_vx_col]
            uniq.iloc[0, uniq.columns.get_loc(new_vy_col)] = uniq.iloc[1][new_vy_col]

        # Map velocities back to all rows (duplicates in time get the same vx/vy)
        df_p = df_p.merge(
            uniq[[time_col, new_vx_col, new_vy_col]],
            on=time_col,
            how="left",
            validate="many_to_one",
        )

        groups.append(df_p)

    df = pd.concat(groups)

    if new_v_col is not None:
        df[new_v_col] = np.sqrt(df[new_vx_col] ** 2 + df[new_vy_col] ** 2)

    return df


if __name__ == '__main__':
    import defensive_network.utility.dataframes
    defensive_network.utility.dataframes.prepare_doctest()
    df_tracking = pd.DataFrame({"datetime_tracking": [0, 1e9, 2e9] * 2, "player_id": [0, 0, 0, 1, 1, 1], "x_tracking": [1, 2, 4] * 2, "y_tracking": [0, 0, 0] * 2})
    df_tracking = add_velocity(df_tracking)
    print(df_tracking)
