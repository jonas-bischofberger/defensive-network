import pandas as pd

import defensive_network.utility.dataframes


def get_average_tracking_positions_off_def(df_tracking, player_col="player_id", ball_player_id="ball", team_col="team_id", team_in_possession_col="ball_poss_team_id", x_col="x_norm", y_col="y_norm"):
    """
    >>> df_tracking = pd.DataFrame({"player_id": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"], "team_id": ["H", "H", "H", "H", "H", "H", "H", "H", "A", "A", "A", "A"], "ball_poss_team_id": ["H", "H", "A", "A", "H", "H", "A", "A", "H", "H", "A", "A"], "x_norm": range(12), "y_norm": range(1, 13)})
    >>> df_tracking
       player_id team_id ball_poss_team_id  x_norm  y_norm
    0          A       H                 H       0       1
    1          A       H                 H       1       2
    2          A       H                 A       2       3
    3          A       H                 A       3       4
    4          B       H                 H       4       5
    5          B       H                 H       5       6
    6          B       H                 A       6       7
    7          B       H                 A       7       8
    8          C       A                 H       8       9
    9          C       A                 H       9      10
    10         C       A                 A      10      11
    11         C       A                 A      11      12
    >>> get_average_tracking_positions_off_def(df_tracking)
    {'def': {'A': (2.5, 3.5), 'B': (6.5, 7.5), 'C': (8.5, 9.5)}, 'off': {'A': (0.5, 1.5), 'B': (4.5, 5.5), 'C': (10.5, 11.5)}}
    """
    i_not_ball = df_tracking[player_col] != ball_player_id

    is_attacking_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_tracking, "is_attacking")
    df_tracking.loc[i_not_ball, is_attacking_col] = df_tracking.loc[i_not_ball, team_col] == df_tracking.loc[i_not_ball, team_in_possession_col]
    data = {}
    for is_attacking, df_tracking_att_def in df_tracking.groupby(is_attacking_col):
        average_positions_off = df_tracking_att_def.groupby(player_col)[[x_col, y_col]].mean()
        average_positions_off = average_positions_off.apply(tuple, axis="columns").to_dict()
        data[{True: "off", False: "def"}[is_attacking]] = average_positions_off
    return data
