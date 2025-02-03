import pandas as pd

import defensive_network.utility.dataframes


def average_positions_in_attack_and_defense(
    df_tracking, player_col="player_id", ball_player_id="ball", team_col="team_id",
    team_in_possession_col="ball_poss_team_id", x_col="x_norm", y_col="y_norm",
):
    """
    Calculate average position of the players in attacking and defending phase separately.

    >>> df_tracking = pd.DataFrame({"player_id": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"], "team_id": ["H", "H", "H", "H", "H", "H", "H", "H", "A", "A", "A", "A"], "ball_poss_team_id": ["H", "H", "A", "A", "H", "H", "A", "A", "H", "H", "A", "A"], "x_norm": range(12), "y_norm": range(1, 13)})
    >>> average_positions_in_attack_and_defense(df_tracking)
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
