import defensive_network.utility.dataframes


def calculate_average_positions(df_tracking, player_col="player_id", ball_player_id="ball", team_col="team_id", team_in_possession_col="ball_poss_team_id", x_col="x_norm", y_col="y_norm", period_col="section"):
    """
    >>> import pandas as pd
    >>> df_tracking = pd.DataFrame
    """
    i_not_ball = df_tracking[player_col] != ball_player_id

    is_attacking_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_tracking, "is_attacking")
    df_tracking.loc[i_not_ball, is_attacking_col] = df_tracking.loc[i_not_ball, team_col] == df_tracking.loc[i_not_ball, team_in_possession_col]
    data = {}
    for is_attacking, df_tracking_att_def in df_tracking.groupby(is_attacking_col):
        average_positions_off = df_tracking_att_def.groupby([player_col, period_col, team_in_possession_col])[[x_col, y_col]].mean()
        average_positions_off = average_positions_off.groupby(level=player_col).mean()
        average_positions_off = average_positions_off.apply(tuple, axis="columns").to_dict()
        data[{True: "off", False: "def"}[is_attacking]] = average_positions_off
    return data

