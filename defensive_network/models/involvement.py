import importlib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import accessible_space
import accessible_space.utility
import accessible_space.interface

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import defensive_network.utility.pitch
import defensive_network.utility.dataframes


def _distance_point_to_segment(px, py, x1, y1, x2, y2):
    """
    >>> float(_distance_point_to_segment(0, 0, 1, 1, 2, 2))
    1.4142135623730951
    >>> float(_distance_point_to_segment(3, 3, 1, 1, 2, 2))
    1.4142135623730951
    >>> float(_distance_point_to_segment(2, 1, 1, 1, 2, 2))
    0.7071067811865476
    >>> float(_distance_point_to_segment(1.5, 1.5, 1, 1, 2, 2))
    0.0
    >>> float(_distance_point_to_segment(1, 1, 1, 1, 2, 2))
    0.0
    """
    # Compute the vector AB and AP
    ABx = x2 - x1
    ABy = y2 - y1
    APx = px - x1
    APy = py - y1

    # Compute the length squared of AB
    AB_length_squared = ABx ** 2 + ABy ** 2

    # Handle the degenerate case where A and B are the same point
    i_degenerate = (AB_length_squared == 0) & (px == px)
    # if AB_length_squared == 0:
    #     return math.sqrt(APx ** 2 + APy ** 2)

    # Compute the projection of AP onto AB, normalized to [0,1]
    t = (APx * ABx + APy * ABy) / AB_length_squared
    t_clamped = np.maximum(0, np.minimum(1, t))

    # Find the closest point on the segment
    closest_x = x1 + t_clamped * ABx
    closest_y = y1 + t_clamped * ABy

    # Compute the distance from P to the closest point
    dx = px - closest_x
    dy = py - closest_y

    res = np.sqrt(dx ** 2 + dy ** 2)
    try:
        res[i_degenerate] = np.sqrt(APx[i_degenerate] ** 2 + APy[i_degenerate] ** 2)
    except TypeError:
        if i_degenerate:
            res = math.sqrt(APx ** 2 + APy ** 2)

    return res


def _dist_to_goal(df_tracking, x_col, y_col):
    y_goal = np.clip(df_tracking[y_col], -7.32 / 2, 7.32 / 2)
    return np.sqrt((df_tracking[x_col] - 52.5) ** 2 + (df_tracking[y_col] - y_goal) ** 2)


def _get_involvement_by_model(
    df_passes, df_tracking,
    tracking_frame_col="full_frame", tracking_team_col="team_id", tracking_player_col="player_id",
    tracking_x_col="x_tracking", tracking_y_col="y_tracking", tracking_player_name_col=None,
    ball_tracking_player_id="BALL",
    event_team_col="team_id_1", event_frame_col="full_frame", event_x_col="x_event", event_y_col="y_event",
    event_receiver_col="player_id_2", value_col="pass_xt", event_target_x_col="x_target", event_target_y_col="y_target",
    model="outplayed", model_radius=5, tracking_defender_meta_cols=None,
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", None)
    >>> df_passes = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [1, 1, 1], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, True, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "pass_xt": [-0.1, 0.1, -0.1]})
    >>> df_passes
       event_id  team_id_1  full_frame  x_event  y_event  x_target  y_target  is_successful  player_id_2  full_frame_rec  pass_xt
    0         0          1           0        0        0        10         0          False            2               1     -0.1
    1         1          1           1        0        0        20         0           True            3               2      0.1
    2         2          1           2        0        0        30         0          False            4               3     -0.1
    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [1, 0, 0, 1, 0, 0, 1, 0, 0], "player_id": [2, 3, 4, 2, 3, 4, 2, 3, 4], "player_name": ["P2", "P3", "P4", "P2", "P3", "P4", "P2", "P3", "P4"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> df_tracking
       full_frame  team_id  player_id player_name  x_tracking  y_tracking
    0           0        1          2          P2           5           1
    1           0        0          3          P3          10           2
    2           0        0          4          P4          15           3
    3           1        1          2          P2           5           4
    4           1        0          3          P3          10           5
    5           1        0          4          P4          15           6
    6           2        1          2          P2           5           7
    7           2        0          3          P3          10           8
    8           2        0          4          P4          15           9
    >>> _get_involvement_by_model(df_passes, df_tracking, model="circle_circle_rectangle", model_radius=7.5, tracking_player_name_col="player_name")#.drop(columns=["team_id_1", "player_id_2", "full_frame_rec", "x_event", "y_event", "x_target", "y_target"])
       defender_id  raw_involvement  raw_contribution  raw_fault  involvement  contribution     fault  event_id  team_id_1  full_frame  x_event  y_event defender_name  defender_x  defender_y        involvement_model  x_target  y_target  is_successful  player_id_2  full_frame_rec  pass_xt  model_radius
    0            3         0.733333          0.733333   0.000000     0.073333      0.073333  0.000000         0          1           0        0        0            P3          10           2  circle_circle_rectangle        10         0          False            2               1     -0.1           7.5
    1            4         0.222540          0.222540   0.000000     0.022254      0.022254  0.000000         0          1           0        0        0            P4          15           3  circle_circle_rectangle        10         0          False            2               1     -0.1           7.5
    2            3         0.333333          0.000000   0.333333     0.033333      0.000000  0.033333         1          1           1        0        0            P3          10           5  circle_circle_rectangle        20         0           True            3               2      0.1           7.5
    3            4         0.200000          0.000000   0.200000     0.020000      0.000000  0.020000         1          1           1        0        0            P4          15           6  circle_circle_rectangle        20         0           True            3               2      0.1           7.5
    4            3         0.000000          0.000000   0.000000     0.000000      0.000000  0.000000         2          1           2        0        0            P3          10           8  circle_circle_rectangle        30         0          False            4               3     -0.1           7.5
    5            4         0.000000          0.000000   0.000000     0.000000      0.000000  0.000000         2          1           2        0        0            P4          15           9  circle_circle_rectangle        30         0          False            4               3     -0.1           7.5
    """
    df_passes = df_passes.copy()

    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["full_frame", "team_id", "player_id", "x_tracking", "y_tracking"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])

    if tracking_defender_meta_cols is not None:
        for col in tracking_defender_meta_cols:
            if col not in df_tracking.columns:
                raise ValueError(f"Missing '{col}' in df_tracking")

    if df_tracking[tracking_team_col].isna().all():
        raise ValueError("No team_id in tracking data - will cause problems!")

    if tracking_player_name_col is None:
        player2name = {player: player for player in df_tracking[tracking_player_col].unique()}
        warnings.warn("TODO")
    else:
        player2name = df_tracking[[tracking_player_col, tracking_player_name_col]].drop_duplicates().set_index(tracking_player_col)[tracking_player_name_col]

    frames_in_events_not_in_tracking = set(df_passes[event_frame_col]) - set(df_tracking[tracking_frame_col])
    if len(frames_in_events_not_in_tracking) != 0:
        warnings.warn(f"Event frames not present in tracking data: {frames_in_events_not_in_tracking} (tracking, e.g.: {df_tracking[tracking_frame_col].iloc[0]}), (events, e.g.: {df_passes[event_frame_col].iloc[0]})")
        df_passes = df_passes[~df_passes[event_frame_col].isin(frames_in_events_not_in_tracking)]

    # assert ball_tracking_player_id in df_tracking[tracking_player_col].unique()

    df_tracking["distance_to_goal"] = _dist_to_goal(df_tracking, tracking_x_col, tracking_y_col)

    unique_frame_col = accessible_space.utility.get_unused_column_name(df_passes.columns, "unique_frame")
    df_passes[unique_frame_col] = np.arange(len(df_passes))

    # ball_present_frames = df_tracking[df_tracking[tracking_player_col] == ball_tracking_player_id][unique_frame_col]

    df_tracking_passes = df_passes[[unique_frame_col, event_frame_col, event_team_col]].merge(df_tracking, how="left", left_on=event_frame_col, right_on=tracking_frame_col)

    df_tracking = df_tracking[df_tracking[tracking_player_col] != ball_tracking_player_id]

    if len(df_passes) != len(df_passes.drop_duplicates([event_frame_col])):
        warnings.warn("Some passes have duplicate frames - may cause issues?")
        # i_duplicates = df_passes[event_frame_col].duplicated(keep=False)
    assert len(df_tracking) == len(df_tracking.drop_duplicates([tracking_frame_col, tracking_player_col]))

    # check no frame-player duplicates
    # duplicates = df_tracking_passes.duplicated([tracking_frame_col, tracking_player_col], keep=False)
    # assert len(df_tracking_passes) == len(df_tracking_passes.drop_duplicates([tracking_frame_col, tracking_player_col]))

    df_tracking_passes = df_tracking_passes[df_tracking_passes[tracking_team_col].notna()]
    df_tracking_passes = df_tracking_passes[df_tracking_passes[event_team_col].notna()]
    teams = df_tracking_passes[tracking_team_col].unique().tolist()

    importlib.reload(accessible_space.interface)
    PLAYER_POS, _, players, player_teams, controlling_teams, frame_to_index, _ = accessible_space.interface.transform_into_arrays(
        df_tracking_passes, frame_col=unique_frame_col, team_col=tracking_team_col, player_col=tracking_player_col,
        x_col=tracking_x_col, y_col=tracking_y_col, controlling_team_col=event_team_col,
        ball_player_id=ball_tracking_player_id, ignore_ball_position=True,  # TODO add back in after publishing accessible_space
        vx_col=None, vy_col=None,
    )

    if len(teams) == 2:
        defending_teams = np.array([teams[0] if controlling_team == teams[1] else teams[1] for controlling_team in controlling_teams])
    elif len(teams) == 1:
        defending_team = teams[0]
        assert all([defending_team != controlling_team for controlling_team in controlling_teams])
        defending_teams = np.array([defending_team for _ in controlling_teams])
    else:
        teams = [team for team in teams if not pd.isna(team)]
        defending_teams = np.array([teams[0] if controlling_team == teams[1] else teams[1] for controlling_team in controlling_teams])

    i_valid_positions_available = df_passes[unique_frame_col].isin(frame_to_index.keys())
    X_PASSER = df_passes.loc[i_valid_positions_available, event_x_col].values
    Y_PASSER = df_passes.loc[i_valid_positions_available, event_y_col].values
    X_RECEIVER = df_passes.loc[i_valid_positions_available, event_target_x_col].values
    Y_RECEIVER = df_passes.loc[i_valid_positions_available, event_target_y_col].values
    INTERCEPTER = df_passes.loc[i_valid_positions_available, event_receiver_col].values
    assert INTERCEPTER.shape == X_PASSER.shape

    if model == "circle_passer":
        DISTANCE_TO_PASSER = np.sqrt((PLAYER_POS[:, :, 0] - X_PASSER[:, np.newaxis]) ** 2 + (PLAYER_POS[:, :, 1] - Y_PASSER[:, np.newaxis]) ** 2)  # F x P
        CLIPPED_DISTANCE_TO_PASSER = np.minimum(DISTANCE_TO_PASSER, model_radius)  # F x P
        INVOLVEMENT = 1 - CLIPPED_DISTANCE_TO_PASSER / model_radius  # F x P
    elif model == "circle_receiver":
        DISTANCE_TO_RECEIVER = np.sqrt((PLAYER_POS[:, :, 0] - X_RECEIVER[:, np.newaxis]) ** 2 + (PLAYER_POS[:, :, 1] - Y_RECEIVER[:, np.newaxis]) ** 2)  # F x P
        CLIPPED_DISTANCE_TO_RECEIVER = np.minimum(DISTANCE_TO_RECEIVER, model_radius)  # F x P
        INVOLVEMENT = 1 - CLIPPED_DISTANCE_TO_RECEIVER / model_radius  # F x P
    elif model == "circle_circle_rectangle":
        DISTANCE_TO_PASSING_LANE = _distance_point_to_segment(PLAYER_POS[:, :, 0], PLAYER_POS[:, :, 1], X_PASSER[:, np.newaxis], Y_PASSER[:, np.newaxis], X_RECEIVER[:, np.newaxis], Y_RECEIVER[:, np.newaxis])  # F x P
        CLIPPED_DISTANCE_TO_PASSING_LANE = np.minimum(DISTANCE_TO_PASSING_LANE, model_radius)  # F x P
        INVOLVEMENT = 1 - CLIPPED_DISTANCE_TO_PASSING_LANE / model_radius  # F x P
    elif model == "intercepter":
        IS_INTERCEPTER = INTERCEPTER[:, np.newaxis] == players[np.newaxis, :]  # F x P
        INVOLVEMENT = IS_INTERCEPTER.astype(float)  # F x P
    else:
        raise ValueError(f"Unknown model: {model}")

    IS_DEFENDER = player_teams[np.newaxis, :] == defending_teams[:, np.newaxis]  # F x P
    INVOLVEMENT[~IS_DEFENDER] = None

    df_involvement = pd.DataFrame(INVOLVEMENT.flatten(), columns=["raw_involvement"])
    F = INVOLVEMENT.shape[0]
    P = INVOLVEMENT.shape[1]
    df_involvement["defender_id"] = list(players) * F  # P x F
    df_involvement["defender_name"] = df_involvement["defender_id"].apply(lambda x: player2name.get(x, x))
    # import streamlit as st
    # st.write("df_involvement")
    # st.write(df_involvement)
    # assert df_involvement["defender_name"].notna().any()
    # st.stop()
    df_involvement["defender_x"] = PLAYER_POS[:, :, 0].flatten()
    df_involvement["defender_y"] = PLAYER_POS[:, :, 1].flatten()

    df_involvement[unique_frame_col] = np.repeat(list(frame_to_index.keys()), P)
    df_involvement["involvement_model"] = model

    df_involvement = df_involvement[df_involvement["raw_involvement"].notna()]

    df_involvement = df_involvement.merge(df_passes, how="left", on=unique_frame_col)
    i_fault = df_involvement[value_col] >= 0
    i_contribution = df_involvement[value_col] < 0
    df_involvement.loc[i_contribution, "raw_contribution"] = df_involvement.loc[i_contribution, "raw_involvement"]
    df_involvement.loc[i_fault, "raw_fault"] = df_involvement.loc[i_fault, "raw_involvement"]

    df_involvement["raw_contribution"] = df_involvement["raw_contribution"].fillna(0)
    df_involvement["raw_fault"] = df_involvement["raw_fault"].fillna(0)

    for col in ["valued_involvement", "valued_contribution", "valued_fault"]:
        df_involvement[col] = df_involvement[col.replace("valued", "raw")] * df_involvement[value_col].abs()

    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "valued_fault", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "valued_contribution", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "valued_involvement", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "raw_fault", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "raw_contribution", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "raw_involvement", 0)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_id", -7)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_name", -7)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_x", -7)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_y", -7)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "involvement_model", -7)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "raw_involvement", -6)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_id", -7)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_name", -7)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_x", -7)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_y", -7)
    # df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "involvement_model", -7)

    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_id", 0)

    # import streamlit as st
    # st.write("df_involvement")
    # st.write(df_involvement)

    if tracking_defender_meta_cols is not None:
        # st.write("tracking_defender_meta_cols")
        # st.write(tracking_defender_meta_cols)
        # dft_player = df_tracking.drop_duplicates(tracking_player_col).set_index(tracking_player_col)#.reset_index()
        for context_col in tracking_defender_meta_cols:
            dft = df_tracking[[tracking_frame_col, tracking_player_col, context_col]].rename(columns={context_col: f"defender_{context_col}", tracking_player_col: "defender_id"})
            # dictionary = dft_player[context_col].dropna().to_dict()
            # df_involvement[f"defender_{context_col}"] = df_involvement["defender_id"].map(dictionary)  # TODO old one
            df_involvement = df_involvement.merge(dft, on=[tracking_frame_col, "defender_id"], how="left")
            # if df_involvement[f"defender_{context_col}"].isna().any():
            #     st.warning(f"Some {context_col} is NaN!")
            # assert df_involvement[f"defender_{context_col}"].notna().all()

    df_involvement = df_involvement.drop(columns=[unique_frame_col])

    # dfg = df_involvement.groupby(["team_id_1", "event_id"]).agg(
    # dfg = df_involvement.groupby(["event_id"]).agg(
    #     n_defenders=("defender_id", "nunique"),
    #     n_defenders_unique=("defender_id", "unique"),
    #     n_defenders_unique_names=("defender_name", "unique"),
    #     pass_xt=("pass_xt", "first"),
    #     n_pass_xt_unique_values=("pass_xt", "nunique"),
    #     pass_xt_unique_values=("pass_xt", "unique"),
    #     involvement_type=("involvement_type", "first"),
    #     # n_involvement_type_unique_values=("involvement_type", "nunique"),
    #     # involvement_type_unique_values=("involvement_type", "unique"),
    #     n_involvement_model_unique_values=("involvement_model", "nunique"),
    #     involvement_model_unique_values=("involvement_model", "unique"),
    #     raw_involvement=("raw_involvement", "sum"),
    #     n_raw_involvement_unique_values=("raw_involvement", "nunique"),
    #     raw_contribution=("raw_contribution", "sum"),
    #     raw_fault=("raw_fault", "sum"),
    #     involvement=("involvement", "sum"),
    #     contribution=("contribution", "sum"),
    #     fault=("fault", "sum"),
    # )

    df_involvement["model_radius"] = model_radius

    return df_involvement


def get_involvement(
    df_passes, df_tracking,
    event_frame_col="full_frame", event_success_col="pass_is_successful", event_intercepted_col="pass_is_intercepted",
    event_team_col="team_id_1", event_raw_x_col="x_event", event_raw_y_col="y_event", event_raw_target_x_col="x_target",
    event_raw_target_y_col="y_target", event_receiver_col="player_id_2", event_value_col="pass_xt",
    tracking_frame_col="full_frame", tracking_x_col="x_tracking", tracking_y_col="y_tracking",
    tracking_team_col="team_id", tracking_player_col="player_id", ball_tracking_player_id="BALL", tracking_player_name_col="player_name",
    involvement_model_success_pos_value="circle_circle_rectangle",
    involvement_model_success_neg_value="circle_passer", involvement_model_out="circle_passer",
    involvement_model_intercepted="intercepter", model_radius=5, tracking_defender_meta_cols=None,
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", None)
    >>> df_event = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [2, 2, 2], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "pass_is_successful": [True, False, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "player_id_1": [1, 2, 3], "event_string": ["pass", "pass", "pass"], "pass_xt": [0.1, -0.1, -0.1], "pass_is_intercepted": [False, False, True], "player_name_1": ["A", "B", "C"]})
    >>> df_event
       event_id  team_id_1  full_frame  x_event  y_event  x_target  y_target  pass_is_successful  player_id_2  full_frame_rec  player_id_1 event_string  pass_xt  pass_is_intercepted player_name_1
    0         0          2           0        0        0        10         0                True            2               1            1         pass      0.1                False             A
    1         1          2           1        0        0        20         0               False            3               2            2         pass     -0.1                False             B
    2         2          2           2        0        0        30         0               False            4               3            3         pass     -0.1                 True             C
    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [1, 1, 1, 1, 1, 1, 2, 2, 2], "player_id": [2, 3, 4, 2, 3, 4, "BALL", "BALL", "BALL"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [0, 0, 0, 0, 0, 0, 0, 0, 0], "player_name": ["A", "B", "C", "A", "B", "C", "BALL", "BALL", "BALL"]})
    >>> df_tracking
       full_frame  team_id player_id  x_tracking  y_tracking player_name
    0           0        1         2           5           0           A
    1           0        1         3          10           0           B
    2           0        1         4          15           0           C
    3           1        1         2           5           0           A
    4           1        1         3          10           0           B
    5           1        1         4          15           0           C
    6           2        2      BALL           5           0        BALL
    7           2        2      BALL          10           0        BALL
    8           2        2      BALL          15           0        BALL
    >>> df_involvement = get_involvement(df_event, df_tracking)
    >>> df_involvement
       involvement_pass_id  defender_id  raw_involvement  raw_contribution  raw_fault  involvement  contribution  fault  event_id  team_id_1  full_frame  x_event  y_event  x_target  y_target  pass_is_successful  player_id_2  full_frame_rec  player_id_1 defender_name  defender_x  defender_y        involvement_model event_string  pass_xt  pass_is_intercepted player_name_1       involvement_type  model_radius
    0                    0          2.0              1.0               0.0        1.0          0.1           0.0    0.1         0          2           0        0        0        10         0                True            2               1            1             A         5.0         0.0  circle_circle_rectangle         pass      0.1                False             A  success_and_pos_value             5
    1                    0          3.0              1.0               0.0        1.0          0.1           0.0    0.1         0          2           0        0        0        10         0                True            2               1            1             B        10.0         0.0  circle_circle_rectangle         pass      0.1                False             A  success_and_pos_value             5
    2                    0          4.0              0.0               0.0        0.0          0.0           0.0    0.0         0          2           0        0        0        10         0                True            2               1            1             C        15.0         0.0  circle_circle_rectangle         pass      0.1                False             A  success_and_pos_value             5
    3                    1          2.0              0.0               0.0        0.0          0.0           0.0    0.0         1          2           1        0        0        20         0               False            3               2            2             A         5.0         0.0            circle_passer         pass     -0.1                False             B                    out             5
    4                    1          3.0              0.0               0.0        0.0          0.0           0.0    0.0         1          2           1        0        0        20         0               False            3               2            2             B        10.0         0.0            circle_passer         pass     -0.1                False             B                    out             5
    5                    1          4.0              0.0               0.0        0.0          0.0           0.0    0.0         1          2           1        0        0        20         0               False            3               2            2             C        15.0         0.0            circle_passer         pass     -0.1                False             B                    out             5
    """
    df_passes["involvement_pass_id"] = range(len(df_passes))  # give unique id to passes

    if event_value_col is None:
        event_value_col = "dummy"
        df_passes[event_value_col] = 1

    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["tracking_frame_col", "tracking_team_col", "tracking_player_col", "tracking_x_col", "tracking_y_col"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_passes, "df_passes", ["event_team_col", "event_frame_col", "event_raw_x_col", "event_raw_y_col", "event_raw_target_x_col", "event_raw_target_y_col", "event_success_col"], [ event_team_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col, event_success_col])
    df_passes[event_success_col] = df_passes[event_success_col].astype(bool)
    df_passes[event_intercepted_col] = df_passes[event_intercepted_col].astype(bool)

    # 1. Successful passes, xT >= 0
    i_success_and_pos_value = df_passes[event_success_col] & (df_passes[event_value_col] >= 0)
    df_passes.loc[i_success_and_pos_value, "involvement_type"] = "success_and_pos_value"

    df_tracking["role_category"] = df_tracking["role"]
    df_involvement_success = _get_involvement_by_model(
        df_passes.loc[i_success_and_pos_value], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col,
        tracking_player_name_col, ball_tracking_player_id, event_team_col,
        event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col,
        event_value_col, event_raw_target_x_col, event_raw_target_y_col,
        model=involvement_model_success_pos_value, model_radius=model_radius,
        tracking_defender_meta_cols=tracking_defender_meta_cols,
    )

    # 2. Successful passes, xT < 0
    i_success_and_neg_value = df_passes[event_success_col] & (df_passes[event_value_col] < 0)
    df_passes.loc[i_success_and_neg_value, "involvement_type"] = "success_and_neg_value"
    df_involvement_success_neg = _get_involvement_by_model(
        df_passes.loc[i_success_and_neg_value], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col,
        tracking_player_name_col, ball_tracking_player_id, event_team_col,
        event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col,
        event_value_col, event_raw_target_x_col, event_raw_target_y_col,
        model=involvement_model_success_neg_value, model_radius=model_radius,
        tracking_defender_meta_cols=tracking_defender_meta_cols,
    )

    # 3. Unsuccessful passes, out
    i_out = (~df_passes[event_success_col]) & (~df_passes[event_intercepted_col])
    df_passes.loc[i_out, "involvement_type"] = "out"
    df_involvement_out = _get_involvement_by_model(
        df_passes.loc[i_out], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col,
        tracking_player_name_col, ball_tracking_player_id, event_team_col,
        event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col,
        event_value_col, event_raw_target_x_col, event_raw_target_y_col,
        model=involvement_model_out, model_radius=model_radius,
        tracking_defender_meta_cols=tracking_defender_meta_cols,
    )

    # 4. Unsuccessful passes, intercepted
    i_intercepted = df_passes[event_intercepted_col]
    df_passes.loc[i_intercepted, "involvement_type"] = "intercepted"
    df_involvement_intercepted = _get_involvement_by_model(
        df_passes.loc[i_intercepted], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col,
        tracking_player_name_col, ball_tracking_player_id, event_team_col,
        event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col,
        event_value_col, event_raw_target_x_col, event_raw_target_y_col,
        model=involvement_model_intercepted, model_radius=model_radius,
        tracking_defender_meta_cols=tracking_defender_meta_cols,
    )

    df_involvement = pd.concat([df_involvement_success, df_involvement_success_neg, df_involvement_out, df_involvement_intercepted], ignore_index=True)
    df_involvement["model_radius"] = model_radius

    import streamlit as st
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "involvement_pass_id", 0)
    # st.write("df_involvement x")
    # st.write(df_involvement)

    return df_involvement


if __name__ == '__main__':
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    # df_passes = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [1, 1, 1], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, True, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "pass_xt": [-0.1, 0.1, -0.1]})
    # df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [0, 0, 1, 0, 0, 1, 0, 0, 1], "player_id": [2, 3, "BALL", 2, 3, "BALL", 2, 3, "BALL"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [1, 2, 3, 4, 5, 6, 7, 8, 9], "vx": [0, 0, 0, 0, 0, 0, 0, 0, 0], "vy": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    # df_faultribution = _get_faultribution_by_model_matrix(df_passes, df_tracking, model="intercepter", model_radius=15)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    df_event = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [2, 2, 2], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "pass_is_successful": [True, False, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "player_id_1": [1, 2, 3], "event_string": ["pass", "pass", "pass"], "pass_xt": [0.1, -0.1, -0.1], "pass_is_intercepted": [False, False, True], "player_name_1": ["A", "B", "C"]})
    df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 0, 1, 2], "team_id": [1, 1, 1, 1, 1, 1, 2, 2, 2], "player_id": [2, 3, 4, 2, 3, 4, "BALL", "BALL", "BALL"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [0, 0, 0, 0, 0, 0, 0, 0, 0], "player_name": ["A", "B", "C", "A", "B", "C", "BALL", "BALL", "BALL"]})
    radius = 9
    df_involvement = get_involvement(df_event, df_tracking, model_radius=radius)
    defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, "circle_circle_rectangle", radius, df_tracking)

    plt.show()
