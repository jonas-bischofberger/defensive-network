import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import accessible_space

import defensive_network.utility.pitch
import defensive_network.utility.dataframes


def distance_point_to_segment(px, py, x1, y1, x2, y2):
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


def plot_passes_with_involvement(
    df_involvement, model, model_radius, df_passes, df_tracking,
    event_id_col="event_id",
    pass_x_col="x_event", pass_y_col="y_event", pass_target_x_col="x_target", pass_target_y_col="y_target",
    pass_frame_col="full_frame", pass_team_col="team_id_1", pass_player_name_col="player_id_1",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x_tracking", tracking_y_col="y_tracking",
    tracking_frame_col="full_frame", tracking_player_name_col="player_name", tracking_vx_col="vx", tracking_vy_col="vy",
    event_string_col="event_string", value_col="pass_xt", ball_tracking_player_id="BALL", n_passes=2
):
    if len(df_involvement) == 0:
        st.warning("plot_passes_with_involvement: No passes found.")
        return
    for pass_nr, (pass_id, df_outplayed_pass) in enumerate(df_involvement.groupby(event_id_col)):
        if pass_nr >= (n_passes - 1):
            break
        try:
            p4ss = df_passes[df_passes[event_id_col] == pass_id].iloc[0]
        except IndexError as e:
            st.warning(f"plot_passes_with_involvement: Pass {pass_id} not found in df_passes.")
            st.write(e)
            continue
        fig = defensive_network.utility.pitch.plot_pass_involvement(
            p4ss, df_outplayed_pass, df_tracking,
            pass_x_col, pass_y_col, pass_target_x_col, pass_target_y_col, pass_frame_col, pass_team_col, pass_player_name_col,
            tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col, tracking_frame_col, tracking_player_name_col,
            tracking_vx_col, tracking_vy_col, ball_tracking_player_id, plot_model=model, model_radius=model_radius
        )
        plt.title(f"Pass {p4ss[event_string_col]} ({p4ss[value_col]:+.3f} {value_col})", fontsize=6)
        st.pyplot(fig, dpi=500)
        plt.close()


def _get_faultribution_by_model(
    df_passes, df_tracking, tracking_frame_col="full_frame", tracking_team_col="team_id", tracking_player_col="player_id",
    tracking_x_col="x_tracking", tracking_y_col="y_tracking", tracking_period_col="period_id",
    tracking_team_in_possession_col="team_in_possession", event_id_col="event_id", event_team_col="team_id_1",
    event_player_col="player_id_1", event_frame_col="full_frame", event_x_col="x_event", event_y_col="y_event",
    event_receiver_col="player_id_2", value_col="pass_xt",
    event_target_frame_col="full_frame_rec", event_target_x_col="x_target", event_target_y_col="y_target", event_outcome_col="is_successful",
    model="outplayed", max_passes_for_debug=20, model_radius=5
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", None)
    >>> df_passes = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [1, 1, 1], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, True, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "pass_xt": [-0.1, 0.1, -0.1]})
    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [0, 0, 0, 0, 0, 0, 0, 0, 0], "player_id": [2, 3, 4, 2, 3, 4, 2, 3, 4], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> _get_faultribution_by_model(df_passes, df_tracking, model="circle_circle_rectangle", model_radius=5)[["pass_nr", "raw_involvement", "raw_contribution", "raw_fault", "involvement", "contribution", "fault", "involvement_model"]]
       pass_nr  raw_involvement  raw_contribution  raw_fault  involvement  contribution  fault        involvement_model
    0        0              0.8               0.8        0.0         0.08          0.08   0.00  circle_circle_rectangle
    1        0              0.6               0.6        0.0         0.06          0.06   0.00  circle_circle_rectangle
    2        0              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    0        1              0.2               0.0        0.2         0.02          0.00   0.02  circle_circle_rectangle
    1        1              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    2        1              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    0        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    1        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    2        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    """
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["full_frame", "team_id", "player_id", "x_tracking", "y_tracking"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])

    df_tracking["distance_to_goal"] = _dist_to_goal(df_tracking, tracking_x_col, tracking_y_col)

    teams = df_tracking[tracking_team_col].unique().tolist()

    dfs = []

#     df_tracking, team_col="team_id", period_col="period_id", team_in_possession_col="team_in_possession", x_col="x",
    for pass_nr, (pass_index, p4ss) in accessible_space.progress_bar(enumerate(df_passes.iterrows())):
        data = []

        if pass_nr > max_passes_for_debug:
            break

        i_frame = df_tracking[tracking_frame_col] == p4ss[event_frame_col]
        i_frame_rec = df_tracking[tracking_frame_col] == p4ss[event_target_frame_col]

        df_tracking_frame = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_frame_col]]
        df_tracking_frame_rec = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_target_frame_col]]

        attacking_team = p4ss[event_team_col]
        defending_team = teams[1] if attacking_team == teams[0] else teams[0]

        i_def = df_tracking_frame[tracking_team_col] == defending_team
        i_def_rec = df_tracking_frame_rec[tracking_team_col] == defending_team

        y_goal = np.clip(p4ss[event_y_col], -7.32 / 2, 7.32 / 2)
        distance_between_ball_and_goal = np.sqrt((p4ss[event_x_col] - 52.5) ** 2 + (p4ss[event_y_col] - y_goal) ** 2)

        df_tracking_frame["is_between_goal_and_ball"] = df_tracking_frame["distance_to_goal"] < distance_between_ball_and_goal
        df_tracking_frame_rec["is_between_goal_and_ball"] = df_tracking_frame_rec["distance_to_goal"] < distance_between_ball_and_goal

        i_between_ball_and_goal = df_tracking_frame["is_between_goal_and_ball"]
        df_between_ball_and_goal = df_tracking_frame.loc[i_def & i_between_ball_and_goal]

        i_between_ball_and_goal_rec = df_tracking_frame_rec["is_between_goal_and_ball"]
        df_between_ball_and_goal_rec = df_tracking_frame_rec.loc[i_def_rec & i_between_ball_and_goal_rec]

        for _, defender_row in df_tracking_frame.loc[i_def].iterrows():
            # add prefix "defender." to defender_row
            defender_row = {f"defender.{key}": value for key, value in defender_row.items()}
            pass_copy = p4ss.copy()

            if model == "outplayed":
                pass_copy["defender"] = defender_row["defender.player_id"]
                pass_copy["is_between_ball_and_goal_at_pass"] = defender_row["defender.player_id"] in df_between_ball_and_goal["player_id"].unique()
                pass_copy["is_between_ball_and_goal_at_reception"] = defender_row["defender.player_id"] in df_between_ball_and_goal_rec["player_id"].unique()
                is_outplayed = pass_copy["is_between_ball_and_goal_at_pass"] and not pass_copy["is_between_ball_and_goal_at_reception"]
                pass_copy["is_outplayed"] = is_outplayed
                pass_copy["raw_faultribution"] = int(is_outplayed)
            elif model == "circle_circle_rectangle":
                pass_copy["distance_to_pass_line"] = distance_point_to_segment(
                    defender_row[f"defender.{tracking_x_col}"], defender_row[f"defender.{tracking_y_col}"],
                    pass_copy[event_x_col], pass_copy[event_y_col], pass_copy[event_target_x_col], pass_copy[event_target_y_col]
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_pass_line"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "circle_passer":
                pass_copy["distance_to_passer"] = np.sqrt(
                    (defender_row[f"defender.{tracking_x_col}"] - pass_copy[event_x_col]) ** 2 +
                    (defender_row[f"defender.{tracking_y_col}"] - pass_copy[event_y_col]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_passer"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "circle_receiver":
                pass_copy["distance_to_receiver"] = np.sqrt(
                    (defender_row[f"defender.{tracking_x_col}"] - pass_copy[event_target_x_col]) ** 2 +
                    (defender_row[f"defender.{tracking_y_col}"] - pass_copy[event_target_y_col]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_receiver"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "intercepter":
                is_intercepter = defender_row[f"defender.{tracking_player_col}"] == pass_copy[event_receiver_col]
                pass_copy["raw_faultribution"] = int(is_intercepter)
            else:
                raise ValueError(f"Unknown model: {model}")

            pass_copy["involvement_model"] = model

            pass_copy = pass_copy.to_dict()

            for key in defender_row:
                pass_copy[key] = defender_row[key]

            # add defender row to apss_copy
            # pass_copy.update(defender_row)

            data.append(pass_copy)

        df_outplayed = pd.DataFrame(data)
        df_outplayed["pass_nr"] = pass_nr

        dfs.append(df_outplayed)

        df_passes.loc[pass_index, "pass_nr"] = pass_nr

    if len(dfs) == 0:
        return pd.DataFrame()

    df_outplayed = pd.concat(dfs)
    df_outplayed["raw_involvement"] = df_outplayed["raw_faultribution"].abs()
    i_fault = df_outplayed[value_col] >= 0
    i_contribution = df_outplayed[value_col] < 0
    df_outplayed.loc[i_contribution, "raw_contribution"] = df_outplayed.loc[i_contribution, "raw_involvement"]
    df_outplayed.loc[i_fault, "raw_fault"] = df_outplayed.loc[i_fault, "raw_involvement"]

    df_outplayed["raw_contribution"] = df_outplayed["raw_contribution"].fillna(0)
    df_outplayed["raw_fault"] = df_outplayed["raw_fault"].fillna(0)

    for col in ["involvement", "contribution", "fault", "faultribution"]:
        df_outplayed[col] = df_outplayed[f"raw_{col}"] * df_outplayed[value_col].abs()

    # put raw_faultribution on fifth to last position in columns
    columns = df_outplayed.columns.tolist()
    raw_faultribution_index = columns.index("raw_faultribution")
    columns = columns[:raw_faultribution_index] + columns[raw_faultribution_index + 1:]
    columns = columns[:len(columns) - 4] + ["raw_faultribution"] + columns[len(columns) - 4:]
    df_outplayed = df_outplayed[columns]

    columns = df_outplayed.columns.tolist()
    involvement_model_index = columns.index("involvement_model")
    columns = columns[:involvement_model_index] + columns[involvement_model_index + 1:]
    columns = columns[:len(columns) - 1] + ["involvement_model"] + columns[len(columns) - 1:]
    df_outplayed = df_outplayed[columns]

    # drop faultribution and raw_faultribution until we figure out if we need it
    df_outplayed = df_outplayed.drop(columns=["faultribution", "raw_faultribution"])

    return df_outplayed


def _get_faultribution_by_model_matrix(
    df_passes, df_tracking, tracking_frame_col="full_frame", tracking_team_col="team_id", tracking_player_col="player_id",
    tracking_x_col="x_tracking", tracking_y_col="y_tracking", tracking_period_col="period_id",
    tracking_team_in_possession_col="team_in_possession", event_id_col="event_id", event_team_col="team_id_1",
    event_player_col="player_id_1", event_frame_col="full_frame", event_x_col="x_event", event_y_col="y_event",
    event_receiver_col="player_id_2", value_col="pass_xt",
    event_target_frame_col="full_frame_rec", event_target_x_col="x_target", event_target_y_col="y_target", event_outcome_col="is_successful",
    ball_tracking_player_id="BALL",
    model="outplayed", max_passes_for_debug=20, model_radius=5, tracking_player_name_col="player_name",
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", None)
    >>> df_passes = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [1, 1, 1], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, True, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "pass_xt": [-0.1, 0.1, -0.1]})
    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [0, 0, 0, 0, 0, 0, 0, 0, 0], "player_id": [2, 3, 4, 2, 3, 4, 2, 3, 4], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> _get_faultribution_by_model(df_passes, df_tracking, model="circle_circle_rectangle", model_radius=5)[["pass_nr", "raw_involvement", "raw_contribution", "raw_fault", "involvement", "contribution", "fault", "involvement_model"]]
       pass_nr  raw_involvement  raw_contribution  raw_fault  involvement  contribution  fault        involvement_model
    0        0              0.8               0.8        0.0         0.08          0.08   0.00  circle_circle_rectangle
    1        0              0.6               0.6        0.0         0.06          0.06   0.00  circle_circle_rectangle
    2        0              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    0        1              0.2               0.0        0.2         0.02          0.00   0.02  circle_circle_rectangle
    1        1              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    2        1              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    0        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    1        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    2        2              0.0               0.0        0.0         0.00          0.00   0.00  circle_circle_rectangle
    """
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["full_frame", "team_id", "player_id", "x_tracking", "y_tracking"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])

    assert ball_tracking_player_id in df_tracking[tracking_player_col].unique()

    df_tracking["distance_to_goal"] = _dist_to_goal(df_tracking, tracking_x_col, tracking_y_col)

    import accessible_space.utility
    unique_frame_col = accessible_space.utility.get_unused_column_name(df_passes.columns, "unique_frame")
    df_passes[unique_frame_col] = np.arange(len(df_passes))

    df_tracking_passes = df_passes[[unique_frame_col, event_frame_col, event_team_col]].merge(df_tracking, how="left", left_on=tracking_frame_col, right_on=tracking_frame_col)

    df_tracking = df_tracking[df_tracking[tracking_player_col] != ball_tracking_player_id]
    teams = df_tracking[tracking_team_col].unique().tolist()

    import accessible_space.interface
    # check no frame-player duplicates
    # duplicates = df_tracking_passes.duplicated([tracking_frame_col, tracking_player_col], keep=False)
    assert len(df_tracking_passes) == len(df_tracking_passes.drop_duplicates([tracking_frame_col, tracking_player_col]))
    PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_index, player_to_index = accessible_space.interface._get_matrix_coordinates(
        df_tracking_passes, frame_col=unique_frame_col, team_col=tracking_team_col, player_col=tracking_player_col,
        x_col=tracking_x_col, y_col=tracking_y_col, controlling_team_col=event_team_col,
        ball_player_id=ball_tracking_player_id,
    )
    assert len(teams) == 2
    defending_teams = np.array([teams[0] if controlling_team == teams[1] else teams[1] for controlling_team in controlling_teams])

    X_PASSER = df_passes[event_x_col].values
    Y_PASSER = df_passes[event_y_col].values
    X_RECEIVER = df_passes[event_target_x_col].values
    Y_RECEIVER = df_passes[event_target_y_col].values
    INTERCEPTER = df_passes[event_receiver_col].values
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
        DISTANCE_TO_PASSING_LANE = distance_point_to_segment(PLAYER_POS[:, :, 0], PLAYER_POS[:, :, 1], X_PASSER[:, np.newaxis], Y_PASSER[:, np.newaxis], X_RECEIVER[:, np.newaxis], Y_RECEIVER[:, np.newaxis])  # F x P
        CLIPPED_DISTANCE_TO_PASSING_LANE = np.minimum(DISTANCE_TO_PASSING_LANE, model_radius)  # F x P
        INVOLVEMENT = 1 - CLIPPED_DISTANCE_TO_PASSING_LANE / model_radius  # F x P
    elif model == "intercepter":
        IS_INTERCEPTER = INTERCEPTER[:, np.newaxis] == players[np.newaxis, :]  # F x P
        INVOLVEMENT = IS_INTERCEPTER.astype(float)  # F x P
    else:
        raise ValueError(f"Unknown model: {model}")

    IS_DEFENDER = player_teams[np.newaxis, :] == defending_teams[:, np.newaxis]  # F x P
    INVOLVEMENT[~IS_DEFENDER] = None

    # elif model == "circle_receiver":
    #     max_distance = model_radius
    #     clipped_distance = min(pass_copy["distance_to_receiver"], max_distance)
    #     pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
    # if model == "circle_circle_rectangle":
    #     pass_copy["distance_to_pass_line"] = distance_point_to_segment(
    #         defender_row[f"defender.{tracking_x_col}"], defender_row[f"defender.{tracking_y_col}"],
    #         pass_copy[event_x_col], pass_copy[event_y_col], pass_copy[event_target_x_col], pass_copy[event_target_y_col]
    #     )
    #     max_distance = model_radius
    #     clipped_distance = min(pass_copy["distance_to_pass_line"], max_distance)
    #     pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
    # elif model == "intercepter":
    #     is_intercepter = defender_row[f"defender.{tracking_player_col}"] == pass_copy[event_receiver_col]
    #     pass_copy["raw_faultribution"] = int(is_intercepter)

    df_involvement = pd.DataFrame(INVOLVEMENT.flatten(), columns=["raw_involvement"])
    F = INVOLVEMENT.shape[0]
    P = INVOLVEMENT.shape[1]
    player2name = df_tracking[[tracking_player_col, tracking_player_name_col]].drop_duplicates().set_index(tracking_player_col)[tracking_player_name_col]
    df_involvement["defender_id"] = list(players) * F  # P x F
    df_involvement["defender_name"] = df_involvement["defender_id"].map(player2name)
    df_involvement["defender_x"] = PLAYER_POS[:, :, 0].flatten()
    df_involvement["defender_y"] = PLAYER_POS[:, :, 1].flatten()

    df_involvement[unique_frame_col] = np.repeat(list(frame_to_index.keys()), P)
    df_involvement["involvement_model"] = model

    df_involvement = df_involvement.merge(df_passes, how="left", on=unique_frame_col)
    i_fault = df_involvement[value_col] >= 0
    i_contribution = df_involvement[value_col] < 0
    df_involvement.loc[i_contribution, "raw_contribution"] = df_involvement.loc[i_contribution, "raw_involvement"]
    df_involvement.loc[i_fault, "raw_fault"] = df_involvement.loc[i_fault, "raw_involvement"]

    df_involvement["raw_contribution"] = df_involvement["raw_contribution"].fillna(0)
    df_involvement["raw_fault"] = df_involvement["raw_fault"].fillna(0)

    for col in ["involvement", "contribution", "fault"]:
        df_involvement[col] = df_involvement[f"raw_{col}"] * df_involvement[value_col].abs()

    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "raw_involvement", -6)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "defender_id", -7)
    df_involvement = defensive_network.utility.dataframes.move_column(df_involvement, "involvement_model", -7)

    return df_involvement

    dfs = []

#     df_tracking, team_col="team_id", period_col="period_id", team_in_possession_col="team_in_possession", x_col="x",
    for pass_nr, (pass_index, p4ss) in accessible_space.progress_bar(enumerate(df_passes.iterrows())):
        data = []


        if pass_nr > max_passes_for_debug:
            break

        i_frame = df_tracking[tracking_frame_col] == p4ss[event_frame_col]
        i_frame_rec = df_tracking[tracking_frame_col] == p4ss[event_target_frame_col]

        df_tracking_frame = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_frame_col]]
        df_tracking_frame_rec = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_target_frame_col]]

        attacking_team = p4ss[event_team_col]
        defending_team = teams[1] if attacking_team == teams[0] else teams[0]

        i_def = df_tracking_frame[tracking_team_col] == defending_team
        i_def_rec = df_tracking_frame_rec[tracking_team_col] == defending_team

        y_goal = np.clip(p4ss[event_y_col], -7.32 / 2, 7.32 / 2)
        distance_between_ball_and_goal = np.sqrt((p4ss[event_x_col] - 52.5) ** 2 + (p4ss[event_y_col] - y_goal) ** 2)

        df_tracking_frame["is_between_goal_and_ball"] = df_tracking_frame["distance_to_goal"] < distance_between_ball_and_goal
        df_tracking_frame_rec["is_between_goal_and_ball"] = df_tracking_frame_rec["distance_to_goal"] < distance_between_ball_and_goal

        i_between_ball_and_goal = df_tracking_frame["is_between_goal_and_ball"]
        df_between_ball_and_goal = df_tracking_frame.loc[i_def & i_between_ball_and_goal]

        i_between_ball_and_goal_rec = df_tracking_frame_rec["is_between_goal_and_ball"]
        df_between_ball_and_goal_rec = df_tracking_frame_rec.loc[i_def_rec & i_between_ball_and_goal_rec]

        for _, defender_row in df_tracking_frame.loc[i_def].iterrows():
            # add prefix "defender." to defender_row
            defender_row = {f"defender.{key}": value for key, value in defender_row.items()}
            pass_copy = p4ss.copy()

            if model == "outplayed":  # TODO re-add x_norm/attacking direction
                raise NotImplementedError("outplayed model not implemented")
                pass_copy["defender"] = defender_row["defender.player_id"]
                pass_copy["is_between_ball_and_goal_at_pass"] = defender_row["defender.player_id"] in df_between_ball_and_goal["player_id"].unique()
                pass_copy["is_between_ball_and_goal_at_reception"] = defender_row["defender.player_id"] in df_between_ball_and_goal_rec["player_id"].unique()
                is_outplayed = pass_copy["is_between_ball_and_goal_at_pass"] and not pass_copy["is_between_ball_and_goal_at_reception"]
                pass_copy["is_outplayed"] = is_outplayed
                pass_copy["raw_faultribution"] = int(is_outplayed)
            elif model == "circle_circle_rectangle":
                pass_copy["distance_to_pass_line"] = distance_point_to_segment(
                    defender_row[f"defender.{tracking_x_col}"], defender_row[f"defender.{tracking_y_col}"],
                    pass_copy[event_x_col], pass_copy[event_y_col], pass_copy[event_target_x_col], pass_copy[event_target_y_col]
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_pass_line"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "circle_passer":
                pass_copy["distance_to_passer"] = np.sqrt(
                    (defender_row[f"defender.{tracking_x_col}"] - pass_copy[event_x_col]) ** 2 +
                    (defender_row[f"defender.{tracking_y_col}"] - pass_copy[event_y_col]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_passer"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "circle_receiver":
                pass_copy["distance_to_receiver"] = np.sqrt(
                    (defender_row[f"defender.{tracking_x_col}"] - pass_copy[event_target_x_col]) ** 2 +
                    (defender_row[f"defender.{tracking_y_col}"] - pass_copy[event_target_y_col]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_receiver"], max_distance)
                pass_copy["raw_faultribution"] = 1 - clipped_distance / max_distance
            elif model == "intercepter":
                is_intercepter = defender_row[f"defender.{tracking_player_col}"] == pass_copy[event_receiver_col]
                pass_copy["raw_faultribution"] = int(is_intercepter)
            else:
                raise ValueError(f"Unknown model: {model}")

            pass_copy["involvement_model"] = model

            pass_copy = pass_copy.to_dict()

            for key in defender_row:
                pass_copy[key] = defender_row[key]

            # add defender row to apss_copy
            # pass_copy.update(defender_row)

            data.append(pass_copy)

        df_outplayed = pd.DataFrame(data)
        df_outplayed["pass_nr"] = pass_nr

        dfs.append(df_outplayed)

        df_passes.loc[pass_index, "pass_nr"] = pass_nr

    if len(dfs) == 0:
        return pd.DataFrame()

    df_outplayed = pd.concat(dfs)
    df_outplayed["raw_involvement"] = df_outplayed["raw_faultribution"].abs()
    i_fault = df_outplayed[value_col] >= 0
    i_contribution = df_outplayed[value_col] < 0
    df_outplayed.loc[i_contribution, "raw_contribution"] = df_outplayed.loc[i_contribution, "raw_involvement"]
    df_outplayed.loc[i_fault, "raw_fault"] = df_outplayed.loc[i_fault, "raw_involvement"]

    df_outplayed["raw_contribution"] = df_outplayed["raw_contribution"].fillna(0)
    df_outplayed["raw_fault"] = df_outplayed["raw_fault"].fillna(0)

    for col in ["involvement", "contribution", "fault", "faultribution"]:
        df_outplayed[col] = df_outplayed[f"raw_{col}"] * df_outplayed[value_col].abs()

    # put raw_faultribution on fifth to last position in columns
    columns = df_outplayed.columns.tolist()
    raw_faultribution_index = columns.index("raw_faultribution")
    columns = columns[:raw_faultribution_index] + columns[raw_faultribution_index + 1:]
    columns = columns[:len(columns) - 4] + ["raw_faultribution"] + columns[len(columns) - 4:]
    df_outplayed = df_outplayed[columns]

    columns = df_outplayed.columns.tolist()
    involvement_model_index = columns.index("involvement_model")
    columns = columns[:involvement_model_index] + columns[involvement_model_index + 1:]
    columns = columns[:len(columns) - 1] + ["involvement_model"] + columns[len(columns) - 1:]
    df_outplayed = df_outplayed[columns]

    # drop faultribution and raw_faultribution until we figure out if we need it
    df_outplayed = df_outplayed.drop(columns=["faultribution", "raw_faultribution"])

    return df_outplayed


# @st.cache_resource
def get_involvement(
    df_passes, df_tracking, event_id_col="event_id", event_string_col="event_string",
    event_success_col="pass_is_successful", event_intercepted_col="pass_is_intercepted",
    event_team_col="team_id_1", event_player_col="player_id_1", event_frame_col="full_frame",
    event_raw_x_col="x_event", event_raw_y_col="y_event", event_raw_target_x_col="x_target", event_raw_target_y_col="y_target",
    event_receiver_col="player_id_2", value_col="pass_xt",
    event_target_frame_col="full_frame_rec",
    event_outcome_col="is_successful", event_player_name_col="player_name_1",
    tracking_frame_col="full_frame", tracking_x_col="x_tracking", tracking_y_col="y_tracking",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_period_col="period_id", tracking_team_in_possession_col="team_in_possession",
    tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="BALL", tracking_player_name_col="player_name",
    involvement_model_success_pos_value="circle_circle_rectangle",
    involvement_model_success_neg_value="circle_passer", involvement_model_out="circle_passer",
    involvement_model_intercepted="intercepter", model_radius=5
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> df_event = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [2, 2, 2], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "pass_is_successful": [True, False, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "player_id_1": [1, 2, 3], "event_string": ["pass", "pass", "pass"], "pass_xt": [0.1, -0.1, -0.1], "pass_is_intercepted": [False, False, True], "player_name_1": ["A", "B", "C"]})
    >>> df_event

    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [1, 1, 1, 1, 1, 1, 2, 2, 2], "player_id": [2, 3, 4, 2, 3, 4, "BALL", "BALL", "BALL"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [0, 0, 0, 0, 0, 0, 0, 0, 0], "player_name": ["A", "B", "C", "A", "B", "C", "BALL", "BALL", "BALL"]})
    >>> df_tracking

    >>> df_involvement = get_involvement(df_event, df_tracking)
    >>> df_involvement

    """
    if value_col is None:
        value_col = "dummy"
        df_passes[value_col] = 1

    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["full_frame", "team_id", "player_id", "x_tracking", "y_tracking"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_passes, "df_passes", ["event_id", "team_id_1", "player_id_1", "full_frame", "x_norm", "y_norm", "x_target", "y_target", "is_successful"], [event_id_col, event_team_col, event_player_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col, event_success_col])
    df_passes[event_success_col] = df_passes[event_success_col].astype(bool)
    df_passes[event_intercepted_col] = df_passes[event_intercepted_col].astype(bool)

    max_passes_for_debug = 3

    # 1. Successful passes, xT >= 0
    i_success_and_pos_value = df_passes[event_success_col] & (df_passes[value_col] >= 0)
    df_passes.loc[i_success_and_pos_value, "involvement_type"] = "success_and_pos_value"
    df_involvement_success = _get_faultribution_by_model_matrix(
        df_passes.loc[i_success_and_pos_value], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col, tracking_period_col, tracking_team_in_possession_col,
        event_id_col, event_team_col, event_player_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col, value_col,
        event_target_frame_col, event_raw_target_x_col, event_raw_target_y_col, event_outcome_col,
        model=involvement_model_success_pos_value, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius, tracking_player_name_col=tracking_player_name_col,
    )
    # with st.expander("Success, xT > 0"):
    #     plot_passes_with_involvement(
    #         df_involvement_success, involvement_model_success_pos_value, model_radius, df_passes, df_tracking,
    #
    #         event_id_col,
    #         event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col,
    #         event_frame_col, event_team_col, event_player_name_col,
    #         tracking_team_col, tracking_player_col, tracking_x_col,
    #         tracking_y_col,
    #         tracking_frame_col, tracking_player_name_col, tracking_vx_col,
    #         tracking_vy_col,
    #         event_string_col, xt_col, ball_tracking_player_id
    #     )
    # st.write("---")

    # 2. Successful passes, xT < 0
    i_success_and_neg_value = df_passes[event_success_col] & (df_passes[value_col] < 0)
    df_passes.loc[i_success_and_neg_value, "involvement_type"] = "success_and_neg_value"
    df_involvement_success_neg = _get_faultribution_by_model_matrix(
        df_passes.loc[i_success_and_neg_value], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col, tracking_period_col, tracking_team_in_possession_col,
        event_id_col, event_team_col, event_player_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col, value_col,
        event_target_frame_col, event_raw_target_x_col, event_raw_target_y_col, event_outcome_col,
        model=involvement_model_success_neg_value, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius, tracking_player_name_col=tracking_player_name_col,
    )

    # with st.expander("Success, xT < 0"):
    #     plot_passes_with_involvement(
    #         df_involvement_success_neg, involvement_model_success_neg_value, model_radius, df_passes, df_tracking,
    #
    #         event_id_col,
    #         event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col,
    #         event_frame_col, event_team_col, event_player_name_col,
    #         tracking_team_col, tracking_player_col, tracking_x_col,
    #         tracking_y_col,
    #         tracking_frame_col, tracking_player_name_col, tracking_vx_col,
    #         tracking_vy_col,
    #         event_string_col, xt_col, ball_tracking_player_id
    #     )
    # st.write("---")

    # 3. Unsuccessful passes, out
    i_out = (~df_passes[event_success_col]) & (~df_passes[event_intercepted_col])
    df_passes.loc[i_out, "involvement_type"] = "out"
    df_involvement_out = _get_faultribution_by_model_matrix(
        df_passes.loc[i_out], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col, tracking_period_col, tracking_team_in_possession_col,
        event_id_col, event_team_col, event_player_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col, value_col,
        event_target_frame_col, event_raw_target_x_col, event_raw_target_y_col, event_outcome_col,
        model=involvement_model_out, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius, tracking_player_name_col=tracking_player_name_col,
    )
    # with st.expander("Out"):
    #     plot_passes_with_involvement(
    #         df_involvement_out, involvement_model_out, model_radius, df_passes, df_tracking,
    #         event_id_col,
    #         event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col,
    #         event_frame_col, event_team_col, event_player_name_col,
    #         tracking_team_col, tracking_player_col, tracking_x_col,
    #         tracking_y_col,
    #         tracking_frame_col, tracking_player_name_col, tracking_vx_col,
    #         tracking_vy_col,
    #         event_string_col, xt_col, ball_tracking_player_id
    #     )
    # st.write("---")

    # 4. Unsuccessful passes, intercepted
    # i_intercepted = df_passes["outcome"] == "intercepted"
    i_intercepted = df_passes[event_intercepted_col]
    df_passes.loc[i_intercepted, "involvement_type"] = "intercepted"
    df_involvement_intercepted = _get_faultribution_by_model_matrix(
        df_passes.loc[i_intercepted], df_tracking,
        tracking_frame_col, tracking_team_col,
        tracking_player_col, tracking_x_col, tracking_y_col, tracking_period_col, tracking_team_in_possession_col,
        event_id_col, event_team_col, event_player_col, event_frame_col, event_raw_x_col, event_raw_y_col, event_receiver_col, value_col,
        event_target_frame_col, event_raw_target_x_col, event_raw_target_y_col, event_outcome_col,
        model=involvement_model_intercepted, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius, tracking_player_name_col=tracking_player_name_col,
    )
    # with st.expander("Intercepted"):
    #     plot_passes_with_involvement(
    #         df_involvement_intercepted, involvement_model_intercepted, model_radius, df_passes, df_tracking,
    #
    #         event_id_col,
    #         event_raw_x_col, event_raw_y_col, event_raw_target_x_col, event_raw_target_y_col,
    #         event_frame_col, event_team_col, event_player_name_col,
    #         tracking_team_col, tracking_player_col, tracking_x_col,
    #         tracking_y_col,
    #         tracking_frame_col, tracking_player_name_col, tracking_vx_col,
    #         tracking_vy_col,
    #         event_string_col, xt_col, ball_tracking_player_id
    #     )
    # st.write("---")

    df_involvement = pd.concat([df_involvement_success, df_involvement_success_neg, df_involvement_out, df_involvement_intercepted])

    df_involvement = df_involvement[df_involvement["involvement"].notna()]  # Drop attackers etc.

    return df_involvement


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    df_passes = pd.DataFrame({"event_id": [0, 1, 2], "team_id_1": [1, 1, 1], "full_frame": [0, 1, 2], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, True, False], "player_id_2": [2, 3, 4], "full_frame_rec": [1, 2, 3], "pass_xt": [-0.1, 0.1, -0.1]})
    df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [0, 0, 1, 0, 0, 1, 0, 0, 1], "player_id": [2, 3, "BALL", 2, 3, "BALL", 2, 3, "BALL"], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [1, 2, 3, 4, 5, 6, 7, 8, 9], "vx": [0, 0, 0, 0, 0, 0, 0, 0, 0], "vy": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    df_faultribution = _get_faultribution_by_model_matrix(df_passes, df_tracking, model="intercepter", model_radius=15)
    st.write("df_faultribution")
    st.write(df_faultribution)
