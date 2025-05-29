import numpy as np
import pandas as pd

import streamlit as st
import defensive_network.utility.dataframes
import collections

ExpectedReceiverResult = collections.namedtuple("ExpectedReceiverResult", ["expected_receiver"])


def get_expected_receiver(
    df_passes, df_tracking, event_frame_col="full_frame", event_team_col="team_id_1", event_player_col="player_id_1",
    event_x_col="x_event", event_y_col="y_event", event_target_x_col="x_target", event_target_y_col="y_target",
    tracking_frame_col="full_frame", tracking_team_col="team_id", tracking_player_col="player_id",
    tracking_x_col="x_tracking", tracking_y_col="y_tracking",
    event_success_col="is_successful", model="power2017"
):
    """
    Adds the expected receiver to a frame of passes based on tracking data, according to the model of Power et al. (2017).

    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df_passes = pd.DataFrame({"full_frame": [0, 1, 2], "team_id_1": [1, 1, 1], "player_id_1": [1, 2, 3], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, False, False]})
    >>> df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [1, 1, 1, 1, 1, 1, 1, 1, 1], "player_id": [2, 3, 4, 2, 3, 4, 2, 3, 4], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    >>> df_passes["expected_receiver"] = get_expected_receiver(df_passes, df_tracking)
    >>> df_passes
       full_frame  team_id_1  player_id_1  x_event  y_event  x_target  y_target  is_successful  expected_receiver
    0           0          1            1        0        0        10         0          False                2.0
    1           1          1            2        0        0        20         0          False                3.0
    2           2          1            3        0        0        30         0          False                2.0
    """
    assert model == "power2017"  # the only implemented model
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_passes, "df_passes", ["event_frame_col", "event_team_col", "event_player_col", "event_x_col", "event_y_col", "event_target_x_col", "event_target_y_col", "event_success_col"], [event_frame_col, event_team_col, event_player_col, event_x_col, event_y_col, event_target_x_col, event_target_y_col, event_success_col])
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_tracking, "df_tracking", ["tracking_frame_col", "tracking_team_col", "tracking_player_col", "tracking_x_col", "tracking_y_col"], [tracking_frame_col, tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col])

    df_passes = df_passes.copy()

    df_passes_unsuccessful = df_passes[~df_passes[event_success_col]]

    frames_in_events_not_in_tracking = set(df_passes_unsuccessful[event_frame_col]) - set(df_tracking[tracking_frame_col])
    if len(frames_in_events_not_in_tracking) != 0:
        st.warning(f"Event frames not present in tracking data: {frames_in_events_not_in_tracking} (tracking, e.g.: {df_tracking[tracking_frame_col].iloc[0]}), (events, e.g.: {df_passes_unsuccessful[event_frame_col].iloc[0]})")
        df_passes_unsuccessful = df_passes_unsuccessful[~df_passes_unsuccessful[event_frame_col].isin(frames_in_events_not_in_tracking)]

    tracking_groups = df_tracking.groupby(tracking_frame_col)

    for pass_index, p4ss in df_passes_unsuccessful.iterrows():
        # df_tracking_frame_attackers = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_frame_col]]
        df_tracking_frame_attackers = tracking_groups.get_group(p4ss[event_frame_col])
        df_tracking_frame_attackers = df_tracking_frame_attackers[
            (df_tracking_frame_attackers[tracking_team_col] == p4ss[event_team_col]) &
            (df_tracking_frame_attackers[tracking_player_col] != p4ss[event_player_col])
        ]
        if pd.isna(p4ss[event_target_x_col]):
            continue

        pass_angle = np.arctan2(p4ss[event_target_y_col] - p4ss[event_y_col], p4ss[event_target_x_col] - p4ss[event_x_col])

        distance_to_pass_endpoint_col, defender_angle_col, angle_to_pass_lane_col = defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "distance_to_pass_endpoint"), defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "defender_angle"), defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "angle_to_pass_lane")
        df_tracking_frame_attackers[distance_to_pass_endpoint_col] = np.sqrt(
            (df_tracking_frame_attackers[tracking_x_col] - p4ss[event_target_x_col]) ** 2 +
            (df_tracking_frame_attackers[tracking_y_col] - p4ss[event_target_y_col]) ** 2
        )
        df_tracking_frame_attackers[defender_angle_col] = np.arctan2(
            df_tracking_frame_attackers[tracking_y_col] - p4ss[event_y_col],
            df_tracking_frame_attackers[tracking_x_col] - p4ss[event_x_col]
        )
        eps = 1e-10
        df_tracking_frame_attackers[angle_to_pass_lane_col] = np.abs(pass_angle - df_tracking_frame_attackers[defender_angle_col])
        min_distance_to_pass_endpoint = np.maximum(eps, df_tracking_frame_attackers[distance_to_pass_endpoint_col].min())
        min_angle_to_pass_lane = np.maximum(eps, df_tracking_frame_attackers[angle_to_pass_lane_col].min())

        expected_receiver_score_distance_col, expected_receiver_score_angle_col, expected_receiver_score_col = defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "expected_receiver_score_distance"), defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "expected_receiver_score_angle"), defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "expected_receiver_score")
        df_tracking_frame_attackers[expected_receiver_score_distance_col] = df_tracking_frame_attackers[distance_to_pass_endpoint_col].div(min_distance_to_pass_endpoint, fill_value=np.inf)
        df_tracking_frame_attackers[expected_receiver_score_angle_col] = df_tracking_frame_attackers[angle_to_pass_lane_col].div(min_angle_to_pass_lane, fill_value=np.inf)
        df_tracking_frame_attackers[expected_receiver_score_col] = df_tracking_frame_attackers[expected_receiver_score_distance_col] * df_tracking_frame_attackers[expected_receiver_score_angle_col]

        # fig = plot_pass(p4ss, df_tracking)
        # st.write(fig)

        expected_receiver_col = defensive_network.utility.dataframes.get_unused_column_name(df_tracking_frame_attackers.columns, "expected_receiver")

        expected_receiver = df_tracking_frame_attackers.loc[df_tracking_frame_attackers[expected_receiver_score_col].idxmin(), tracking_player_col]
        df_passes.loc[pass_index, expected_receiver_col] = expected_receiver

    return df_passes["expected_receiver"]


if __name__ == '__main__':
    defensive_network.utility.dataframes.prepare_doctest()
    df_passes = pd.DataFrame({"full_frame": [0, 1, 2], "team_id_1": [1, 1, 1], "player_id_1": [1, 2, 3], "x_event": [0, 0, 0], "y_event": [0, 0, 0], "x_target": [10, 20, 30], "y_target": [0, 0, 0], "is_successful": [False, False, False]})
    df_tracking = pd.DataFrame({"full_frame": [0, 0, 0, 1, 1, 1, 2, 2, 2], "team_id": [1, 1, 1, 1, 1, 1, 1, 1, 1], "player_id": [2, 3, 4, 2, 3, 4, 2, 3, 4], "x_tracking": [5, 10, 15, 5, 10, 15, 5, 10, 15], "y_tracking": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    res = get_expected_receiver(df_passes, df_tracking)
    df_passes["expected_receiver"] = res.expected_receiver
    st.write(df_passes)
