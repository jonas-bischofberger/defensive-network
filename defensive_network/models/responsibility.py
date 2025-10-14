import collections
import sys
import os

import numpy as np

sys.path.append(os.path.join(__file__, "../../.."))

import importlib
import streamlit as st

import defensive_network.models.involvement
import defensive_network.utility.pitch
import defensive_network.utility.dataframes
import defensive_network.utility.pitch


importlib.reload(defensive_network.models.involvement)
importlib.reload(defensive_network.utility.pitch)


ResponsibilityResult = collections.namedtuple("ResponsibilityResult", ["raw_responsibility", "raw_relative_responsibility", "valued_responsibility", "valued_relative_responsibility"])


def get_responsibility_model(df_involvement, responsibility_context_cols=["role_category_1", "network_receiver_role_category", "defender_role_category"], involvement_col="raw_involvement", value_col="pass_xt"):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> from defensive_network.tests.data import df_events, df_tracking
    >>> df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])
    >>> get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
                                                                        raw_responsibility  n_passes     value  valued_responsibility
    player_position unified_receiver_position defender_player_position
    DF              MF                        MF                                  0.500000         1  0.500000               0.250000
                                              ST                                  0.500000         2  0.500000               0.250000
    MF              ST                        DF                                  0.000000         3  0.166667               0.000000
                                              MF                                  0.573490         4  0.050000               0.028674
                                              ST                                  0.148903         5 -0.020000              -0.002978
    ST              MF                        DF                                  0.000000         1 -0.100000              -0.000000
                                              MF                                  0.416905         1 -0.100000              -0.041690
                                              ST                                  0.416905         1 -0.100000              -0.041690
                    ST                        DF                                  1.000000         1 -0.200000              -0.200000
                                              MF                                  0.000000         1 -0.200000              -0.000000
                                              ST                                  0.000000         1 -0.200000              -0.000000
    """
    dfg_responsibility_model = df_involvement.groupby(responsibility_context_cols).agg(
        raw_responsibility=(involvement_col, "mean"),
        n_passes=(involvement_col, "count"),
        value=(value_col, "mean"),
    )
    dfg_responsibility_model["valued_responsibility"] = dfg_responsibility_model["raw_responsibility"] * dfg_responsibility_model["value"]
    return dfg_responsibility_model


def get_responsibility(df_passes, dfg_responsibility_model, event_id_col="involvement_pass_id", value_col="pass_xt", context_cols=["role_category_1", "network_receiver_role_category", "defender_role_category"]):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> from defensive_network.tests.data import df_events, df_tracking
    >>> df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])
    >>> dfg_responsibility = get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
    >>> df_involvement["raw_responsibility"], df_involvement["raw_relative_responsibility"], df_involvement["valued_responsibility"], df_involvement["valued_relative_responsibility"] = get_responsibility(df_involvement, dfg_responsibility, context_cols=["player_position", "unified_receiver_position", "defender_player_position"])
    >>> df_involvement.head(3)
       involvement_pass_id defender_id  raw_involvement  raw_contribution  raw_fault  valued_involvement  valued_contribution  valued_fault  frame_id  frame_id_rec  x_event  y_event player_id_1 player_position player_id_2 receiver_position team_id_1 team_id_2  pass_is_successful  pass_xt  pass_is_intercepted  x_target  y_target expected_receiver defender_name  defender_x  defender_y        involvement_model expected_receiver_position unified_receiver unified_receiver_position      event_string       involvement_type defender_player_name defender_player_position  model_radius  raw_responsibility  raw_relative_responsibility  valued_responsibility  valued_relative_responsibility
    0                    0           x              0.5               0.0        0.5                 0.1                  0.0           0.1         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         x(ST)           5          -5  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                x(ST)                       ST            10            0.148903                     0.206125               0.029781                        0.041225
    1                    0           y              0.5               0.0        0.5                 0.1                  0.0           0.1         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         y(MF)           5           5  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                y(MF)                       MF            10            0.573490                     0.793875               0.114698                        0.158775
    2                    0           z              0.0               0.0        0.0                 0.0                  0.0           0.0         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         z(DF)          40           0  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                z(DF)                       DF            10            0.000000                     0.000000               0.000000                        0.000000
    >>> _ = defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, tracking_frame_col="frame_id", pass_frame_col="frame_id", n_passes=1000000)
    """
    n_passes = df_passes.shape[0]

    dfg_responsibility_model = dfg_responsibility_model.reset_index()

    df_passes = df_passes[[event_id_col, value_col] + context_cols]
    df_passes["_index"] = df_passes.index
    df_passes = df_passes.merge(dfg_responsibility_model, on=context_cols, how="left")
    df_passes = df_passes.set_index("_index")

    df_passes["raw_relative_responsibility"] = df_passes.groupby(event_id_col)["raw_responsibility"].transform(lambda x: x / x.sum())
    df_passes["valued_responsibility"] = df_passes["raw_responsibility"] * df_passes[value_col].abs()
    df_passes["valued_relative_responsibility"] = df_passes["raw_relative_responsibility"] * df_passes[value_col].abs()

    assert len(df_passes) == n_passes, f"Number of passes changed during responsibility calculation: {len(df_passes)} != {n_passes}. Check context_cols."

    return ResponsibilityResult(df_passes["raw_responsibility"], df_passes["raw_relative_responsibility"], df_passes["valued_responsibility"], df_passes["valued_relative_responsibility"])

def main():
    from defensive_network.tests.data import df_events, df_tracking
    import defensive_network.utility.general

    defensive_network.utility.general.start_streamlit_profiler()

    st.write("df_events")
    st.write(df_events)

    df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])
    dfg_responsibility = get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
    st.write("dfg_responsibility")
    st.write(dfg_responsibility)

    df_involvement["raw_responsibility"] = get_responsibility(df_involvement, dfg_responsibility)
    st.write("df_involvement")
    st.write(df_involvement)

    defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, tracking_frame_col="frame_id", pass_frame_col="frame_id", n_passes=20000)


if __name__ == '__main__':
    main()
