import collections
import sys
import os

import numpy as np

sys.path.append(os.path.join(__file__, "../../.."))

import importlib
import io
import pandas as pd
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
    >>> from defensive_network.tests.test_data import df_events, df_tracking
    >>> df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])
    >>> get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
                                                                        responsibility  n_passes
    player_position unified_receiver_position defender_player_position
    DF              MF                        MF                              0.500000         1
                                              ST                              0.500000         2
    MF              ST                        DF                              0.000000         3
                                              MF                              0.573490         4
                                              ST                              0.148903         5
    ST              MF                        DF                              0.000000         1
                                              MF                              0.416905         1
                                              ST                              0.416905         1
                    ST                        DF                              1.000000         1
                                              MF                              0.000000         1
                                              ST                              0.000000         1
    """
    dfg_responsibility_model = df_involvement.groupby(responsibility_context_cols).agg(
        raw_responsibility=(involvement_col, "mean"),
        n_passes=(involvement_col, "count"),
        value=(value_col, "mean")
    )
    dfg_responsibility_model["valued_responsibility"] = dfg_responsibility_model["raw_responsibility"] * dfg_responsibility_model["value"]
    return dfg_responsibility_model


def get_responsibility(df_passes, dfg_responsibility_model, event_id_col="involvement_pass_id", value_col="pass_xt", context_cols=["role_category_1", "network_receiver_role_category", "defender_role_category"]):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> from defensive_network.tests.test_data import df_events, df_tracking
    >>> df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])
    >>> dfg_responsibility = get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
    >>> df_involvement["responsibility"], df_involvement["sample_size_for_responsibility"] = get_responsibility(df_involvement, dfg_responsibility)
    >>> df_involvement.head(3)
       involvement_pass_id defender_id  raw_involvement  raw_contribution  raw_fault  involvement  contribution  fault  frame_id  frame_id_rec  x_event  y_event player_id_1 player_position player_id_2 receiver_position team_id_1 team_id_2  pass_is_successful  pass_xt  pass_is_intercepted  x_target  y_target expected_receiver defender_name  defender_x  defender_y        involvement_model expected_receiver_position unified_receiver unified_receiver_position      event_string       involvement_type defender_player_name defender_player_position  model_radius  responsibility  sample_size_for_responsibility
    0                    0           x              0.5               0.0        0.5          0.1           0.0    0.1         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         x(ST)           5          -5  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                x(ST)                       ST            10        0.148903                             5.0
    1                    0           y              0.5               0.0        0.5          0.1           0.0    0.1         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         y(MF)           5           5  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                y(MF)                       MF            10        0.573490                             4.0
    2                    0           z              0.0               0.0        0.0          0.0           0.0    0.0         0             3        0        0           a              MF           b                ST         H         H                True      0.2                False        10         0               NaN         z(DF)          40           0  circle_circle_rectangle                        NaN                b                        ST  a (MF) -> b (ST)  success_and_pos_value                z(DF)                       DF            10        0.000000                             3.0
    >>> defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, tracking_frame_col="frame_id", pass_frame_col="frame_id", n_passes=1000000)
    [<Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>, <Figure size 640x480 with 1 Axes>]
    """
    n_passes = df_passes.shape[0]

    dfg_responsibility_model = dfg_responsibility_model.reset_index()

    df_passes = df_passes[[event_id_col, value_col] + context_cols]

    df_passes["_index"] = df_passes.index
    df_passes = df_passes.merge(dfg_responsibility_model, on=context_cols, how="left")
    df_passes = df_passes.set_index("_index")

    # st.write("df_passes")
    # st.write(df_passes)

    # def foo(row):
    #     try:
    #         r = dfg_responsibility_model.loc[tuple(row)]
    #         # st.write(f"r {tuple(row)})")
    #         # st.write(r)
    #         # st.stop()
    #         return r
    #     except KeyError as e:
    #         return None
    #
    # try:
    #     # st.write("dfg_responsibility_model.index.names")
    #     # st.write(dfg_responsibility_model.index.names)
    #     # st.write("df_passes[dfg_responsibility_model.index.names]")
    #     # st.write(df_passes[dfg_responsibility_model.index.names])
    #     responsibility = df_passes[dfg_responsibility_model.index.names].apply(lambda row: foo(row), axis=1)
    #     st.write("A turn")
    # except KeyError:
    #     # use columns before "responsibility" as index
    #     i_responsibility_col = dfg_responsibility_model.columns.get_loc("responsibility")
    #     dfg_responsibility_model = dfg_responsibility_model.set_index(dfg_responsibility_model.columns[:i_responsibility_col].tolist())
    #     st.write("B turn")
    #     st.write("df_passes")
    #     st.write(df_passes)
    #     # st.write(df_passes[dfg_responsibility_model.index.names])
    #     st.write("dfg_responsibility_model")
    #     st.write(dfg_responsibility_model)
    #     responsibility = df_passes[dfg_responsibility_model.index.names].apply(lambda row: foo(row), axis=1)
    #     st.write("responsibility")
    #     st.write(responsibility)
    #

    df_passes["raw_relative_responsibility"] = df_passes.groupby(event_id_col)["raw_responsibility"].transform(lambda x: x / x.sum())
    df_passes["valued_responsibility"] = df_passes["raw_responsibility"] * df_passes[value_col].abs()
    df_passes["valued_relative_responsibility"] = df_passes["raw_relative_responsibility"] * df_passes[value_col].abs()

    assert len(df_passes) == n_passes, f"Number of passes changed during responsibility calculation: {len(df_passes)} != {n_passes}. Check context_cols."

    return ResponsibilityResult(df_passes["raw_responsibility"], df_passes["raw_relative_responsibility"], df_passes["valued_responsibility"], df_passes["valued_relative_responsibility"])

def main():
    from defensive_network.tests.data import df_events, df_tracking
    import defensive_network.utility.general

    defensive_network.utility.general.start_streamlit_profiler()

    # (base_path, selected_tracking_matches, xt_model, expected_receiver_model, formation_model,
    # involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out,
    # involvement_model_intercepted, model_radius, selected_player_col, selected_player_name_col,
    # selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col,
    # selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col,
    # use_tracking_average_position, selected_value_col, plot_involvement_examples, n_examples_per_type,
    # show_def_full_metrics, remove_passes_with_zero_involvement, defender_col, defender_name_col) = defensive_network.utility.scripts.select_defensive_network_options()
    #
    # for slugified_match_string in selected_tracking_matches:
    #     df_tracking, df_events = defensive_network.parse.cdf.get_match_data(
    #         base_path, slugified_match_string, xt_model=xt_model,
    #         expected_receiver_model=expected_receiver_model, formation_model=formation_model
    #     )
    #
    #     st.write("df_events", df_events.shape)
    #     st.write(df_events.set_index("event_string"))

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
