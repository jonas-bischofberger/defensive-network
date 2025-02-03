import os

import collections
import streamlit as st
import defensive_network.parse.dfb

StreamlitDefensiveNetworkOptions = collections.namedtuple("StreamlitDefensiveNetworkOptions", [
    "base_path", "selected_tracking_matches", "xt_model", "expected_receiver_model", "formation_model",
    "involvement_model_success_pos_value", "involvement_model_success_neg_value", "involvement_model_out",
    "involvement_model_intercepted", "model_radius", "selected_player_col", "selected_player_name_col",
                "selected_receiver_col", "selected_receiver_name_col", "selected_expected_receiver_col",
                "selected_expected_receiver_name_col", "selected_tracking_player_col", "selected_tracking_player_name_col",
                "use_tracking_average_position", "selected_value_col", "plot_involvement_examples", "n_examples_per_type",
                "show_def_full_metrics", "remove_passes_with_zero_involvement", "defender_col", "defender_name_col",
])

def _select_matches(base_path):
    df_meta = defensive_network.parse.dfb.get_all_meta(base_path)
    # df_lineups = defensive_network.parse.dfb.get_all_lineups(base_path)
    all_tracking_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.get_tracking_fpath(base_path, "")))
    all_event_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.get_event_fpath(base_path, "")))

    all_files = [file for file in all_tracking_files if file.replace("parquet", "csv") in all_event_files]
    all_slugified_match_strings = [os.path.splitext(file)[0] for file in all_files]

    default = [match_string for match_string in ["3-liga-2023-2024-20-st-sc-verl-viktoria-koln"] if match_string in all_slugified_match_strings]

    slugified_match_string_to_match_string = df_meta.set_index("slugified_match_string")["match_string"].to_dict()
    selceted_tracking_slugified_match_strings = st.multiselect("Select tracking files", all_slugified_match_strings, default=default, format_func=lambda x: slugified_match_string_to_match_string.get(x, x))
    return selceted_tracking_slugified_match_strings


def select_defensive_network_options() -> StreamlitDefensiveNetworkOptions:
    all_involvement_models = ["circle_circle_rectangle", "circle_passer", "circle_receiver", "intercepter"]
    all_xt_models = ["ma2024", "the_athletic"]
    all_expected_receiver_models = ["power2017"]
    all_formation_models = ["average_pos"]

    with st.expander("Settings"):
        base_path = st.text_input("Base path", "2324")

        involvement_types = ["intercepted", "out", "success_and_neg_value", "success_and_pos_value"]

        # Select options
        selected_tracking_matches = _select_matches(base_path)
        xt_model = st.selectbox("Select xT model", all_xt_models)
        expected_receiver_model = st.selectbox("Select expected receiver model", all_expected_receiver_models)
        formation_model = st.selectbox("Select formation model", all_formation_models)
        involvement_model_success_pos_value = st.selectbox("Select involvement model for successful passes with positive value", all_involvement_models, index=0)
        involvement_model_success_neg_value = st.selectbox("Select involvement model for successful passes with negative value", all_involvement_models, index=1)
        involvement_model_out = st.selectbox("Select involvement model for passes that are interrupted by referee (e.g. goes out of the pitch)", all_involvement_models, index=1)
        involvement_model_intercepted = st.selectbox("Select involvement model for passes that are intercepted", all_involvement_models, index=3)
        model_radius = st.number_input("Circle model radius", min_value=0, value=5)
        selected_player_mode = st.selectbox("Select player column for networks", [("player_id_1", "player_name_1", "player_id_2", "player_name_2", "expected_receiver", "expected_receiver_name", "player_id", "player_name", "Player"), ("role_1", "role_name_1", "role_2", "role_name_2", "expected_receiver_role", "expected_receiver_role_name", "role", "role_name", "Role")], format_func=lambda x: x[-1], index=1)
        selected_player_col, selected_player_name_col, selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col, selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col = selected_player_mode[0], selected_player_mode[1], selected_player_mode[2], selected_player_mode[3], selected_player_mode[4], selected_player_mode[5], selected_player_mode[6], selected_player_mode[7]
        use_tracking_average_position = st.toggle("Use average player position in tracking data for passing networks")
        selected_value_col = st.selectbox("Value column for networks", ["pass_xt", None], format_func=lambda x: str(x))
        plot_involvement_examples = st.multiselect("Plot involvement examples", involvement_types, default=[])
        n_examples_per_type = st.number_input("Number of examples", min_value=0, value=3)
        show_def_full_metrics = st.toggle("Show individual offensive metrics for each defender", value=False)
        remove_passes_with_zero_involvement = st.toggle("Remove passes with 0 involvement from networks", value=True)
        defender_col = st.selectbox("Defender for defensive networks", ["defender_id"], format_func=lambda x: {"defender_id": "Defender (player)"}[x])
        defender_name_col = {"defender_id": "defender_name"}[defender_col]

        return StreamlitDefensiveNetworkOptions(base_path, selected_tracking_matches, xt_model, expected_receiver_model, formation_model,
                involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out,
                involvement_model_intercepted, model_radius, selected_player_col, selected_player_name_col,
                selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col,
                selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col,
                use_tracking_average_position, selected_value_col, plot_involvement_examples, n_examples_per_type,
                show_def_full_metrics, remove_passes_with_zero_involvement, defender_col, defender_name_col)
