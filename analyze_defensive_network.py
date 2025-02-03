import collections
import importlib

import networkx.exception
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import functools
import networkx as nx

import sys
import os

import defensive_network.utility.pitch
import defensive_network.utility.dataframes
import defensive_network.utility.general
import defensive_network.models.involvement
import defensive_network.models.passing_network
import defensive_network.models.expected_receiver
import defensive_network.parse.dfb
import defensive_network.models.average_position

# import defensive_network.scripts.create_dfb_tracking_animations

# importlib.reload(defensive_network.scripts.create_dfb_tracking_animations)
importlib.reload(defensive_network.parse.dfb)

assert "defensive_network" in sys.modules
assert "defensive_network.utility" in sys.modules
importlib.reload(defensive_network.models.involvement)
importlib.reload(defensive_network.models)


PRELOADED_MODULES = set()

def init() :
    # local imports to keep things neat
    from sys import modules

    global PRELOADED_MODULES

    # sys and importlib are ignored here too
    PRELOADED_MODULES = set(modules.values())

def reload() :
    from sys import modules
    import importlib

    for module in set(modules.values()) - PRELOADED_MODULES :
        try :
            importlib.reload(module)
        except :
            # there are some problems that are swept under the rug here
            pass

init()

import warnings

# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

warnings.filterwarnings("ignore")  # hack to suppress weird streamlit bug, see: https://discuss.streamlit.io/t/keyerror-warnings/2474/14

# dummy data
df_tracking = pd.DataFrame({
    "frame_id": [0, 0, 0, 0],
    "player_id": ["a", "b", "x", "ball"],
    "x_tracking": [0, -50, 50, 0],
    "y_tracking": [0, 0, 0, 0],
    "vx": [0, 0, 0, 0],
    "vy": [0, 0, 0, 0],
    "team_id": ["H", "H", "A", None],
    "player_color": ["blue", "blue", "red", "black"],
    "team_in_possession": ["H"] * 4,
    "player_in_possession": ["a"] * 4,
    # "attacking_direction": [1] * 4,
})

### Plotting
# plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])
# plt.show()

df_pass_safe = pd.DataFrame({
    "frame_id": [0],
    "player_id": ["a"],
    "team_id": ["H"],
    "x": [0],
    "y": [0],
    "x_target": [-50],
    "y_target": [0],
    "v0": [15],
})
df_pass_risky = df_pass_safe.copy()
df_pass_risky["x_target"] = 50


def _select_matches(base_path):
    df_meta = defensive_network.parse.dfb.get_all_meta(base_path)
    df_lineups = defensive_network.parse.dfb.get_all_lineups(base_path)
    all_tracking_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.get_tracking_fpath(base_path, "")))
    all_event_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.get_event_fpath(base_path, "")))

    all_files = [file for file in all_tracking_files if file.replace("parquet", "csv") in all_event_files]
    all_slugified_match_strings = [os.path.splitext(file)[0] for file in all_files]

    default = "3-liga-2023-2024-20-st-sc-verl-viktoria-koln"

    slugified_match_string_to_match_string = df_meta.set_index("slugified_match_string")["match_string"].to_dict()
    selceted_tracking_slugified_match_strings = st.multiselect("Select tracking files", all_slugified_match_strings, default=[default], format_func=lambda x: slugified_match_string_to_match_string.get(x, x))
    return selceted_tracking_slugified_match_strings



def defensive_network_dashboard():
    defensive_network.utility.general.start_streamlit_profiler()

    base_path = st.text_input("Base path", "data_reduced")

    # create_animation = st.toggle("Create animation", value=False)
    create_animation = False

    selected_tracking_matches = _select_matches(base_path)
    xt_model = st.selectbox("Select xT model", ["ma2024", "the_athletic"])
    expected_receiver_model = st.selectbox("Select expected receiver model", ["power2017"])
    formation_model = st.selectbox("Select formation model", ["average_pos"])

    for slugified_match_string in selected_tracking_matches:
        df_tracking, df_event = defensive_network.parse.dfb.get_match_data(
            base_path, slugified_match_string, xt_model=xt_model,
            expected_receiver_model=expected_receiver_model, formation_model=formation_model
        )

        df_passes = df_event[df_event["event_type"] == "pass"]
        # player2name = df_tracking[["player_id", "player_name"]].set_index("player_id")["player_name"].to_dict()

        if create_animation:
            video_fpath = os.path.join(os.path.dirname(__file__), f"{slugified_match_string}.mp4")
            with st.spinner("Creating animation..."):
                defensive_network.scripts.create_dfb_tracking_animations.create_animation(df_tracking, df_event, video_fpath)

        df_passes = df_passes.dropna(subset=["frame"])

        # df_tracking = defensive_network.add_velocity(df_tracking)

        df_tracking["v"] = np.sqrt(df_tracking["vx"] ** 2 + df_tracking["vy"] ** 2)

        # i_pass = df_event["event_type"] == "pass"
        # df_tracking["vx"] = 0  # TODO add speed
        # df_tracking["vy"] = 0
        # ret = accessible_space.get_expected_pass_completion(
        #     df_event.loc[i_pass], df_tracking,
        #     tracking_x_col="x_tracking", tracking_y_col="y_tracking", tracking_frame_col="full_frame",
        #     event_frame_col="full_frame", event_start_x_col="x_event", event_start_y_col="y_event", event_player_col="player_id_1", event_team_col="team_id_1",
        #     ball_tracking_player_id="BALL",
        # )
        # df_event.iloc[i_pass, "xc"] = ret.xc

        # for pass_nr, (_, p4ss) in accessible_space.progress_bar(enumerate(df_passes.iterrows())):
        #     fig = plot_pass(p4ss, df_tracking)
        #     plt.title(f"Pass {pass_nr}: {p4ss['event_string']}")
        #     st.write(fig)
        #
        #     if pass_nr > 3:
        #         break

        # max_passes_for_debug = st.number_input("Max passes for debug", min_value=0, value=3)
        all_models = ["circle_circle_rectangle", "circle_passer", "circle_receiver", "intercepter"]
        involvement_model_success_pos_value = st.selectbox("Select involvement model for successful passes with positive value", all_models, index=0)
        involvement_model_success_neg_value = st.selectbox("Select involvement model for successful passes with negative value", all_models, index=1)
        involvement_model_out = st.selectbox("Select involvement model for passes that are interrupted by referee (e.g. goes out of the pitch)", all_models, index=1)
        involvement_model_intercepted = st.selectbox("Select involvement model for passes that are intercepted", all_models, index=3)
        model_radius = st.number_input("Circle model radius", min_value=0, value=5)
        selected_player_mode = st.selectbox("Select player column for networks", [("player_id_1", "player_name_1", "player_id_2", "player_name_2", "expected_receiver", "expected_receiver_name", "player_id", "player_name", "Player"), ("role_1", "role_name_1", "role_2", "role_name_2", "expected_receiver_role", "expected_receiver_role_name", "role", "role_name", "Role")], format_func=lambda x: x[-1], index=1)
        selected_player_col, selected_player_name_col, selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col, selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col = selected_player_mode[0], selected_player_mode[1], selected_player_mode[2], selected_player_mode[3], selected_player_mode[4], selected_player_mode[5], selected_player_mode[6], selected_player_mode[7]
        selectedtrackingplayer2name = df_tracking[[selected_tracking_player_col, selected_tracking_player_name_col]].set_index(selected_tracking_player_col)[selected_tracking_player_name_col].to_dict()

        st.write("df_tracking")
        st.write(df_tracking.head())

        use_tracking_average_position = st.toggle("Use average player position in tracking data for passing networks")
        average_positions = None
        if use_tracking_average_position:
            average_positions = defensive_network.models.average_position.calculate_average_positions(df_tracking)["off"]

        st.write("average_positions")
        st.write(average_positions)

        df_tracking = df_tracking[df_tracking[selected_tracking_player_col].notna()]
        df_passes = df_passes[df_passes["frame"].isin(df_tracking["frame"])]

        @st.cache_resource
        def _get_involvement(
            value_col, player_col, receiver_col, tracking_player_col, tracking_player_name_col, slugified_match_string, involvement_model_success_pos_value,
            involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted,
            model_radius
        ):
            df_involvement = defensive_network.models.involvement.get_involvement(
                df_passes, df_tracking, event_player_col=player_col, event_receiver_col=receiver_col, tracking_player_col=tracking_player_col,
                involvement_model_success_pos_value=involvement_model_success_pos_value,
                involvement_model_success_neg_value=involvement_model_success_neg_value,
                involvement_model_out=involvement_model_out, involvement_model_intercepted=involvement_model_intercepted,
                model_radius=model_radius, value_col=value_col, tracking_player_name_col=tracking_player_name_col,
            )
            return df_involvement

        selected_value_col = st.selectbox("Value column for networks", ["pass_xt", None], format_func=lambda x: str(x))

        df_involvement = _get_involvement(selected_value_col, selected_player_col, selected_receiver_col, selected_tracking_player_col, selected_tracking_player_name_col, slugified_match_string, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius)

        st.write("df_involvement")
        st.write(df_involvement)

        plot_involvement_examples = st.multiselect("Plot involvement examples", df_involvement["involvement_type"].unique(), default=df_involvement["involvement_type"].unique())
        n_examples_per_type = st.number_input("Number of examples", min_value=0, value=3)
        if len(plot_involvement_examples) > 0:
            for involvement_type, df_involvement_type in df_involvement.groupby("involvement_type"):
                if involvement_type not in plot_involvement_examples:
                    continue
                st.write(f"### {involvement_type}")
                with st.expander(f"### {involvement_type}"):
                    importlib.reload(defensive_network.models.involvement)
                    defensive_network.models.involvement.plot_passes_with_involvement(
                        df_involvement_type, df_involvement_type["involvement_model"].iloc[0], model_radius, df_passes, df_tracking,
                        n_passes=n_examples_per_type
                    )

        st.write("---")

        plot_offensive_network = functools.partial(
            defensive_network.models.passing_network.plot_passing_network, show_colorbar=False, node_size_multiplier=20,
            arrow_width_multiplier=1, label_col="value_passes", arrow_color_col="value_passes", annotate_top_n_edges=5,
            label_format_string="{:.3f}",
        )
        plot_defensive_network = functools.partial(defensive_network.models.passing_network.plot_passing_network,
            show_colorbar=False, node_size_multiplier=20, arrow_width_multiplier=1,
            # colormap=matplotlib.cm.get_cmap("PuBuGn"),
            colormap=matplotlib.cm.get_cmap("coolwarm"), min_color_value_edges=-0.05, max_color_value_edges=0.05,
            min_color_value_nodes=-0.5, max_color_value_nodes=0.5, annotate_top_n_edges=5, label_col="edge_label",
            label_format_string="{:.3f}", arrow_color_col="value_passes",
        )

        show_def_full_metrics = st.toggle("Show individual offensive metrics for each defender", value=False)
        remove_passes_with_zero_involvement = st.toggle("Remove passes with 0 involvement from networks", value=True)

        total_minutes = 0
        for period, df_period in df_tracking.groupby("section"):
            period_minutes = (df_period["datetime_tracking"].max() - df_period["datetime_tracking"].min()).total_seconds() / 60
            total_minutes += period_minutes

        for team, df_involvement_team in df_involvement.groupby("team_id_1"):
            team_name = df_involvement_team["team_name_1"].iloc[0]
            st.write(f"### {team_name}")
            for involvement_type_col in ["involvement", "contribution", "fault"]:
                st.write(f"## {involvement_type_col.capitalize()} networks")
                with st.expander(involvement_type_col.capitalize()):
                    # selected_player_col, selected_receiver_col
                    df_involvement_team["network_receiver"] = df_involvement_team[selected_expected_receiver_col].where(df_involvement_team[selected_expected_receiver_col].notna(), df_involvement_team[selected_receiver_col])
                    df_involvement_team["network_receiver"] = df_involvement_team["network_receiver"].where(df_involvement_team["outcome"] != "intercepted", None)
                    df_involvement_team["network_receiver_name"] = df_involvement_team[selected_expected_receiver_name_col].where(df_involvement_team[selected_expected_receiver_name_col].notna(), df_involvement_team[selected_receiver_name_col])
                    df_involvement_team["network_receiver_name"] = df_involvement_team["network_receiver_name"].where(df_involvement_team["outcome"] != "intercepted", None)

                    networks = defensive_network.models.passing_network.get_defensive_networks(
                        df_involvement_team, value_col=selected_value_col, involvement_type_col=involvement_type_col,
                        player_col=selected_player_col, player_name_col=selected_player_name_col,
                        receiver_col="network_receiver", receiver_name_col="network_receiver_name",
                        total_minutes=total_minutes, average_positions=average_positions,
                        remove_passes_with_zero_involvement=remove_passes_with_zero_involvement,
                    )
                    metrics = defensive_network.models.passing_network.analyse_defensive_networks(networks)

                    columns = st.columns(2)
                    fig = plot_offensive_network(df_nodes=networks.off_network.df_nodes, df_edges=networks.off_network.df_edges)
                    columns[0].write("xT network")
                    columns[0].write(fig)
                    columns[0].write(metrics.off_network[1])
                    fig = plot_offensive_network(df_nodes=networks.off_involvement_type_network.df_nodes, df_edges=networks.off_involvement_type_network.df_edges)
                    columns[1].write(f"Defensive {involvement_type_col} network")
                    columns[1].write(fig)
                    columns[1].write(metrics.off_involvement_type_network[1])

                    columns = st.columns(3)

                    for defender_nr, (defender, network) in enumerate(networks.def_networks.items()):
                        defender_name = df_involvement_team[df_involvement_team["defender_id"] == defender]["defender_name"].iloc[0]
                        columns[defender_nr % 3].write(f"{defender_name}")

                        # fig = plot_defensive_network(df_nodes=networks.off_network.df_nodes, df_edges=network.df_edges)
                        fig = plot_defensive_network(df_nodes=network.df_nodes, df_edges=network.df_edges)
                        columns[defender_nr % 3].write(fig)
                        if show_def_full_metrics:
                            # columns[defender_nr % 3].write("df_metrics_def")
                            # columns[defender_nr % 3].write(metrics.def_networks[defender][1].to_dict())
                            # st.write(metrics.def_networks[defender][1])
                            try:
                                columns[defender_nr % 3].write(f'Weighted Density: {metrics.def_networks[defender][1].loc["Weighted Density"]}')
                                columns[defender_nr % 3].write(f'Total Degree: {metrics.def_networks[defender][1].loc["Total Degree"]}')
                            except KeyError as e:
                                columns[defender_nr % 3].write(e)
                            # columns[defender_nr % 3].write(metrics.def_networks[defender][1].to_dict())
                            # columns[defender_nr % 3].write("sum")
                            # columns[defender_nr % 3].write(metrics.def_network_sums.loc[defender].to_dict())
                    metrics.def_network_sums["defender_name"] = metrics.def_network_sums.index.map(selectedtrackingplayer2name)

                    st.write(metrics.def_network_sums)
                    st.write(metrics.def_network_sums.set_index("defender_name"))

                    for x_col, y_col in [
                        ("Weighted Density", "Total Degree"),
                        ("Unweighted Density", "Total Degree"),
                        ("Unweighted Density", "Weighted Density"),
                    ]:
                        x = metrics.def_network_sums[x_col]
                        y = metrics.def_network_sums[y_col]

                        # scatter plot
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.scatter(x, y)
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        st.pyplot(fig)

    defensive_network.utility.general.stop_streamlit_profiler()

#
# def demo_dashboard():
#     from accessible_space.tests.resources import df_passes, df_tracking
#     df_passes = df_passes.copy()
#     df_tracking = df_tracking.copy()
#     df_tracking.loc[df_tracking["player_id"] == "Y", "x"] -= 5
#     # change location of player X in frame 6 to (10, 30)
#     df_tracking.loc[(df_tracking["player_id"] == "X") & (df_tracking["frame_id"] == 6), "x"] = 27
#     df_tracking.loc[(df_tracking["player_id"] == "X") & (df_tracking["frame_id"] == 6), "y"] = 30
#
#     res_xt = defensive_network.models.value.get_expected_threat(df_passes, xt_model="ma2024", event_x_col="x", event_y_col="y", pass_end_x_col="x_target", pass_end_y_col="y_target", event_success_col="pass_outcome")
#     df_passes["xt"] = res_xt.delta_xt
#
#     res_xr = defensive_network.models.expected_receiver.get_expected_receiver(df_passes, df_tracking, event_frame_col="frame_id", event_team_col="team_id", event_player_col="player_id", event_x_col="x", event_y_col="y", event_target_x_col="x_target", event_target_y_col="y_target", event_success_col="pass_outcome", tracking_frame_col="frame_id", tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x", tracking_y_col="y")
#     df_passes["expected_receiver"] = res_xr.expected_receiver
#
#     df_passes["is_intercepted"] = [False, False, True]
#     df_passes["event_id"] = np.arange(len(df_passes))
#     df_involvement = defensive_network.models.involvement.get_involvement(
#         df_passes, df_tracking, event_success_col="pass_outcome", event_intercepted_col="is_intercepted", xt_col="xt",
#         tracking_frame_col="frame_id", tracking_team_col="team_id", tracking_player_col="player_id",
#         event_frame_col="frame_id", event_team_col="team_id", event_player_col="player_id", event_x_col="x", event_y_col="y", event_receiver_col="receiver_id", value_col="xt",
#         event_target_frame_col="target_frame_id", event_id_col="event_id", event_target_x_col="x_target", event_target_y_col="y_target",
#         tracking_x_col="x", tracking_y_col="y",
#     )


# def main(run_as_streamlit_app=True, fnc=demo_dashboard):
#     if run_as_streamlit_app:
#         key_argument = "run_dashboard"
#         if len(sys.argv) == 2 and sys.argv[1] == key_argument:
#             fnc()
#         else:  # if script is called directly, call it again with streamlit
#             subprocess.run(['streamlit', 'run', os.path.abspath(__file__), key_argument], check=True)
#     else:
#         fnc()


if __name__ == '__main__':
    defensive_network_dashboard()

    # fnc = st.selectbox("Select function", [demo_dashboard, defensive_network_dashboard], format_func=lambda x: x.__name__, index=1)
    # main(True, fnc)
