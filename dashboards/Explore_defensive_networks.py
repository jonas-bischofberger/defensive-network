import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import importlib
import pandas as pd
import streamlit as st

import defensive_network.models.average_position
import defensive_network.models.expected_receiver
import defensive_network.models.involvement
import defensive_network.models.passing_network
import defensive_network.parse.cdf
import defensive_network.utility.dataframes
import defensive_network.utility.general
import defensive_network.utility.pitch
import defensive_network.utility.dashboards
import defensive_network.models.formation
import defensive_network.utility.video


# import defensive_network.scripts.create_dfb_tracking_animations

assert "defensive_network" in sys.modules
assert "defensive_network.utility" in sys.modules


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


def defensive_network_dashboard():
    defensive_network.utility.general.start_streamlit_profiler()

    (base_path, selected_tracking_matches, xt_model, expected_receiver_model, formation_model,
    involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out,
    involvement_model_intercepted, model_radius, selected_player_col, selected_player_name_col,
    selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col,
    selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col,
    use_tracking_average_position, selected_value_col, plot_involvement_examples, n_examples_per_type,
    show_def_full_metrics, remove_passes_with_zero_involvement, defender_col, defender_name_col) = defensive_network.utility.dashboards.select_defensive_network_options()

    # selected_tracking_matches = ["3-liga-2023-2024-16-st-fc-ingolstadt-04-rot-weiss-essen", "3-liga-2023-2024-16-st-hallescher-fc-1-fc-saarbrucken"]

    remove_throw_ins = st.toggle("Remove throw-ins", value=True)

    for slugified_match_string in selected_tracking_matches:
        df_tracking, df_event = defensive_network.parse.cdf.get_match_data(
            base_path, slugified_match_string, xt_model=xt_model,
            expected_receiver_model=expected_receiver_model, formation_model=formation_model
        )
        defensive_network.utility.general.stop_streamlit_profiler()
        # st.stop()

        df_tracking = df_tracking.loc[df_tracking[selected_tracking_player_col].notna()]
        selectedtrackingplayer2name = df_tracking[[selected_tracking_player_col, selected_tracking_player_name_col]].set_index(selected_tracking_player_col)[selected_tracking_player_name_col].to_dict()

        df_passes = df_event[df_event["event_type"] == "pass"].dropna(subset=["frame"])
        df_passes = df_passes[df_passes["frame"].isin(df_tracking["frame"])]
        df_passes = df_passes[~df_passes["is_in_setpiece_phase"]]

        if remove_throw_ins:
            df_passes = df_passes[df_passes["event_subtype"] != "throw_in"]

        # df_passes_sorted_by_xt = df_passes.sort_values("pass_xt", ascending=False)
        # for i in defensive_network.utility.general.progress_bar(list(range(len(df_passes_sorted_by_xt)-11, len(df_passes_sorted_by_xt)))):
        #     defensive_network.utility.video.pass_video(df_tracking, df_passes_sorted_by_xt.iloc[i:i+1], os.path.join(os.path.dirname(__file__), f"{slugified_match_string}_top_xt_{i}.mp4"))
        # st.stop()

        i_successful = df_passes["outcome"] == "successful"
        df_passes.loc[i_successful, "network_receiver"] = df_passes.loc[i_successful, selected_receiver_col]
        df_passes.loc[i_successful, "network_receiver_name"] = df_passes.loc[i_successful, selected_receiver_name_col]
        df_passes.loc[~i_successful, "network_receiver"] = df_passes.loc[~i_successful, selected_expected_receiver_col]
        df_passes.loc[~i_successful, "network_receiver_name"] = df_passes.loc[~i_successful, selected_expected_receiver_name_col]

        # st.write("Pernot")
        # st.write(df_passes[df_passes["network_receiver_name"] == "Pernot"])

        # defensive_network.utility.general.stop_streamlit_profiler()

        average_positions = None
        if use_tracking_average_position:
            average_positions = defensive_network.models.average_position.get_average_tracking_positions_off_def(df_tracking)["off"]

        total_minutes = 0
        for period, df_period in df_tracking.groupby("section"):
            period_minutes = (df_period["datetime_tracking"].max() - df_period["datetime_tracking"].min()).total_seconds() / 60
            total_minutes += period_minutes

        @st.cache_resource
        def _get_involvement(
            value_col, receiver_col, tracking_player_col, tracking_player_name_col, involvement_model_success_pos_value,
            involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted,
            model_radius
        ):
            df_involvement = defensive_network.models.involvement.get_involvement(
                df_passes, df_tracking, event_receiver_col=receiver_col,
                tracking_player_col=tracking_player_col, involvement_model_success_pos_value=involvement_model_success_pos_value,
                involvement_model_success_neg_value=involvement_model_success_neg_value,
                involvement_model_out=involvement_model_out, involvement_model_intercepted=involvement_model_intercepted,
                model_radius=model_radius, event_value_col=value_col, tracking_player_name_col=tracking_player_name_col,
            )
            return df_involvement

        df_involvement = _get_involvement(selected_value_col, selected_receiver_col, selected_tracking_player_col, selected_tracking_player_name_col, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius)

        defensive_network.utility.general.stop_streamlit_profiler()

        if len(plot_involvement_examples) > 0:
            for involvement_type, df_involvement_type in df_involvement.groupby("involvement_type"):
                if involvement_type not in plot_involvement_examples:
                    continue
                st.write(f"### {involvement_type}")
                with st.expander(f"### {involvement_type}"):
                    defensive_network.utility.pitch.plot_passes_with_involvement(
                        df_involvement_type, df_involvement_type["involvement_model"].iloc[0], model_radius, df_passes, df_tracking,
                        n_passes=n_examples_per_type
                    )

        st.write("---")

        for team, df_involvement_team in df_involvement.groupby("team_id_1"):
            team_name = df_involvement_team["team_name_1"].iloc[0]
            st.write(f"### {team_name}")
            for involvement_type_col in ["involvement", "contribution", "fault"]:
                st.write(f"## {involvement_type_col.capitalize()} networks {'(xT > 0)' if involvement_type_col == 'fault' else '(xT < 0)' if involvement_type_col == 'contribution' else ''}")
                with st.expander(involvement_type_col.capitalize()):
                    # df_involvement_team["network_receiver"] = df_involvement_team[selected_expected_receiver_col].where(df_involvement_team[selected_expected_receiver_col].notna(), df_involvement_team[selected_receiver_col])
                    # df_involvement_team["network_receiver_name"] = df_involvement_team[selected_expected_receiver_name_col].where(df_involvement_team[selected_expected_receiver_name_col].notna(), df_involvement_team[selected_receiver_name_col])

                    networks = defensive_network.models.passing_network.get_defensive_networks(
                        df_involvement_team, value_col=selected_value_col, involvement_type_col=involvement_type_col,
                        player_col=selected_player_col, player_name_col=selected_player_name_col,
                        receiver_col="network_receiver", receiver_name_col="network_receiver_name",
                        defender_id_col=defender_col, defender_name_col=defender_name_col,
                        total_minutes=total_minutes, average_positions=average_positions,
                        remove_passes_with_zero_involvement=remove_passes_with_zero_involvement,
                        x_to_col="x_target", y_to_col="y_target",
                    )
                    metrics = defensive_network.models.passing_network.analyse_defensive_networks(networks)

                    columns = st.columns(2)
                    fig = defensive_network.models.passing_network.plot_offensive_network(df_nodes=networks.off_network.df_nodes, df_edges=networks.off_network.df_edges)
                    columns[0].write("xT network")
                    columns[0].write(fig)
                    # columns[0].write(networks.off_network.df_nodes)
                    # columns[0].write(networks.off_network.df_edges)
                    columns[0].write(metrics.off_network[1])
                    fig = defensive_network.models.passing_network.plot_offensive_network(df_nodes=networks.off_network_only_positive.df_nodes, df_edges=networks.off_network_only_positive.df_edges)
                    columns[0].write("xT network only positive")
                    columns[0].write(fig)
                    # columns[0].write(networks.off_network_only_positive.df_nodes)
                    # columns[0].write(networks.off_network_only_positive.df_edges)
                    columns[0].write(metrics.off_network_only_positive[1])
                    fig = defensive_network.models.passing_network.plot_offensive_network(df_nodes=networks.off_involvement_type_network.df_nodes, df_edges=networks.off_involvement_type_network.df_edges)
                    columns[1].write(f"Defensive {involvement_type_col} network")
                    columns[1].write(fig)
                    columns[1].write(metrics.off_involvement_type_network[1])
                    # columns[1].write(networks.off_involvement_type_network.df_nodes)
                    # columns[1].write(networks.off_involvement_type_network.df_edges)

                    columns = st.columns(3)

                    for defender_nr, (_, network) in enumerate(networks.def_networks.items()):
                        # defender_name = df_involvement_team[df_involvement_team["defender_id"] == defender]["defender_name"].iloc[0]
                        # columns[defender_nr % 3].write(f"{network.defender_name}")

                        # fig = plot_defensive_network(df_nodes=networks.off_network.df_nodes, df_edges=network.df_edges)
                        fig = defensive_network.models.passing_network.plot_defensive_network(df_nodes=network.df_nodes, df_edges=network.df_edges, title=network.defender_name)
                        columns[defender_nr % 3].write(fig)

                        if show_def_full_metrics:
                            # columns[defender_nr % 3].write("df_metrics_def")
                            # columns[defender_nr % 3].write(metrics.def_networks[defender][1].to_dict())
                            st.write(metrics.def_networks[network.defender].df_nodes.set_index("name"))
                            st.write(metrics.def_networks[network.defender].df_edges.set_index(["from_name", "to_name"]))
                            try:
                                columns[defender_nr % 3].write(f'Weighted Density: {metrics.def_networks[network.defender][1].loc["Weighted Density"]}')
                                columns[defender_nr % 3].write(f'Total Degree: {metrics.def_networks[network.defender][1].loc["Total Degree"]}')
                            except KeyError as e:
                                columns[defender_nr % 3].write(e)
                            # columns[defender_nr % 3].write(metrics.def_networks[defender][1].to_dict())
                            # columns[defender_nr % 3].write("sum")
                            # columns[defender_nr % 3].write(metrics.def_network_sums.loc[defender].to_dict())
                    metrics.def_network_sums["defender_name"] = metrics.def_network_sums.index.map(selectedtrackingplayer2name)

                    st.write(metrics.def_network_sums.set_index("defender_name"))

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
