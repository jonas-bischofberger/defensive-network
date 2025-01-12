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

import defensive_network.utility
import defensive_network.models.involvement
import defensive_network.models
import defensive_network.models.passing_network
import defensive_network.parse.dfb

import defensive_network.scripts.create_dfb_tracking_animations

importlib.reload(defensive_network.scripts.create_dfb_tracking_animations)
importlib.reload(defensive_network.parse.dfb)

import wfork_streamlit_profiler

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


def get_network_metrics(matrix):
    if len(matrix) == 0:
        return pd.DataFrame()

    matrix = matrix.fillna(0)  # TODO: appropriate?

    def get_network(matrix):  # 创建一个有向加权图
        G = nx.DiGraph()
        # 将邻接矩阵转换为边列表，并添加到图中
        for player in matrix.index:
            # print(player)
            for recipient in matrix.columns:
                # print(recipient)
                weight = matrix.loc[player, recipient]
                if weight != 0:
                    G.add_edge(player, recipient, weight=weight)
        return G

    G = get_network(matrix)
    matrix_inverted = 1 / matrix  # 权重变化取倒数
    matrix_inverted = matrix_inverted.replace([np.inf, -np.inf], 0)
    G_inverted = get_network(matrix_inverted)

    try:
        reciprocity = nx.reciprocity(G)
    except networkx.exception.NetworkXError as e:
        st.warning(e)
        reciprocity = np.nan
    neighbor = nx.average_neighbor_degree(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    try:
        eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=10000)
    except (networkx.exception.PowerIterationFailedConvergence, networkx.exception.NetworkXPointlessConcept) as e:
        st.warning(e)
        eigenvector = np.nan
    try:
        closeness_centrality = nx.closeness_centrality(G_inverted, distance='weight')
    except ValueError as e:
        st.warning(e)
        closeness_centrality = np.nan
    betweenness_centrality = nx.betweenness_centrality(G_inverted, weight='weight')

    degree = nx.degree_centrality(G)#, weight="weight")  # TODO how did we get weighted degree centrality???
    indegree = nx.in_degree_centrality(G)#, weight="weight")
    outdegree = nx.out_degree_centrality(G)#, weight="weight")

    degree = {player: value for player, value in G.degree(weight="weight")}
    indegree = {player: value for player, value in G.in_degree(weight="weight")}
    outdegree = {player: value for player, value in G.out_degree(weight="weight")}

    df_metrics = pd.DataFrame({'Reciprocity': reciprocity,
                               'Neighbor': neighbor,
                               'Clustering': clustering,
                               'Eigenvector': eigenvector,
                               'Closeness': closeness_centrality,
                               'Betweenness': betweenness_centrality,
                               "Degree centrality": degree,
                               "In-degree centrality": indegree,
                               "Out-degree centrality": outdegree,
                               })
    return df_metrics


def analyse_network(network):
    df_nodes, df_edges = network.df_nodes, network.df_edges
    matrix = df_edges["value_passes"].unstack(level=1)
    return get_network_metrics(matrix)


def defensive_network_dashboard():
    profiler = wfork_streamlit_profiler.Profiler()
    profiler.start()

    base_path = st.text_input("Base path", "C:/Users/Jonas/Downloads/dfl_test_data/2324")

    # create_animation = st.toggle("Create animation", value=False)
    create_animation = False

    selected_tracking_matches = _select_matches(base_path)
    xt_model = st.selectbox("Select xT model", ["ma2024", "the_athletic"])
    expected_receiver_model = st.selectbox("Select expected receiver model", ["power2017"])

    for slugified_match_string in selected_tracking_matches:
        df_tracking, df_event = defensive_network.parse.dfb.get_match_data(base_path, slugified_match_string, xt_model=xt_model, expected_receiver_model=expected_receiver_model)
        df_passes = df_event[df_event["event_type"] == "pass"]
        player2name = df_tracking[["player_id", "player_name"]].set_index("player_id")["player_name"].to_dict()

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
        max_passes_for_debug = st.number_input("Max passes for debug", min_value=0, value=3)
        all_models = ["circle_circle_rectangle", "circle_passer", "circle_receiver", "intercepter"]
        involvement_model_success_pos_value = st.selectbox("Select involvement model for successful passes with positive value", all_models, index=0)
        involvement_model_success_neg_value = st.selectbox("Select involvement model for successful passes with negative value", all_models, index=1)
        involvement_model_out = st.selectbox("Select involvement model for passes that are interrupted by referee (e.g. goes out of the pitch)", all_models, index=1)
        involvement_model_intercepted = st.selectbox("Select involvement model for passes that are intercepted", all_models, index=3)
        model_radius = st.number_input("Circle model radius", min_value=0, value=5)

        @st.cache_resource
        def _get_involvement(value_col, slugified_match_string, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius):
            df_involvement = defensive_network.models.get_involvement(
                df_passes, df_tracking,
                involvement_model_success_pos_value=involvement_model_success_pos_value,
                involvement_model_success_neg_value=involvement_model_success_neg_value,
                involvement_model_out=involvement_model_out, involvement_model_intercepted=involvement_model_intercepted,
                model_radius=model_radius, value_col=value_col,
            )
            return df_involvement

        selected_value_col = st.selectbox("Value column for networks", ["pass_xt", None], format_func=lambda x: str(x))

        df_involvement = _get_involvement(selected_value_col, slugified_match_string, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius)

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
                    defensive_network.models.plot_passes_with_involvement(
                        df_involvement_type, df_involvement_type["involvement_model"].iloc[0], model_radius, df_passes, df_tracking,
                        n_passes=n_examples_per_type
                    )

        st.write("---")

        plot_offensive_network = functools.partial(
            defensive_network.models.plot_passing_network, show_colorbar=False, node_size_multiplier=20,
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

        def _analyse_network(network):
            df_metrics = analyse_network(network)
            df_metrics["player_name"] = df_metrics.index.map(player2name)
            return df_metrics.set_index("player_name")

        DefensiveNetworkMetrics = collections.namedtuple("DefensiveNetworkMetrics", ["off_network", "off_involvement_type_network", "def_networks", "def_network_sums"])
        def analyse_networks(networks: DefensiveNetworks):
            df_metrics_off = _analyse_network(networks.off_network)
            df_metrics_off_involvement_type = _analyse_network(networks.off_involvement_type_network)
            def_metrics = {}
            def_sums = {}
            for defender, def_network in networks.def_networks.items():
                df_metrics_def = _analyse_network(def_network)
                def_metrics[defender] = df_metrics_def
                def_sums[defender] = df_metrics_def.sum()

            df_def_sums = pd.DataFrame(def_sums).T
            df_def_sums["player_name"] = df_def_sums.index.map(player2name)
            df_def_sums = df_def_sums.set_index("player_name")

            return DefensiveNetworkMetrics(df_metrics_off, df_metrics_off_involvement_type, def_metrics, df_def_sums)

        show_def_full_metrics = st.toggle("Show individual offensive metrics for each defender", value=False)

        for team, df_involvement_team in df_involvement.groupby("team_id_1"):
            team_name = df_involvement_team["team_name_1"].iloc[0]
            st.write(f"### {team_name}")
            for invplvement_type_col in ["involvement", "contribution", "fault"]:
                st.write(f"## {invplvement_type_col.capitalize()} networks")
                with st.expander(invplvement_type_col.capitalize()):
                    networks = get_defensive_network(df_involvement_team, value_col=selected_value_col, involvement_type_col=invplvement_type_col)
                    metrics = analyse_networks(networks)

                    columns = st.columns(2)
                    fig, ax = plot_offensive_network(df_nodes=networks.off_network.df_nodes, df_edges=networks.off_network.df_edges)
                    columns[0].write("xT network")
                    columns[0].write(fig)
                    columns[0].write(metrics.off_network)
                    fig, ax = plot_offensive_network(df_nodes=networks.off_involvement_type_network.df_nodes, df_edges=networks.off_involvement_type_network.df_edges)
                    columns[1].write(f"Offensive {invplvement_type_col} network")
                    columns[1].write(fig)
                    columns[1].write(metrics.off_involvement_type_network)

                    columns = st.columns(3)

                    for defender_nr, (defender, network) in enumerate(networks.def_networks.items()):
                        defender_name = df_involvement_team[df_involvement_team["defender_id"] == defender]["defender_name"].iloc[0]
                        columns[defender_nr % 3].write(f"{defender_name}")

                        fig, ax = plot_defensive_network(df_nodes=networks.off_network.df_nodes, df_edges=network.df_edges)
                        columns[defender_nr % 3].write(fig)
                        if show_def_full_metrics:
                            columns[defender_nr % 3].write("df_metrics_def")
                            columns[defender_nr % 3].write(metrics.def_networks[defender])
                            columns[defender_nr % 3].write("sum")
                            columns[defender_nr % 3].write(metrics.def_network_sums.loc[defender_name])

                    st.write("df_sum")
                    st.write(metrics.def_network_sums)

                    # fig, ax = defensive_network.models.plot_passing_network(
        # df_nodes=df_nodes, df_edges=df_edges, show_colorbar=False, node_size_multiplier=20, arrow_width_multiplier=1,
        # label_col=edge_value_col, arrow_color_col=edge_value_col, annotate_top_n_edges=5, label_format_string="{:.3f}",
    # )


    # df_passes = pd.concat([df_pass_risky, df_pass_safe])
    # resp = outplayed(df_tracking, df_passes)

    profiler.stop()


DefensiveNetworks = collections.namedtuple("DefensiveNetworkResult", ["off_network", "off_involvement_type_network", "def_networks"])
Network = collections.namedtuple("AttackingNetwork", ["df_nodes", "df_edges"])


def get_defensive_network(df_passes_with_defenders, value_col="pass_xt", involvement_type_col="involvement") -> DefensiveNetworks:
    if value_col is None:
        df_passes_with_defenders["dummy"] = 1
        value_col = "dummy"

    df_passes_with_defenders["network_receiver"] = df_passes_with_defenders["expected_receiver"].where(df_passes_with_defenders["expected_receiver"].notna(), df_passes_with_defenders["player_id_2"])
    df_passes_with_defenders["network_receiver"] = df_passes_with_defenders["network_receiver"].where(df_passes_with_defenders["outcome"] != "intercepted", None)

    df_passes_with_defenders_dedup = df_passes_with_defenders.groupby("event_id").agg({
        "x_event": "first",
        "y_event": "first",
        "player_id_1": "first",
        "network_receiver": "first",
        "player_name_1": "first",
        "player_name_2": "first",
        involvement_type_col: "sum",
        "x_target": "first",
        "y_target": "first",
        value_col: "first",
    })
    df_passes_with_defenders_dedup = df_passes_with_defenders_dedup[df_passes_with_defenders_dedup["network_receiver"].notna()]

    df_nodes, df_edges = defensive_network.models.get_passing_network_df(
        df_passes_with_defenders_dedup.reset_index(),
        x_col="x_event", y_col="y_event", from_col="player_id_1", to_col="network_receiver", from_name_col="player_name_1", to_name_col="player_name_2",
        value_col=value_col, x_to_col="x_target", y_to_col="y_target", dedup_cols=["event_id"],
    )
    off_network = Network(df_nodes, df_edges)

    df_nodes, df_edges = defensive_network.models.get_passing_network_df(
        df_passes_with_defenders_dedup.reset_index(),
        x_col="x_event", y_col="y_event", from_col="player_id_1", to_col="network_receiver", from_name_col="player_name_1", to_name_col="player_name_2",
        value_col=involvement_type_col, x_to_col="x_target", y_to_col="y_target", dedup_cols=["event_id"],
    )
    off_involvement_type_network = Network(df_nodes, df_edges)

    edge_value_col = "value_passes"
    defensive_networks = {}
    for defender_nr, (defender, df_defender) in enumerate(df_passes_with_defenders.reset_index().groupby("defender_id")):
        df_nodes, df_edges = defensive_network.models.passing_network.get_passing_network_df(
            df_defender.reset_index(),
            x_col="x_event", y_col="y_event", from_col="player_id_1", to_col="network_receiver",
            from_name_col="player_name_1", to_name_col="player_name_2",
            value_col=involvement_type_col, x_to_col="x_target", y_to_col="y_target", dedup_cols=["event_id"],
        )
        df_edges["edge_label"] = df_edges[edge_value_col].apply(lambda x: x if x != 0 else None)
        defensive_networks[defender] = Network(df_nodes, df_edges)

    res = DefensiveNetworks(off_network, off_involvement_type_network, defensive_networks)
    return res


def demo_dashboard():
    from accessible_space.tests.resources import df_passes, df_tracking
    df_passes = df_passes.copy()
    df_tracking = df_tracking.copy()
    df_tracking.loc[df_tracking["player_id"] == "Y", "x"] -= 5
    # change location of player X in frame 6 to (10, 30)
    df_tracking.loc[(df_tracking["player_id"] == "X") & (df_tracking["frame_id"] == 6), "x"] = 27
    df_tracking.loc[(df_tracking["player_id"] == "X") & (df_tracking["frame_id"] == 6), "y"] = 30

    res_xt = defensive_network.models.get_expected_threat(df_passes, xt_model="ma2024", event_x_col="x", event_y_col="y", pass_end_x_col="x_target", pass_end_y_col="y_target", event_success_col="pass_outcome")
    df_passes["xt"] = res_xt.delta_xt

    res_xr = defensive_network.models.get_expected_receiver(df_passes, df_tracking, event_frame_col="frame_id", event_team_col="team_id", event_player_col="player_id", event_x_col="x", event_y_col="y", event_target_x_col="x_target", event_target_y_col="y_target", event_success_col="pass_outcome", tracking_frame_col="frame_id", tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x", tracking_y_col="y")
    df_passes["expected_receiver"] = res_xr.expected_receiver

    df_passes["is_intercepted"] = [False, False, True]
    df_passes["event_id"] = np.arange(len(df_passes))
    df_involvement = defensive_network.models.get_involvement(
        df_passes, df_tracking, event_success_col="pass_outcome", event_intercepted_col="is_intercepted", xt_col="xt",
        tracking_frame_col="frame_id", tracking_team_col="team_id", tracking_player_col="player_id",
        event_frame_col="frame_id", event_team_col="team_id", event_player_col="player_id", event_x_col="x", event_y_col="y", event_receiver_col="receiver_id", value_col="xt",
        event_target_frame_col="target_frame_id", event_id_col="event_id", event_target_x_col="x_target", event_target_y_col="y_target",
        tracking_x_col="x", tracking_y_col="y",
    )


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
