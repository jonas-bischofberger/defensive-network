import os.path
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st

import matplotlib.cm
import matplotlib.pyplot as plt
import defensive_network.utility.dataframes
import defensive_network.utility.general
import defensive_network.utility.pitch

import collections
import networkx as nx
import networkx.exception
import numpy as np
import pandas as pd
import functools

DefensiveNetworks = collections.namedtuple("DefensiveNetworks", ["off_network", "off_network_only_positive", "off_network_only_successful", "off_involvement_type_network", "def_networks"])
Network = collections.namedtuple("Network", ["df_nodes", "df_edges"])
DefensiveNetwork = collections.namedtuple("DefensiveNetwork", ["df_nodes", "df_edges", "defender", "defender_name"])
DefensiveNetworkMetrics = collections.namedtuple("DefensiveNetworkMetrics", ["off_network", "off_network_only_positive", "off_involvement_type_network", "def_networks", "def_network_sums", "def_network_team_sums", "def_network_team_means"])
Metrics = collections.namedtuple("Metrics", ["player", "team"])


def _get_average_positions(df_passes, x_col, y_col, from_col, x_to_col=None, y_to_col=None, to_col=None, entities=None):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"x": [-10] * 3 + [5, 5], "y": [0] * 3 + [10, -25], "x_target": [0] * 5, "y_target": [3] * 5, "from": ["A"] * 3 + ["A", "B"], "to": ["B"] * 3 + ["B", "C"], "xT": list(np.arange(5) / 28)})
    >>> df
        x   y  x_target  y_target from to        xT
    0 -10   0         0         3    A  B  0.000000
    1 -10   0         0         3    A  B  0.035714
    2 -10   0         0         3    A  B  0.071429
    3   5  10         0         3    A  B  0.107143
    4   5 -25         0         3    B  C  0.142857
    >>> _get_average_positions(df, "x", "y", "from", "x_target", "y_target", "to", ["A", "B", "C", "D"])
          x    y
    A -6.25  2.5
    B  1.00 -2.6
    C  0.00  3.0
    D   NaN  NaN
    """
    df_agg = df_passes.groupby(from_col).agg({x_col: "sum", y_col: "sum", from_col: "count"}).rename(columns={x_col: "x", y_col: "y", from_col: "count"})
    if x_to_col is not None:
        dfpos_to = df_passes.groupby(to_col).agg({x_to_col: "sum", y_to_col: "sum", to_col: "count"}).rename(columns={x_to_col: "x", y_to_col: "y", to_col: "count"})
        df_agg = dfpos_to.add(df_agg, fill_value=0)
        df_agg["count"] = df_agg["count"].astype(int)
    average_positions = df_agg[["x", "y"]].div(df_agg["count"], axis=0)  # avg positions are not weighted by value, but could be
    if len(average_positions) < len(entities):
        # If there are players who did not pass the ball, add them to the average positions
        average_positions = average_positions.reindex(entities).fillna(np.nan)
    # assert (average_positions["x"].dropna() < 52.5).all()
    # assert (average_positions["x"].dropna() > -52.5).all()
    return average_positions


def get_defensive_networks(
    df_passes_with_defenders, pass_id_col="event_id", value_col=None, involvement_type_col="involvement", player_col="player_id_1",
    player_name_col=None, receiver_col="network_receiver", receiver_name_col=None,
    x_col="x_event", y_col="y_event", x_to_col=None, y_to_col=None,
    defender_id_col="defender_id", defender_name_col="defender_name",
    average_positions=None, total_minutes=90, remove_passes_with_zero_involvement=True, entities=None, entity_to_name=None,
    success_col="pass_is_successful",
) -> DefensiveNetworks:
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"event_id": [0, 1, 1, 2, 3], "defender_id": ["DA", "DA", "DB", "DA", "DB"], "involvement": [0.2, 0.3, 0.1, 0.4, 0.5], "x": [-10] * 3 + [5, 5], "y": [0] * 3 + [10, -25], "player_id_1": ["A"] * 3 + ["A", "B"], "to": ["B"] * 3 + ["B", "C"], "pass_xt": list(np.arange(5) / 28), "pass_is_successful": [True] * 5})
    >>> df
       event_id defender_id  involvement   x   y player_id_1 to   pass_xt  pass_is_successful
    0         0          DA          0.2 -10   0           A  B  0.000000                True
    1         1          DA          0.3 -10   0           A  B  0.035714                True
    2         1          DB          0.1 -10   0           A  B  0.071429                True
    3         2          DA          0.4   5  10           A  B  0.107143                True
    4         3          DB          0.5   5 -25           B  C  0.142857                True
    >>> networks = get_defensive_networks(df, x_col="x", y_col="y", x_to_col=None, y_to_col=None, receiver_col="to", receiver_name_col="to", total_minutes=3)
    >>> networks.off_involvement_type_network.df_nodes

    >>> networks.off_involvement_type_network.df_edges

    """
    if value_col is None:
        df_passes_with_defenders["dummy"] = 1
        value_col = "dummy"

    if player_name_col is None:
        warnings.warn("TODO")
        player_name_col = player_col

    if receiver_name_col is None:
        warnings.warn("TODO")
        receiver_name_col = receiver_col

    aggregates = {
        x_col: "first",
        y_col: "first",
        player_col: "first",
        receiver_col: "first",
        player_name_col: "first",
        receiver_name_col: "first",
        involvement_type_col: "sum",
        x_to_col: "first",
        y_to_col: "first",
        value_col: "first",
        success_col: "first",
    }
    aggregates = {k: v for k, v in aggregates.items() if k in df_passes_with_defenders.columns}
    df_passes_defender_sums = df_passes_with_defenders.groupby(pass_id_col).agg(aggregates)
    # df_passes_defender_sums = df_passes_defender_sums[df_passes_defender_sums[receiver_col].notna()]  # TODO... ?

    if entities is None:
        entities = defensive_network.utility.general.uniquify_keep_order(df_passes_with_defenders[player_col].tolist() + df_passes_with_defenders[receiver_col].tolist())

    if defender_name_col not in df_passes_with_defenders.columns:
        defender_to_name = {defender_id: defender_id for defender_id in df_passes_with_defenders[defender_id_col].unique()}
    else:
        defender_to_name = df_passes_with_defenders[[defender_id_col, defender_name_col]].set_index(defender_id_col)[defender_name_col].to_dict()

    if entity_to_name is None:
        if player_name_col == player_col:
            entity_to_name = {entity: entity for entity in entities}
        else:
            entity_to_name = df_passes_with_defenders[[player_col, player_name_col]].set_index(player_col)[player_name_col].to_dict()
            entity_to_name.update(df_passes_with_defenders[[receiver_col, receiver_name_col]].set_index(receiver_col)[receiver_name_col].to_dict())

    df_nodes_off, df_edges_off = get_passing_network(
        df_passes_defender_sums.reset_index(),
        x_col=x_col, y_col=y_col, x_to_col=x_to_col, y_to_col=y_to_col, from_col=player_col, to_col=receiver_col,
        from_name_col=player_name_col, to_name_col=receiver_name_col, value_col=value_col,
        entity_to_average_position=average_positions, total_minutes=total_minutes, entities=entities, entity_to_name=entity_to_name,
        remove_passes_with_zero_value=False,  # for the normal xT network, we shouldn't ignore passes with 0 value (= 0 xT or 0 DAS Gained or 0 outplayed opponents, ...) because 0 can be a normal value!
    )
    off_network = Network(df_nodes_off, df_edges_off)

    df_passes_defender_sums_only_successful = df_passes_defender_sums.copy()
    i_success = df_passes_defender_sums_only_successful[success_col]
    # df_passes_defender_sums_only_successful = df_passes_defender_sums_only_successful[df_passes_defender_sums_only_successful[success_col]]
    df_passes_defender_sums_only_successful.loc[~i_success, value_col] = 0
    df_nodes_off_only_successful, df_edges_off_only_successful = get_passing_network(
        df_passes_defender_sums_only_successful.reset_index(),
        x_col=x_col, y_col=y_col, from_col=player_col, to_col=receiver_col, from_name_col=player_name_col,
        to_name_col=receiver_name_col, value_col=value_col, x_to_col=x_to_col, y_to_col=y_to_col,
        entity_to_average_position=average_positions, total_minutes=total_minutes, entities=entities, entity_to_name=entity_to_name,
        remove_passes_with_zero_value=False,  # for the normal xT network, we shouldn't ignore passes with 0 value (= 0 xT or 0 DAS Gained or 0 outplayed opponents, ...) because 0 can be a normal value!
    )
    off_network_only_successful = Network(df_nodes_off_only_successful, df_edges_off_only_successful)

    df_passes_defender_sums_only_positive = df_passes_defender_sums.copy()
    df_passes_defender_sums_only_positive[value_col] = df_passes_defender_sums_only_positive[value_col].clip(lower=0)
    df_nodes_only_positive, df_edges_only_positive = get_passing_network(
        df_passes_defender_sums_only_positive.reset_index(),
        x_col=x_col, y_col=y_col, from_col=player_col, to_col=receiver_col, from_name_col=player_name_col,
        to_name_col=receiver_name_col, value_col=value_col, x_to_col=x_to_col, y_to_col=y_to_col,
        entity_to_average_position=average_positions, total_minutes=total_minutes, entities=entities, entity_to_name=entity_to_name,
        remove_passes_with_zero_value=False,  # for the normal xT network, we shouldn't ignore passes with 0 value (= 0 xT or 0 DAS Gained or 0 outplayed opponents, ...) because 0 can be a normal value!
    )
    off_network_only_positive = Network(df_nodes_only_positive, df_edges_only_positive)

    df_nodes_off_inv, df_edges_off_inv = get_passing_network(
        df_passes_defender_sums.reset_index(),
        x_col=x_col, y_col=y_col, from_col=player_col, to_col=receiver_col, from_name_col=player_name_col,
        to_name_col=receiver_name_col, value_col=involvement_type_col, x_to_col=x_to_col, y_to_col=y_to_col,
        entity_to_average_position=average_positions, total_minutes=total_minutes,
        remove_passes_with_zero_value=remove_passes_with_zero_involvement, entities=entities, entity_to_name=entity_to_name,
    )
    off_involvement_type_network = Network(df_nodes_off_inv, df_edges_off_inv)

    edge_value_col = "value_passes"
    defensive_networks = {}
    for defender_nr, (defender, df_defender) in enumerate(df_passes_with_defenders.reset_index().groupby(defender_id_col)):
        df_nodes_def, df_edges_def = get_passing_network(
            df_defender.reset_index(),
            x_col=x_col, y_col=y_col, from_col=player_col, to_col=receiver_col,
            from_name_col=player_name_col, to_name_col=receiver_name_col,
            value_col=involvement_type_col, x_to_col=x_to_col, y_to_col=y_to_col,
            entity_to_average_position=average_positions, total_minutes=total_minutes,
            remove_passes_with_zero_value=remove_passes_with_zero_involvement, entities=entities, entity_to_name=entity_to_name,
        )
        df_edges_def["edge_label"] = df_edges_def[edge_value_col].apply(lambda x: x if x != 0 else None)
        defensive_networks[defender] = DefensiveNetwork(df_nodes_def, df_edges_def, defender, defender_to_name[defender])

    res = DefensiveNetworks(off_network, off_network_only_positive, off_network_only_successful, off_involvement_type_network, defensive_networks)
    return res


def get_passing_network(
    df_passes: pd.DataFrame,
    from_col: str,  # column with unique (!) ID or name of the entity (player/position/...) who passes the ball
    to_col: str,  # column with unique (!) ID or name of the entity (player/position/...) who receives the ball
    x_col: str = None,  # column with x position of the pass
    y_col: str = None,  # column with y position of the pass

    entities: list = None,  # list of all existing entities, if None will be inferred
    entity_to_average_position: dict = None,
    entity_to_name: dict = None,

    total_minutes: float = None,  # total minutes covered by the passed dataframe
    norm_minutes: int = 90,  # minutes to normalize to
    from_name_col: str = None,  # column with the name of the player/position/... who passes the ball, if None is given - from_col is used
    to_name_col: str = None,  # column with the name of the player/position/... who receives the ball, if None is given - to_col is used
    value_col: str = None,  # column with the value of the pass (e.g. xGCgain, xT, ...), if None is given - all passes have value = 1
    x_to_col: str = None,  # x position column of the receiving player (optional additional information for average positions)
    y_to_col: str = None,  # y position column of the receiving player (optional additional information for average positions)

    remove_passes_with_zero_value: bool = True
) -> Network:
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"x": [-10] * 3 + [5, 5], "y": [0] * 3 + [10, -25], "from": ["A"] * 3 + ["A", "B"], "to": ["B"] * 3 + ["B", "C"], "xT": list(np.arange(5) / 28)})
    >>> network = get_passing_network(df, "from", "to", "x", "y", value_col="xT", total_minutes=5, norm_minutes=90)
    >>> network.df_nodes
      entity name  x_avg      y_avg  num_passes  num_passes_per_time  num_receptions  num_receptions_per_time  num_passes_and_receptions  num_passes_and_receptions_per_time  value_passes  value_passes_per_time  value_receptions  value_receptions_per_time  value_passes_and_receptions  value_passes_and_receptions_per_time  value_per_pass  value_per_reception  value_per_pass_and_reception
    0      A    A   -5.0   3.333333         3.0                 54.0             0.0                      0.0                        3.0                                54.0      0.214286               3.857143          0.000000                   0.000000                     0.214286                              3.857143        0.071429             0.000000                      0.071429
    1      B    B    5.0 -25.000000         1.0                 18.0             3.0                     54.0                        4.0                                72.0      0.142857               2.571429          0.214286                   3.857143                     0.357143                              6.428571        0.142857             0.071429                      0.089286
    2      C    C    NaN        NaN         0.0                  0.0             1.0                     18.0                        1.0                                18.0      0.000000               0.000000          0.142857                   2.571429                     0.142857                              2.571429        0.000000             0.142857                      0.142857
    >>> network.df_edges
      from to from_name to_name  num_passes  num_passes_per_time  value_passes  value_passes_per_time  value_per_pass
    0    A  B         A       B           3                 54.0      0.214286               3.857143        0.071429
    1    B  C         B       C           1                 18.0      0.142857               2.571429        0.142857
    """
    if value_col is None:
        # If no value column is given, use a value of 1 for each pass
        value_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_passes, "value")
        df_passes[value_col] = 1

    if remove_passes_with_zero_value:
        df_passes = df_passes[df_passes[value_col] != 0]

    if from_name_col is None:
        from_name_col = from_col
    if to_name_col is None:
        to_name_col = to_col

    # Infer entity information if not provided
    if entities is None:
        entities = list(set(df_passes[from_col].dropna().unique()) | set(df_passes[to_col].dropna().unique()))
    if entity_to_name is None:
        entity_to_name = dict(zip(df_passes[from_col], df_passes[from_name_col]))
        entity_to_name.update(dict(zip(df_passes[to_col], df_passes[to_name_col])))

    # Compute average positions if not provided
    if entity_to_average_position is not None:
        average_positions = pd.DataFrame(entity_to_average_position).T.rename(columns={0: "x", 1: "y"})
    elif x_col is not None and y_col is not None:
        average_positions = _get_average_positions(df_passes, x_col, y_col, from_col, x_to_col, y_to_col, to_col, entities)
    else:
        raise ValueError("Either x_col and y_col or entity_to_average_position must be given to compute average positions")

    # Aggregate value and number of passes and receptions
    df_origins = df_passes.groupby(from_col, observed=False).agg({value_col: "sum", from_col: "count"}).rename(columns={value_col: "value_passes", from_col: "num_passes"}).reindex(entities).fillna(0)
    df_receptions = df_passes.groupby(to_col).agg({value_col: "sum", to_col: "count"}).rename(columns={value_col: "value_receptions", to_col: "num_receptions"}).reindex(entities).fillna(0)

    df_nodes = df_origins.merge(df_receptions, left_index=True, right_index=True)
    df_nodes["num_passes_and_receptions"] = df_nodes["num_passes"] + df_nodes["num_receptions"]
    df_nodes["value_passes_and_receptions"] = df_nodes["value_passes"] + df_nodes["value_receptions"]
    df_nodes["entity"] = df_nodes.index
    df_nodes["name"] = df_nodes.index.map(entity_to_name)
    df_nodes["x_avg"] = average_positions["x"]
    df_nodes["y_avg"] = average_positions["y"]

    df_nodes["value_per_pass"] = (df_nodes["value_passes"] / df_nodes["num_passes"]).fillna(0)
    df_nodes["value_per_reception"] = (df_nodes["value_receptions"] / df_nodes["num_receptions"]).fillna(0)
    df_nodes["value_per_pass_and_reception"] = (df_nodes["value_passes_and_receptions"] / df_nodes["num_passes_and_receptions"]).fillna(0)

    # assert (df_nodes["x_avg"].dropna() < 52.5).all()
    # assert (df_nodes["x_avg"].dropna() > -52.5).all()

    # Aggregate edges
    df_edges = df_passes.groupby([from_col, to_col]).agg({from_col: "count", value_col: "sum"}).rename(columns={from_col: "num_passes", value_col: "value_passes"})
    df_edges["from_name"] = df_edges.index.get_level_values(0).map(entity_to_name)
    df_edges["to_name"] = df_edges.index.get_level_values(1).map(entity_to_name)
    df_edges["value_per_pass"] = (df_edges["value_passes"] / df_edges["num_passes"]).fillna(0)
    # df_edges["median_pass_value"] = df_passes.groupby([from_col, to_col]).agg({value_col: "median"}).rename(columns={value_col: "median_pass_value"})
    # df_edges["median_sum"] = df_edges["median_pass_value"] * df_edges["num_passes"]

    # Sort
    df_edges = df_edges.sort_index()
    df_nodes = df_nodes.sort_index()
    df_edges["from"] = df_edges.index.get_level_values(0)
    df_edges["to"] = df_edges.index.get_level_values(1)

    # Weights per minutes
    if total_minutes is not None:
        for col in ["num_passes", "value_passes", "num_receptions", "value_receptions", "num_passes_and_receptions", "value_passes_and_receptions"]:
            df_nodes[col + "_per_time"] = df_nodes[col] / total_minutes * norm_minutes
        for col in ["num_passes", "value_passes"]:
            df_edges[col + "_per_time"] = df_edges[col] / total_minutes * norm_minutes

    node_cols = ["entity", "name", "x_avg", "y_avg", "num_passes","num_passes_per_time", "num_receptions", "num_receptions_per_time", "num_passes_and_receptions", "num_passes_and_receptions_per_time", "value_passes", "value_passes_per_time", "value_receptions", "value_receptions_per_time", "value_passes_and_receptions", "value_passes_and_receptions_per_time", "value_per_pass", "value_per_reception", "value_per_pass_and_reception"]
    edge_cols = ["from", "to", "from_name", "to_name", "num_passes", "num_passes_per_time", "value_passes", "value_passes_per_time", "value_per_pass"]

    df_nodes = df_nodes.reset_index(drop=True)[[col for col in node_cols if col in df_nodes.columns]]
    df_edges = df_edges.reset_index(drop=True)[[col for col in edge_cols if col in df_edges.columns]]

    return Network(df_nodes, df_edges)


def plot_passing_network(
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame = None,

    x_col: str = "x_avg",
    y_col: str = "y_avg",
    name_col: str = "name",
    node_size_col: str = "num_passes_per_time",  # Use None for fixed size
    node_color_col: str = "value_per_pass",  # Use None for fixed color
    other_node_color_col: str = None,  # "other_value",  # Use None for no other value
    node_size_multiplier: float = 3.0,
    node_min_size: float = 100,

    arrow_width_col: str = "num_passes_per_time",  # Use None for fixed width
    arrow_color_col: str = "value_per_pass",  # Use None for fixed color
    arrow_width_multiplier: float = 0.1,  # fixed width of the arrows

    label_col: str = None,  # column to use for labels, None for no labels
    label_format_string: str = "{:.2f}",  # format string for labels
    threshold_col: str = "num_passes",  # column to use for threshold
    threshold: float = 0.0,  # minimum value to show an edge
    alternative_threshold_col: str = None,  # column to use for alternative threshold
    alternative_threshold: float = 0.0,  # minimum value to show an edge

    fixed_node_color: str = "black",  # fixed color of the nodes
    fixed_arrow_width: float = 1,  # fixed width of the arrows
    fixed_arrow_color: str = "black",  # fixed color of the arrows
    annotate_top_n_edges: int = 5,  # annotate the top n edges

    colorbar_label: str = "",  # label for the colorbar (e.g. "Expected Threat per Pass")
    max_color_value_edges: float = None,  # maximum value for color scale
    min_color_value_edges: float = 0.0,  # minimum value for color scale
    max_color_value_nodes: float = None,  # maximum value for color scale
    min_color_value_nodes: float = 0.0,  # minimum value for color scale
    show_colorbar: bool = False,  # show colorbar

    colormap: str = None,

    ignore_nodes_without_position: bool = False,
    ignore_nodes_without_passes_or_receptions: bool = True,

    title: str = None,
    zoom: bool = False,  # zooms a little bit into the plot so that there is less empty space in the plot
):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"x": [-10] * 3 + [5, 5], "y": [0] * 3 + [10, -25], "from": ["A"] * 3 + ["A", "B"], "to": ["B"] * 3 + ["B", "C"], "xT": list(np.arange(5) / 28)})
    >>> df_nodes, df_edges = get_passing_network(df, "from", "to", "x", "y", value_col="xT", total_minutes=5, norm_minutes=90)
    >>> plot_passing_network(df_nodes=df_nodes, df_edges=df_edges, show_colorbar=False)
    <Figure size 933.333x600 with 1 Axes>
    >>> plt.show()
    """
    if colormap is None:
        colormap = matplotlib.cm.get_cmap("YlOrBr")

    fig, ax = defensive_network.utility.pitch.plot_football_pitch(color="black", figsize=(14 / 1.5, 9 / 1.5))

    # Color scaling
    if max_color_value_edges is None:
        if arrow_color_col is not None:
            max_color_value_edges = df_edges[arrow_color_col].max()  # Use the highest value in the data
        else:
            max_color_value_edges = 0.0  # If fixed: We don't need to scale the color anyway (?)
    if max_color_value_nodes is None:
        if node_color_col is not None:
            if other_node_color_col is None:
                max_color_value_nodes = df_nodes[node_color_col].max()
            else:

                max_color_value_nodes = max(df_nodes[[node_color_col, other_node_color_col]].sum(axis=1))
        else:
            max_color_value_nodes = 0.0

    if not ignore_nodes_without_position:
        df_nodes[x_col] = df_nodes[x_col].fillna(0)
        df_nodes[y_col] = df_nodes[y_col].fillna(0)
    if ignore_nodes_without_passes_or_receptions:
        i_zero = df_nodes["num_passes_and_receptions"] == 0
        df_nodes.loc[i_zero, x_col] = np.nan
        df_nodes.loc[i_zero, y_col] = np.nan

    normalize_edges = plt.Normalize(min_color_value_edges, max_color_value_edges)
    normalize_nodes = plt.Normalize(min_color_value_nodes, max_color_value_nodes)

    custom_width_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_edges, "width")
    custom_color_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_edges, "color")
    custom_size_col = defensive_network.utility.dataframes.get_new_unused_column_name(df_nodes, "size")

    # Set width and color for edges
    if arrow_width_col is not None:
        df_edges[custom_width_col] = df_edges[arrow_width_col] * arrow_width_multiplier
    else:
        df_edges[custom_width_col] = fixed_arrow_width
    if arrow_color_col is not None:
        df_edges[custom_color_col] = df_edges[arrow_color_col].apply(lambda x: colormap(normalize_edges(x)))
    else:
        df_edges[custom_color_col] = fixed_arrow_color

    # Set color and size for nodes
    if node_color_col is not None:
        if other_node_color_col is None:
            df_nodes[custom_color_col] = df_nodes[node_color_col].apply(lambda x: colormap(normalize_nodes(x)))
        else:
            df_nodes[custom_color_col] = df_nodes[[node_color_col, other_node_color_col]].sum(axis=1).apply(lambda x: colormap(normalize_nodes(x)))
    else:
        df_nodes[custom_color_col] = fixed_node_color
    if node_size_col is not None:
        df_nodes[custom_size_col] = df_nodes[node_size_col]
    else:
        df_nodes[custom_size_col] = 1

    df_nodes[custom_size_col] = np.maximum(1, df_nodes[custom_size_col]**1.5 * node_size_multiplier / 15)# - node_min_size)

    df_edges["sort_value"] = df_edges[arrow_color_col]#.abs()

    for edge_nr, (_, row) in enumerate(df_edges.sort_values(by="sort_value", ascending=True).iterrows()):
        if (alternative_threshold_col is not None and row[alternative_threshold_col] < alternative_threshold) and (threshold_col is not None and row[threshold_col] < threshold):
            continue
        i_from = df_nodes["entity"] == row["from"]
        i_to = df_nodes["entity"] == row["to"]
        x_avg = df_nodes.loc[i_from, x_col].iloc[0]
        y_avg = df_nodes.loc[i_from, y_col].iloc[0]
        x2_avg = df_nodes.loc[i_to, x_col].iloc[0]
        y2_avg = df_nodes.loc[i_to, y_col].iloc[0]

        if label_col is not None and edge_nr >= len(df_edges) - annotate_top_n_edges and row[label_col] is not None and not pd.isna(row[label_col]):
            formatted_label_str = label_format_string.format(row[label_col])
        else:
            formatted_label_str = None

        if row[arrow_color_col] >= 0:
            label_color = df_edges[df_edges[arrow_color_col] == df_edges[arrow_color_col].max()][custom_color_col].values[0]
        else:
            label_color = df_edges[df_edges[arrow_color_col] == df_edges[arrow_color_col].min()][custom_color_col].values[0]

        defensive_network.utility.pitch.plot_position_arrow(
            row["from"],
            row["to"],
            plot_players=False,
            label=formatted_label_str,
            label_color=label_color,
            # label=f"{row['possession_attack_xg']['mean']:.2f}",
            arrow_width=row[custom_width_col],
            arrow_color=row[custom_color_col],
            custom_xy=(x_avg, y_avg),
            custom_x2y=(x2_avg, y2_avg),
        )

    for node in df_nodes.index:
        # size = df_nodes.loc[node, "raw_size"] * node_size_multiplier
        defensive_network.utility.pitch.plot_position(
            node,
            color=df_nodes.loc[node, custom_color_col],
            size=df_nodes.loc[node, custom_size_col],
            custom_x=df_nodes.loc[node, x_col],
            custom_y=df_nodes.loc[node, y_col],
            label=df_nodes.loc[node, name_col],
            label_size=12,
        )

    if show_colorbar:
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize_edges, cmap=colormap), label=colorbar_label, cax=plt.gca())#, fraction=0.046, pad=0.04)

    if title is not None:
        plt.title(title)

    if zoom:
        x_padding = -5
        y_padding = 5
        i_ok = ~df_nodes["x_avg"].isin([-np.inf, np.nan, np.inf]) & ~df_nodes["y_avg"].isin([-np.inf, np.nan, np.inf])
        x_min = df_nodes.loc[i_ok, "x_avg"].min()
        x_max = df_nodes.loc[i_ok, "x_avg"].max()
        y_min = df_nodes.loc[i_ok, "y_avg"].min()
        y_max = df_nodes.loc[i_ok, "y_avg"].max()
        if len(df_nodes) > 0 and not pd.isna(x_min) and not pd.isna(x_max) and not pd.isna(y_min) and not pd.isna(y_max):
            plt.gca().set_xlim(x_min - x_padding, x_max + x_padding)
            plt.gca().set_ylim(y_min - y_padding, y_max + y_padding)

    return fig


def analyse_network(df_nodes, df_edges, silent=True):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"x": [-10] * 3 + [5, 5], "y": [0] * 3 + [10, -25], "from": ["A"] * 3 + ["A", "B"], "to": ["B"] * 3 + ["B", "C"], "xT": list(np.arange(5) / 28)})
    >>> df_nodes, df_edges = get_passing_network(df, "from", "to", "x", "y", value_col="xT", total_minutes=5, norm_minutes=90)
    >>> df_player_metrics, team_metrics = analyse_network(df_nodes, df_edges)
    >>> df_player_metrics
       Reciprocity  Average Neighbor Degree  Clustering coefficient  Eigenvector centrality  Closeness centrality  Betweenness centrality  Degree centrality  In-degree centrality  Out-degree centrality name
    A          0.0                      1.0                       0                0.000014              0.000000                     0.0           0.214286              0.000000               0.214286    A
    B          0.0                      0.0                       0                0.006443              0.107143                     0.5           0.357143              0.214286               0.142857    B
    C          0.0                      0.0                       0                0.999979              0.107143                     0.0           0.142857              0.142857               0.000000    C
    >>> team_metrics
    Unweighted Density    0.333333
    Weighted Density      0.059524
    Team reciprocity      0.000000
    Total Degree          0.714286
    dtype: float64
    """
    entity_to_name = dict(zip(df_nodes["entity"], df_nodes["name"]))
    matrix = df_edges.pivot(index='from', columns='to', values='value_passes')
    if len(matrix) == 0:
        return pd.DataFrame(), pd.Series()

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
        reciprocity = nx.reciprocity(G, G.nodes)
    except networkx.exception.NetworkXError as e:
        if not silent:
            st.warning(e)
        reciprocity = np.nan

    average_neighbor_degree = nx.average_neighbor_degree(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')
    unweighted_density = nx.density(G)
    try:
        eigenvector_centrality = None  # nx.eigenvector_centrality(G, weight='weight', max_iter=10000)  # too inefficient!
    except (networkx.exception.PowerIterationFailedConvergence, networkx.exception.NetworkXPointlessConcept) as e:
        if not silent:
            st.warning(e)
        eigenvector_centrality = np.nan
    try:
        closeness_centrality = nx.closeness_centrality(G_inverted, distance='weight')
    except ValueError as e:
        if not silent:
            st.warning(e)
        closeness_centrality = np.nan
    betweenness_centrality = nx.betweenness_centrality(G_inverted, weight='weight')

    degree = {player: value for player, value in G.degree(weight="weight")}
    indegree = {player: value for player, value in G.in_degree(weight="weight")}
    outdegree = {player: value for player, value in G.out_degree(weight="weight")}

    df_player_metrics = pd.DataFrame({
        'Reciprocity': reciprocity, 'Average Neighbor Degree': average_neighbor_degree,
        'Clustering coefficient': clustering, 'Eigenvector centrality': eigenvector_centrality,
        'Closeness centrality': closeness_centrality, 'Betweenness centrality': betweenness_centrality,
        "Degree centrality": degree, "In-degree centrality": indegree, "Out-degree centrality": outdegree,
    })
    df_player_metrics["name"] = df_player_metrics.index.map(entity_to_name)

    def _weighted_density(G):
        n_nodes = 11#len(G.nodes)  # TODO nodes in the def network should always be 11 (or at all)
        total_weight = sum([G.edges[edge]["weight"] for edge in G.edges])
        return total_weight / (n_nodes * (n_nodes - 1))

    def _unweighted_density(G):
        n_nodes = 11
        return len(G.edges) / (n_nodes * (n_nodes - 1))

    weighted_density = _weighted_density(G)

    team_reciprocity = nx.reciprocity(G)
    team_degree = sum([value for player, value in G.degree(weight="weight")])

    team_metrics = pd.Series({
        "Unweighted Density": _unweighted_density(G), "Weighted Density": weighted_density,
        "Team reciprocity": team_reciprocity, "Total Degree": team_degree,
    })

    return Metrics(player=df_player_metrics, team=team_metrics)


def analyse_defensive_networks(networks: DefensiveNetworks):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df = pd.DataFrame({"x": [-10] * 3 + [5, 5, 0], "y": [0] * 3 + [10, -25, 0], "from": ["A"] * 3 + ["A", "B", "B"], "to": ["B"] * 3 + ["B", "C", "A"], "xT": list(np.arange(6) / 28)})
    >>> networks = get_defensive_networks(df, value_col="xT", total_minutes=5)
    >>> analyse_defensive_networks(networks)

    """
    df_metrics_off = analyse_network(networks.off_network.df_nodes, networks.off_network.df_edges)
    df_metrics_off_only_positive = analyse_network(networks.off_network_only_positive.df_nodes, networks.off_network_only_positive.df_edges)
    df_metrics_off_involvement_type = analyse_network(networks.off_involvement_type_network.df_nodes, networks.off_involvement_type_network.df_edges)
    def_metrics = {}
    def_sums = {}
    for defender, def_network in networks.def_networks.items():
        df_metrics_player_def, team_metrics_def = analyse_network(def_network.df_nodes, def_network.df_edges)
        def_metrics[defender] = (df_metrics_player_def, team_metrics_def)
        # def_sums[defender] = df_metrics_player_def.sum()
        def_sums[defender] = team_metrics_def

    df_def_sums = pd.DataFrame(def_sums).T
    # df_def_sums["player_name"] = df_def_sums.index.map(selectedtrackingplayer2name)
    # df_def_sums = df_def_sums.set_index("player_name")

    df_def_team_sums = df_def_sums.sum()
    df_def_team_means = df_def_sums.mean()

    return DefensiveNetworkMetrics(df_metrics_off, df_metrics_off_only_positive, df_metrics_off_involvement_type, def_metrics, df_def_sums, df_def_team_sums, df_def_team_means)


def plot_defensive_networks(networks: DefensiveNetworks, max_value=5):
    fig = plot_passing_network(df_nodes=networks.off_network.df_nodes, df_edges=networks.off_network.df_edges, node_color_col="value_passes_and_receptions", arrow_color_col="value_passes")
    st.write(fig)
    st.write(networks.off_network.df_nodes)
    st.write(networks.off_network.df_edges)

    # max_nodes_involvement = max(networks.off_involvement_type_network.df_nodes["value_passes"])
    # max_edges_involvement = max(networks.off_involvement_type_network.df_edges["value_passes"])

    fig = plot_passing_network(df_nodes=networks.off_involvement_type_network.df_nodes, df_edges=networks.off_involvement_type_network.df_edges, max_color_value_nodes=max_value, max_color_value_edges=max_value, node_color_col="value_passes", arrow_color_col="value_passes")
    st.write(fig)
    st.write(networks.off_involvement_type_network.df_nodes)
    st.write(networks.off_involvement_type_network.df_edges)

    for defender, network in networks.def_networks.items():
        fig = plot_passing_network(df_nodes=network.df_nodes, df_edges=network.df_edges, max_color_value_nodes=max_value, max_color_value_edges=max_value, node_color_col="value_passes", arrow_color_col="value_passes")
        st.write("#### " + defender)
        st.write(fig)
        st.write(network.df_nodes)
        st.write(network.df_edges)

    metrics = analyse_defensive_networks(networks)

    st.write("metrics")
    st.write("metrics.off_network.team")
    st.write(metrics.off_network.team)
    st.write("metrics.off_network.player")
    st.write(metrics.off_network.player)
    st.write("metrics.off_involvement_type_network.team")
    st.write(metrics.off_involvement_type_network.team)
    st.write("metrics.off_involvement_type_network.player")
    st.write(metrics.off_involvement_type_network.player)
    st.write("metrics.def_network_sums")
    st.write(metrics.def_network_sums)


plot_offensive_network = functools.partial(
    plot_passing_network, show_colorbar=False, node_size_multiplier=30,#, node_size_multiplier=20,
    arrow_width_multiplier=1,#100,
    label_col="value_passes_per_time", arrow_color_col="value_passes_per_time", annotate_top_n_edges=5,
    arrow_width_col="num_passes_per_time", node_size_col="num_passes_per_time",
    label_format_string="{:.3f}", ignore_nodes_without_position=False, node_color_col="value_passes_per_time",
    # arrow_width_col = "value_passes_per_time",
    ignore_nodes_without_passes_or_receptions=True, max_color_value_nodes=0.1,
)
plot_defensive_network = functools.partial(plot_passing_network,    arrow_width_col="num_passes_per_time", node_size_col="num_passes_per_time",


    show_colorbar=False, node_size_multiplier=200, arrow_width_multiplier=3,
    # colormap=matplotlib.cm.get_cmap("PuBuGn"),
    colormap=matplotlib.cm.get_cmap("coolwarm"), min_color_value_edges=-0.05, max_color_value_edges=0.05,
    min_color_value_nodes=-0.5, max_color_value_nodes=0.2, annotate_top_n_edges=5, label_col="edge_label",
    label_format_string="{:.3f}", arrow_color_col="value_passes_per_time", ignore_nodes_without_position=False, node_color_col="value_passes_per_time",
    ignore_nodes_without_passes_or_receptions=True,
)


if __name__ == '__main__':
    defensive_network.utility.dataframes.prepare_doctest()
    df = pd.DataFrame({
        "event_id": [0, 1, 1, 2, 3],
        "defender_id": ["DA", "DA", "DB", "DA", "DB"],
        "defender_name": ["DA", "DA", "DB", "DA", "DB"],
        "involvement": [0.2, 0.3, 0.1, 0.4, 0.5],
        "x": [-10] * 3 + [5, 5],
        "y": [0] * 3 + [10, -25],
        "x_target": [-40] * 3 + [45, 50],
        "y_target": [0] * 5,
        "from": ["A"] * 3 + ["A", "B"],
        "to": ["B"] * 3 + ["B", "C"],
        "pass_xt": list(np.arange(5) / 28)
    })
    networks = get_defensive_networks(df, player_col="from", player_name_col="from", x_col="x", y_col="y", receiver_col="to", receiver_name_col="to", total_minutes=3, defender_name_col="defender_name", x_to_col="x_target", y_to_col="y_target")

    plot_defensive_networks(networks)
