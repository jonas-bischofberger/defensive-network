import os.path
import sys
import streamlit as st

import numpy as np
# import utility.log

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utility.pitch
import utility.general

# import utility.general
import pandas as pd

# import utility.vis.pitch
# import utility.dataframes
# import utility.impect
# import utility.dataframes

# import parse.from_s3


# def create_and_plot_passing_network(
#         df_passes: pd.DataFrame,  # DataFrame where each row is 1 pass
#         x_col: str,  # column with x position of the pass
#         y_col: str,  # column with y position of the pass
#         from_col: str,  # column with unique (!) ID or name of the player/position/... who passes the ball
#         to_col: str,  # column with unique (!) ID or name of the player/position/... who receives the ball
#         from_name_col: str = None,  # column with the name of the player/position/... who passes the ball
#         to_name_col: str = None,  # column with the name of the player/position/... who receives the ball
#         value_col: str = None,  # column with the value of the pass (e.g. xGCgain, xT, ...), if None is given - all passes have value = 1
#         x_to_col: str = None,  # x position column of the receiving player (additional information for average positions)
#         y_to_col: str = None,  # y position column of the receiving player (additional information for average positions)
#         colorbar_label: str = "",  # label for the colorbar (e.g. "Expected Threat per Pass")
#         min_value_to_plot: float = 0.0,  # minimum value per pass to plot a pass
#         show_labels: bool = False,  # show values next to arrows
#         node_size = "count",  # "count", "value", "value/count", "fixed", size of the nodes, "count" = determine by number of passes, "value" = determine by total value, "value/count" = determine by average value per pass, "fixed" = fixed size
#         node_color = "value/count",  # "count", "value", "value/count", "fixed", color of the nodes, see above
#         add_receptions_to_node_size_and_color: bool = True,
#         arrow_width = "count",  # "count", "value", "value/count", "fixed", width of the arrows, see above
#         arrow_color = "value/count",  # "count", "value", "value/count", "fixed", color of the arrows, see above
#         node_size_scale: float = 1,  # fixed size of the nodes
#         arrow_width_scale: float = 1,  # fixed width of the arrows
#         fixed_node_color: str = "black",  # fixed color of the nodes
#         fixed_arrow_width: float = 1,  # fixed width of the arrows
#         fixed_arrow_color: str = "black",  # fixed color of the arrows
#         max_color_value: float = None,  # maximum value for color scale
#         show_colorbar: bool = True,  # show colorbar
# ):
#     """
#     >>> utility.dataframes.set_unlimited_pandas_display_options()
#     >>> df = pd.DataFrame({
#     ...     "x": [-10] * 10 + [-15, 5, 5, 5],
#     ...     "y": [0] * 10 + [-1, 10, -15, -14],
#     ...     "from": ["A"] * 10 + ["A", "B", "B", "C"],
#     ...     "to": ["B"] * 10 + ["B", "A", "D", "A"],
#     ...     "xT": list(np.arange(14) / 28),
#     ... })
#     >>> df[["from_name", "to_name"]] = df[[ "from", "to"]].applymap(lambda x: f"Player {x}")
#     >>> df
#          x   y from to        xT from_name   to_name
#
#     >>> create_and_plot_passing_network(df, "x", "y", "from", "to", from_name_col="from_name", to_name_col="to_name", value_col="xT")
#
#     """
#     # df_nodes, df_edges = get_passing_network_df(
#     #     df_passes,
#     #     x_col,
#     #     y_col,
#     #     from_col,
#     #     to_col,
#     #     from_name_col=from_name_col,
#     #     to_name_col=to_name_col,
#     #     value_col=value_col,
#     #     x_to_col=x_to_col,
#     #     y_to_col=y_to_col,
#     # )
#     # fig = plot_passing_network(
#     #     df_nodes,
#     #     x_col=x_col,
#     #     y_col=y_col,
#     #     name_col="name",
#     #     df_edges=df_edges,
#     #
#     # )
#
#     if from_name_col is None:
#         from_name_col = from_col
#     if to_name_col is None:
#         to_name_col = to_col
#     if value_col is None:
#         # If no value column is given, use a value of 1 for each pass
#         value_col = utility.dataframes.get_new_unused_column_name(df_passes, "value")
#         df_passes[value_col] = 1
#
#     entity2name = dict(zip(df_passes[from_col], df_passes[from_name_col]))
#     entity2name.update(dict(zip(df_passes[to_col], df_passes[to_name_col])))
#
#     # Draw pitch
#     fig = plt.figure(figsize=(14/1.5, 9/1.5))
#     utility.vis.pitch.plot_football_pitch(color="grey", figsize=(14/1.5, 9/1.5))
#
#     entities = list(set(df_passes[from_col].unique()) | set(df_passes[to_col].unique()))
#
#     # Compute average positions
#     df_agg = df_passes.groupby(from_col).agg({x_col: "sum", y_col: "sum", to_col: "count"}).rename(columns={x_col: "x", y_col: "y", to_col: "count"})
#     if x_to_col is not None:
#         dfpos_to = df_passes.groupby(to_col).agg({x_to_col: "sum", y_to_col: "sum", to_col: "count"}).rename(columns={x_to_col: "x", y_to_col: "y", to_col: "count"})
#         df_agg = dfpos_to.add(df_agg, fill_value=0)
#         df_agg["count"] = df_agg["count"].astype(int)
#     average_positions = df_agg[["x", "y"]].div(df_agg["count"], axis=0)  # avg positions are not weighted by value, but could be
#     if len(average_positions) < len(entities):
#         # If there are players who did not pass the ball, add them to the average positions
#         average_positions = average_positions.reindex(entities).fillna(0)
#
#     # Aggregate value + number of passes and receptions
#     df_passes = df_passes.groupby(from_col).agg({value_col: "sum", from_col: "count"}).rename(columns={value_col: "value", from_col: "count"})
#     df_passes = df_passes.reindex(entities).fillna(0)
#     df_passes["value/count"] = (df_passes["value"] / df_passes["count"]).fillna(0)
#     df_receptions = df_passes.groupby(to_col).agg({value_col: "sum", to_col: "count"}).rename(columns={value_col: "value", to_col: "count"})
#     df_receptions = df_receptions.reindex(entities).fillna(0)
#     df_receptions["value/count"] = (df_receptions["value"] / df_receptions["count"]).fillna(0)
#     df_passes_and_receptions = df_passes.add(df_receptions, fill_value=0)
#
#     # Aggregate edges
#     df_edges = df_passes.groupby([from_col, to_col]).agg({from_col: "count", value_col: "sum"}).rename(columns={from_col: "count", value_col: "value"})
#     df_edges["value/count"] = (df_edges["value"] / df_edges["count"]).fillna(0)
#
#     # Color scaling
#     if max_color_value is None:
#         if arrow_color != "fixed":
#             max_color_value = df_edges[arrow_color].max()  # Use the highest value in the data
#         else:
#             max_color_value = 0.0  # If fixed: We don't need to scale the color anyway (?)
#     colormap = plt.cm.get_cmap('coolwarm')
#     normalize = plt.Normalize(0.00, max_color_value)
#
#     # Set width and color for edges
#     if arrow_width != "fixed":
#         df_edges["width"] = df_edges[arrow_width] * arrow_width_scale
#     else:
#         df_edges["width"] = fixed_arrow_width
#     if arrow_color != "fixed":
#         df_edges["color"] = df_edges[arrow_color].apply(lambda x: colormap(normalize(x)))
#     else:
#         df_edges["color"] = fixed_arrow_color
#
#     # Set color and size for nodes
#     if node_color != "fixed":
#         df_passes["color"] = df_passes[node_color].apply(lambda x: colormap(normalize(x)))
#         df_receptions["color"] = df_receptions[node_color].apply(lambda x: colormap(normalize(x)))
#         df_passes_and_receptions["color"] = df_passes_and_receptions[node_color].apply(lambda x: colormap(normalize(x)))
#     else:
#         df_passes["color"] = fixed_node_color
#         df_receptions["color"] = fixed_node_color
#         df_passes_and_receptions["color"] = fixed_node_color
#     if node_size != "fixed":
#         df_passes["raw_size"] = df_passes[node_size]
#         df_receptions["raw_size"] = df_receptions[node_size]
#         df_passes_and_receptions["raw_size"] = df_passes_and_receptions[node_size]
#     else:
#         df_passes["raw_size"] = 1
#         df_receptions["raw_size"] = 1
#         df_passes_and_receptions["raw_size"] = 1
#
#     df_node = df_passes if not add_receptions_to_node_size_and_color else df_passes_and_receptions
#     for entity in entities:
#         size = df_node.loc[entity, "raw_size"] * node_size_scale
#         color = df_node.loc[entity, "color"]
#         utility.vis.pitch.plot_position(
#             entity,
#             color=color,
#             size=size,
#             custom_x=average_positions.loc[entity, "x"],
#             custom_y=average_positions.loc[entity, "y"],
#             label=entity2name[entity],
#         )
#     for (from_entity, to_entity), row in df_edges.sort_values(by="count").iterrows():
#         if row["value"] < min_value_to_plot:
#             continue
#         x_avg = average_positions.loc[from_entity, "x"]
#         y_avg = average_positions.loc[from_entity, "y"]
#         x2_avg = average_positions.loc[to_entity, "x"]
#         y2_avg = average_positions.loc[to_entity, "y"]
#
#         utility.vis.pitch.plot_position_arrow(
#             from_entity,
#             to_entity,
#             plot_players=False,
#             label=f"{row[arrow_color]:.3f}" if show_labels else None,
#             # label=f"{row['possession_attack_xg']['mean']:.2f}",
#             arrow_width=row["width"],
#             arrow_color=row["color"],
#             custom_xy=(x_avg, y_avg),
#             custom_x2y=(x2_avg, y2_avg),
#         )
#     if show_colorbar:
#         plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap), label=colorbar_label)
#     plt.show()
#     return fig


def get_passing_network_df(
    df_passes: pd.DataFrame,
    x_col: str,  # column with x position of the pass
    y_col: str,  # column with y position of the pass
    from_col: str,  # column with unique (!) ID or name of the player/position/... who passes the ball
    to_col: str,  # column with unique (!) ID or name of the player/position/... who receives the ball
    net_minutes = None,
    from_name_col: str = None,  # column with the name of the player/position/... who passes the ball, if None is given - from_col is used
    to_name_col: str = None,  # column with the name of the player/position/... who receives the ball, if None is given - to_col is used
    value_col: str = None,  # column with the value of the pass (e.g. xGCgain, xT, ...), if None is given - all passes have value = 1
    x_to_col: str = None,  # x position column of the receiving player (optional additional information for average positions)
    y_to_col: str = None,  # y position column of the receiving player (optional additional information for average positions)
    additional_node_values: dict = None,  # additional value for nodes (e.g. pxT through dribbling, shooting etc.)
    dedup_cols = None,  # column to de-duplicate
):
    """
    >>> utility.dataframes.set_unlimited_pandas_display_options()
    >>> df = pd.DataFrame({
    ...     "x": [-10] * 10 + [-15, 5, 5, 5],
    ...     "y": [0] * 10 + [-1, 10, -15, -14],
    ...     "from": ["A"] * 10 + ["A", "B", "B", "C"],
    ...     "to": ["B"] * 10 + ["B", "A", "D", "A"],
    ...     "xT": list(np.arange(14) / 28),
    ... })
    >>> df[["from_name", "to_name"]] = df[[ "from", "to"]].applymap(lambda x: f"Player {x}")
    >>> df
         x   y from to        xT from_name   to_name
    0  -10   0    A  B  0.000000  Player A  Player B
    1  -10   0    A  B  0.035714  Player A  Player B
    2  -10   0    A  B  0.071429  Player A  Player B
    3  -10   0    A  B  0.107143  Player A  Player B
    4  -10   0    A  B  0.142857  Player A  Player B
    5  -10   0    A  B  0.178571  Player A  Player B
    6  -10   0    A  B  0.214286  Player A  Player B
    7  -10   0    A  B  0.250000  Player A  Player B
    8  -10   0    A  B  0.285714  Player A  Player B
    9  -10   0    A  B  0.321429  Player A  Player B
    10 -15  -1    A  B  0.357143  Player A  Player B
    11   5  10    B  A  0.392857  Player B  Player A
    12   5 -15    B  D  0.428571  Player B  Player D
    13   5 -14    C  A  0.464286  Player C  Player A
    >>> df_nodes, df_edges = get_passing_network_df(df, "x", "y", "from", "to", from_name_col="from_name", to_name_col="to_name", value_col="xT")
    >>> df_nodes
           name      x_avg      y_avg  num_passes  value_passes  num_receptions  value_receptions  num_passes_and_receptions  value_passes_and_receptions  value_per_pass  value_per_reception  value_per_pass_and_reception  other_value
    A  Player A -10.454545  -0.090909        11.0      1.964286             2.0          0.857143                       13.0                     2.821429        0.178571             0.428571                      0.217033            0
    B  Player B   5.000000  -2.500000         2.0      0.821429            11.0          1.964286                       13.0                     2.785714        0.410714             0.178571                      0.214286            0
    C  Player C   5.000000 -14.000000         1.0      0.464286             0.0          0.000000                        1.0                     0.464286        0.464286             0.000000                      0.464286            0
    D  Player D   0.000000   0.000000         0.0      0.000000             1.0          0.428571                        1.0                     0.428571        0.000000             0.428571                      0.428571            0
    >>> df_edges
             num_passes  value_passes  value_per_pass  median_pass_value  median_sum from_name   to_name
    from to
    A    B           11      1.964286        0.178571           0.178571    1.964286  Player A  Player B
    B    A            1      0.392857        0.392857           0.392857    0.392857  Player B  Player A
         D            1      0.428571        0.428571           0.428571    0.428571  Player B  Player D
    C    A            1      0.464286        0.464286           0.464286    0.464286  Player C  Player A
    """
    if value_col is None:
        # If no value column is given, use a value of 1 for each pass
        value_col = utility.general.get_new_unused_column_name(df_passes, "value")
        df_passes[value_col] = 1

    if from_name_col is None:
        from_name_col = from_col
    if to_name_col is None:
        to_name_col = to_col

    entity2name = dict(zip(df_passes[from_col], df_passes[from_name_col]))
    entity2name.update(dict(zip(df_passes[to_col], df_passes[to_name_col])))

    # Draw pitch
    entities = list(set(df_passes[from_col].unique()) | set(df_passes[to_col].unique()))

    # Compute average positions
    df_agg = df_passes.groupby(from_col).agg({x_col: "sum", y_col: "sum", to_col: "count"}).rename(columns={x_col: "x", y_col: "y", to_col: "count"})
    if x_to_col is not None:
        dfpos_to = df_passes.groupby(to_col).agg({x_to_col: "sum", y_to_col: "sum", to_col: "count"}).rename(columns={x_to_col: "x", y_to_col: "y", to_col: "count"})
        df_agg = dfpos_to.add(df_agg, fill_value=0)
        df_agg["count"] = df_agg["count"].astype(int)
    average_positions = df_agg[["x", "y"]].div(df_agg["count"], axis=0)  # avg positions are not weighted by value, but could be
    if len(average_positions) < len(entities):
        # If there are players who did not pass the ball, add them to the average positions
        average_positions = average_positions.reindex(entities).fillna(0)

    # Aggregate value + number of passes and receptions
    df_passes_to_group_origin = df_passes if dedup_cols is None else df_passes.drop_duplicates(subset=dedup_cols + [from_col])
    df_origin = df_passes_to_group_origin.groupby(from_col).agg({value_col: "sum", from_col: "count"}).rename(columns={value_col: "value", from_col: "count"})
    df_origin = df_origin.reindex(entities).fillna(0)
    # df_origin["value/count"] = (df_origin["value"] / df_origin["count"]).fillna(0)
    df_rec_to_group_origin = df_passes if dedup_cols is None else df_passes.drop_duplicates(subset=dedup_cols + [to_col])
    df_receptions = df_rec_to_group_origin.groupby(to_col).agg({value_col: "sum", to_col: "count"}).rename(columns={value_col: "value", to_col: "count"})
    df_receptions = df_receptions.reindex(entities).fillna(0)
    # df_receptions["value/count"] = (df_receptions["value"] / df_receptions["count"]).fillna(0)
#    df_origin_and_receptions = df_origin.add(df_receptions, fill_value=0).rename(columns={"value": "value_pass_and_reception", "count": "", "value/count": "value/count_origin"})

    df_origin = df_origin.rename(columns={"value": "value_passes", "count": "num_passes"})
    df_receptions = df_receptions.rename(columns={"value": "value_receptions", "count": "num_receptions"})
    df_nodes = df_origin.merge(df_receptions, left_index=True, right_index=True)
    df_nodes["num_passes_and_receptions"] = df_nodes["num_passes"] + df_nodes["num_receptions"]
    df_nodes["value_passes_and_receptions"] = df_nodes["value_passes"] + df_nodes["value_receptions"]

    df_nodes["name"] = df_nodes.index.map(entity2name)
    df_nodes["x_avg"] = average_positions["x"]
    df_nodes["y_avg"] = average_positions["y"]

    df_nodes["value_per_pass"] = (df_nodes["value_passes"] / df_nodes["num_passes"]).fillna(0)
    df_nodes["value_per_reception"] = (df_nodes["value_receptions"] / df_nodes["num_receptions"]).fillna(0)
    df_nodes["value_per_pass_and_reception"] = (df_nodes["value_passes_and_receptions"] / df_nodes["num_passes_and_receptions"]).fillna(0)

    # Reorder
    df_nodes = df_nodes[[
        "name", "x_avg", "y_avg", "num_passes", "value_passes", "num_receptions", "value_receptions",
        "num_passes_and_receptions", "value_passes_and_receptions", "value_per_pass", "value_per_reception",
        "value_per_pass_and_reception"
    ]]
    df_nodes["other_value"] = 0
    if additional_node_values is not None:
        for k, v in additional_node_values.items():
            df_nodes.loc[k, "other_value"] = v

    # Aggregate edges
    df_passes_to_group_edges = df_passes if dedup_cols is None else df_passes.drop_duplicates(subset=dedup_cols + [from_col, to_col])
    df_edges = df_passes_to_group_edges.groupby([from_col, to_col]).agg({from_col: "count", value_col: "sum"}).rename(columns={from_col: "num_passes", value_col: "value_passes"})
    df_edges["value_per_pass"] = (df_edges["value_passes"] / df_edges["num_passes"]).fillna(0)
    df_edges["median_pass_value"] = df_passes_to_group_edges.groupby([from_col, to_col]).agg({value_col: "median"}).rename(columns={value_col: "median_pass_value"})
    df_edges["median_sum"] = df_edges["median_pass_value"] * df_edges["num_passes"]

    # sort nodes by from_col
    df_edges = df_edges.sort_index()
    df_nodes.index = df_nodes.index.map(str)
    df_nodes = df_nodes.sort_index()

    df_edges["from_name"] = df_edges.index.get_level_values(0).map(entity2name)
    df_edges["to_name"] = df_edges.index.get_level_values(1).map(entity2name)

    if net_minutes is not None:
        df_edges["num_passes_per90"] = df_edges["num_passes"] / net_minutes * 90
        df_edges["value_passes_per90"] = df_edges["value_passes"] / net_minutes * 90
        df_nodes["num_passes_per90"] = df_nodes["num_passes"] / net_minutes * 90
        df_nodes["value_passes_per90"] = df_nodes["value_passes"] / net_minutes * 90
        df_nodes["num_receptions_per90"] = df_nodes["num_receptions"] / net_minutes * 90
        df_nodes["value_receptions_per90"] = df_nodes["value_receptions"] / net_minutes * 90
        df_nodes["num_passes_and_receptions_per90"] = df_nodes["num_passes_and_receptions"] / net_minutes * 90
        df_nodes["value_passes_and_receptions_per90"] = df_nodes["value_passes_and_receptions"] / net_minutes * 90

    return df_nodes, df_edges


def plot_passing_network(
    df_nodes: pd.DataFrame,
    x_col: str = "x_avg",
    y_col: str = "y_avg",
    name_col: str = "name",
    node_size_col: str = "num_passes",  # Use None for fixed size
    node_color_col: str = "value_per_pass",  # Use None for fixed color
    other_node_color_col: str = "other_value",  # Use None for no other value
    node_size_multiplier: float = 1.0,
    node_min_size: float = 100,

    df_edges: pd.DataFrame = None,
    arrow_width_col: str = "num_passes",  # Use None for fixed width
    arrow_color_col: str = "value_per_pass",  # Use None for fixed color
    arrow_width_multiplier: float = 1,  # fixed width of the arrows

    label_col: str = None,  # column to use for labels, None for no labels
    threshold_col: str = "num_passes",  # column to use for threshold
    threshold: float = 0.0,  # minimum value to show an edge
    alternative_threshold_col: str = None,  # column to use for alternative threshold
    alternative_threshold: float = 0.0,  # minimum value to show an edge

    fixed_node_color: str = "black",  # fixed color of the nodes
    fixed_arrow_width: float = 1,  # fixed width of the arrows
    fixed_arrow_color: str = "black",  # fixed color of the arrows

    colorbar_label: str = "",  # label for the colorbar (e.g. "Expected Threat per Pass")
    max_color_value_edges: float = None,  # maximum value for color scale
    min_color_value_edges: float = 0.0,  # minimum value for color scale
    max_color_value_nodes: float = None,  # maximum value for color scale
    min_color_value_nodes: float = 0.0,  # minimum value for color scale
    show_colorbar: bool = True,  # show colorbar

    colormap: str = None,
):
    """
    >>> utility.dataframes.set_unlimited_pandas_display_options()
    >>> df = pd.DataFrame({
    ...     "x": [-50] * 10 + [-30, 25, 25, 25],
    ...     "y": [0] * 10 + [-10, 20, -25, -24],
    ...     "from": ["A"] * 10 + ["A", "B", "B", "C"],
    ...     "to": ["B"] * 10 + ["B", "A", "D", "A"],
    ...     "xT": list(np.arange(14) / 28),
    ... })
    >>> df[["from_name", "to_name"]] = df[[ "from", "to"]].applymap(lambda x: f"Player {x}")
    >>> df_nodes, df_edges = get_passing_network_df(df, "x", "y", "from", "to", from_name_col="from_name", to_name_col="to_name", value_col="xT")
    >>> df_nodes
           name      x_avg      y_avg  num_passes  value_passes  num_receptions  value_receptions  num_passes_and_receptions  value_passes_and_receptions  value_per_pass  value_per_reception  value_per_pass_and_reception  other_value
    A  Player A -10.454545  -0.090909        11.0      1.964286             2.0          0.857143                       13.0                     2.821429        0.178571             0.428571                      0.217033            0
    B  Player B   5.000000  -2.500000         2.0      0.821429            11.0          1.964286                       13.0                     2.785714        0.410714             0.178571                      0.214286            0
    C  Player C   5.000000 -14.000000         1.0      0.464286             0.0          0.000000                        1.0                     0.464286        0.464286             0.000000                      0.464286            0
    D  Player D   0.000000   0.000000         0.0      0.000000             1.0          0.428571                        1.0                     0.428571        0.000000             0.428571                      0.428571            0
    >>> df_edges
             num_passes  value_passes  value_per_pass  median_pass_value  median_sum from_name   to_name
    from to
    A    B           11      1.964286        0.178571           0.178571    1.964286  Player A  Player B
    B    A            1      0.392857        0.392857           0.392857    0.392857  Player B  Player A
         D            1      0.428571        0.428571           0.428571    0.428571  Player B  Player D
    C    A            1      0.464286        0.464286           0.464286    0.464286  Player C  Player A
    >>> plot_passing_network(df_nodes=df_nodes, df_edges=df_edges)
    (<Figure size 933.333x600 with 2 Axes>, <Axes: >)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm

    if colormap is None:
        colormap = matplotlib.cm.get_cmap("YlOrBr")

    fig, ax = utility.pitch.plot_football_pitch(color="black", figsize=(14 / 1.5, 9 / 1.5))

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

    normalize_edges = plt.Normalize(min_color_value_edges, max_color_value_edges)
    normalize_nodes = plt.Normalize(min_color_value_nodes, max_color_value_nodes)

    custom_width_col = utility.general.get_new_unused_column_name(df_edges, "width")
    custom_color_col = utility.general.get_new_unused_column_name(df_edges, "color")
    custom_size_col = utility.general.get_new_unused_column_name(df_nodes, "size")

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
        # df_receptions[custom_color_col] = df_receptions[node_color].apply(lambda x: colormap(normalize(x)))
        # df_passes_and_receptions[custom_color_col] = df_passes_and_receptions[node_color].apply(lambda x: colormap(normalize(x)))
    else:
        df_nodes[custom_color_col] = fixed_node_color
        # df_passes[custom_color_col] = fixed_node_color
        # df_receptions[custom_color_col] = fixed_node_color
        # df_passes_and_receptions[custom_color_col] = fixed_node_color
    if node_size_col is not None:
        df_nodes[custom_size_col] = df_nodes[node_size_col]
        # df_passes["raw_size"] = df_passes[node_size]
        # df_receptions["raw_size"] = df_receptions[node_size]
        # df_passes_and_receptions["raw_size"] = df_passes_and_receptions[node_size]
    else:
        df_nodes[custom_size_col] = 1
        # df_passes["raw_size"] = 1
        # df_receptions["raw_size"] = 1
        # df_passes_and_receptions["raw_size"] = 1

    df_nodes[custom_size_col] = np.maximum(1, df_nodes[custom_size_col]**1.5 * node_size_multiplier / 15)# - node_min_size)

    # df_node = df_passes if not add_receptions_to_node_size_and_color else df_passes_and_receptions
    # for node in df_nodes.index:
    #     # size = df_nodes.loc[node, "raw_size"] * node_size_multiplier
    #     # color = df_nodes.loc[node, custom_color_col]
    #     utility.vis.pitch.plot_position(
    #         node,
    #         color=df_nodes.loc[node, custom_color_col],
    #         size=df_nodes.loc[node, custom_size_col],
    #         custom_x=df_nodes.loc[node, x_col],
    #         custom_y=df_nodes.loc[node, y_col],
    #         label=df_nodes.loc[node, name_col],
    #     )

    for (from_entity, to_entity), row in df_edges.sort_values(by=arrow_color_col, ascending=True).iterrows():
        # if row[threshold_col] < threshold:
        #     continue
        if (alternative_threshold_col is not None and row[alternative_threshold_col] < alternative_threshold) and (threshold_col is not None and row[threshold_col] < threshold):
            continue
        x_avg = df_nodes.loc[str(from_entity), x_col]
        y_avg = df_nodes.loc[str(from_entity), y_col]
        x2_avg = df_nodes.loc[str(to_entity), x_col]
        y2_avg = df_nodes.loc[str(to_entity), y_col]

        utility.pitch.plot_position_arrow(
            from_entity,
            to_entity,
            plot_players=False,
            label=f"{row[label_col]:.3f}" if label_col is not None else None,
            # label=f"{row['possession_attack_xg']['mean']:.2f}",
            arrow_width=row[custom_width_col],
            arrow_color=row[custom_color_col],
            custom_xy=(x_avg, y_avg),
            custom_x2y=(x2_avg, y2_avg),
        )

    for node in df_nodes.index:
        # size = df_nodes.loc[node, "raw_size"] * node_size_multiplier
        # color = df_nodes.loc[node, custom_color_col]
        utility.pitch.plot_position(
            node,
            color=df_nodes.loc[node, custom_color_col],
            size=df_nodes.loc[node, custom_size_col],
            custom_x=df_nodes.loc[node, x_col],
            custom_y=df_nodes.loc[node, y_col],
            label=df_nodes.loc[node, name_col],
            label_size=12,
        )

    if show_colorbar:
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalize_edges, cmap=colormap), label=colorbar_label, fraction=0.046, pad=0.04)

    return fig, ax


def plottt(df_nodes, df_edges, title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plot_passing_network(
        df_nodes,
        df_edges=df_edges,
        node_size_col="num_passes_and_receptions_per90",
        node_color_col="value_passes_and_receptions_per90",
        # node_size_multiplier=5.0,
        node_size_multiplier=3.0,
        max_color_value_nodes=1.25,

        arrow_width_col="num_passes_per90",
        arrow_color_col="value_passes_per90",
        # arrow_width_multiplier=0.3,
        arrow_width_multiplier=0.5,
        max_color_value_edges=0.25,

        colorbar_label="pxT Pass (per 90 minutes)",
    )

    # formation_str = formation if "undefined" not in formation.lower() else "Undefined Formation"
    # plt.title(f"{squad_name} - {player_name} - {formation_str} - {formation2mins[formation]:.0f} minutes")
    plt.title(title)

    # xlim to max
    x_padding = -5
    y_padding = 5
    i_ok = ~df_nodes["x_avg"].isin([-np.inf, np.nan, np.inf])

    x_min = df_nodes.loc[i_ok, "x_avg"].min()
    x_max = df_nodes.loc[i_ok, "x_avg"].max()
    y_min = df_nodes.loc[i_ok, "y_avg"].min()
    y_max = df_nodes.loc[i_ok, "y_avg"].max()
    if len(df_nodes) > 0:
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    # xlim to max
    # x_padding = -5
    # y_padding = 5
    # ax.set_xlim(df_nodes["x_avg"].min() - x_padding, df_nodes["x_avg"].max() + x_padding)
    # ax.set_ylim(df_nodes["y_avg"].min() - y_padding, df_nodes["y_avg"].max() + y_padding)

    # plt.savefig(f"C:/misc/" + f"{utility.general.slugify(title)}"[:200] + ".png")
    # print(f"Wrote to C:/misc/" + f"{utility.general.slugify(title)}"[:200] + ".png")
    # plt.show()

    return fig


# def pn_by_events(df_relevant_actions, formation2mins, formation2grossmins, remove_backpasses_for_pxt=True, value_metric="pxt", squad_name=None, player_name=None, min_player_share=0.8):
#     import matplotlib
#     import matplotlib.pyplot as plt
#
#     df_formation = pd.read_excel(os.path.join(os.path.dirname(__file__), "../assets/2024-03-15_impect_formations/formations_ids.xlsx"))
#
#     if remove_backpasses_for_pxt:
#         df_relevant_actions["PXT_PASS_progressive"] = df_relevant_actions["PXT_PASS"].clip(0, None)
#     df_relevant_actions["PXT_DRIBBLE_progressive"] = df_relevant_actions["PXT_DRIBBLE"].clip(0, None)
#     # df_relevant_actions["position_with_side"] = df_relevant_actions["player.position"] + "_" + df_relevant_actions["player.position_side"]
#
#     # Get receiving position
#     # df_matchinfos = utility.impect.get_matchinfos(df_relevant_actions["match_id"].iloc[0])
#
#     df_relevant_actions_passes = df_relevant_actions[df_relevant_actions["actionType"] != "DRIBBLE"]
#
#     # dfg = df_relevant_actions[["live_formation_string", "duration"]].groupby("live_formation_string").sum() / 60
#     # formation2mins = dfg.to_dict()["duration"]
#     # for key in dfg_mapping:
#     #     if key not in formation2mins:
#     #         formation2mins[key] = 0
#     #     formation2mins[key] += dfg_mapping[key]
#
#     figs = []
#     minutes = []
#     dfs_nodes = []
#     dfs_edges = []
#
#     if player_name is None:
#         # dfg = df_relevant_actions.groupby("live_formation_string")
#         dfg = df_relevant_actions.groupby("live_formation_id")
#     else:
#         dfg = df_relevant_actions.groupby("live_formation_id_and_player_pos")
#
#     for formation, df_events_in_formation in dfg:
#
#         # df_match_formations = df_relevant_actions[df_relevant_actions["live_formation_string"] == formation]
#         # if "13112020001" not in formation:
#         #     continue
#
#         # assert len(df_events_in_formation["live_formation_id"].unique()) == 1
#
#         for formation_players, df_events_in_formation_with_besetzung in df_events_in_formation.groupby("live_formation_players"):
#             besetzung = json.loads(formation_players)
#             player2position_with_side = {int(k): f"{v[0]}_{v[1]}" for k, v in besetzung.items()}
#             df_events_in_formation.loc[df_events_in_formation["live_formation_players"] == formation_players, "pass_receiver_position_with_side"] = df_events_in_formation["pass.receiver.playerId"].replace(player2position_with_side)
#
#         # formation_players = json.loads(df_events_in_formation["live_formation_players"].iloc[0])
#         # formation_players = {int(k): f"{v[0]}_{v[1]}" for k, v in formation_players.items()}
#         # player2position_with_side = formation_players
#         #
#         # player2position_with_side = df_events_in_formation[["player.id", "position_with_side"]].drop_duplicates().set_index("player.id")["position_with_side"].to_dict()
#         # player2position_with_side = {int(k): v for k, v in player2position_with_side.items()}
#
#         # df_events_in_formation["position_with_side"] = df_events_in_formation["player.id"].replace(player2position_with_side)
#
#         # assert len(player2position_with_side) == 11
#         # TODO dynamic
#         # df_events_in_formation["pass_receiver_position_with_side"] = df_events_in_formation["pass.receiver.playerId"].replace(player2position_with_side)
#
#         def get_most_common_besetzung(df_events):
#             data = {}
#             for formation_players, df_events_in_formation_with_besetzung in df_events.groupby("live_formation_players"):
#                 formation_players_dict = json.loads(formation_players)
#                 for player, pos in formation_players_dict.items():
#                     pos_with_side = f"{pos[0]}_{pos[1]}"
#                     index = (player, pos_with_side)
#                     if index not in data:
#                         data[index] = 0
#                     data[index] += df_events_in_formation_with_besetzung["duration"].sum() / 60
#             df = pd.Series(data, name="minutes").reset_index().rename(columns={"level_0": "player.id", "level_1": "position_with_side"}).set_index(["position_with_side", "player.id"])["minutes"]
#             # df["position_with_side"] = df["position"] + "_" + df["position_side"]
#             # df_minutes_counts =
#             df_shares = df / df.groupby("position_with_side").sum()
#
#             # df_pos_counts = df_events[["player.id", "position_with_side"]].groupby('position_with_side')['player.id'].value_counts()
#             # df_pos_rec_counts = df_events[["pass.receiver.playerId", "pass_receiver_position_with_side"]].groupby('pass_receiver_position_with_side')['pass.receiver.playerId'].value_counts()
#             # df_pos_rec_counts.index = df_pos_rec_counts.index.rename(["position_with_side", "player.id"])
#             #
#             # df_pos_counts = df_pos_counts.add(df_pos_rec_counts, fill_value=0)
#             # df_pos_counts_shares = df_pos_counts / df_pos_counts.groupby('position_with_side').sum()
#
#             def get_player_name(player_id, all_player_ids):
#                 # name_segments = utility.impect.player_id_to_name(int(player_id), use_display_name=False).split(" ")
#                 #
#                 # if len(name_segments) == 1:
#                 #     return name_segments[0]
#                 #
#                 # final_name = ""
#                 # for segment_nr, segment in enumerate(name_segments):
#                 #     if segment_nr == 0:
#                 #         final_name += f"{segment[0]}. "
#                 #     elif segment_nr == len(name_segments) - 1:
#                 #         final_name += f"{segment}"
#                 #     elif segment in ["van", "de", "der", "ter", "la", "Van", "De", "Der", "Ter", "La"]:
#                 #         final_name += f"{segment} "
#                 #
#                 # return final_name
#
#                 all_player_ids = utility.general.uniquify_keep_order2(all_player_ids)
#                 player_name = utility.impect.player_id_to_name(int(player_id), use_display_name=False).split(" ")[-1]
#                 potentially_duplicate_names = [utility.impect.player_id_to_name(int(player_id), use_display_name=False).split(" ")[-1] for player_id in all_player_ids]
#
#                 # st.write(potentially_duplicate_names)
#                 # st.write(player_name)
#
#                 # if potentially_duplicate_names.count(player_name) == 1:
#                 #     return player_name
#
#                 name_segments = utility.impect.player_id_to_name(int(player_id), use_display_name=False).split(" ")
#
#                 final_name = ""
#                 for segment_nr, segment in enumerate(name_segments):
#                     if segment_nr == 0 and potentially_duplicate_names.count(player_name) > 1:
#                         final_name += f"{segment[0]}. "
#                     elif segment_nr == len(name_segments) - 1:
#                         final_name += f"{segment}"
#                     elif segment in ["van", "de", "di", "da", "der", "ter", "la", "Van", "De", "Da", "Di", "Der", "Ter", "La"]:
#                         final_name += f"{segment} "
#
#                 return final_name
#
#                 # st.write(name_segments)
#                 # st.write(name_segments[0][0] + ". " + name_segments[-1])
#                 #
#                 # return name_segments[0] + ". " + name_segments[-1]
#
#                 # st.write(potentially_duplicate_names)
#                 # st.write(duplicate_names)
#                 # st.stop()
#
#             share_threshold = min_player_share #0.71
#             pos2player = {}
#             for i, pos in enumerate(df_shares.index.get_level_values(0)):
#                 df_pos_counts_shares_pos = df_shares.loc[pos].sort_values(ascending=False)
#                 total_accounted_share = 0.0
#                 accounting_players = []
#                 for player, share in df_pos_counts_shares_pos.items():
#                     # player_name = utility.impect.player_id_to_name(int(player), use_display_name=False).split(" ")[-1]
#                     player_name = get_player_name(player, df_shares.index.get_level_values(1))
#                     total_accounted_share += share
#                     accounting_players.append(player_name)
#                     if total_accounted_share >= share_threshold:
#                         break
#                 pos2player[pos] = "/".join(accounting_players)
#             return pos2player
#
#         pos2player = get_most_common_besetzung(df_events_in_formation)
#
#         df_events_in_formation["position_with_side_names"] = df_events_in_formation["position_with_side"].replace(pos2player)
#         df_events_in_formation["pass_receiver_position_with_side_names"] = df_events_in_formation["pass_receiver_position_with_side"].replace(pos2player)
#
#         i_bla = df_events_in_formation["position_with_side"].isin(["LEFT_WINGER_LEFT"])
#         df_events_in_formation.loc[i_bla,
#             ["event_string", "live_formation_id", "live_formation_string", "player.commonname", "live_formation_players", "position_with_side", "pass.receiver.playerId", "pass_receiver_position_with_side",
#                 "start.adjCoordinates.x", "start.adjCoordinates.y", "position_with_side_names",
#              "pass_receiver_position_with_side_names", "end.adjCoordinates.x",
#              "end.adjCoordinates.y", "PXT_PASS_progressive", "SHOT_XG_chain", "match_id", "possession_id"]
#         ]
#
#         # df_events_in_formation = df_events_in_formation[
#         #     (df_events_in_formation["position_with_side"].isin(pos2player.keys())) & (df_events_in_formation["pass_receiver_position_with_side"].isin(pos2player.keys()))
#         # ]
#
#         # df_events_in_formation = df_events_in_formation[df_events_in_formation["pass_receiver_position_with_side"].notna()]  # only successful passes?
#
#         position2addvalue = None#{position: df_events_in_formation[df_events_in_formation["position_with_side"] == position][["PXT_DRIBBLE_progressive"]].sum().sum() / formation2mins[formation] * 60 for position in pos2player.keys()}
#         df_events_in_formation_passes = df_events_in_formation[df_events_in_formation["actionType"] != "DRIBBLE"]
#
#         # df_events_in_formation_passes["position_with_side"] = pd.Categorical(df_events_in_formation_passes["position_with_side"], categories=pos2player.keys())
#         # df_events_in_formation_passes["pass_receiver_position_with_side"] = pd.Categorical(df_events_in_formation_passes["pass_receiver_position_with_side"], categories=pos2player.keys())
#         # df_events_in_formation_passes["position_with_side_names"] = pd.Categorical(df_events_in_formation_passes["position_with_side"], categories=pos2player.values())
#         # df_events_in_formation_passes["pass_receiver_position_with_side_names"] = pd.Categorical(df_events_in_formation_passes["pass_receiver_position_with_side"], categories=pos2player.values())
#
#         if value_metric in ["pxt", "xgchain"]:
#             df_nodes, df_edges = get_passing_network_df(df_events_in_formation_passes,
#                 "start.adjCoordinates.x",
#                 "start.adjCoordinates.y",
#                 "position_with_side",
#                 "pass_receiver_position_with_side",
#                 x_to_col=None,#"end.adjCoordinates.x",  # BUGGY!
#                 y_to_col=None,#"end.adjCoordinates.y",  # BUGGY!
#                 value_col=("PXT_PASS_progressive" if remove_backpasses_for_pxt else "PXT_PASS") if value_metric == "pxt" else "SHOT_XG_chain",  # SHOT_XG_chain
#                 # value_col="OPP_PXT_PASS",
#                 from_name_col="position_with_side_names",
#                 # from_name_col="position_with_side",
#                 to_name_col="pass_receiver_position_with_side_names",
#                 # to_name_col="pass_receiver_position_with_side",
#                 dedup_cols=["match_id", "possession_id"] if value_metric == "xgchain" else None,
#                 net_minutes=formation2mins[formation],
#                 additional_node_values=position2addvalue,
#             )
#             # st.write("df_nodes", df_nodes.shape, formation)
#             # st.write(df_nodes)
#
#             df_edges = df_edges.sort_values("value_passes", ascending=False)
#             # get top 30 edges
#             df_edges_filtered = df_edges[
#                 (df_edges["num_passes_per90"] > 5) |
#                 (df_edges["value_passes_per90"] > 0.05)
#             ]
#             utility.dataframes.set_unlimited_pandas_display_options()
#
#             # if len(df_edges_filtered) == 0:
#             #     df_edges_filtered = df_edges
#             try:
#                 # table_formation = round(float(formation.split("-")[0]))
#                 table_formation = str(formation)
#                 formation_parts = table_formation.split("-")
#                 formation_id = int(float(table_formation.split("-")[0]))
#                 assert df_formation["custom_name"].is_unique
#                 formation_name = df_formation[df_formation["FormationID"] == formation_id]["custom_name"].iloc[0]
#
#                 if len(formation_parts) > 1:
#                     provider_position = [position for position in utility.impect.POSITIONS_V2 if position in formation_parts[1]][0]
#
#                     # POSITIONS_V2
#
#                     formation_name += f" {utility.impect.POSITIONS_V2[provider_position].global_position}"
#             except (IndexError, ValueError) as e:
#                 formation_name = "Unbekannte Formation"
#
#             # formation_name = formation
#
#             if player_name is None:
#                 title = f"{squad_name} - Häufigste Formation: {formation_name}"
#             else:
#                 title = f"{squad_name} - {player_name} - {formation_name} - {formation2grossmins[formation]:.0f} (net: {formation2mins[formation]:.0f}) minutes"
#
#             fig = plottt(df_nodes, df_edges_filtered, title=title)
#             figs.append(fig)
#             minutes.append(formation2mins[formation])
#             dfs_nodes.append(df_nodes)
#             dfs_edges.append(df_edges)
#
#         elif value_metric in ["def_pxt"]:
#             assert len(df_events_in_formation["match_id"].unique()) == 1
#             match_id = df_events_in_formation["match_id"].iloc[0]
#
#             df_matchinfos = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/matches_v5/{match_id}.csv")
#             team2players = df_matchinfos.groupby("squadId")["playerId"].unique().to_dict()
#
#             attacking_squad = df_events_in_formation["squadId"].iloc[0]
#             assert len(df_events_in_formation["squadId"].unique()) == 1
#             defending_squad = [squad for squad in team2players.keys() if squad != attacking_squad][0]
#
#             def_pxt_pass_cols = [col for col in df_events_in_formation.columns if "DEF_PXT_PASS_of_player" in col and utility.general.extract_numbers_as_int(col) in team2players[defending_squad]]
#
#             for def_pxt_col in def_pxt_pass_cols:
#                 # df_events_in_formation_passes_for_def_pxt = df_events_in_formation_passes[df_events_in_formation_passes[def_pxt_col].notna()]
#                 df_nodes, df_edges = get_passing_network_df(
#                     df_events_in_formation_passes,
#                     "start.adjCoordinates.x",
#                     "start.adjCoordinates.y",
#                     "position_with_side",
#                     "pass_receiver_position_with_side",
#                     x_to_col=None, # "end.adjCoordinates.x",
#                     y_to_col=None, # "end.adjCoordinates.y",
#                     value_col=def_pxt_col,
#                     # value_col="OPP_PXT_PASS",
#                     from_name_col="position_with_side_names",
#                     # from_name_col="position_with_side",
#                     to_name_col="pass_receiver_position_with_side_names",
#                     # to_name_col="pass_receiver_position_with_side",
#                     dedup_cols=None,
#                     net_minutes=formation2mins[formation],
#                     additional_node_values=position2addvalue,
#                 )
#                 # df_edges = df_edges.sort_values("value_passes", ascending=False)
#                 # # get top 30 edges
#                 df_edges_filtered = df_edges[
#                     # (df_edges["num_passes_per90"] > 10) |
#                     (df_edges["value_passes_per90"] != 0.0)
#                 ]
#                 utility.dataframes.set_unlimited_pandas_display_options()
#
#                 def_pxt_player = utility.impect.player_id_to_name(utility.general.extract_numbers_as_int(def_pxt_col)[-1], use_display_name=False)
#
#                 match_string = utility.impect.match_id_to_string(match_id, short=True)
#                 # if len(df_edges_filtered) == 0:
#                 #     df_edges_filtered = df_edges
#
#                 # plottt(df_nodes, df_edges_filtered, title=f"{def_pxt_player} vs {squad_name} - ({match_string}) - {formation2mins[formation]:.0f} minutes")
#                 title = f"{def_pxt_player} vs {squad_name} - ({match_string}) - {formation2grossmins[formation]:.0f} (net: {formation2mins[formation]:.0f}) minutes"
#                 filetitle = f"{squad_name} - {def_pxt_player} - ({match_string})"
#                 fig, ax = plot_passing_network(
#                     df_nodes,
#                     df_edges=df_edges_filtered,
#                     node_size_col="num_passes_per90",
#                     node_color_col="value_passes_and_receptions_per90",
#                     node_size_multiplier=5.0,
#                     max_color_value_nodes=0.25,
#                     min_color_value_nodes=-0.25,
#
#                     arrow_width_col="num_passes_per90",
#                     arrow_color_col="value_passes_per90",
#                     arrow_width_multiplier=0.3,
#                     max_color_value_edges=0.3,
#                     min_color_value_edges=-0.3,
#
#                     colorbar_label="Allowed pxT (DEF-pxT) through passes in total (per 90 minutes)",
#                     colormap=matplotlib.cm.get_cmap("coolwarm"),
#                 )
#                 # formation_str = formation if "undefined" not in formation.lower() else "Undefined Formation"
#                 # plt.title(f"{squad_name} - {player_name} - {formation_str} - {formation2mins[formation]:.0f} minutes")
#                 plt.title(title)
#
#                 # xlim to max
#                 x_padding = -5
#                 y_padding = 5
#                 ax.set_xlim(df_nodes["x_avg"].min() - x_padding, df_nodes["x_avg"].max() + x_padding)
#                 ax.set_ylim(df_nodes["y_avg"].min() - y_padding, df_nodes["y_avg"].max() + y_padding)
#                 # xlim to max
#                 x_padding = -5
#                 y_padding = 5
#                 ax.set_xlim(df_nodes["x_avg"].min() - x_padding, df_nodes["x_avg"].max() + x_padding)
#                 ax.set_ylim(df_nodes["y_avg"].min() - y_padding, df_nodes["y_avg"].max() + y_padding)
#
#                 plt.savefig(f"C:/misc/" + f"{utility.general.slugify(filetitle)}"[:200] + ".png")
#                 plt.show()
#
#         else:
#             raise ValueError(f"Unknown value_metric {value_metric}")
#
#     plt.show()
#
#     return figs, minutes, dfs_nodes, dfs_edges
#
#
# def _process_events_single_match(match_id, match2relevant_periods, match2periodpos, formation2mins, formation2grossmins, phases, squad_id):
#     try:
#         pass
#         # processing.process.impect.preprocess_matchinfos_and_events(match_ids=[match_id], overwrite_if_exists=True, download_if_raw_file_not_present=False, do_not_overwrite_younger_than=None)
#
#     except botocore.client.ClientError as e:
#         return None
#
#     try:
#         df_events = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/events/{round(match_id)}.csv")
#         # df_matchinfos = utility.impect.get_matchinfos(match["matchId"])
#         df_events["match_string"] = utility.impect.match_id_to_string(match_id)
#         if df_events["live_formation_players"].str.contains("UNKNOWN").sum() != 0:
#             st.warning("UNKNOWN formation players detected. Reprocessing...")
#             processing.process.impect.preprocess_matchinfos_and_events(match_ids=[match_id], overwrite_if_exists=True, download_if_raw_file_not_present=False, do_not_overwrite_younger_than=None)
#             df_events = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/events/{round(match_id)}.csv")
#             assert df_events["live_formation_players"].str.contains("UNKNOWN").sum() == 0
#             st.info("UNKNOWN formation players fixed!")
#
#     except botocore.client.ClientError as e:
#         return None
#     # df_passes_and_dribbles = df_events.loc[(
#     #     (df_events["actionType"] == "PASS") | (df_events["actionType"] == "DRIBBLE")
#     # ), [col for col in df_events.columns if "_of_player_" not in col and "_of_squad_" not in col]]
#     # df_passes_and_dribbles = df_events.loc[df_events["actionType"].isin(["PASS", "DRIBBLE", "KICK_OFF", "CLEARANCE", "THROW_IN", "GOAL_KICK", "SHOT", "BLOCK", "CORNER", "OFFSIDE"]),
#     #                          [col for col in df_events.columns if "_of_player_" not in col and "_of_squad_" not in col]]
#
#     if match2relevant_periods is not None:
#         df_events["is_in_relevant_period"] = False
#         for (start, end) in match2relevant_periods[match_id]:
#             i_segment = ((df_events["gameTime.gameTimeInSec"] >= start) & (df_events["gameTime.gameTimeInSec"] <= end))
#             df_events["is_in_relevant_period"] = df_events["is_in_relevant_period"] | i_segment
#             df_events.loc[i_segment, "network_player_segment_position"] = match2periodpos[match_id][0][2]
#             df_events.loc[i_segment, "network_player_segment_position_side"] = match2periodpos[match_id][0][3]
#
#         df_events["live_formation_string_and_player_pos"] = df_events["live_formation_string"] + "-" + df_events["network_player_segment_position"] + "_" + df_events["network_player_segment_position_side"]
#         df_events["live_formation_id_and_player_pos"] = df_events["live_formation_id"].astype(str) + "-" + df_events["network_player_segment_position"] + "_" + df_events["network_player_segment_position_side"]
#         formation_col = "live_formation_id_and_player_pos"
#     else:
#         df_events["is_in_relevant_period"] = True
#         formation_col = "live_formation_id"
#         # formation_col = "live_formation_string"
#
#     # dfg = df_events[["live_formation_string", "duration"]].groupby("live_formation_string").sum() / 60
#     # dfg_mapping = dfg.to_dict()["duration"]
#     # for key in dfg_mapping:
#     #     if key not in formation2mins:
#     #         formation2mins[key] = 0
#     #     formation2mins[key] += dfg_mapping[key]
#
#     # df_events[["event_string", "formation_ok", "is_in_relevant_period", "is_pass_for_pn", "is_dribble_for_pn", "actionType", "pass.receiver.type", "pass.receiver.playerId", "player.position_side", "start.adjCoordinates.x", "live_formation_players", "live_formation_string"]]
#
#     # pathological_formation_strings = [formation for formation in df_events["live_formation_players"].dropna().unique().tolist() if "UNKNOWN" in formation]
#     # pathological_unknown_counts = {formation: formation.count("UNKNOWN") for formation in pathological_formation_strings}
#     # for formation, pathological_counts in pathological_unknown_counts.items():
#     #     continue
#     #     if pathological_counts == 1:
#     #         formation_dict = json.loads(formation)
#     #         unknown_position = [v for k, v in formation_dict.items() if "UNKNOWN" in v][0][0]
#     #         unknown_position_alternatives = [v for k, v in formation_dict.items() if unknown_position == v[0]]
#     #         present_position_sides = [v[1] for v in unknown_position_alternatives]
#     #         formation_str = df_events[df_events["live_formation_players"] == formation]["live_formation_string"].iloc[0]
#     #
#     #         if unknown_position in ["GOALKEEPER"] and len(present_position_sides) == 1:  # 1 GK
#     #             target = "CENTRE"
#     #         elif unknown_position in ["RIGHT_WINGER", "RIGHT_WINGBACK_DEFENDER"] and len(present_position_sides) == 1:  # 1 RW/RB
#     #             target = "RIGHT"
#     #         elif unknown_position in ["LEFT_WINGER", "LEFT_WINGBACK_DEFENDER"] and len(present_position_sides) == 1:  # 1 LW/LB
#     #             target = "LEFT"
#     #         elif unknown_position in ["CENTER_FORWARD", "ATTACKING_MIDFIELD"] and len(present_position_sides) == 1:  # 1 striker
#     #             target = "CENTRE"
#     #         elif unknown_position == "CENTRAL_DEFENDER" and len(present_position_sides) == 3 and "CENTRE_RIGHT" in present_position_sides and "CENTRE_LEFT" in present_position_sides:  # 3 CBs
#     #             target = "CENTRE"
#     #         elif unknown_position == "DEFENSE_MIDFIELD" and len(present_position_sides) == 2 and "CENTRE" in present_position_sides:  # 2 DMs
#     #             target = "CENTRE"
#     #         elif unknown_position == "DEFENSE_MIDFIELD" and len(present_position_sides) == 1:  # 1 DM
#     #             target = "CENTRE"
#     #         elif "CENTRE_RIGHT" in present_position_sides and "CENTRE_LEFT" not in present_position_sides:
#     #             target = "CENTRE_LEFT"
#     #         elif "CENTRE_LEFT" in present_position_sides and "CENTRE_RIGHT" not in present_position_sides:
#     #             target = "CENTRE_RIGHT"
#     #         else:
#     #             if "undefined" in df_events[df_events["live_formation_players"] == formation]["live_formation_string"].iloc[0]:
#     #                 continue
#     #             utility.log.warning(f"Formation {formation_str} has 1 UNKNOWN player but no clear target position found {unknown_position_alternatives}")
#     #             st.write(f"Formation {formation_str} has 1 UNKNOWN player but no clear target position found {unknown_position_alternatives}")
#     #             continue
#     #
#     #         df_events.loc[df_events["live_formation_players"] == formation, "live_formation_players"] = formation.replace("UNKNOWN", target)
#     #
#     #         # print(f"Formation {formation} has 1 UNKNOWN player")
#
#     # pathological_formation_strings2 = [formation for formation in df_events["live_formation_players"].dropna().unique().tolist() if "UNKNOWN" in formation]
#     # pathological_unknown_counts2 = {formation: formation.count("UNKNOWN") for formation in pathological_formation_strings2}
#
#     df_events["is_pass_for_pn"] = (df_events["actionType"].isin(["PASS", "KICK_OFF", "CLEARANCE", "THROW_IN", "GOAL_KICK"]) & (df_events["pass.receiver.type"] == "TEAMMATE"))
#     df_events["is_dribble_for_pn"] = df_events["actionType"] == "DRIBBLE"
#     # df_events["formation_ok"] = (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool))
#
#     # first: remove irrelevant periods (selected player not on pitch) and invalid formations
#     df_events_prefiltered_gross = df_events[
#         # (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool))
#         (df_events["is_in_relevant_period"])
#         # (df_events["phase"].isin(phases)) &
#         # (df_events["squadId"] == squad_id)
#     ]
#     formation2grossmins_match = df_events_prefiltered_gross[[formation_col, "duration"]].groupby(formation_col).sum() / 60
#     df_events_prefiltered = df_events_prefiltered_gross[
#         (~df_events_prefiltered_gross["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool)) &
#         (df_events_prefiltered_gross["is_in_relevant_period"])
#         # (df_events["phase"].isin(phases)) &
#         # (df_events["squadId"] == squad_id)
#     ]
#
#     # formation2grossmins_match = df_events_prefiltered[[formation_col, "duration"]].groupby(formation_col).sum() / 60
#     for formation in formation2grossmins_match.index:
#         if formation not in formation2grossmins:
#             formation2grossmins[formation] = 0
#         formation2grossmins[formation] += formation2grossmins_match.loc[formation, "duration"]
#     # then: filter out irrelevant match phases
#     df_events_prefiltered = df_events_prefiltered[
#         # (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool)) &
#         # (df_events["is_in_relevant_period"]) &
#         (df_events_prefiltered["phase"].isin(phases)) &
#         (df_events_prefiltered["squadId"] == squad_id)
#     ]
#
#     formation2mins_match = df_events_prefiltered[[formation_col, "duration"]].groupby(formation_col).sum() / 60
#     for formation in formation2mins_match.index:
#         if formation not in formation2mins:
#             formation2mins[formation] = 0
#         # formation2mins[formation] += formation2mins_match.loc[formation, "duration"]
#     # then: remove irrelevant and invalid actions
#     df_events_filtered = df_events_prefiltered[
#         # (df_events["is_in_relevant_period"])
#         # (df_events["player.position_side"].notna()) &
#         (df_events_prefiltered["player.position_side"].notna()) &
#         (df_events_prefiltered["start.adjCoordinates.x"].notna()) &
#         # (df_events["squadId"] == squad_id) &
#         # (df_events["phase"].isin(phases)) &
#         # (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool)) &  # this loses much information - should be changed once Impect delivers live formation data or manually
#         #
#         ((df_events_prefiltered["is_pass_for_pn"] | df_events_prefiltered["is_dribble_for_pn"]))
#     ].copy()
#
#     df_events_filtered["position_with_side"] = df_events_filtered["player.position"].str.cat(df_events_filtered["player.position_side"], sep="_")
#
#     return df_events_filtered, formation2mins_match, formation2grossmins_match
#
#
# def _process_all_df_per_player(df_events, squad_id, phases=("IN_POSSESSION", "SECOND_BALL", "ATTACKING_TRANSITION")):
#     df_events = df_events[df_events["player_on_pitch_squad_id"] == squad_id]  # necessary?
#
#     df_events["network_player_segment_position"] = df_events["player_on_pitch_segment_position"]
#     df_events["network_player_segment_position_side"] = df_events["player_on_pitch_segment_segment_position_side"]
#     df_events["live_formation_string_and_player_pos"] = df_events["live_formation_string"] + "-" + df_events["network_player_segment_position"] + "_" + df_events["network_player_segment_position_side"]
#     df_events["live_formation_id_and_player_pos"] = df_events["live_formation_id"].astype(str) + "-" + df_events["network_player_segment_position"] + "_" + df_events["network_player_segment_position_side"]
#     formation_col = "live_formation_id_and_player_pos"
#     df_events["is_in_relevant_period"] = True
#
#     df_events["is_pass_for_pn"] = (df_events["actionType"].isin(["PASS", "KICK_OFF", "CLEARANCE", "THROW_IN", "GOAL_KICK"]) &
#                                    (df_events["pass.receiver.type"] == "TEAMMATE"))
#     df_events["is_dribble_for_pn"] = df_events["actionType"] == "DRIBBLE"
#
#     formation2grossmins = (df_events[[formation_col, "duration"]].groupby(formation_col).sum() / 60).to_dict()["duration"]
#     formation2mins = {k: 0 for k in formation2grossmins.keys()}
#
#     df_events_prefiltered = df_events[
#         # (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool)) &
#         (df_events["phase"].isin(phases)) &
#         (df_events["squadId"] == squad_id)
#     ]
#
#     formation2mins.update((df_events_prefiltered[[formation_col, "duration"]].groupby(formation_col).sum() / 60).to_dict()["duration"])
#
#     df_events_filtered = df_events_prefiltered[
#         (df_events_prefiltered["player.position_side"].notna()) &
#         (df_events_prefiltered["start.adjCoordinates.x"].notna()) &
#         ((df_events_prefiltered["is_pass_for_pn"] | df_events_prefiltered["is_dribble_for_pn"]))
#     ].copy()
#
#     df_events_filtered["position_with_side"] = df_events_filtered["player.position"].str.cat(df_events_filtered["player.position_side"], sep="_")
#
#     assert set(formation2mins.keys()) == set(formation2grossmins.keys())
#
#     return df_events_filtered, formation2mins, formation2grossmins
#
#
#     # player_on_pitch"] = int(player_id)
#     #                     df_events.loc[i_in_segment, "player_on_pitch_segment_position"] = row["segment_position"]
#     #                     df_events.loc[i_in_segment, "player_on_pitch_segment_segment_position_side"] = row["segment_position_side"]
#     #                     df_events.loc[i_in_segment, "player_on_pitch_squad_id
#
#
# def _process_all_df_per_squad(df_events, squad_id, phases=("IN_POSSESSION", "SECOND_BALL", "ATTACKING_TRANSITION")):
#     formation_col = "live_formation_id"
#
#     df_events["is_in_relevant_period"] = True
#
#     df_events["is_pass_for_pn"] = (df_events["actionType"].isin(["PASS", "KICK_OFF", "CLEARANCE", "THROW_IN", "GOAL_KICK"]) &
#                                    (df_events["pass.receiver.type"] == "TEAMMATE"))
#     df_events["is_dribble_for_pn"] = df_events["actionType"] == "DRIBBLE"
#
#     formation2grossmins = (df_events[[formation_col, "duration"]].groupby(formation_col).sum() / 60).to_dict()["duration"]
#     formation2mins = {k: 0 for k in formation2grossmins.keys()}
#
#     df_events_prefiltered = df_events[
#         # (~df_events["live_formation_players"].str.contains("UNKNOWN").fillna(True).astype(bool)) &
#         (df_events["phase"].isin(phases)) &
#         (df_events["squadId"] == squad_id)
#     ]
#
#     formation2mins.update((df_events_prefiltered[[formation_col, "duration"]].groupby(formation_col).sum() / 60).to_dict()["duration"])
#
#     df_events_filtered = df_events_prefiltered[
#         (df_events_prefiltered["player.position_side"].notna()) &
#         (df_events_prefiltered["start.adjCoordinates.x"].notna()) &
#         ((df_events_prefiltered["is_pass_for_pn"] | df_events_prefiltered["is_dribble_for_pn"]))
#     ].copy()
#
#     df_events_filtered["position_with_side"] = df_events_filtered["player.position"].str.cat(df_events_filtered["player.position_side"], sep="_")
#
#     assert set(formation2mins.keys()) == set(formation2grossmins.keys())
#
#     return df_events_filtered, formation2mins, formation2grossmins
#
#
#
# memory = joblib.Memory(location="joblib_cache_passing_network", verbose=0)
#
# # def get_events_for_pn(squad_id=None, player_id=None, df_schedule=None, phases=("IN_POSSESSION", "SECOND_BALL", "ATTACKING_TRANSITION")):
#
# # @memory.cache
# def get_events_for_pn(squad_id=None, player_id=None, match_ids=None, phases=("IN_POSSESSION", "SECOND_BALL", "ATTACKING_TRANSITION")):
#     # if player_id is not None:
#     #     df_events = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/partitioned_events/per_player_on_pitch/player_on_pitch={round(player_id)}.csv")
#     #     return _process_all_df_per_player(df_events, squad_id, phases)
#     # elif squad_id is not None:
#     #     df_events = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/partitioned_events/per_squad_id/squad_id={round(squad_id)}.csv")
#     #     return _process_all_df_per_squad(df_events, squad_id, phases)
#
#     assert squad_id is not None or player_id is not None
#     squad_name = utility.impect.squad_id_to_name(squad_id)
#     # if df_schedule is None:
#     #     df_schedule = utility.impect.get_schedule()
#     df_schedule = utility.impect.get_filtered_schedule(match_ids=match_ids)
#
#     if squad_id is not None:
#         df_schedule_filtered = df_schedule[
#             (df_schedule["homeSquadId"] == squad_id) | (df_schedule["awaySquadId"] == squad_id)
#         ].sort_values("scheduledDate")
#     else:
#         df_schedule_filtered = df_schedule.sort_values("scheduledDate")
#
#     assert len(df_schedule_filtered) > 0, f"No matches found for squad {squad_name} ({squad_id})"
#
#     if player_id is not None:
#         df_player_matchsums = parse.from_s3.read_csv_s3(f"s3://vfb-datalake/tracking_general_processing/impect/preprocessed/v5/partitioned_matchsums_BY_POSITION/per_playerId/playerId={round(player_id)}.csv")
#         if squad_id is not None:
#             df_player_matchsums = df_player_matchsums[(df_player_matchsums["squadId"] == squad_id)]
#
#         df_schedule_filtered = df_schedule_filtered[df_schedule_filtered["matchId"].isin(df_player_matchsums["match_id"])]
#         df_player_matchsums = df_player_matchsums[df_player_matchsums["match_id"].isin(df_schedule_filtered["matchId"])]
#
#         match2start = df_player_matchsums.groupby("match_id").apply(lambda df: df["segment_start_gametime"].tolist())
#         match2end = df_player_matchsums.groupby("match_id").apply(lambda df: df["segment_end_gametime"].tolist())
#         match2relevant_periods = {match: list(zip(start, end)) for match, start, end in zip(match2start.index, match2start, match2end)}
#         match2periodpos = df_player_matchsums.groupby("match_id").apply(lambda df: df[["segment_start_gametime", "segment_end_gametime", "segment_position", "segment_position_side"]].values.tolist())
#     else:
#         match2relevant_periods = None
#         match2periodpos = None
#
#     formation2mins = {}
#     formation2grossmins = {}
#
#     # dfs = []
#     data = []
#     for _, match in tqdm.tqdm(df_schedule_filtered.iterrows(), total=len(df_schedule_filtered), desc=f"Processing matches for {squad_name} ({squad_id})"):
#         # processing.process.impect.preprocess_matchinfos_and_events([match["matchId"]], overwrite_if_exists=True)
#         res = _process_events_single_match(match["matchId"], match2relevant_periods, match2periodpos, formation2mins, formation2grossmins, phases, squad_id)
#
#         if res is None:
#             continue
#
#         data.append(res)
#         # st.write("Collecting data for PN:", utility.impect.match_id_to_string(match["matchId"], short=True))
#         # st.write("Collecting data for PN:", f'{res[0]["duration"].sum() / 60} minutes')
#
#     # data = utility.general.multiprocess(_process_events_single_match, [{"match_id": match["matchId"], "match2relevant_periods": match2relevant_periods, "match2periodpos": match2periodpos, "formation2mins": formation2mins, "formation2grossmins": formation2grossmins, "phases": phases, "squad_id": squad_id} for _, match in df_schedule_filtered.iterrows()], n_jobs=6)
#
#     dfs = [d[0] for d in data]
#     for d in data:
#         formation2mins_match = d[1]
#         formation2grossmins_match = d[2]
#         for formation in formation2mins_match.index:
#             if formation not in formation2mins:
#                 formation2mins[formation] = 0
#             formation2mins[formation] += formation2mins_match.loc[formation, "duration"]
#         for formation in formation2grossmins_match.index:
#             if formation not in formation2grossmins:
#                 formation2grossmins[formation] = 0
#             formation2grossmins[formation] += formation2grossmins_match.loc[formation, "duration"]
#
#     # all events have same cols
#     df_relevant_actions = pd.concat(dfs, axis=0, ignore_index=True)
#
#     # st.write("df_relevant_actions")
#     # st.write(df_relevant_actions)
#
#     assert len(df_relevant_actions["squad_name"].dropna().unique()) == 1
#
#     return df_relevant_actions, formation2mins, formation2grossmins
#
#
# def create_auto_team_passing_network(squad_id=None, player_id=None, df_schedule=None, phases=("IN_POSSESSION", "SECOND_BALL", "ATTACKING_TRANSITION"), remove_backpasses_for_pxt=True, value_metric="pxt", min_player_share=0.7):
#     """
#     >>> create_auto_team_passing_network(46, utility.impect.get_filtered_schedule(iteration_ids=[743, 742]))
#     """
#     # df_relevant_actions, formation2mins, formation2grossmins = get_events_for_pn(squad_id, player_id, df_schedule, phases)
#     if df_schedule is None:
#         df_schedule = utility.impect.get_schedule()
#
#     df_relevant_actions, formation2mins, formation2grossmins = get_events_for_pn(squad_id, player_id, df_schedule["matchId"].tolist(), phases)
#
#     player_name = utility.impect.player_id_to_name(player_id, use_display_name=False) if player_id is not None else None
#
#     figs, minutes, dfs_nodes, dfs_edges = pn_by_events(
#         df_relevant_actions, remove_backpasses_for_pxt=remove_backpasses_for_pxt, value_metric=value_metric,
#         formation2mins=formation2mins, formation2grossmins=formation2grossmins, player_name=player_name,
#         squad_name=utility.impect.squad_id_to_name(squad_id), min_player_share=min_player_share,
#     )
#     return figs, minutes, dfs_nodes, dfs_edges
