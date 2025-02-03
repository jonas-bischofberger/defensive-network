import collections
import os

import numpy as np
import pandas as pd

import defensive_network.utility.dataframes

XT_WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/xt_weights"))

XTResult = collections.namedtuple("XTResult", ["xt_before", "xt_after", "delta_xt"])


def get_expected_threat(
    df_events, event_x_col="x_norm", event_y_col="y_norm", pass_end_x_col="x_target_norm",
    pass_end_y_col="y_target_norm", event_success_col="is_successful", xt_model="ma2024", attacking_direction_col=None,
) -> XTResult:
    """
    Add expected threat to passes.

    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df_events = pd.DataFrame({"x_norm": [0, -30, 0], "y_norm": [0, 10, 0], "x_target_norm": [-30, 25, 10], "y_target_norm": [0, 0, 0], "is_successful": [True, True, False]})
    >>> df_events
       x_norm  y_norm  x_target_norm  y_target_norm  is_successful
    0       0       0            -30              0           True
    1     -30      10             25              0           True
    2       0       0             10              0          False
    >>> res = get_expected_threat(df_events, xt_model="ma2024")
    >>> df_events["xt_before"], df_events["xt_after"], df_events["delta_xt"] = res.xt_before, res.xt_after, res.delta_xt
    >>> df_events
       x_norm  y_norm  x_target_norm  y_target_norm  is_successful  xt_before  xt_after  delta_xt
    0       0       0            -30              0           True   0.015593  0.004577 -0.011016
    1     -30      10             25              0           True   0.004604  0.037778  0.033174
    2       0       0             10              0          False   0.015593  0.000000 -0.015593
    """
    defensive_network.utility.dataframes.check_presence_of_required_columns(df_events, "df_events", ["event_x_col", "event_y_col", "pass_end_x_col", "pass_end_y_col", "event_success_col"], [event_x_col, event_y_col, pass_end_x_col, pass_end_y_col, event_success_col])
    df_events[event_success_col] = df_events[event_success_col].astype(bool)

    available_xt_models = [file.split(".")[0] for file in os.listdir(XT_WEIGHTS_DIR) if file.endswith(".xlsx")]
    assert xt_model in available_xt_models, f"xt_model must be one of {available_xt_models}"
    xt_file = os.path.join(os.path.dirname(__file__), "../assets/xt_weights", f"{xt_model}.xlsx")

    df_events = df_events.copy()
    if attacking_direction_col is not None:
        df_events[event_x_col] = df_events[event_x_col] * df_events[attacking_direction_col]
        df_events[pass_end_x_col] = df_events[pass_end_x_col] * df_events[attacking_direction_col]
        df_events[event_y_col] = df_events[event_y_col] * df_events[attacking_direction_col]
        df_events[pass_end_y_col] = df_events[pass_end_y_col] * df_events[attacking_direction_col]

    # Get xT transition matrix
    df_xt = pd.read_excel(xt_file, header=None)
    num_x_cells = len(df_xt.columns)
    num_y_cells = len(df_xt.index)
    dx_cell = 105 / num_x_cells
    dy_cell = 68 / num_y_cells

    # Get cell index from x and y coordinates
    x_cell_index_col, y_cell_index_col, x_cell_index_after_col, y_cell_index_after_col = defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "x_cell_index"), defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "y_cell_index"), defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "x_cell_index_after"), defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "y_cell_index_after")
    df_events[x_cell_index_col] = np.clip(((df_events[event_x_col] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
    df_events[y_cell_index_col] = np.clip(((df_events[event_y_col] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)
    df_events[x_cell_index_after_col] = np.clip(((df_events[pass_end_x_col] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
    df_events[y_cell_index_after_col] = np.clip(((df_events[pass_end_y_col] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)

    # assign xT values based on cell index and compute xT of passes
    xt_before_col, xt_after_col, pass_xt_col = defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "xt_before"), defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "xt_after"), defensive_network.utility.dataframes.get_unused_column_name(df_events.columns, "pass_xt")
    df_events[xt_before_col] = 0
    df_events[xt_after_col] = 0
    i_valid_before = df_events["x_cell_index"].notnull() & df_events["y_cell_index"].notnull()  # sometimes we have no cell index because x and y coordinates are missing!
    df_events.loc[i_valid_before, xt_before_col] = df_events.loc[i_valid_before, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index"]), int(x["x_cell_index"])], axis=1)
    i_valid_end = df_events["x_cell_index_after"].notnull() & df_events["y_cell_index_after"].notnull()
    df_events.loc[i_valid_end, xt_after_col] = df_events.loc[i_valid_end, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index_after"]), int(x["x_cell_index_after"])], axis=1)

    # Important: xT after an unsuccessful pass is 0!
    df_events.loc[~df_events[event_success_col], xt_after_col] = 0

    df_events[pass_xt_col] = df_events[xt_after_col] - df_events[xt_before_col]

    return XTResult(df_events[xt_before_col], df_events[xt_after_col], df_events[pass_xt_col])
