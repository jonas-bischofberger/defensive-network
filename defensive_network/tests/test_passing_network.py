import pandas as pd
import defensive_network.models.passing_network as pn


def test_passing_network_df_average_positions():
    df = pd.DataFrame({"x": [1, 2, 6], "y": [-2, 0, 5], "from": ["a", "a", "b"], "to": ["b", "b", "a"]})
    df_nodes, _ = pn.get_passing_network(df, "from", "to", "x", "y")
    assert df_nodes.loc["a"]["x_avg"] == 1.5
    assert df_nodes.loc["a"]["y_avg"] == -1
    assert df_nodes.loc["b"]["x_avg"] == 6
    assert df_nodes.loc["b"]["y_avg"] == 5

    static_positions = {"a": (0, 7), "b": (10, 2)}
    df_nodes, _ = pn.get_passing_network(df, "from", "to", "x", "y", entity_to_average_position=static_positions)
    assert df_nodes.loc["a"]["x_avg"] == 0
    assert df_nodes.loc["a"]["y_avg"] == 7
    assert df_nodes.loc["b"]["x_avg"] == 10
    assert df_nodes.loc["b"]["y_avg"] == 2
