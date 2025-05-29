import pandas as pd
import defensive_network.models.passing_network as pn


def test_average_positions():
    df = pd.DataFrame({"x": [1, 2, 6], "y": [-2, 0, 5], "from": ["a", "a", "b"], "to": ["b", "b", "a"]})
    df_nodes, _ = pn.get_passing_network(df, "from", "to", "x", "y")

    assert sorted(df_nodes["entity"].tolist()) == ["a", "b"]
    assert df_nodes.loc[df_nodes["entity"] == "a", "x_avg"].iloc[0] == 1.5
    assert df_nodes.loc[df_nodes["entity"] == "a", "y_avg"].iloc[0] == -1
    assert df_nodes.loc[df_nodes["entity"] == "b", "x_avg"].iloc[0] == 6
    assert df_nodes.loc[df_nodes["entity"] == "b", "y_avg"].iloc[0] == 5

    static_positions = {"a": (0, 7), "b": (10, 2)}
    df_nodes, _ = pn.get_passing_network(df, "from", "to", "x", "y", entity_to_average_position=static_positions)
    assert sorted(df_nodes["entity"].tolist()) == ["a", "b"]
    assert df_nodes.loc[df_nodes["entity"] == "a", "x_avg"].iloc[0] == 0
    assert df_nodes.loc[df_nodes["entity"] == "a", "y_avg"].iloc[0] == 7
    assert df_nodes.loc[df_nodes["entity"] == "b", "x_avg"].iloc[0] == 10
    assert df_nodes.loc[df_nodes["entity"] == "b", "y_avg"].iloc[0] == 2
