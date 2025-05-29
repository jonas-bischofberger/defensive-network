import collections
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import scipy.optimize

import defensive_network.utility.general
import defensive_network.utility.dataframes

FormationResult = collections.namedtuple("FormationResult", ["role", "role_name", "formation_instance", "role_category"])


def _plot_roles(df_tracking, x_col="x_norm", y_col="y_norm", role_col="role", role_name_col=None):
    """
    >>> df_tracking = pd.DataFrame({"role": ["A", "A", "B", "B", "C", "C"], "role_name": ["Role A", "Role A", "Role B", "Role B", "Role C", "Role C"], "x_norm": [0, 10, 20, 30, 40, 50], "y_norm": [0, 0, 30, 35, -20, 10]})
    >>> df_tracking
      role role_name  x_norm  y_norm
    0    A    Role A       0       0
    1    A    Role A      10       0
    2    B    Role B      20      30
    3    B    Role B      30      35
    4    C    Role C      40     -20
    5    C    Role C      50      10
    >>> _plot_roles(df_tracking, role_name_col="role_name")
    <Figure size 640x480 with 1 Axes>
    >>> plt.show() # doctest: +SKIP
    """
    if role_name_col is None:
        role_name_col = role_col
    assert role_col in df_tracking.columns
    assert role_name_col in df_tracking.columns

    dfg = df_tracking.groupby(role_col).agg({x_col: "mean", y_col: "mean", role_name_col: "first"})
    try:
        dfg = dfg.reset_index()
    except ValueError:
        pass
    plt.figure()
    plt.xlim(-52.5, 52.5)
    plt.ylim(-34, 34)
    roles = dfg[role_col].unique()
    dfg["role_index"] = dfg[role_col].apply(lambda x: list(roles).index(x))

    plt.scatter(dfg[x_col], dfg[y_col], c=dfg["role_index"])
    for i, role_name in enumerate(dfg[role_name_col]):
        plt.annotate(role_name, (dfg[x_col].iloc[i], dfg[y_col].iloc[i]-0.5), fontsize=8, color="black", ha="center", va="top")
    # plt.legend()
    return plt.gcf()


def detect_formation(
    df_tracking, frame_col="full_frame", x_col="x_norm", y_col="y_norm", player_col="player_id", team_col="team_id",
    player_name_col="player_name", team_name_col="team_id", ball_team="BALL", model="average_pos", plot_formation=False,
):
    """
    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df_tracking = pd.DataFrame({"full_frame": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "x_norm": [10, 20, 30, 40, 50, -10, -20, -30, -40, -50, 0], "y_norm": [10, -10, 30, -30, 0, -10, -20, -30, -5, -25, 15], "player_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "team_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"], "player_name": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"], "ball_poss_team_id": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]})
    >>> result = detect_formation(df_tracking)
    >>> df_tracking["role"], df_tracking["role_name"], df_tracking["formation_instance"], df_tracking["role_category"] = result.role, result.role_name, result.formation_instance, result.role_category
    >>> df_tracking
        full_frame  x_norm  y_norm  player_id team_id player_name ball_poss_team_id      role role_name formation_instance     role_category
    0            1      10      10          1       A          P1                 A  A_off0.0        P1              A_off        goalkeeper
    1            1      20     -10          2       A          P2                 A  A_off1.0        P2              A_off  central_defender
    2            1      30      30          3       A          P3                 A  A_off2.0        P3              A_off     left_defender
    3            1      40     -30          4       A          P4                 A  A_off3.0        P4              A_off      right_winger
    4            1      50       0          5       A          P5                 A  A_off4.0        P5              A_off           striker
    5            1     -10     -10          6       B          P6                 A  B_def0.0        P6              B_def  central_defender
    6            1     -20     -20          7       B          P7                 A  B_def1.0        P7              B_def  central_midfield
    7            1     -30     -30          8       B          P8                 A  B_def2.0        P8              B_def       left_winger
    8            1     -40      -5          9       B          P9                 A  B_def3.0        P9              B_def      right_winger
    9            1     -50     -25         10       B         P10                 A  B_def4.0       P10              B_def           striker
    10           1       0      15         11       B         P11                 A  B_def5.0       P11              B_def        goalkeeper
    """
    df_tracking = df_tracking.copy()

    if ball_team is not None:
        df_tracking = df_tracking[df_tracking[team_col] != ball_team]

    df_tracking = df_tracking[df_tracking[x_col].notna() & df_tracking[y_col].notna()]

    with st.spinner("Sorting..."):
        df_tracking = df_tracking.sort_values([frame_col, team_col, player_col])

    for col in [frame_col, x_col, y_col, player_col, team_col]:
        assert col in df_tracking.columns, f"{col} not in df.columns ({df_tracking.columns})"

    dfs = []

    for team_nr, (team, df_team) in enumerate(df_tracking.groupby(team_col)):
        if plot_formation:
            columns = st.columns(2)
        team_name = df_team[team_name_col].iloc[0]
        for in_possession_nr, in_possession in enumerate([True, False]):
            in_poss_str = "off" if in_possession else "def"
            formation_instance = f"{team_name}_{in_poss_str}"

            if in_possession:
                df = df_team[df_team["ball_poss_team_id"] == team]
            else:
                df = df_team[df_team["ball_poss_team_id"] != team]
                df[x_col] *= -1
                df[x_col] *= -1

            if len(df) == 0:
                continue

            # check if all frames have exactly 11 players. If it fails, throw away frames with duplicate numbers
            df["n_unique_players"] = df.groupby(frame_col)[player_col].transform("nunique")
            df = df[df["n_unique_players"] <= 11]

            binsize = 1.5

            def get_default_role_assignment(df, frame_col, player_col, x_col="x_norm", y_col="y_norm", role_prefix=""):
                df_configuration = pd.pivot_table(df, index=frame_col, columns=player_col, values=x_col, aggfunc="count")
                df_configuration["configuration_id"] = pd.factorize(df_configuration.apply(tuple, axis=1))[0]

                unique_configurations = df_configuration.drop_duplicates("configuration_id").reset_index(drop=True)
                unique_configuration_lists = unique_configurations[[col for col in unique_configurations.columns if col != "configuration_id"]].apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()

                with st.spinner("Mapping..."):
                    df["configuration_id"] = df[frame_col].map(df_configuration["configuration_id"])

                dfg_player_means = df.groupby(player_col).agg(x_mean=(x_col, "mean"), y_mean=(y_col, "mean"))

                current_config = set(unique_configuration_lists[0])
                current_config_id = unique_configurations["configuration_id"].iloc[0]
                current_player2role = {player: player_nr for player_nr, player in enumerate(current_config)}
                i_current_configuration = df["configuration_id"] == current_config_id
                df.loc[i_current_configuration, "role"] = df.loc[i_current_configuration, player_col].map(current_player2role)

                for config_nr, next_config in enumerate(unique_configuration_lists[1:]):
                    next_config = set(next_config)
                    current_config_id = unique_configurations["configuration_id"].iloc[config_nr + 1]
                    out_subs = list(current_config - next_config)
                    in_subs = list(next_config - current_config)
                    assert all([out_sub in current_player2role for out_sub in out_subs])
                    assert not any([in_sub in current_player2role for in_sub in in_subs])

                    if len(in_subs) == 1:
                        assert in_subs[0] not in current_player2role
                        assert len(current_player2role) == 11
                        current_player2role[in_subs[0]] = current_player2role.pop(out_subs[0])
                        assert len(current_player2role) == 11
                    elif len(in_subs) > 1:
                        in_sub_pos_means = dfg_player_means.loc[in_subs, ["x_mean", "y_mean"]]
                        out_sub_pos_means = dfg_player_means.loc[out_subs, ["x_mean", "y_mean"]]
                        cost_matrix = np.linalg.norm(in_sub_pos_means.values[:, np.newaxis, :] - out_sub_pos_means.values[np.newaxis, :, :], axis=-1)
                        optimal_insub, optimal_outsub = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                        for in_sub_index, out_sub_index in zip(optimal_insub, optimal_outsub):
                            in_sub = in_subs[in_sub_index]
                            out_sub = out_subs[out_sub_index]
                            assert out_sub in current_player2role
                            current_player2role[in_sub] = current_player2role.pop(out_sub)
                    else:
                        raise NotImplementedError(f"len(in_subs) == {len(in_subs)}")

                    i_config = df["configuration_id"] == current_config_id
                    df.loc[i_config, "role"] = df.loc[i_config, player_col].map(current_player2role)
                    assert df.loc[i_config, "role"].notna().all()

                    current_config = next_config

                df["role"] = role_prefix + df["role"].astype(str)
                return df["role"]

            df["role"] = get_default_role_assignment(df, frame_col, player_col, x_col, y_col, role_prefix=formation_instance)

            role2players = df.groupby("role")[player_name_col].apply(set).to_dict()

            # check if some player is nan
            if any([pd.isna(player) for players in role2players.values() for player in players]):
                st.warning(f"Some players are missing in the role2players mapping: {role2players}")

            role2role_name = {role: '/'.join([player.split(". ")[-1] for player in list(players) if not pd.isna(player)]) for role, players in role2players.items()}
            df["role_name"] = df["role"].map(role2role_name)
            df["formation_instance"] = formation_instance
            if plot_formation:
                with columns[in_possession_nr]:
                    st.write(f"#### {formation_instance}")
                    st.write(_plot_roles(df))
                plt.close()

            df_tracking.loc[df.index, "role"] = df["role"]
            df_tracking.loc[df.index, "role_name"] = df["role_name"]
            df_tracking.loc[df.index, "formation_instance"] = df["formation_instance"]

            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    # assert frame-player-combo is unique
    assert len(df[[frame_col, player_col]].drop_duplicates()) == len(df)

    # df_tracking = df_tracking.merge(df, on=[frame_col, player_col], how="left")
    df_tracking["role_category"] = get_role_category(df_tracking)

    return FormationResult(df_tracking["role"], df_tracking["role_name"], df_tracking["formation_instance"], df_tracking["role_category"])


@st.cache_resource
def _read_parquet(fpath):
    return pd.read_parquet(fpath)


def get_role_category(df_tracking, role_col="role", formation_col="formation_instance", x_col="x_norm", y_col="y_norm"):
    df_tracking = df_tracking.copy()
    original_index = df_tracking.index
    assert (original_index == df_tracking.index).all()

    df_tracking["is_def_formation"] = df_tracking[formation_col].str.endswith("_def")
    df_tracking["x_norm_form"] = df_tracking[x_col] * (1 - 2 * df_tracking["is_def_formation"])
    df_tracking["y_norm_form"] = df_tracking[y_col] * (1 - 2 * df_tracking["is_def_formation"])
    del x_col, y_col  # do not use anymore

    # Identify and remove goalkeepers
    dfg_gk_role = df_tracking.groupby([formation_col, role_col]).agg({"x_norm_form": "mean", "y_norm_form": "mean"}).reset_index()
    # with st.spinner("Getting gk (likely the bottleneck)"):
    #     dfg_gk_role = dfg_gk_role.groupby(formation_col).apply(lambda x: x[x["x_norm_form"] == x["x_norm_form"].min()])[role_col].reset_index().drop(columns="level_1")

    dfg_gk_role = dfg_gk_role.groupby(formation_col)[["formation_instance", "role", "x_norm_form"]].apply(
        lambda x: x[x["x_norm_form"] == x["x_norm_form"].min()]
    )[role_col].reset_index().drop(columns="level_1")

    # st.write("dfg_gk_role")
    # st.write(dfg_gk_role)
    # st.stop()

    dfg_gk_role["role_category"] = "goalkeeper"
    dfg_gk_role["is_gk"] = dfg_gk_role["role_category"] == "goalkeeper"
    gk_roles = dfg_gk_role[role_col].unique()
    # TODO bottleneck 1
    assert (original_index == df_tracking.index).all()

    # assert dfg_gk_role has no duplicates
    assert len(dfg_gk_role) == len(dfg_gk_role.drop_duplicates(subset=[formation_col, role_col]))

    df_tracking = df_tracking.reset_index().merge(dfg_gk_role[[formation_col, role_col, "is_gk"]], on=[formation_col, role_col], how="left")
    df_tracking = df_tracking.set_index("index")  # somehow merge doesnt preserve index
    assert (original_index == df_tracking.index).all()

    with pd.option_context('future.no_silent_downcasting', True):
        df_tracking["is_gk"] = df_tracking["is_gk"].fillna(False).astype(bool)
    # assert len(df_tracking["is_gk"].dropna().unique()) == 2
    # df_tracking["is_gk"] = df_tracking[[formation_col, role_col]].apply(tuple, axis=1).map(dfg_gk_role.set_index([formation_col, role_col])["role_category"]) == "goalkeeper"

    dfg_roles = df_tracking[~df_tracking["is_gk"]].groupby([formation_col, role_col]).agg({"x_norm_form": "mean", "y_norm_form": "mean"}).reset_index()
    dfg_centroid = dfg_roles.groupby(formation_col).agg(x_centroid=("x_norm_form", "mean"), y_centroid=("y_norm_form", "mean"), x_min=("x_norm_form", "min"), x_max=("x_norm_form", "max"), y_min=("y_norm_form", "min"), y_max=("y_norm_form", "max"))

    dfg_centroid["x_formation_0.2_percentile"] = dfg_centroid["x_min"] + 0.15 * (dfg_centroid["x_max"] - dfg_centroid["x_min"])
    dfg_centroid["x_formation_0.4_percentile"] = dfg_centroid["x_min"] + 0.4 * (dfg_centroid["x_max"] - dfg_centroid["x_min"])
    dfg_centroid["x_formation_0.6_percentile"] = dfg_centroid["x_min"] + 0.6 * (dfg_centroid["x_max"] - dfg_centroid["x_min"])
    dfg_centroid["x_formation_0.8_percentile"] = dfg_centroid["x_min"] + 0.8 * (dfg_centroid["x_max"] - dfg_centroid["x_min"])
    dfg_centroid["y_formation_0.2_percentile"] = dfg_centroid["y_min"] + 0.2 * (dfg_centroid["y_max"] - dfg_centroid["y_min"])
    dfg_centroid["y_formation_0.4_percentile"] = dfg_centroid["y_min"] + 0.4 * (dfg_centroid["y_max"] - dfg_centroid["y_min"])
    dfg_centroid["y_formation_0.6_percentile"] = dfg_centroid["y_min"] + 0.6 * (dfg_centroid["y_max"] - dfg_centroid["y_min"])
    dfg_centroid["y_formation_0.8_percentile"] = dfg_centroid["y_min"] + 0.85 * (dfg_centroid["y_max"] - dfg_centroid["y_min"])

    dfg_roles = dfg_roles.merge(dfg_centroid, on=formation_col, how="left")

    with st.spinner("Calculating lane numbers (likely not the bottleneck)"):
        dfg_roles["horizontal_lane_number"] = dfg_roles.apply(lambda x: 1 if x["x_norm_form"] < x["x_formation_0.2_percentile"] else 2 if x["x_norm_form"] < x["x_formation_0.4_percentile"] else 3 if x["x_norm_form"] < x["x_formation_0.6_percentile"] else 4 if x["x_norm_form"] < x["x_formation_0.8_percentile"] else 5, axis=1)
        dfg_roles["vertical_lane_number"] = dfg_roles.apply(lambda x: 1 if x["y_norm_form"] < x["y_formation_0.2_percentile"] else 2 if x["y_norm_form"] < x["y_formation_0.4_percentile"] else 3 if x["y_norm_form"] < x["y_formation_0.6_percentile"] else 4 if x["y_norm_form"] < x["y_formation_0.8_percentile"] else 5, axis=1)

    lane_numbers_to_position_labels = {
        (1, 1): "right_defender",
        (1, 2): "central_defender",
        (1, 3): "central_defender",
        (1, 4): "central_defender",
        (1, 5): "left_defender",

        (2, 1): "right_defender",
        (2, 2): "central_midfield",
        (2, 3): "central_midfield",
        (2, 4): "central_midfield",
        (2, 5): "left_defender",

        (3, 1): "right_winger",
        (3, 2): "central_midfield",
        (3, 3): "central_midfield",
        (3, 4): "central_midfield",
        (3, 5): "left_winger",

        (4, 1): "right_winger",
        (4, 2): "central_midfield",
        (4, 3): "central_midfield",
        (4, 4): "central_midfield",
        (4, 5): "left_winger",

        (5, 1): "right_winger",
        (5, 2): "striker",
        (5, 3): "striker",
        (5, 4): "striker",
        (5, 5): "left_winger",
    }
    assert len(dfg_roles) > 0
    dfg_roles["role_category"] = dfg_roles.apply(lambda x: lane_numbers_to_position_labels[(x["horizontal_lane_number"], x["vertical_lane_number"])], axis=1)
    assert (original_index == df_tracking.index).all()

    # add gk role
    dfg_roles = pd.concat([dfg_roles, dfg_gk_role], axis=0).reset_index(drop=True)

    # TODO bottleneck 2
    # df_tracking["role_category"] = df_tracking[[formation_col, role_col]].apply(tuple, axis=1).map(dfg_roles.set_index([formation_col, role_col])["role_category"])
    # st.write("df_tracking")
    # st.write(df_tracking[[formation_col, role_col]].head(5))
    # st.write(dfg_roles[[formation_col, role_col, "role_category"]])
    df_tracking = df_tracking.reset_index().merge(dfg_roles[[formation_col, role_col, "role_category"]], on=[formation_col, role_col], how="left").set_index("index")  # somehow merge doesnt preserve index
    # st.write(df_tracking.head(5))
    assert "role_category" in df_tracking.columns
    # df_tracking = df_tracking.merge(dfg_gk_role[[formation_col, role_col, "role_category", "is_gk"]], on=[formation_col, role_col], how="left")
    # df_tracking = df_tracking.set_index("index")

    # st.write(df_tracking[[formation_col, role_col, "role_category"]].value_counts())

    assert "goalkeeper" in df_tracking["role_category"].unique()
    assert set(df_tracking.loc[df_tracking["role_category"] == "goalkeeper", "role"]) == set(gk_roles)

    # assert that every role has a unique category
    assert (original_index == df_tracking.index).all()

    return df_tracking["role_category"]


def main2():
    data = {
        "full_frame": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "x_norm": [10, 20, 30, 40, 50, -10, -20, -30, -40, -50, 0],
        "y_norm": [10, -10, 30, -30, 0, -10, -20, -30, -5, -25, 15],
        "player_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "team_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        "player_name": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"],
        "ball_poss_team_id": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
    }
    df_tracking = pd.DataFrame(data)
    result = detect_formation(df_tracking)
    df_tracking["role"] = result.role
    df_tracking["role_name"] = result.role_name
    df_tracking["formation_instance"] = result.formation_instance
    df_tracking["role_category"] = result.role_category
    # get_formation_plots(df_tracking, role_name_col="role_name")
    plots = get_formation_plots(df_tracking)

    for formation, fig in plots.items():
        st.write("formation", formation)
        st.write(fig)


def get_formation_plots(df_tracking, formation_col="formation_instance", x_col="x_norm", y_col="y_norm", role_col="role", role_name_col="role_category"):
    plots = {}
    for formation, df_formation in df_tracking.groupby(formation_col):
        # st.write("-------------")
        # st.write(formation)
        fig = _plot_roles(df_formation, x_col=x_col, y_col=y_col, role_col=role_col, role_name_col=role_name_col)
        # st.write(_plot_roles(df_formation, x_col=x_col, y_col=y_col, role_col=role_col, role_name_col=role_col))
        # st.write("-------------")
        plots[formation] = fig
    return plots


if __name__ == '__main__':
    defensive_network.utility.general.start_streamlit_profiler()
    main2()


def main1():
    defensive_network.utility.general.start_streamlit_profiler()

    # df = _read_parquet("C:/Users/Jonas/Downloads/dfl_test_data/2324/preprocessed/tracking/3-liga-2023-2024-20-st-sc-verl-viktoria-koln.parquet")
    df_tracking = _read_parquet(os.path.join(os.path.dirname(__file__), "../../data_reduced/preprocessed/tracking/3-liga-2023-2024-20-st-sc-verl-viktoria-koln.parquet"))
    df_tracking = df_tracking.drop(columns=["role", "role_name", "formation_instance"])
    assert "role" not in df_tracking.columns
    res = detect_formation(df_tracking)
    df_tracking["role"] = res.role
    df_tracking["role_name"] = res.role_name
    df_tracking["formation_instance"] = res.formation_instance
    df_tracking["role_category"] = res.role_category
    defensive_network.utility.general.stop_streamlit_profiler()

    for formation, df_tracking_formation in df_tracking.groupby("formation_instance"):
        st.write("-------------")
        st.write(formation)
        st.write(_plot_roles(df_tracking_formation, role_name_col="role_category"))
        st.write(_plot_roles(df_tracking_formation, role_name_col="role"))
        st.write("-------------")

