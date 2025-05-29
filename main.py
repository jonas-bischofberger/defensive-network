import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import scipy.optimize
import streamlit_profiler

import defensive_network.tests.test_passing_network


def plot_roles(df, x_col="x_norm", y_col="y_norm", role_col="role", role_name_col="role_name"):
    role_col = "role"
    role_name_col = "role_name"
    assert role_col in df.columns
    assert role_name_col in df.columns
    # role_col = "player_index"
    dfg = df.groupby(role_col).agg({x_col: "mean", y_col: "mean", role_name_col: "first"}).reset_index()
    plt.figure()
    # plt.xlim(-52.5, 52.5)
    # plt.ylim(-34, 34)
    roles = dfg[role_col].unique()
    dfg["role_index"] = dfg[role_col].apply(lambda x: list(roles).index(x))
    plt.scatter(dfg[x_col], dfg[y_col], c=dfg["role_index"])
    for i, txt in enumerate(dfg[role_name_col]):
        plt.annotate(txt, (dfg[x_col].iloc[i], dfg[y_col].iloc[i]-0.5), fontsize=8, color="black", ha="center", va="top")
    plt.legend()
    st.write(plt.gcf())
    plt.close()


def get_default_role_assignment(df, frame_col, player_col, x_col="x_norm", y_col="y_norm", role_prefix=""):
    df_configuration = df.groupby(frame_col)[player_col].apply(set).reset_index()
    df["configuration"] = df[frame_col].map(df_configuration.set_index(frame_col)[player_col])

    unique_configurations = df_configuration[player_col].drop_duplicates().tolist()
    dfg_player_means = df.groupby(player_col).agg(x_mean=(x_col, "mean"), y_mean=(y_col, "mean"))

    current_config = unique_configurations[0]
    current_player2role = {player: player_nr for player_nr, player in enumerate(current_config)}
    i_current_configuration = df["configuration"] == current_config
    df.loc[i_current_configuration, "role"] = df.loc[i_current_configuration, player_col].map(current_player2role)
    for next_config in unique_configurations[1:]:
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

        i_config = df["configuration"] == next_config
        df.loc[i_config, "role"] = df.loc[i_config, player_col].map(current_player2role)

        current_config = next_config

    df["role"] = role_prefix + df["role"].astype(str)
    return df["role"]


FormationResult = collections.namedtuple("FormationResult", ["role", "role_name", "formation_instance"])


def detect_formation(
    df_tracking, frame_col="full_frame", x_col="x_norm", y_col="y_norm", player_col="player_id", team_col="team_id",
    player_name_col="player_name", team_name_col="team_id", ball_team="BALL",
):
    profiler = streamlit_profiler.Profiler()
    profiler.start()

    df_tracking = df_tracking.copy()

    if ball_team is not None:
        df_tracking = df_tracking[df_tracking[team_col] != ball_team]

    df_tracking = df_tracking.sort_values([frame_col, team_col, player_col])
    df_tracking = df_tracking[df_tracking[x_col].notna() & df_tracking[y_col].notna()]

    for col in [frame_col, x_col, y_col, player_col, team_col]:
        assert col in df_tracking.columns, f"{col} not in df.columns ({df_tracking.columns})"

    dfs = []

    for team_nr, (team, df_team) in enumerate(df_tracking.groupby(team_col)):
        team_name = df_team[team_name_col].iloc[0]
        for in_possession in [True, False]:
            in_poss_str = "off" if in_possession else "def"
            formation_instance = f"{team_name}_{in_poss_str}"

            st.write("##", formation_instance)
            if in_possession:
                df = df_team[df_team["ball_poss_team_id"] == team]
            else:
                df = df_team[df_team["ball_poss_team_id"] != team]
                df["x_norm"] *= -1
                df["y_norm"] *= -1

            assert len(df) > 0

            # check if all frames have exactly 11. If it fails, throw away frames with duplicate numbers
            df["n_unique_players"] = df.groupby(frame_col)[player_col].transform("nunique")
            df = df[df["n_unique_players"] == 11]

            binsize = 1.5

            df["role"] = get_default_role_assignment(df, frame_col, player_col, x_col, y_col, role_prefix=formation_instance)

            role2players = df.groupby("role")[player_name_col].apply(set).to_dict()
            role2role_name = {role: '/'.join([player.split(". ")[-1] for player in list(players)]) for role, players in role2players.items()}
            df["role_name"] = df["role"].map(role2role_name)
            df["formation_instance"] = formation_instance
            plot_roles(df)

            df_tracking.loc[df.index, "role"] = df["role"]
            df_tracking.loc[df.index, "role_name"] = df["role_name"]
            df_tracking.loc[df.index, "formation_instance"] = df["formation_instance"]

            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    # assert frame-player-combo is unique
    assert len(df[[frame_col, player_col]].drop_duplicates()) == len(df)

    # df_tracking = df_tracking.merge(df, on=[frame_col, player_col], how="left")

    profiler.stop()

    return FormationResult(df_tracking["role"], df_tracking["role_name"], df_tracking["formation_instance"])

        # def test():
        #     st.write("role2role_name")
        #     st.write(role2role_name)
        #
        #     xy_mean = df.groupby([team_col]).agg(x_mean=(x_col, "mean"), y_mean=(y_col, "mean"))
        #     st.write("xy_mean")
        #     st.write(xy_mean)
        #
        #     df = df.merge(xy_mean, on=team_col, how="left")
        #     # df["x_mean"] = 0
        #     # df["y_mean"] = 0
        #
        #     # df["x_adj"] = df[x_col] - df["x_mean"]
        #     # df["y_adj"] = df[y_col] - df["y_mean"]
        #     # df["x_bin"] = df["x_adj"] // binsize * binsize
        #     # df["y_bin"] = df["y_adj"] // binsize * binsize
        #     # df["xy_bin"] = df["x_bin"].astype(str) + "_" + df["y_bin"].astype(str)
        #
        #     # players = df[player_col].unique()
        #     # df["player_index"] = df[player_col].apply(lambda x: list(players).index(x))
        #     # player2role = {player: player_nr for player_nr, player in enumerate(players)}
        #     # df["role"] = df[player_col].map(player2role)
        #     # roles = df["role"].unique().tolist()
        #     # N = len(roles)
        #     # df = df.sort_values(frame_col)
        #
        #     # plot
        #
        #     plot_roles(df)
        #
        #     def get_FPC(df, rows=frame_col, cols=player_col):
        #         pivoted = df.pivot(index=rows, columns=cols, values=[x_col, y_col])
        #         pivoted = pivoted.sort_index(axis=1)
        #         pivoted = pivoted.fillna(0)
        #         numpy_array = pivoted.to_numpy()
        #         F, P = pivoted.index.size, len(pivoted.columns.levels[1])
        #         numpy_array = numpy_array.reshape(F, P, 2)
        #         return numpy_array
        #
        #     def df2matrix(df):
        #         frames = sorted(df['frame'].unique())
        #         players = sorted(df['player'].unique())
        #         roles = sorted(df['role'].unique())
        #
        #         # Create a mapping for indexing
        #         frame_idx = {frame: i for i, frame in enumerate(frames)}
        #         player_idx = {player: i for i, player in enumerate(players)}
        #         role_idx = {role: i for i, role in enumerate(roles)}
        #
        #         # Initialize an empty array with NaN
        #         result = np.full((len(frames), len(players), len(roles), 2), np.nan)
        #
        #         # Populate the array
        #         for _, row in df.iterrows():
        #             f = frame_idx[row['frame']]
        #             p = player_idx[row['player']]
        #             r = role_idx[row['role']]
        #             result[f, p, r, :] = [row['x'], row['y']]
        #
        #         # Resulting array
        #         return result
        #
        #     PLAYER_ROLES = df.pivot(index=frame_col, columns=player_col, values="role").to_numpy()  # F x P (Ri)
        #
        #     for i in defensive_network.utility.progress_bar(range(1)):
        #         POSITIONS = get_FPC(df)  # F x P x C
        #
        #         cost_function = "distance_to_role_mean"
        #
        #         F_range = list(range(POSITIONS.shape[0]))
        #
        #         ROLE_POSITIONS = np.empty_like(POSITIONS)  # F x P x C
        #         for i in range(POSITIONS.shape[0]):
        #             ROLE_POSITIONS[i, :, :] = POSITIONS[i, PLAYER_ROLES[i], :]  # Reorder second axis for each row
        #
        #         ROLE_MEANS = ROLE_POSITIONS.mean(axis=0)  # R x C
        #
        #         if cost_function == "distance_to_role_mean":
        #             COST_MATRICES = np.linalg.norm(POSITIONS[:, :, np.newaxis, :] - ROLE_MEANS[np.newaxis, np.newaxis, :, :], axis=-1)  # F x P x R
        #         elif cost_function == "entropy":
        #             pass
        #         else:
        #             raise NotImplementedError(f"{cost_function}")
        #
        #         all_optimal_roles = []
        #         for frame in range(COST_MATRICES.shape[0]):
        #             COST_MATRIX = COST_MATRICES[frame, :, :]  # P x R
        #             optimal_players, optimal_roles = scipy.optimize.linear_sum_assignment(COST_MATRIX, maximize=True)  # R
        #             assert list(optimal_players) == list(range(len(optimal_players)))
        #             all_optimal_roles.append(optimal_roles)
        #
        #         ALL_OPTIMAL_ROLES = np.array(all_optimal_roles)  # F x R
        #         PLAYER_ROLES = ALL_OPTIMAL_ROLES  # F x R
        #
        #         df["role"] = df[[frame_col, "player_index"]].apply(lambda x: PLAYER_ROLES[int(x[frame_col]), int(x["player_index"])], axis=1)
        #
        #     df["player_index"] = df[player_col].apply(lambda x: list(players).index(x))
        #     df["role"] = df[[frame_col, "player_index"]].apply(lambda x: PLAYER_ROLES[int(x[frame_col]), int(x["player_index"])], axis=1)
        #     plot_roles(df)
        #
        #     dfs.append(df)
        #
        #     break

#    return dfs


@st.cache_resource
def _read_parquet(fpath):
    return pd.read_parquet(fpath)


if __name__ == '__main__':
    defensive_network.tests.test_passing_network.test_average_positions()
    st.stop()

    df = _read_parquet("C:/Users/Jonas/Downloads/dfl_test_data/2324/preprocessed/tracking/3-liga-2023-2024-20-st-sc-verl-viktoria-koln.parquet")
    # C:\Users\Jonas\Downloads\dfl_test_data\2324\preprocessed\tracking
    df = df.drop(columns=["role", "role_name", "formation_instance"])
    assert "role" not in df.columns
    res = detect_formation(df)
    df["role"] = res.role
    df["role_name"] = res.role_name
    df["formation_instance"] = res.formation_instance

    for formation, df_formation in df.groupby("formation_instance"):
        plot_roles(df_formation)
