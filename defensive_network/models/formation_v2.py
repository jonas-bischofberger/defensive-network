import copy
import sys
import os

import shapely

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.optimize
import collections
import importlib

import matplotlib.pyplot as plt

import defensive_network.utility.general
import defensive_network.utility.pitch
importlib.reload(defensive_network.utility.pitch)
importlib.reload(defensive_network.utility.general)

FormationDetectionResult = collections.namedtuple("FormationDetectionResult", ["formation", "position", "confidence", "n_frames", "ball_in_phase_id", "df_all_role_assignments"])


def gaussian_smooth_time(df, time_col, value_col, sigma_seconds):
    """
    Apply time-aware Gaussian smoothing to a DataFrame (sigma in seconds).

    Parameters:
    - df: pandas DataFrame
    - time_col: str, name of the datetime column
    - value_col: str, name of the numeric value column to smooth
    - sigma_seconds: float, standard deviation of the Gaussian kernel (in seconds)

    Returns:
    - pandas Series with smoothed values (indexed same as input df)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)

    # Convert timestamps to Unix time in seconds
    times = df[time_col].view(np.int64) // 10 ** 9
    values = df[value_col].values

    smoothed_values = []
    for i, t in enumerate(times):
        deltas = times - t  # in seconds
        weights = np.exp(-0.5 * (deltas / sigma_seconds) ** 2)
        weights /= weights.sum()
        smoothed = np.dot(weights, values)
        smoothed_values.append(smoothed)

    return pd.Series(smoothed_values, index=df.index, name=f'{value_col}_smoothed')

import numpy as np

def bounded_voronoi(points, plot):
    mp = shapely.MultiPoint([(x, y) for x, y in zip(points[:, 0], points[:, 1])])

    shapely.voronoi_polygons(mp).normalize()
    polygons = shapely.voronoi_polygons(mp).geoms

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='blue', ec='black')

        # plot points
        plt.scatter(points[:, 0], points[:, 1], color='red', s=10)
        st.write(fig)

        plt.close()

    return polygons


def add_in_play_phase_id(df_tracking, frame_col="full_frame", ball_status_col="ball_status", section_col="section"):
    if "ball_in_play_phase_id" in df_tracking.columns:
        df_tracking = df_tracking.drop(columns=["ball_in_play_phase_id"])
    df_ball_status = df_tracking[[frame_col, ball_status_col, section_col]].drop_duplicates(subset=[frame_col]).copy()
    df_ball_status["ball_status_phase_begins"] = (df_ball_status[ball_status_col].shift(1) != df_ball_status[ball_status_col]) | (df_ball_status[section_col].shift(1) != df_ball_status[section_col])
    df_ball_status["ball_in_play_phase_id"] = (df_ball_status["ball_status_phase_begins"].cumsum() - 1) / 2
    df_ball_status.loc[df_ball_status[ball_status_col] == 0, "ball_in_play_phase_id"] = pd.NA
    frame2id = df_ball_status[[frame_col, "ball_in_play_phase_id"]].set_index(frame_col).to_dict()["ball_in_play_phase_id"]
    return df_tracking[frame_col].map(frame2id)


# @st.cache_resource
def _get_average(
    df_tracking, ball_id="BALL", is_gk_col="is_gk", team_in_possession_col="ball_poss_team_id", team_col="team_id",
    x_norm_col="x_norm", y_norm_col="y_norm", full_frame_col="full_frame", player_col="player_id", plot=False,
):
    # Filter out the ball and goalkeepers
    i_no_ball_no_gk = (df_tracking[player_col] != ball_id) & (df_tracking[is_gk_col] == False)
    # df_tracking = df_tracking[df_tracking[player_col] != ball_id]
    # df_tracking = df_tracking[~df_tracking[is_gk_col]]

    # Normalize coordinates
    df_tracking["is_defending"] = (df_tracking[team_col] != df_tracking[team_in_possession_col]).map({True: 1, False: -1})
    df_tracking["x_norm_formation"] = df_tracking[x_norm_col] * -df_tracking["is_defending"]
    df_tracking["y_norm_formation"] = df_tracking[y_norm_col] * -df_tracking["is_defending"]
    df_centroid = df_tracking.loc[i_no_ball_no_gk].groupby([full_frame_col, "team_id"]).agg(
        x_centroid=("x_norm_formation", "mean"),
        y_centroid=("y_norm_formation", "mean"),
        x_std=("x_norm_formation", "std"),
        y_std=("y_norm_formation", "std"),
    )
    df_tracking = df_tracking.merge(df_centroid, on=[full_frame_col, "team_id"], how="left", suffixes=("", "_centroid"))
    df_tracking["x_norm_formation_z"] = (df_tracking["x_norm_formation"] - df_tracking["x_centroid"]) / df_tracking["x_std"]
    df_tracking["y_norm_formation_z"] = (df_tracking["y_norm_formation"] - df_tracking["y_centroid"]) / df_tracking["y_std"]

    i_no_ball_no_gk = (df_tracking[player_col] != ball_id) & (df_tracking[is_gk_col] == False)
    dft = df_tracking.loc[i_no_ball_no_gk]
    nan_frames = dft.loc[dft["x_norm_formation_z"].isna() | dft["y_norm_formation_z"].isna()][full_frame_col].unique()
    assert df_tracking.loc[i_no_ball_no_gk, "x_norm_formation_z"].notna().all() and df_tracking.loc[i_no_ball_no_gk, "y_norm_formation_z"].notna().all()
    assert len(nan_frames) == 0
    plot = False

    # Detect formation by ball-in phase
    data = []
    for (team_in_possession, ball_in_play_phase_id), df_frame_grouped_team in defensive_network.utility.general.progress_bar(df_tracking.loc[i_no_ball_no_gk].groupby(["team_id", "ball_in_play_phase_id"]), desc="Getting averaging positions", total=len(df_tracking[["team_id", "ball_in_play_phase_id"]].drop_duplicates())):
        # st.write(team_in_possession, ball_in_play_phase_id)
        # for team_nr, (team, df_frame_grouped_team) in enumerate(df_frame_grouped.groupby("team_id")):
        df_average = df_frame_grouped_team.groupby("player_id").agg(
            team_id=(team_col, "first"),
            # team_name=("team_name", "first"),
            # player_name=("player_name", "first"),
            x_norm_formation_z=("x_norm_formation_z", "mean"),
            y_norm_formation_z=("y_norm_formation_z", "mean"),
            n_frames=(full_frame_col, "nunique"),
        ).reset_index()

        df_average["ball_in_play_phase_id"] = ball_in_play_phase_id

        data.append(df_average)
        if plot:
            fig = defensive_network.utility.pitch.plot_tracking_frame(
                df_average, attacking_team=team_in_possession, tracking_x_col="x_norm_formation_z", tracking_y_col="y_norm_formation_z",
                dy_name=0.125,
            )
            plt.plot([-3, 3], [0, 0], color="black", linestyle="--", linewidth=1, alpha=0.3)
            plt.plot([0, 0], [-3, 3], color="black", linestyle="--", linewidth=1, alpha=0.3)
            plt.title(f"Average positions for {team} in ball-in-play phase {ball_in_play_phase_id} (team {team_nr + 1})")

    if plot:
        st.write(fig)
        plt.close()

    df_average = pd.concat(data, ignore_index=True)

    AVERAGE_POSITIONS = df_average[["x_norm_formation_z", "y_norm_formation_z"]].values
    TEMPLATE_POSITIONS = df_positions.set_index("position")[["x", "y"]].values
    positions = df_positions["position"].values

    DISTANCES = scipy.spatial.distance.cdist(TEMPLATE_POSITIONS, AVERAGE_POSITIONS, metric="minkowski", p=2)
    MIN_INDICES = np.argmin(DISTANCES, axis=0)
    CLOSEST_POSITIONS = positions[MIN_INDICES]
    df_average["position"] = CLOSEST_POSITIONS

    st.write("df_average")
    st.write(df_average)

    assert df_tracking.loc[(~df_tracking["is_gk"]) & (df_tracking["player_id"] != "BALL"), "x_norm_formation_z"].notna().all()

    return df_average, df_tracking


positions = {
    # 4-3-1-2, 6.ST SVW - SGE
    "LB": [(-0.5, 1.25)],#, (0.0, 1.5)],#, (-0.25, 1.375)],
    "RB": [(-0.5, -1.25)],#, (0.0, -1.5)],#, (-0.25, -1.375)],
    "LCB-4": [(-1.0, 0.5),],
    "RCB-4": [(-1.0, -0.5),],
    "ZDM": [(-0.25, 0.0),],
    "LZM": [(0.35, 0.75)],#, (0.5, 0.825)],#, (0.25, 0.7825)],
    "RZM": [(0.35, -0.75)],#, (0.5, -0.825)],#, (0.25, -0.7825)],
    "ZOM": [(0.75, 0.0),],
    "LS": [(1.25, 0.5),],
    "RS": [(1.25, -0.5),],

    # 3-5-2, 6.ST SVW - SGE
    "LCB-3": [(-1.0, 0.75),],
    "RCB-3": [(-1.0, -0.75),],
    "CB-3": [(-1.25, 0.0),],
    "LWB": [(0.0, 1.5),],
    "RWB": [(0.0, -1.5),],

    # 4-4-2
    "LDM": [(-0.25, 0.5),],
    "RDM": [(-0.25, -0.5),],
    "LW": [(0.75, 1.5),],
    "RW": [(0.75, -1.5),],

    # 4-2-3-1
    "ST": [(1.25, 0.0),],
}
position_data = [{"position": pos, "x": x, "y": y} for pos in positions.keys() for x, y in positions[pos]]
df_positions = pd.DataFrame.from_dict(position_data)

def plot_template_positions():
    # plot all positions
    plt.figure()
    plt.title("Template positions")
    for pos, locations in positions.items():
        for (x, y) in locations:
            plt.scatter(x, y, label=pos)
            plt.text(x + 0.05, y + 0.05, pos, fontsize=8)
    st.write(plt.gcf())

plot_template_positions()

# df_positions = pd.DataFrame.from_dict(positions, orient="index", columns=["x", "y"]).reset_index()
df_positions["index"] = range(len(df_positions))

templates = {
    "4-3-3": ["LB", "LCB-4", "RCB-4", "RB", "ZDM", "LZM", "RZM", "LW", "RW", "ST"],
    "4-3-1-2": ["LB", "LCB-4", "RCB-4", "RB", "ZDM", "LZM", "RZM", "ZOM", "LS", "RS"],
    "4-2-3-1": ["LB", "LCB-4", "RCB-4", "RB", "LDM", "RDM", "LW", "RW", "ZOM", "ST"],
    "4-4-2": ["LB", "LCB-4", "RCB-4", "RB", "LDM", "RDM", "LW", "RW", "LS", "RS"],
    "3-4-3": ["LCB-3", "RCB-3", "CB-3", "LWB", "RWB", "LDM", "RDM", "LW", "RW", "ST"],
    "3-5-2": ["LCB-3", "RCB-3", "CB-3", "LWB", "RWB", "ZDM", "LZM", "RZM", "LS", "RS"],
    "3-4-1-2": ["LCB-3", "RCB-3", "CB-3", "LWB", "RWB", "LDM", "RDM", "ZOM", "LS", "RS"],

    # 10 players
    "4-4-1": ["LB", "LCB-4", "RCB-4", "RB", "LDM", "RDM", "LW", "RW", "ZOM"],
    "4-3-2": ["LB", "LCB-4", "RCB-4", "RB", "ZDM", "LZM", "RZM", "LS", "RS"],
    "3-4-2": ["LCB-3", "RCB-3", "CB-3", "LWB", "RWB", "LDM", "RDM", "LS", "RS"],

    # 3 players (test)
    # "2-1": ["LCB-4", "RCB-4", "ST"],
    # "1-2": ["CB-3", "LW", "RW"],
}

import pandas as pd
from itertools import product
from typing import List


def generate_position_combinations(formation):
    unique_positions = templates[formation]
    filtered_df = df_positions[df_positions['position'].isin(unique_positions)]
    grouped = filtered_df.groupby('position')

    # Check for missing positions
    missing = [pos for pos in unique_positions if pos not in grouped.groups]
    if missing:
        raise ValueError(f"The following positions are missing in the DataFrame: {missing}")

    rows_per_position = [grouped.get_group(pos).to_dict('records') for pos in unique_positions]
    all_combinations = list(product(*rows_per_position))

    # # Optionally limit the number of combinations
    # if limit is not None:
    #     from random import sample
    #     all_combinations = sample(all_combinations, min(limit, len(all_combinations)))

    # Convert to DataFrames
    combination_dfs = [pd.DataFrame(combo) for combo in all_combinations]

    return combination_dfs


# build F x P matrix where an entry is 1 if the position is in the template, otherwise 0
formation_matrix = pd.DataFrame(0, index=templates.keys(), columns=df_positions["index"].tolist())
for template, positions in templates.items():
    for position in positions:
        formation_matrix.loc[template, position] = 1

# turn df_positions into a P x 2 matrix where the first column is the x position and the second column is the y position
POS = df_positions.set_index("index")[["x", "y"]].values


def plot_formation_segments(df_role_assignment):
    for team_id, df_team in df_role_assignment.groupby("team_id"):
        plt.figure()

        team_name = df_team["team_name"].iloc[0]
        plt.title(f"Formation segments for {team_name}")

        # formations as yticks
        plt.yticks(range(len(templates)), list(templates.keys()), rotation=45)
        plt.ylim(-0.5, len(templates) - 0.5)

        cum_frame = 0
        for phase_id, df_team_phase in df_team.groupby(["ball_in_play_phase_id", "team_id"]):
            start_frame = cum_frame
            d_frame = df_team_phase["n_frames"].iloc[0]
            stop_frame = cum_frame + d_frame
            df_formation = df_team_phase.groupby("formation").agg(
                # n_frames=("n_frames", "sum"),
                # confidence=("confidence", "mean"),
                # distance=("distance", "mean"),
                smoothed_confidence=("smoothed_confidence", "mean"),
                # is_best_formation=("is_best_formation", "any"),
            )
            # best_formation = df_formation["distance"].idxmin()
            best_formation = df_formation["smoothed_confidence"].idxmax()

            # best formation index = index within templates dict
            best_formation_index = list(templates.keys()).index(best_formation)

            colors = ["red", "blue", "green", "yellow", "orange", "purple", "black"] * 3
            # plt.plot([start_frame, stop_frame], [best_formation_index, best_formation_index], label=best_formation, alpha=1, linewidth=30, color=colors[best_formation_index])
            start_time = df_team_phase["time_start"].iloc[0]
            stop_time = df_team_phase["time_end"].iloc[0]
            # plt.fill_between([start_frame, stop_frame], best_formation_index - 0.4, best_formation_index + 0.4,
            plt.fill_between([start_time, stop_time], best_formation_index - 0.4, best_formation_index + 0.4,
                             label=best_formation, alpha=0.5, color=colors[best_formation_index])

            # rotate xticks
            plt.xticks(rotation=90)

            cum_frame += d_frame

        st.write(plt.gcf())
        plt.close()


def detect_formation(df_tracking, show_average_positions=False, plot_phase_by_phase=False, do_formation_segment_plot=False):
    dft_copy = copy.deepcopy(df_tracking)
    dft_copy_index = dft_copy.index
    assert "formation" not in dft_copy.columns

    df_tracking = df_tracking[(df_tracking["player_id"] != "BALL") & (df_tracking["is_gk"] == False)]

    # Get ball in play phases
    df_tracking["ball_in_play_phase_id"] = add_in_play_phase_id(df_tracking)
    dft_copy["ball_in_play_phase_id"] = add_in_play_phase_id(dft_copy)
    assert dft_copy.loc[dft_copy["ball_status"] == 1, "ball_in_play_phase_id"].notna().all(), "All frames must have a ball in play phase ID"
    assert "formation" not in dft_copy.columns

    df_tracking = df_tracking[df_tracking["ball_in_play_phase_id"].notna()]

    with st.spinner("Calculating average positions..."):
        df_average, df_tracking = _get_average(df_tracking)

    assert df_tracking["x_norm_formation_z"].notna().all()

    if show_average_positions:
        dfg = df_average.groupby(["team_id", "player_id"]).agg(
            x_norm_formation_z=("x_norm_formation_z", "mean"), y_norm_formation_z=("y_norm_formation_z", "mean"),
        )

        for team_id, df_team in dfg.groupby(level=["team_id"]):
            # fig = defensive_network.utility.pitch.plot_pitch()
            team_name = df_tracking[df_tracking["team_id"] == team_id]["team_name"].iloc[0]
            fig = defensive_network.utility.pitch.plot_tracking_frame(
                df_team.reset_index(), attacking_team=team_id, tracking_x_col="x_norm_formation_z",
                tracking_y_col="y_norm_formation_z", dy_name=0.125,
            )
            plt.title(f"Average positions for {team_name}")
            st.write(fig)
            plt.close()

    data = []
    i = 0
    for (ball_in_phase_id, team_id), df_team in defensive_network.utility.general.progress_bar(df_average.reset_index().groupby(["ball_in_play_phase_id", "team_id"]), total=len(df_average[["ball_in_play_phase_id", "team_id"]].drop_duplicates()), desc="Detecting formations"):
        i += 1
        # if i != 9:
        #     continue
        # st.write("i", i)
        time_start = df_tracking[df_tracking["ball_in_play_phase_id"] == ball_in_phase_id]["datetime_tracking"].min()
        time_end = df_tracking[df_tracking["ball_in_play_phase_id"] == ball_in_phase_id]["datetime_tracking"].max()

        n_frames = df_team["n_frames"].iloc[0]
        df_player_pos = df_team.set_index("player_id")[["x_norm_formation_z", "y_norm_formation_z"]]
        PLAYER_POS = df_player_pos.values

        role_assignment_data = []

        if False:  # len(df_team) not in [10, 9, 3]:
            raise ValueError(f"Expected 10 players per team, but found {len(df_team)}. This might be due to missing players or goalkeepers in the data.")
            for _, row in df_team.iterrows():
                role_assignment_data.append({
                    "ball_in_play_phase_id": ball_in_phase_id,
                    "team_id": team_id,
                    "formation": "unclear",
                    "player_id": row["player_id"],
                    "position": "unclear",
                    "distance": None,
                    "confidence": None,
                    "n_frames": n_frames,
                    "time_start": time_start,
                    "time_end": time_end,
                })
            df_role_assignment = pd.DataFrame(role_assignment_data)
            continue

        else:
            # linear sum assignment
            assert len(templates) > 0
            possible_templates = {template: positions for template, positions in templates.items() if len(positions) == len(df_team) or len(df_team) >= 11 and len(positions) == 10}
            if len(possible_templates) == 0:
                st.write("df_team", df_team.shape)
                st.write(df_team)
                raise ValueError(f"No suitable template found for team {team_id} with {len(df_team)} players. This might be due to missing players or goalkeepers in the data.")
            for template, positions in possible_templates.items():
                unique_positions = set(positions)
                if len(df_team) <= 10 and len(unique_positions) != len(df_team):
                    continue

                dfs_combination = generate_position_combinations(template)
                assert len(dfs_combination) > 0, f"No position combinations found for template {template} with positions {positions}"
                for df_positions_template in dfs_combination:
                    df_positions_template = df_positions_template.reset_index()
                    POS_TEMPLATE = df_positions_template[["x", "y"]].values
                    distance_mode = "distance"

                    if distance_mode == "distance":
                        DISTANCE_TEMPLATE = scipy.spatial.distance.cdist(POS_TEMPLATE, PLAYER_POS, metric="minkowski", p=2)

                    elif distance_mode == "voronoi":
                        from scipy.spatial import Voronoi
                        from shapely.geometry import Point, Polygon

                        # @st.cache_resource
                        def _get_voronoi(POS_TEMPLATE):
                            return bounded_voronoi(POS_TEMPLATE, False)#, bounds=[(-200, -200), (200, -200), (200, 200), (-200, 200)], radius=1000)

                        voronoi_cells = _get_voronoi(POS_TEMPLATE)

                        # Binary distance matrix: 0 if player is inside the cell, 1 otherwise
                        DISTANCE_TEMPLATE = np.ones((len(POS_TEMPLATE), len(PLAYER_POS)))

                        for i, cell in enumerate(voronoi_cells):
                            if cell is None:
                                continue  # skip unbounded cell, leave distances as 1
                            for j, player_pos in enumerate(PLAYER_POS):
                                point = Point(player_pos)
                                if cell.contains(point):
                                    DISTANCE_TEMPLATE[i, j] = 0

                        # plot it
                        if plot_phase_by_phase:
                            fig = plt.figure(figsize=(10, 6))
                            for i, cell in enumerate(voronoi_cells):
                                if cell is not None:
                                    x, y = cell.exterior.xy
                                    plt.fill(x, y, alpha=0.5, fc='lightblue', ec='black', label=f"Cell {i}")
                            plt.scatter(POS_TEMPLATE[:, 0], POS_TEMPLATE[:, 1], color='red', label='Template Positions')
                            plt.scatter(PLAYER_POS[:, 0], PLAYER_POS[:, 1], color='green', label='Player Positions')
                            plt.title(f"Voronoi Cells for {template}")
                            plt.xlim(-2.5, 2.5)
                            plt.ylim(-2.5, 2.5)
                            plt.legend()
                            st.write(fig)
                            plt.close()

                        DISTANCE_TEMPLATE += 0.1 * scipy.spatial.distance.cdist(POS_TEMPLATE, PLAYER_POS, metric="euclidean")

                        # st.write("DISTANCE_TEMPLATE")
                        # st.write(DISTANCE_TEMPLATE.shape)

                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(DISTANCE_TEMPLATE)

                    template_distance = DISTANCE_TEMPLATE[row_ind, col_ind].mean()
                    for ri, ci in zip(row_ind, col_ind):
                        confidence = 1 - DISTANCE_TEMPLATE[ri, ci]
                        role_assignment_data.append({
                            "ball_in_play_phase_id": ball_in_phase_id,
                            "team_id": team_id,
                            "formation": template,
                            "player_id": df_team.iloc[ci]["player_id"],
                            "position": df_positions_template.iloc[ri]["position"],
                            "distance": DISTANCE_TEMPLATE[ri, ci],
                            "confidence": confidence,
                            "n_frames": n_frames,
                            "time_start": time_start,
                            "time_end": time_end,
                        })

                    confidence = 1 - template_distance

            df_role_assignment = pd.DataFrame(role_assignment_data)

            assert len(df_role_assignment) > 0, "No role assignments found. This might be due to missing players or goalkeepers in the data."
            assert df_role_assignment["player_id"].notna().all(), "Not all players have a role assigned. Check the data and the role assignment."
            assert df_role_assignment["position"].notna().all(), "Not all players have a position assigned. Check the data and the role assignment."
            assert df_role_assignment["formation"].notna().all(), "Not all players have a formation assigned. Check the data and the role assignment."

        data.append(df_role_assignment)

    with st.spinner("Combining role assignments..."):
        df_role_assignment = pd.concat(data)
        df_role_assignment = df_role_assignment.merge(df_tracking[["player_id", "team_id", "player_name", "team_name"]].drop_duplicates(), on=["player_id", "team_id"], how="left")
        df_role_assignment["time_middle"] = df_role_assignment["time_start"] + (df_role_assignment["time_end"] - df_role_assignment["time_start"]) / 2
        df_role_assignment["weighted_confidence"] = df_role_assignment["confidence"] * df_role_assignment["n_frames"]

    # apply gaussian smoothing using time_middle
    # df_role_assignment_indexed = df_role_assignment.set_index("time_middle")
    with st.spinner("Applying Gaussian smoothing..."):
        n_total = len(df_role_assignment[["formation", "team_id"]].drop_duplicates())
        for (formation, team_id), df_formation in defensive_network.utility.general.progress_bar(df_role_assignment.groupby(["formation", "team_id"]), desc="Smoothing role assignments", total=n_total):
            df_formation["smoothed_confidence"] = gaussian_smooth_time(df_formation, time_col="time_middle", value_col="weighted_confidence", sigma_seconds=60*7.5)
            i_ra = (df_role_assignment["formation"] == formation) & (df_role_assignment["team_id"] == team_id)
            df_role_assignment.loc[i_ra, "smoothed_confidence"] = df_formation["smoothed_confidence"]

    if "is_best_formation" in df_role_assignment.columns:
        df_role_assignment = df_role_assignment.drop(columns=["is_best_formation"])
    df_role_assignment["is_best_formation"] = False
    df_role_assignment["is_best_formation"] = df_role_assignment.groupby(["ball_in_play_phase_id", "team_id"])["smoothed_confidence"].transform("max") == df_role_assignment["smoothed_confidence"]

    df_role_assignment_best = df_role_assignment[df_role_assignment["is_best_formation"]]

    df_tracking = df_tracking.merge(df_role_assignment_best.drop(columns=["team_name", "player_name", "formation"]), on=["ball_in_play_phase_id", "team_id", "player_id"], how="left")

    ### Its ok to have NaN values, e.g. when more than 11 players are on the pitch
    # i_ball_status_1 = (df_tracking["ball_status"] == 1) & (~df_tracking["is_gk"]) & (df_tracking["player_id"] != "BALL")
    # assert df_tracking.loc[i_ball_status_1, "formation"].notna().all(), "Not all formations could be detected. Check the data and the role assignment."

    if do_formation_segment_plot:
        plot_formation_segments(df_role_assignment)

    i_ball = df_tracking["player_id"] == "BALL"
    i_gk_or_ball = df_tracking["is_gk"] | (df_tracking["player_id"] == "BALL")
    i_gk = df_tracking["is_gk"]

    df_tracking.loc[df_tracking["is_gk"], "position"] = "GK"

    # fill in formation info for goalkeepers
    dfg_formation = df_role_assignment_best.groupby(["ball_in_play_phase_id", "team_id"]).agg(
        formation=("formation", "first"),
    )
    df_tracking = df_tracking.merge(dfg_formation, on=["ball_in_play_phase_id", "team_id"], how="left")
    # dft_copy = dft_copy.merge(dfg_formation, on=["ball_in_play_phase_id", "team_id""], how="left")

    # assign remaining nan positions with nearest position
    i_pos_nan = df_tracking["position"].isna() & ~df_tracking["is_gk"] & ~i_ball

    # assert len(df_tracking.loc[i_pos_nan]) < 50000
    remaining_pos = []
    if i_pos_nan.any():
        dfg_grouped = df_tracking.loc[i_pos_nan].groupby(["ball_in_play_phase_id", "team_id", "player_id"]).agg(
            x_norm_formation_z=("x_norm_formation_z", "mean"),
            y_norm_formation_z=("y_norm_formation_z", "mean"),
        )
        for (ball_in_phase_id, team_id), df_tracking_grouped in defensive_network.utility.general.progress_bar(dfg_grouped.reset_index().groupby(["ball_in_play_phase_id", "team_id"]), desc="Assigning remaining positions", total=len(dfg_grouped)):
            def foo(row):
                # find the nearest position in df_positions
                distances = ((df_positions["x"] - row["x_norm_formation_z"])**2 + (df_positions["y"] - row["y_norm_formation_z"])**2)
                nearest_index = distances.idxmin()

                return df_positions.iloc[nearest_index]["position"]

            df_tracking_grouped["position"] = df_tracking_grouped.apply(foo, axis=1)

            for _, row in df_tracking_grouped.iterrows():
                # append to remaining_pos
                remaining_pos.append({
                    "ball_in_play_phase_id": row["ball_in_play_phase_id"],
                    "team_id": row["team_id"],
                    "player_id": row["player_id"],
                    "position": row["position"],
                })

        # assert df_tracking.loc[i_pos_nan, "position"].notna().all(), "Not all positions could be assigned. Check the data and the role assignment."

    # st.write(pd.DataFrame(remaining_pos))
    # df_remaining_pos = pd.DataFrame(remaining_pos).set_index(["ball_in_play_phase_id", "team_id", "player_id"]).drop_duplicates()
    # assert df_tracking.loc[~i_ball, "position"].notna().all()i_nan = ~i_ball & (df_tracking["formation"].isna() | df_tracking["position
    # assert dft_copy.loc[i_gk, "formation"].notna().all(), "Not all goalkeepers have a formation assigned. Check the data and the role assignment."
    # assert df_tracking.loc[(~df_tracking["is_gk"]) & (df_tracking["player_id"] != "BALL"), "x_norm_formation_z"].notna().all()

    # merge information back to dft_copy
    # df_role_assignment_best_dict =

    assert "formation" not in dft_copy.columns

    dfg_formation = df_role_assignment_best.groupby(["ball_in_play_phase_id", "team_id"]).agg(
        formation=("formation", "first"),
    )
    formation_mapping = dfg_formation["formation"].to_dict()
    dft_copy["formation"] = dft_copy.apply(lambda row: formation_mapping.get((row["ball_in_play_phase_id"], row["team_id"]), None), axis=1)

    position_mapping = df_average.set_index(["ball_in_play_phase_id", "team_id", "player_id"])["position"].to_dict()
    dft_copy["position"] = dft_copy.apply(lambda row: position_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), None), axis=1)
    assert dft_copy["position"].notna().any()
    # st.write("position_mapping")
    # st.write(position_mapping)
    # ixxx = (dft_copy["ball_in_play_phase_id"] == 80.5) & (dft_copy["team_id"] == "909b58975c636de982a6e9db835944ca") & (dft_copy["player_id"] == "770f7ea7a742740313620742d5b7bce3")
    # st.write(dft_copy.loc[ixxx])
    # assert dft_copy.loc[ixxx, "position"].notna().all()
 # 909b58975c636de982a6e9db835944ca	770f7ea7a742740313620742d5b7bce3
    # 909b58975c636de982a6e9db835944ca	770f7ea7a742740313620742d5b7bce3
    # assert "formation" in dft_copy.columns
    # df_tracking = df_tracking.merge(dfg_formation, on=["ball_in_play_phase_id", "team_id""], how="left")
    # dft_copy.loc[dft_copy["player_id"] != "BALL""] = (dft_copy.loc[dft_copy["player_id"] != "BALL", "team_id"] != dft_copy.loc[dft_copy["player_id"] != "BALL", "ball_poss_team_id"])
    # dft_copy = dft_copy.merge(df_role_assignment_best.drop(columns=["team_name", "player_name", "formation"]), on=["ball_in_play_phase_id", "team_id", "player_id"], how="left")

    for (ball_in_phase_id, team_id), formation in defensive_network.utility.general.progress_bar(dfg_formation["formation"].items(), desc="Assigning positions to players", total=len(dfg_formation)):
        df_average_phase_team = df_average[(df_average["ball_in_play_phase_id"] == ball_in_phase_id) & (df_average["team_id"] == team_id)]

        df_formation_positions = df_positions[df_positions["position"].isin(templates[formation])]
        # find optimal assignment of positions to players based on the template using the  Hungarian algorithm
        POS_TEMPLATE = df_formation_positions[["x", "y"]].values
        PLAYER_POS = df_average_phase_team[["x_norm_formation_z", "y_norm_formation_z"]].values
        DISTANCE_TEMPLATE = scipy.spatial.distance.cdist(POS_TEMPLATE, PLAYER_POS, metric="minkowski", p=2)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(DISTANCE_TEMPLATE)
        player2positions = {df_average_phase_team["player_id"].iloc[ci]: df_formation_positions["position"].iloc[ri] for ri, ci in zip(row_ind, col_ind)}

        # st.write("player2positions")
        # st.write(player2positions)

        # assign positions to players in dft_copy
        i_phase_team = (dft_copy["ball_in_play_phase_id"] == ball_in_phase_id) & (dft_copy["team_id"] == team_id) & (dft_copy["player_id"] != "BALL") & (~dft_copy["is_gk"])
        dft_copy.loc[i_phase_team, "position_2"] = dft_copy.loc[i_phase_team, "player_id"].map(player2positions)
        dft_copy.loc[i_phase_team, "position"] = dft_copy.loc[i_phase_team, "position_2"].where(dft_copy.loc[i_phase_team, "position_2"].notna(), dft_copy.loc[i_phase_team, "position"])

        # df_not_assigned = dft_copy.loc[i_phase_team & dft_copy["position"].isna()]
        # st.write("df_not_assigned")
        # st.write(df_not_assigned)
        # st.write("df_average")
        # st.write(df_average)

        # df_not_assigned = df_not_assigned.merge(df_average, on=["ball_in_play_phase_id", "team_id", "player_id", "position"], how="left")
        # st.write("df_not_assigned")
        # st.write(df_not_assigned)
        #
        # dft_copy["position"] = dft_copy["position"].merge()
        #
        # st.stop()

        # if len(df_not_assigned) > 0:
            # NOT_ASSIGNED_POS = df_not_assigned[["x_norm_formation_z", "y_norm_formation_z"]].values
            # DISTANCES = ((POS_TEMPLATE - NOT_ASSIGNED_POS[:, None])**2).sum(axis=2)
            # nearest_indices = DISTANCES.argmin(axis=1)
            # nearest_positions = df_formation_positions["position"].iloc[nearest_indices].values
            #
            # # Create a mask for the current phase and team
            # mask = i_phase_team & dft_copy["player_id"].isin(df_not_assigned["player_id"])
            #
            # # Build a DataFrame for merging on player_id
            # assignments = pd.DataFrame({
            #     "player_id": df_not_assigned["player_id"].values,
            #     "position": nearest_positions
            # })
            #
            # # Update dft_copy using a left join
            # dft_copy = dft_copy.merge(assignments, on="player_id", how="left", suffixes=('', '_new'))
            # dft_copy.loc[mask, "position"] = dft_copy.loc[mask, "position_new"]
            # dft_copy = dft_copy.drop(columns=["position_new"])

            # st.write(f"Check {ball_in_phase_id=} {team_id=}")

            # for player, position in zip(df_not_assigned["player_id"], nearest_positions):
            #     dft_copy.loc[i_phase_team & (dft_copy["player_id"] == player), "position"] = position
            #
            #     st.write(f"Check {ball_in_phase_id=} {team_id=} {row['player_id']=} {position=}")

        # for _, row in defensive_network.utility.general.progress_bar(df_not_assigned.iterrows(), total=len(df_not_assigned), desc="Assigning nearest position to players without position"):
        #     # if a player is not assigned a position, assign the nearest position from the template
        #     df_average_row = df_average_phase_team[df_average_phase_team["player_id"] == row["player_id"]]
        #     average_x = df_average_row["x_norm_formation_z"].values[0]
        #     average_y = df_average_row["y_norm_formation_z"].values[0]
        #     distances = ((POS_TEMPLATE - np.array([average_x, average_y]))**2).sum(axis=1)
        #     nearest_index = distances.argmin()
        #     nearest_position = df_formation_positions["position"].iloc[nearest_index]
        #     dft_copy.loc[i_phase_team & (dft_copy["player_id"] == row["player_id"]), "position"] = nearest_position
        #
        #     ball_in_phase_id = row["ball_in_play_phase_id"]
        #     team_id = row["team_id"]
        #
        #     st.write(f"Check {ball_in_phase_id=} {team_id=} {row['player_id']=} {nearest_position=}")

        # st.write(dft_copy.loc[i_phase_team, ["ball_in_play_phase_id", "team_id", "player_id", "position"]])

        assert dft_copy.loc[i_phase_team, "position"].notna().all()


    dft_copy = dft_copy.drop(columns=["position_2"])


    # assign_manually = False
    # if assign_manually:
    #     with st.spinner("Assigning positions..."):
    #         for (ball_in_phase_id, team_id, player_id), _ in df_role_assignment_best.set_index(["ball_in_play_phase_id", "team_id", "player_id"]).iterrows():
    #             for col in ["position", "smoothed_confidence", "n_frames"]:
    #                 dft_copy.loc[
    #                     (dft_copy["ball_in_play_phase_id"] == ball_in_phase_id) &
    #                     (dft_copy["team_id"] == team_id) &
    #                     (dft_copy["player_id"] == player_id),
    #                     col
    #                 ] = df_role_assignment_best.loc[
    #                     (df_role_assignment_best["ball_in_play_phase_id"] == ball_in_phase_id) &
    #                     (df_role_assignment_best["team_id"] == team_id) &
    #                     (df_role_assignment_best["player_id"] == player_id),
    #                     col
    #                 ].values[0]
    # else:
    #     assert (dft_copy.index == dft_copy_index).all(), "Index of dft_copy has changed. This might be due to a merge operation that changed the index."
    #     dft_copy["_row_id_temp"] = dft_copy.index
    #     dft_copy = dft_copy.merge(df_role_assignment_best.drop(columns=["team_name", "player_name", "formation"]).drop_duplicates(["ball_in_play_phase_id", "team_id", "player_id"]), on=["ball_in_play_phase_id", "team_id", "player_id"], how="left")
    #     dft_copy = dft_copy.set_index("_row_id_temp")
    #     assert len(dft_copy.index) == len(dft_copy_index), "Merge changed the number of rows."
    #     assert (dft_copy.index == dft_copy_index).all(), "Index of dft_copy has changed. This might be due to a merge operation that changed the index."

    # dft_copy["position"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("position", None), axis=1)
    # dft_copy["smoothed_confidence"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("smoothed_confidence", None), axis=1)
    # dft_copy["n_frames"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("n_frames", None), axis=1)
    # dft_copy["position"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("position", None), axis=1)
    # dft_copy["smoothed_confidence"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("smoothed_confidence", None), axis=1)
    # dft_copy["n_frames"] = dft_copy.apply(lambda row: role_assignment_mapping.get((row["ball_in_play_phase_id"], row["team_id"], row["player_id"]), {}).get("n_frames", None), axis=1)

    i_gk = dft_copy["is_gk"]
    dft_copy.loc[i_gk, "position"] = "GK"

    assert "formation" in dft_copy.columns

    # add remaining positions
    # with st.spinner("Assigning remaining positions..."):
    #     st.write("df_remaining_pos")
    #     st.write(df_remaining_pos)
    #     assign_manually = True
    #     if assign_manually:
    #         for (ball_in_play_phase_id, team_id, player_id), row in df_remaining_pos.iterrows():
    #             dft_copy.loc[(dft_copy["ball_in_play_phase_id"] == ball_in_play_phase_id) &
    #                         (dft_copy["team_id"] == team_id) &
    #                         (dft_copy["player_id"] == player_id), "position"] = row["position"]
    #     else:
    #         dft_copy["_row_id_temp"] = dft_copy.index
    #         dft_copy = dft_copy.merge(df_remaining_pos.reset_index(), on=["ball_in_play_phase_id", "team_id", "player_id"], how="left")
    #         dft_copy["position"] = dft_copy["position_x"].fillna(dft_copy["position_y"])
    #         dft_copy = dft_copy.drop(columns=["position_x", "position_y"])
    #         dft_copy = dft_copy.set_index("_row_id_temp")

    assert (dft_copy_index == dft_copy.index).all()

    # res = FormationDetectionResult(dft_copy["formation"], dft_copy["position"], dft_copy["smoothed_confidence"], dft_copy["n_frames"], dft_copy["ball_in_play_phase_id"], df_role_assignment)
    res = FormationDetectionResult(dft_copy["formation"], dft_copy["position"], None, None, dft_copy["ball_in_play_phase_id"], df_role_assignment)

    i_nofo = (dft_copy["player_id"] != "BALL") & (dft_copy["ball_status"] == 1)
    assert dft_copy.loc[i_nofo, "formation"].notna().all(), "Not all formations could be detected. Check the data and the role assignment."
    assert dft_copy.loc[i_nofo, "position"].notna().all(), "Not all positions could be assigned. Check the data and the role assignment."
    assert dft_copy.loc[dft_copy["ball_status"] == 1, "ball_in_play_phase_id"].notna().all(), "Not all formations could be detected. Check the data and the role assignment."

    # st.write(dft_copy[dft_copy["ball_in_play_phase_id"] == 4])

    return res


@st.cache_resource
def _get_test_data(slugified_match_string, frames=25000):
    df_tracking = pd.read_parquet(
        f"Y:/w_raw/finalized/tracking/{slugified_match_string}.parquet",
        filters=[("frame", "<", frames)],
        engine="pyarrow"
    )

    # this will be moved to preprocessing
    match_id = df_tracking["match_id"].iloc[0]
    df_lineups = pd.read_csv("Y:/w_raw/lineups.csv")
    df_lineups = df_lineups[df_lineups["match_id"] == match_id]
    df_lineups["is_gk"] = df_lineups["position"] == "GK"
    player2isgk = df_lineups.set_index("player_id")["is_gk"].to_dict()
    team2teamname = df_lineups[df_lineups["team_name"].notna()].set_index("team_id")["team_name"].to_dict()
    df_tracking["team_name"] = df_tracking["team_id"].map(team2teamname)
    df_tracking["is_gk"] = df_tracking["player_id"].map(player2isgk).fillna(False).astype(bool)

    return df_tracking


def main():
    slugified_match_string = "bundesliga-2023-2024-21-st-sgs-essen-1-fc-koln"
    df_tracking = _get_test_data(slugified_match_string, 1000000000)
    # df_tracking = _get_test_data(slugified_match_string, 10000)
    try:
        df_tracking = df_tracking.drop(columns=["formation"])
        df_tracking = df_tracking.drop(columns=["position"])
    except KeyError:
        pass

    # from defensive_network.tests.data import df_tracking, df_events
    # df_tracking = df_tracking.copy()
    # df_tracking["full_frame"] = df_tracking["frame_id"]
    # df_tracking["section"] = "first_half"
    # df_tracking["ball_status"] = 1
    # df_tracking["is_gk"] = False
    # df_tracking["ball_poss_team_id"] = "H"
    # df_tracking["x_norm"] = df_tracking["x_tracking"]
    # df_tracking["y_norm"] = df_tracking["y_tracking"]
    # df_tracking["frame"] = df_tracking["frame_id"]
    # df_tracking["team_name"] = df_tracking["team_id"]
    # df_tracking["datetime_tracking"] = pd.to_datetime(df_tracking["frame_id"], unit="s", origin="unix")

    res = detect_formation(df_tracking)
    df_tracking["formation"] = res.formation
    df_tracking["position"] = res.position
    df_tracking["smoothed_confidence"] = res.confidence
    df_tracking["n_frames"] = res.n_frames
    df_tracking["ball_in_play_phase_id"] = res.ball_in_phase_id

    i_nofo = (df_tracking["player_id"] != "BALL") & (df_tracking["ball_status"] == 1)
    assert df_tracking.loc[i_nofo, "formation"].notna().all(), "Not all formations could be detected. Check the data and the role assignment."
    assert df_tracking.loc[i_nofo, "position"].notna().all(), "Not all positions could be assigned. Check the data and the role assignment."
    assert df_tracking.loc[df_tracking["ball_status"] == 1, "ball_in_play_phase_id"].notna().all(), "Not all formations could be detected. Check the data and the role assignment."

    plot_formation_segments(res.df_all_role_assignments)
    del res
    # df_tracking["formation_confidence"] = res.confidence
    # df_tracking["ball_on_player_phase_n_frames"] = res.n_frames
    # df_tracking["ball_in_play_phase_id"] = res.ball_in_phase_id

    def plot_formations(df_tracking):
        df_tracking = df_tracking[df_tracking["player_id"] != "BALL"]

        df_averages = df_tracking.groupby(["ball_in_play_phase_id", "team_id", "player_id"]).agg(
            x_tracking=("x_tracking", "mean"),
            y_tracking=("y_tracking", "mean"),
            position=("position", "first"),
            formation=("formation", "first"),
            player_name=("player_name", "first"),
        ).reset_index()

        cols = st.columns(3)
        for (ball_phase_id, team_id), df_team in df_averages.groupby(["ball_in_play_phase_id", "team_id"]):

            fig = defensive_network.utility.pitch.plot_tracking_frame(
                df_team, attacking_team=team_id, tracking_x_col="x_tracking", tracking_y_col="y_tracking",
                tracking_player_name_col="position",
            )
            plt.title(f"Formations for {team_id} in phase {ball_phase_id}: {df_team['formation'].iloc[0]}")
            cols[0].write(fig)
            plt.close()
            fig = defensive_network.utility.pitch.plot_tracking_frame(
                df_team, attacking_team=team_id, tracking_x_col="x_tracking", tracking_y_col="y_tracking",
                tracking_player_name_col="player_id",
            )
            plt.title(f"Formations for {team_id} in phase {ball_phase_id}: {df_team['formation'].iloc[0]}")
            cols[1].write(fig)
            plt.close()
            fig = defensive_network.utility.pitch.plot_tracking_frame(
                df_team, attacking_team=team_id, tracking_x_col="x_tracking", tracking_y_col="y_tracking",
                tracking_player_name_col="player_name",
            )
            plt.title(f"Formations for {team_id} in phase {ball_phase_id}: {df_team['formation'].iloc[0]}")
            cols[2].write(fig)
            plt.close()

    plot_formations(df_tracking)


if __name__ == '__main__':
    main()
