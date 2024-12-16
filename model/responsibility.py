import importlib
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
# import accessible_space  # use to infer attacking direction e.g.
import accessible_space
import matplotlib.pyplot as plt

import warnings

import streamlit_profiler

warnings.filterwarnings("ignore")  # hack to suppress weird streamlit bug, see: https://discuss.streamlit.io/t/keyerror-warnings/2474/14

importlib.reload(accessible_space)


def plot_pass(  # todo move to accessible space
    p4ss, df_tracking,
    pass_x_col="x_event", pass_y_col="y_event", pass_end_x_col="x_target", pass_end_y_col="y_target",
    pass_frame_col="full_frame", pass_team_col="team_id_1", pass_player_name_col="player_name_1",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x_tracking",
    tracking_y_col="y_tracking", tracking_frame_col="full_frame", tracking_player_name_col="player_name",
    tracking_vx_col=None, tracking_vy_col=None, ball_tracking_player_id="BALL", plot_defenders=True,
    plot_expected_receiver=True,
):

    import accessible_space.interface
    # if ball_tracking_player_id not in df_tracking[tracking_player_col]:
    #     return
    accessible_space.interface._check_ball_in_tracking_data(df_tracking, tracking_player_col, ball_tracking_player_id)

    plt.figure()
    # plot penalty boxse
    # left penalty box
    y_box = 16.5 + 7.32 / 2
    x0 = -52.5
    x_box = -52.5+16.5
    plt.plot([x0, x_box], [y_box, y_box], color='grey')
    plt.plot([x_box, x_box], [-y_box, y_box], color='grey')
    plt.plot([x0, x_box], [-y_box, -y_box], color='grey')

    # right penalty box
    x0 = 52.5
    x_box = 52.5-16.5
    plt.plot([x0, x_box], [y_box, y_box], color='grey')
    plt.plot([x_box, x_box], [-y_box, y_box], color='grey')
    plt.plot([x0, x_box], [-y_box, -y_box], color='grey')


    df_frame = df_tracking[df_tracking[tracking_frame_col] == p4ss[pass_frame_col]]

    df_frame_without_ball = df_frame[df_frame[tracking_player_col] != ball_tracking_player_id]

    factor=1

    for team, df_frame_team in df_frame_without_ball.groupby(tracking_team_col):
        is_defending_team = team != p4ss[pass_team_col]
        if is_defending_team and not plot_defenders:
            continue
        x = df_frame_team[tracking_x_col].tolist()
        y = df_frame_team[tracking_y_col].tolist()
        color = "red" if not is_defending_team else "blue"
        plt.scatter(x, y, c=color)

        if tracking_vx_col is not None and tracking_vy_col is not None:
            vx = df_frame_team["vx"].tolist()
            vy = df_frame_team["vy"].tolist()
            for i in range(len(x)):
                plt.arrow(x=x[i], y=y[i], dx=vx[i] / 5, dy=vy[i] / 5, head_width=0.5*factor, head_length=0.5*factor, fc="black", ec="black")

        if tracking_player_name_col is not None:
            for i, txt in enumerate(df_frame_team[tracking_player_name_col]):
                plt.annotate(txt, (x[i], y[i]-2.25), fontsize=5*factor, ha="center", va="center", color=color)

    # plot passing start point with colored X
    plt.scatter(p4ss[pass_x_col], p4ss[pass_y_col], c="red", marker="x", s=30*factor)

    # plot ball position
    try:
        df_frame_ball = df_frame[df_frame[tracking_player_col] == ball_tracking_player_id]
        assert len(df_frame_ball) == 1, f"Expected exactly one ball position, got {len(df_frame_ball)}"  # sanity check
        x_ball = df_frame_ball[tracking_x_col].iloc[0]
        y_ball = df_frame_ball[tracking_y_col].iloc[0]
        plt.scatter(x_ball, y_ball, c="black", marker="x", s=50*factor)
    except AssertionError as e:
        st.write(e)

    if plot_expected_receiver:
        expected_receiver = p4ss["expected_receiver"]
        if not pd.isna(expected_receiver):
            df_tracking_expected_receiver = df_frame[df_frame[tracking_player_col] == expected_receiver]
            assert len(df_tracking_expected_receiver) > 0
            x = df_tracking_expected_receiver[tracking_x_col].iloc[0]
            y = df_tracking_expected_receiver[tracking_y_col].iloc[0]
            plt.scatter(x, y, c="yellow", marker="x", s=25*factor, label="expected receiver")
            plt.legend()

    plt.plot([-52.5, 52.5], [-34, -34], c="grey")
    plt.plot([-52.5, 52.5], [34, 34], c="grey")
    plt.plot([-52.5, -52.5], [-34, 34], c="grey")
    plt.plot([52.5, 52.5], [-34, 34], c="grey")
    plt.axis("equal")

    # plot middle circle
    circle = plt.Circle((0, 0), 9.15, color='grey', fill=False)
    plt.gca().add_artist(circle)

    # plot middle line
    plt.plot([0, 0], [-34, 34], color='grey')

    # remove axes and set xlim and ylim
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xlim(-52.5-5, 52.5+5)
    plt.ylim(-34-5, 34+5)

    # plot pass arrow
    plt.arrow(x=p4ss[pass_x_col], y=p4ss[pass_y_col], dx=p4ss[pass_end_x_col] - p4ss[pass_x_col],
              dy=p4ss[pass_end_y_col] - p4ss[pass_y_col], head_width=2*factor, head_length=3*factor, fc="black", ec="black")

    return plt.gcf()


def plot_pass_involvement(  # todo move to accessible space
    p4ss, df_involvement, df_tracking,
    pass_x_col="x_event", pass_y_col="y_event", pass_end_x_col="x_target", pass_end_y_col="y_target",
    pass_frame_col="full_frame", pass_team_col="team_id_1", pass_player_name_col="player_name_1",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x_tracking",
    tracking_y_col="y_tracking", tracking_frame_col="full_frame", tracking_player_name_col="player_name",
    tracking_vx_col=None, tracking_vy_col=None, ball_tracking_player_id="BALL",
    plot_model="circle_circle_rectangle", plot_expected_receiver=True, model_radius=5,
):
    fig = plot_pass(p4ss, df_tracking, plot_defenders=False, plot_expected_receiver=plot_expected_receiver)
    for _, row in df_involvement.iterrows():
        involvement = row["involvement"]

        # scale involvement from 0.1 to 1.0
        alpha_lower_threshold = 0.05
        alpha = alpha_lower_threshold + (1-alpha_lower_threshold) * involvement

        try:
            plt.scatter(row["defender.x_tracking"], row["defender.y_tracking"], c="blue", marker="o", s=50, alpha=alpha)
        except ValueError as e:
            st.write(e)

        # add number to involvement
        plt.annotate(f"{involvement:.2f}", (row["defender.x_tracking"], row["defender.y_tracking"]), fontsize=3, ha="center", va="center", color="black")

        # add defender name
        plt.annotate(row["defender.player_name"], (row["defender.x_tracking"], row["defender.y_tracking"]-2.25), fontsize=5, ha="center", va="center", color="blue")

    def _plot_passer_circle(model_radius):
        circle = plt.Circle((p4ss[pass_x_col], p4ss[pass_y_col]), model_radius, color='blue', fill=False)
        fig.gca().add_artist(circle)

    def _plot_receiver_circle(_model_radius):
        circle = plt.Circle((p4ss[pass_end_x_col], p4ss[pass_end_y_col]), _model_radius, color='blue', fill=False)
        fig.gca().add_artist(circle)

    if plot_model == "circle_passer":
        _plot_passer_circle(model_radius)

    if plot_model == "circle_receiver":
        _plot_receiver_circle(model_radius)

    if plot_model == "circle_circle_rectangle":
        _plot_passer_circle(model_radius)
        _plot_receiver_circle(model_radius)

        # plot line segment from passer to receiver (but 5m to the right)
        perpendicular_vec = np.array([p4ss[pass_end_y_col] - p4ss[pass_y_col], p4ss[pass_x_col] - p4ss[pass_end_x_col]])
        perpendicular_vec = perpendicular_vec / np.linalg.norm(perpendicular_vec) * model_radius
        plt.plot([p4ss[pass_x_col] + perpendicular_vec[0], p4ss[pass_end_x_col] + perpendicular_vec[0]], [p4ss[pass_y_col] + perpendicular_vec[1], p4ss[pass_end_y_col] + perpendicular_vec[1]], color='blue')

        # 5m to left
        perpendicular_vec = np.array([p4ss[pass_end_y_col] - p4ss[pass_y_col], p4ss[pass_x_col] - p4ss[pass_end_x_col]])
        perpendicular_vec = perpendicular_vec / np.linalg.norm(perpendicular_vec) * model_radius
        plt.plot([p4ss[pass_x_col] - perpendicular_vec[0], p4ss[pass_end_x_col] - perpendicular_vec[0]], [p4ss[pass_y_col] - perpendicular_vec[1], p4ss[pass_end_y_col] - perpendicular_vec[1]], color='blue')

    return fig


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


BASE_PATH = "C:/Users/Jonas/Downloads/dfl_test_data/2324"
lineup_fpath = os.path.abspath(os.path.join(BASE_PATH, "lineup.csv"))
meta_fpath = os.path.abspath(os.path.join(BASE_PATH, "meta.csv"))
event_path = os.path.abspath(os.path.join(BASE_PATH, "events"))
tracking_path = os.path.abspath(os.path.join(BASE_PATH, "tracking"))


@st.cache_resource
def _get_all_meta():
    return pd.read_csv(meta_fpath)


@st.cache_resource
def _get_all_lineups():
    return pd.read_csv(lineup_fpath)



def _select_matches():
    df_meta = _get_all_meta()
    df_lineups = _get_all_lineups()
    all_files = os.listdir(tracking_path)
    all_slugified_match_strings = [os.path.splitext(file)[0] for file in all_files]
    selceted_tracking_slugified_match_strings = st.multiselect("Select tracking files", all_slugified_match_strings)
    return selceted_tracking_slugified_match_strings


def get_involvement(df_passes, df_tracking, model="outplayed", max_passes_for_debug=20, model_radius=5):
    def _dist_to_goal(df_tracking):
        y_goal = np.clip(df_tracking["y_norm"], -7.32 / 2, 7.32 / 2)
        return np.sqrt((df_tracking["x_norm"] - 52.5) ** 2 + (df_tracking["y_norm"] - y_goal) ** 2)
        # x_goal = 52.5
        # goal_width = 7.32
        # y_goal = np.clip(y_norm, -goal_width / 2, goal_width / 2)
        # return np.sqrt((x_norm - x_goal) ** 2 + (y_norm - y_goal) ** 2)

    df_tracking["distance_to_goal"] = _dist_to_goal(df_tracking)

    teams = df_tracking["team_id"].unique().tolist()

    dfs = []

#     df_tracking, team_col="team_id", period_col="period_id", team_in_possession_col="team_in_possession", x_col="x",
    for pass_nr, (pass_index, p4ss) in accessible_space.progress_bar(enumerate(df_passes.iterrows())):
        data = []
        if pass_nr > max_passes_for_debug:
            break

        i_frame = df_tracking["full_frame"] == p4ss["full_frame"]
        i_frame_rec = df_tracking["full_frame"] == p4ss["full_frame_rec"]

        df_tracking_frame = df_tracking[df_tracking["full_frame"] == p4ss["full_frame"]]
        df_tracking_frame_rec = df_tracking[df_tracking["full_frame"] == p4ss["full_frame_rec"]]

        attacking_team = p4ss["team_id_1"]
        defending_team = teams[1] if attacking_team == teams[0] else teams[0]

        i_def = df_tracking_frame["team_id"] == defending_team
        i_def_rec = df_tracking_frame_rec["team_id"] == defending_team

        y_goal = np.clip(p4ss["y_event"], -7.32 / 2, 7.32 / 2)
        distance_between_ball_and_goal = np.sqrt((p4ss["x_norm"] - 52.5) ** 2 + (p4ss["y_norm"] - y_goal) ** 2)

        df_tracking_frame["is_between_goal_and_ball"] = df_tracking_frame["distance_to_goal"] < distance_between_ball_and_goal
        df_tracking_frame_rec["is_between_goal_and_ball"] = df_tracking_frame_rec["distance_to_goal"] < distance_between_ball_and_goal

        i_between_ball_and_goal = df_tracking_frame["is_between_goal_and_ball"]
        df_between_ball_and_goal = df_tracking_frame.loc[i_def & i_between_ball_and_goal]

        i_between_ball_and_goal_rec = df_tracking_frame_rec["is_between_goal_and_ball"]
        df_between_ball_and_goal_rec = df_tracking_frame_rec.loc[i_def_rec & i_between_ball_and_goal_rec]

        for _, defender_row in df_tracking_frame.loc[i_def].iterrows():
            # add prefix "defender." to defender_row
            defender_row = {f"defender.{key}": value for key, value in defender_row.items()}
            pass_copy = p4ss.copy()

            if model == "outplayed":
                pass_copy["defender"] = defender_row["defender.player_id"]
                pass_copy["is_between_ball_and_goal_at_pass"] = defender_row["defender.player_id"] in df_between_ball_and_goal["player_id"].unique()
                pass_copy["is_between_ball_and_goal_at_reception"] = defender_row["defender.player_id"] in df_between_ball_and_goal_rec["player_id"].unique()
                is_outplayed = pass_copy["is_between_ball_and_goal_at_pass"] and not pass_copy["is_between_ball_and_goal_at_reception"]
                pass_copy["is_outplayed"] = is_outplayed
                pass_copy["involvement"] = int(is_outplayed)
            elif model == "circle_circle_rectangle":
                def distance_point_to_segment(px, py, x1, y1, x2, y2):
                    # Compute the vector AB and AP
                    ABx = x2 - x1
                    ABy = y2 - y1
                    APx = px - x1
                    APy = py - y1

                    # Compute the length squared of AB
                    AB_length_squared = ABx ** 2 + ABy ** 2

                    # Handle the degenerate case where A and B are the same point
                    if AB_length_squared == 0:
                        return math.sqrt(APx ** 2 + APy ** 2)

                    # Compute the projection of AP onto AB, normalized to [0,1]
                    t = (APx * ABx + APy * ABy) / AB_length_squared
                    t_clamped = max(0, min(1, t))

                    # Find the closest point on the segment
                    closest_x = x1 + t_clamped * ABx
                    closest_y = y1 + t_clamped * ABy

                    # Compute the distance from P to the closest point
                    dx = px - closest_x
                    dy = py - closest_y

                    return math.sqrt(dx ** 2 + dy ** 2)

                pass_copy["distance_to_pass_line"] = distance_point_to_segment(
                    defender_row["defender.x_tracking"], defender_row["defender.y_tracking"],
                    pass_copy["x_event"], pass_copy["y_event"], pass_copy["x_target"], pass_copy["y_target"]
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_pass_line"], max_distance)
                pass_copy["involvement"] = 1 - clipped_distance / max_distance
            elif model == "circle_passer":
                pass_copy["distance_to_passer"] = np.sqrt(
                    (defender_row["defender.x_tracking"] - pass_copy["x_event"]) ** 2 +
                    (defender_row["defender.y_tracking"] - pass_copy["y_event"]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_passer"], max_distance)
                pass_copy["involvement"] = 1 - clipped_distance / max_distance
            elif model == "circle_receiver":
                pass_copy["distance_to_receiver"] = np.sqrt(
                    (defender_row["defender.x_tracking"] - pass_copy["x_target"]) ** 2 +
                    (defender_row["defender.y_tracking"] - pass_copy["y_target"]) ** 2
                )
                max_distance = model_radius
                clipped_distance = min(pass_copy["distance_to_receiver"], max_distance)
                pass_copy["involvement"] = 1 - clipped_distance / max_distance
            elif model == "intercepter":
                is_intercepter = defender_row["defender.player_id"] == pass_copy["player_id_2"]
                pass_copy["involvement"] = int(is_intercepter)

            pass_copy["involvement_model"] = model

            pass_copy = pass_copy.to_dict()

            for key in defender_row:
                pass_copy[key] = defender_row[key]

            # add defender row to apss_copy
            # pass_copy.update(defender_row)

            data.append(pass_copy)

        df_outplayed = pd.DataFrame(data)
        df_outplayed["pass_nr"] = pass_nr

        dfs.append(df_outplayed)

        df_passes.loc[pass_index, "pass_nr"] = pass_nr

    df_outplayed = pd.concat(dfs)

    return df_outplayed

def add_xt(
    df_events, event_x_col="x_norm", event_y_col="y_norm", pass_end_x_col="x_target_norm", pass_end_y_col="y_target_norm",
    event_outcome_col="event_outcome", unsuccessful_outcome_values=("unsuccessful",), xt_model="ma2024",
):
    # xt_files = os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/xt_weights")))
    # full_xt_files = [os.path.join(os.path.dirname(__file__), "../assets/xt_weights", file) for file in xt_files]
    # xt_file = full_xt_files[0]
    xt_file = os.path.join(os.path.dirname(__file__), "../assets/xt_weights", f"{xt_model}.xlsx")

    # Get xT transition matrix
    df_xt = pd.read_excel(xt_file, header=None)
    num_x_cells = len(df_xt.columns)
    num_y_cells = len(df_xt.index)
    dx_cell = 105 / num_x_cells
    dy_cell = 68 / num_y_cells

    # Get cell index from x and y coordinates
    df_events["x_cell_index"] = np.clip(((df_events[event_x_col] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
    df_events["y_cell_index"] = np.clip(((df_events[event_y_col] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)
    df_events["x_cell_index_after"] = np.clip(((df_events[pass_end_x_col] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
    df_events["y_cell_index_after"] = np.clip(((df_events[pass_end_y_col] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)

    # assign xT values based on cell index and compute xT of passes
    df_events["xt_before"] = 0
    df_events["xt_after"] = 0
    i_valid_before = df_events["x_cell_index"].notnull() & df_events["y_cell_index"].notnull()  # sometimes we have no cell index because x and y coordinates are missing!
    df_events.loc[i_valid_before, "xt_before"] = df_events.loc[i_valid_before, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index"]), int(x["x_cell_index"])], axis=1)
    i_valid_end = df_events["x_cell_index_after"].notnull() & df_events["y_cell_index_after"].notnull()
    df_events.loc[i_valid_end, "xt_after"] = df_events.loc[i_valid_end, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index_after"]), int(x["x_cell_index_after"])], axis=1)

    # Important: xT after an unsuccessful pass is 0!
    df_events.loc[df_events[event_outcome_col].isin(unsuccessful_outcome_values), "xt_after"] = 0

    df_events["pass_xt"] = df_events["xt_after"] - df_events["xt_before"]

    return df_events


def get_expected_receiver(
    df_passes, df_tracking, event_frame_col="full_frame", tracking_frame_col="full_frame", event_team_col="team_id_1",
    tracking_team_col="team_id", event_player_col="player_id_1", tracking_player_col="player_id",
):
    i_unsuccessful = df_passes["event_outcome"].isin(["unsuccessful"])

    df_passes_unsuccessful = df_passes[i_unsuccessful]

    for pass_index, p4ss in df_passes_unsuccessful.iterrows():
        df_tracking_frame_attackers = df_tracking[df_tracking[tracking_frame_col] == p4ss[event_frame_col]]
        df_tracking_frame_attackers = df_tracking_frame_attackers[
            (df_tracking_frame_attackers[tracking_team_col] == p4ss[event_team_col]) &
            (df_tracking_frame_attackers[tracking_player_col] != p4ss[event_player_col])
        ]

        # is_intercepted = p4ss["pass_is_intercepted"]  # todo move to preprocessing
        # assert p4ss["pass_is_intercepted"] or p4ss["pass_is_out"]
        # st.write("is_intercepted")
        # st.write(is_intercepted)

        if pd.isna(p4ss["x_target"]):
            # st.warning("x_target is NaN")
            continue

        pass_angle = np.arctan2(p4ss["y_target"] - p4ss["y_event"], p4ss["x_target"] - p4ss["x_event"])

        df_tracking_frame_attackers["distance_to_pass_endpoint"] = np.sqrt(
            (df_tracking_frame_attackers["x_tracking"] - p4ss["x_target"]) ** 2 +
            (df_tracking_frame_attackers["y_tracking"] - p4ss["y_target"]) ** 2
        )
        df_tracking_frame_attackers["angle_to_pass_lane"] = np.abs(pass_angle - np.arctan2(
            df_tracking_frame_attackers["y_tracking"] - p4ss["y_event"],
            df_tracking_frame_attackers["x_tracking"] - p4ss["x_event"]
        ))
        min_distance_to_pass_endpoint = df_tracking_frame_attackers["distance_to_pass_endpoint"].min()
        min_angle_to_pass_lane = df_tracking_frame_attackers["angle_to_pass_lane"].min()
        df_tracking_frame_attackers["expected_receiver_score"] = df_tracking_frame_attackers["distance_to_pass_endpoint"] / min_distance_to_pass_endpoint * df_tracking_frame_attackers["angle_to_pass_lane"] / min_angle_to_pass_lane

        # get player with smallest score
        expected_receiver = df_tracking_frame_attackers.loc[df_tracking_frame_attackers["expected_receiver_score"].idxmin(), "player_id"]

        df_passes.loc[pass_index, "expected_receiver"] = expected_receiver

    return df_passes


@st.cache_resource
def _get_match_data(slugified_match_string, xt_model="ma2024", expected_receiver_model="power2017"):
    df_meta = _get_all_meta()
    match_id = df_meta[df_meta["slugified_match_string"] == slugified_match_string]["match_id"].iloc[0]
    df_lineup = _get_all_lineups()
    df_lineup_match = df_lineup[df_lineup["match_id"] == match_id]
    playerid2name = df_lineup_match[['player_id', 'short_name']].set_index('player_id').to_dict()['short_name']
    team2name = df_lineup_match.loc[df_lineup_match["team_name"].notna(), ['team_id', 'team_name']].set_index('team_id').to_dict()['team_name']

    fpath_tracking = os.path.join(tracking_path, f"{slugified_match_string}.parquet")
    df_tracking = pd.read_parquet(fpath_tracking)
    df_tracking["full_frame"] = df_tracking["section"].str.cat(df_tracking["frame"].astype(float).astype(str), sep="-")
    df_tracking["player_name"] = df_tracking["player_id"].map(playerid2name)

    fpath_event = os.path.join(event_path, f"{slugified_match_string}.csv")
    df_event = pd.read_csv(fpath_event)
    df_event["full_frame"] = df_event["section"].str.cat(df_event["frame"].astype(float).astype(str), sep="-")
    df_event["full_frame_rec"] = df_event["section"].str.cat(df_event["frame_rec"].astype(float).astype(str), sep="-")
    df_event["player_name_1"] = df_event["player_id_1"].map(playerid2name)
    df_event["player_name_2"] = df_event["player_id_2"].map(playerid2name)
    df_event["team_name_1"] = df_event["team_id_1"].map(team2name)
    df_event["team_name_2"] = df_event["team_id_2"].map(team2name)
    df_event["event_string"] = df_event["full_frame"].astype(str) + ": " + df_event["event_type"] + " " + df_event["player_name_1"] + " (" + df_event["team_name_1"] + ") -> " + df_event["player_name_2"] + " (" + df_event["team_name_2"] + ") (" + df_event["event_outcome"].astype(str) + ")"

    df_tracking["playing_direction"] = accessible_space.infer_playing_direction(
        df_tracking, team_col="team_id", period_col="section", team_in_possession_col="ball_poss_team_id", x_col="x_tracking"
    )
    df_tracking["x_norm"] = df_tracking["x_tracking"] * df_tracking["playing_direction"]
    df_tracking["y_norm"] = df_tracking["y_tracking"] * df_tracking["playing_direction"]
    fr2playing_direction = df_tracking.set_index("full_frame")["playing_direction"].to_dict()
    df_event["playing_direction"] = df_event["full_frame"].map(fr2playing_direction)
    df_event["x_norm"] = df_event["x_event"] * df_event["playing_direction"]
    df_event["y_norm"] = df_event["y_event"] * df_event["playing_direction"]
    df_event["x_target_norm"] = df_event["x_target"] * df_event["playing_direction"]
    df_event["y_target_norm"] = df_event["y_target"] * df_event["playing_direction"]

    df_event = add_xt(df_event, xt_model=xt_model)
    i_pass = df_event["event_type"] == "pass"
    df_event.loc[i_pass, "pass_is_intercepted"] = (df_event.loc[i_pass, "event_outcome"] == "unsuccessful") & ~pd.isna(df_event.loc[i_pass, "player_id_2"])
    df_event.loc[i_pass, "pass_is_out"] = (df_event.loc[i_pass, "event_outcome"] == "unsuccessful") & pd.isna(df_event.loc[i_pass, "player_id_2"])
    df_event.loc[i_pass, "pass_is_successful"] = df_event.loc[i_pass, "event_outcome"] == "successfully_completed"

    # assert all three exist
    assert df_event.loc[i_pass, "pass_is_intercepted"].sum() > 0
    assert df_event.loc[i_pass, "pass_is_out"].sum() > 0
    assert df_event.loc[i_pass, "pass_is_successful"].sum() > 0

    df_event.loc[i_pass, "outcome"] = df_event.loc[i_pass].apply(lambda x: "successful" if x["pass_is_successful"] else ("intercepted" if x["pass_is_intercepted"] else "out"), axis=1)
    df_event.loc[i_pass, "expected_receiver"] = get_expected_receiver(df_event.loc[i_pass, :], df_tracking)
    df_event.loc[i_pass, "expected_receiver_name"] = df_event.loc[i_pass, "expected_receiver"].map(playerid2name)

    df_tracking = df_tracking[df_tracking["team_id"] != "referee"]

    return df_tracking, df_event


def main():
    profiler = streamlit_profiler.Profiler()
    profiler.start()

    selected_tracking_matches = _select_matches()
    xt_model = st.selectbox("Select xT model", ["ma2024", "the_athletic"])
    expected_receiver_model = st.selectbox("Select expected receiver model", ["power2017"])

    for slugified_match_string in selected_tracking_matches:
        df_tracking, df_event = _get_match_data(slugified_match_string, xt_model=xt_model, expected_receiver_model=expected_receiver_model)
        df_passes = df_event[df_event["event_type"] == "pass"]

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
        model_radius = st.number_input("Model radius", min_value=0, value=5)

        # @st.cache_resource
        def _get_involvement(max_passes_for_debug, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius=model_radius):

            def plot_passes_with_involvement(df_involvement, model, model_radius):
                for pass_id, df_outplayed_pass in df_involvement.groupby("event_id"):
                    p4ss = df_passes[df_passes["event_id"] == pass_id].iloc[0]
                    fig = plot_pass_involvement(p4ss, df_outplayed_pass, df_tracking, plot_model=model, model_radius=model_radius)
                    plt.title(f"Pass {p4ss['event_string']} ({p4ss['pass_xt']:+.3f} xT)", fontsize=6)
                    st.pyplot(fig, dpi=500)
                    plt.close()

            # 1. Successful passes, xT > 0
            i_success_and_pos_value = (df_passes["outcome"] == "successful") & (df_passes["pass_xt"] > 0)
            df_involvement_success = get_involvement(df_passes.loc[i_success_and_pos_value], df_tracking, model=involvement_model_success_pos_value, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius)
            with st.expander("Success, xT > 0"):
                plot_passes_with_involvement(df_involvement_success, involvement_model_success_pos_value, model_radius)
            st.write("---")

            # 2. Successful passes, xT < 0
            i_success_and_neg_value = (df_passes["outcome"] == "successful") & (df_passes["pass_xt"] < 0)
            df_involvement_success_neg = get_involvement(df_passes.loc[i_success_and_neg_value], df_tracking, model=involvement_model_success_neg_value, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius)

            with st.expander("Success, xT < 0"):
                plot_passes_with_involvement(df_involvement_success_neg, involvement_model_success_neg_value, model_radius)
            st.write("---")

            # 3. Unsuccessful passes, out
            i_out = df_passes["outcome"] == "out"
            df_involvement_out = get_involvement(df_passes.loc[i_out], df_tracking, model=involvement_model_out, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius)
            with st.expander("Out"):
                plot_passes_with_involvement(df_involvement_out, involvement_model_out, model_radius)
            st.write("---")

            # 4. Unsuccessful passes, intercepted
            i_intercepted = df_passes["outcome"] == "intercepted"
            df_involvement_intercepted = get_involvement(df_passes.loc[i_intercepted], df_tracking, model=involvement_model_intercepted, max_passes_for_debug=max_passes_for_debug, model_radius=model_radius)
            with st.expander("Intercepted"):
                plot_passes_with_involvement(df_involvement_intercepted, involvement_model_intercepted, model_radius)
            st.write("---")

            df_involvement = pd.concat([df_involvement_success, df_involvement_success_neg, df_involvement_out, df_involvement_intercepted])

            return df_involvement

        df_involvement = _get_involvement(max_passes_for_debug, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted)

        for team, df_involvement_team in df_involvement.groupby("team_id_1"):
            team_name = df_involvement_team["team_name_1"].iloc[0]
            st.write(f"### {team_name}")
            df_network = get_defensive_network(df_involvement_team)

    profiler.stop()

    # df_passes = pd.concat([df_pass_risky, df_pass_safe])
    # resp = outplayed(df_tracking, df_passes)


def get_defensive_network(df_passes_with_defenders):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import model.passing_network

    df_passes_with_defenders["network_receiver"] = df_passes_with_defenders["expected_receiver"].where(df_passes_with_defenders["expected_receiver"].notna(), df_passes_with_defenders["player_id_2"])
    df_passes_with_defenders["network_receiver"] = df_passes_with_defenders["network_receiver"].where(df_passes_with_defenders["outcome"] != "intercepted", None)

    df_passes_with_defenders_dedup = df_passes_with_defenders.drop_duplicates(subset=["event_id"])
    df_passes_with_defenders_dedup = df_passes_with_defenders_dedup[df_passes_with_defenders_dedup["network_receiver"].notna()]

    df_nodes, df_edges = model.passing_network.get_passing_network_df(
        df_passes_with_defenders_dedup.reset_index(),
        x_col="x_event", y_col="y_event", from_col="player_id_1", to_col="network_receiver", from_name_col="player_name_1", to_name_col="player_name_2",
        value_col="pass_xt", x_to_col="x_target", y_to_col="y_target", dedup_cols=["event_id"],
    )
    # df_nodes["y_avg"] /= 1.75
    # df_nodes["x_avg"] /= 1.25

    value_col = "value_passes"
    fig, ax = model.passing_network.plot_passing_network(
        df_nodes=df_nodes, df_edges=df_edges, show_colorbar=False, node_size_multiplier=200, arrow_width_multiplier=10,
        label_col=value_col, arrow_color_col=value_col, annotate_top_n_edges=5555, label_format_string="{:.3f}",
    )
    st.write(fig)

    n_cols = 2
    columns = st.columns(n_cols)
    df_passes_with_defenders["valued_involvement"] = df_passes_with_defenders["involvement"] * df_passes_with_defenders["pass_xt"]
    for defender_nr, (defender, df_defender) in enumerate(df_passes_with_defenders.reset_index().groupby("defender.player_id")):
        defender_name = df_defender["defender.player_name"].iloc[0]

        import matplotlib
        import utility.pitch

        importlib.reload(model.passing_network)
        importlib.reload(utility.pitch)

        _, df_edges = model.passing_network.get_passing_network_df(
            df_defender.reset_index(),
            x_col="x_event", y_col="y_event", from_col="player_id_1", to_col="network_receiver",
            from_name_col="player_name_1", to_name_col="player_name_2",
            value_col="valued_involvement", x_to_col="x_target", y_to_col="y_target", dedup_cols=["event_id"],
        )

        value_col = "value_passes"
        df_edges["edge_label"] = df_edges[value_col].apply(lambda x: x if x != 0 else None)
        fig, ax = model.passing_network.plot_passing_network(
            df_nodes=df_nodes, df_edges=df_edges, show_colorbar=False, node_size_multiplier=200, arrow_width_multiplier=10,
                                 # colormap=matplotlib.cm.get_cmap("PuBuGn"),
            colormap=matplotlib.cm.get_cmap("coolwarm"), min_color_value_edges=-0.05, max_color_value_edges=0.05,
            min_color_value_nodes=-0.5, max_color_value_nodes=0.5, annotate_top_n_edges=5555, label_col="edge_label",
            label_format_string="{:.3f}", arrow_color_col=value_col,
        )
        with columns[defender_nr % n_cols]:
            plt.title(f"Defender {defender_name}")
            st.write(fig)



    # def get_passing_network_df(
    #         df_passes: pd.DataFrame,
    #         x_col: str,  # column with x position of the pass
    #         y_col: str,  # column with y position of the pass
    #         from_col: str,  # column with unique (!) ID or name of the player/position/... who passes the ball
    #         to_col: str,  # column with unique (!) ID or name of the player/position/... who receives the ball
    #         net_minutes=None,
    #         from_name_col: str = None,
    #         # column with the name of the player/position/... who passes the ball, if None is given - from_col is used
    #         to_name_col: str = None,
    #         # column with the name of the player/position/... who receives the ball, if None is given - to_col is used
    #         value_col: str = None,
    #         # column with the value of the pass (e.g. xGCgain, xT, ...), if None is given - all passes have value = 1
    #         x_to_col: str = None,
    #         # x position column of the receiving player (optional additional information for average positions)
    #         y_to_col: str = None,
    #         # y position column of the receiving player (optional additional information for average positions)
    #         additional_node_values: dict = None,
    #         # additional value for nodes (e.g. pxT through dribbling, shooting etc.)
    #         dedup_cols=None,  # column to de-duplicate
    # )


if __name__ == '__main__':
    main()
