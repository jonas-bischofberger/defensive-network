import math

import accessible_space
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def_x = -30
mf_x = 0
att_x = 40
location_by_position = {
    'GK': (-50, 0),
    'LWB': (def_x + 30, 30),
    'LB5': (def_x + 30, 30),
    'LB': (def_x + 7.5, 30),
    'LCB': (def_x + 5, 15),
    'CB': (def_x, 0),
    'RCB': (def_x + 5, -15),
    'RB': (def_x + 7.5, -30),
    'RWB': (def_x + 30, -30),
    'RB5': (def_x + 30, -30),

    'LW': (mf_x + 30, 30),
    'LAMF': (mf_x + 20, 20),
    'LCMF': (mf_x, 10),
    'CMF': (mf_x, 0),
    'RCMF': (mf_x, -10),
    'RAMF': (mf_x + 20, -20),
    'RW': (mf_x + 30, -30),
    'DMF': (mf_x - 10, 0),
    'RDMF': (mf_x - 10, -10),
    'LDMF': (mf_x - 10, 10),
    'AMF': (mf_x + 20, 0),

    'CF': (att_x, -5),
    'SS': (att_x - 10, 5),
}


def plot_football_pitch(color='black', linewidth=1, alpha=0.3, figsize=(16,9)):
    """
    >>> plot_football_pitch()
    <AxesSubplot:>
    >>> plt.show()
    """
    semi_pitch_length = 52.5
    semi_pitch_width = 34
    penalty_box_width = 40.32
    penalty_box_length = 16.5
    outfield_padding = 5
    middle_circle_radius = 9.15

    # Plot the pitch
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    pitch = plt.Rectangle((-semi_pitch_length, -semi_pitch_width), width=semi_pitch_length*2, height=semi_pitch_width*2,
                          fill=True, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(pitch)

    # Plot the middle circle
    middle_circle = plt.Circle((0, 0), radius=middle_circle_radius, fill=False, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(middle_circle)

    # Plot the middle line
    middle_line = plt.Line2D([0, 0], [-semi_pitch_width, semi_pitch_width], color=color, linewidth=linewidth, alpha=alpha)
    ax.add_line(middle_line)

    # Plot the penalty boxes
    left_penalty_box = plt.Rectangle((-semi_pitch_length, -penalty_box_width/2), width=penalty_box_length, height=penalty_box_width,
                                        fill=False, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(left_penalty_box)
    right_penalty_box = plt.Rectangle((semi_pitch_length-penalty_box_length, -penalty_box_width/2), width=penalty_box_length, height=penalty_box_width,
                                        fill=False, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(right_penalty_box)

    plt.xlim(-semi_pitch_length - outfield_padding, semi_pitch_length + outfield_padding)
    plt.ylim(-semi_pitch_width - outfield_padding, semi_pitch_width + outfield_padding)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # Equal scaling of x and y-axis
    plt.axis('equal')

    return fig, ax

def plot_position(position: str, label: str = None, color="blue", size=100, custom_x=None, custom_y=None, label_size=12):
    """
    >>> plot_football_pitch()
    >>> for position in ['AMF', 'CB', 'CF', 'DMF', 'GK', 'LAMF', 'LB', 'LWB', 'LCB', 'LCB', 'LCMF', 'LCMF', 'LCMF', 'LW', 'LWB', 'LW', 'RAMF', 'RB', 'RWB', 'RCB', 'RCB', 'RCMF', 'RCMF', 'RCMF', 'RW', 'RWB', 'RW', 'SS', 'CF']:
    ...     plot_position(position)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if label is None:
        label = position
    if custom_x is not None and custom_y is not None:
        location = (custom_x, custom_y)
    else:
        location = location_by_position.get(position, None)

    if location is not None:
        # path_collection = plt.scatter(*location, marker='o', color=color, s=size, edgecolors="black", linewidths=1)
        plt.scatter(*location, marker='o', color=color, s=size, edgecolors="black", linewidths=1)
        ydelta = 2.75 + math.sqrt(size) / 20
        # ydelta = math.sqrt((size/130) / math.pi) + 0.5
        plt.text(location[0], location[1]-ydelta, label, color="black", fontsize=label_size, ha="center", va="top")
    else:
        pass


def plot_position_arrow(start_position: str, end_position: str, label: str = "", arrow_width=1.5, bidirectional=False,
                        arrow_color="blue", label_color="blue", position_color="blue", include_label=True,
                        plot_players=False, custom_xy=None, custom_x2y=None):
    """
    >>> plot_football_pitch()
    >>> plot_position_arrow('LW', 'RW', "0.52", plot_players=True)
    >>> plot_position_arrow('RW', 'LW', "0.13", arrow_width=6)
    >>> plt.show()
    """
    import matplotlib.patches
    import matplotlib.pyplot as plt

    x1, y1 = location_by_position[start_position] if custom_xy is None else custom_xy
    x2, y2 = location_by_position[end_position] if custom_x2y is None else custom_x2y

    if not bidirectional:
        arrow = matplotlib.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=25, connectionstyle='arc3,rad=0.2', linewidth=arrow_width, color=arrow_color)
    else:
        arrow = matplotlib.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='<->', mutation_scale=40, connectionstyle='arc3,rad=0.0', linewidth=arrow_width, color=arrow_color)

    if not isinstance(x1, float):
        raise ValueError("x1 is not a float")

    plt.gca().add_patch(arrow)

    # Plot text next to the arrow, but closer to the x1,y1 position than to the x2,y2 position

    arrow_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    text_r = 0.1 * arrow_length
    # Get vector that points to the right hand side from the perspective of the arrow
    arrow_vector = np.array([x2 - x1, y2 - y1])
    arrow_vector = arrow_vector / np.linalg.norm(arrow_vector)

    right_hand_side_vector = np.array([arrow_vector[1], -arrow_vector[0]])

    text_pos = np.array([x1, y1]) + text_r * arrow_vector + 0 * right_hand_side_vector

    xt, yt = text_pos[0], text_pos[1]

    # Plot the text, aligned to the right if the text is right of the arrow, and aligned to the left if the text is left of the arrow
    if include_label and label != 0.0:  # TODO FIX
        plt.text(xt, yt, label, horizontalalignment='right' if xt < x1 else 'left',
                 bbox=dict(boxstyle='round', color="black", alpha=0.8), color="white", weight='bold', fontsize=6,
                 # color=label_color,
                 )

    # plt.text((x1 + x2) / 3, (y1 + y2) / 3, label, color='red')
    if plot_players:
        plot_position(start_position, color=position_color)
        plot_position(end_position, color=position_color)


def plot_pass(
    p4ss, df_tracking,
    pass_x_col="x_event", pass_y_col="y_event", pass_end_x_col="x_target", pass_end_y_col="y_target",
    pass_frame_col="full_frame", pass_team_col="team_id_1", pass_player_name_col="player_name_1",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x_tracking",
    tracking_y_col="y_tracking", tracking_frame_col="full_frame", tracking_player_name_col="player_name",
    tracking_vx_col=None, tracking_vy_col=None, ball_tracking_player_id="BALL", plot_defenders=True,
    plot_expected_receiver=True,
):

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
            try:
                vx = df_frame_team["vx"].tolist()
                vy = df_frame_team["vy"].tolist()
                for i in range(len(x)):
                    plt.arrow(x=x[i], y=y[i], dx=vx[i] / 5, dy=vy[i] / 5, head_width=0.5*factor, head_length=0.5*factor, fc="black", ec="black")
            except KeyError as e:
                st.warning(e)

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

    if plot_expected_receiver and "expected_receiver" in p4ss:
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


def plot_pass_involvement(
    p4ss, df_involvement, df_tracking,
    pass_x_col="x_event", pass_y_col="y_event", pass_end_x_col="x_target", pass_end_y_col="y_target",
    pass_frame_col="full_frame", pass_team_col="team_id_1", pass_player_name_col="player_name_1",
    tracking_team_col="team_id", tracking_player_col="player_id", tracking_x_col="x_tracking",
    tracking_y_col="y_tracking", tracking_frame_col="full_frame", tracking_player_name_col="player_name",
    tracking_vx_col=None, tracking_vy_col=None, ball_tracking_player_id="BALL",
    plot_model="circle_circle_rectangle", plot_expected_receiver=True, model_radius=5,
):
    fig = plot_pass(
        p4ss, df_tracking,
        pass_x_col, pass_y_col, pass_end_x_col, pass_end_y_col,
        pass_frame_col, pass_team_col, pass_player_name_col,
        tracking_team_col, tracking_player_col, tracking_x_col, tracking_y_col, tracking_frame_col, tracking_player_name_col,
        tracking_vx_col, tracking_vy_col, ball_tracking_player_id,
        plot_defenders=False, plot_expected_receiver=plot_expected_receiver
    )
    for _, row in df_involvement.iterrows():
        involvement = row["raw_involvement"]

        # scale alpha
        alpha_lower_threshold = 0.05
        alpha = alpha_lower_threshold + (1-alpha_lower_threshold) * involvement

        if np.isnan(alpha):
            continue

        try:
            plt.scatter(row["defender_x"], row["defender_y"], c="blue", marker="o", s=50, alpha=alpha)
        except ValueError as e:
            st.write(e)

        # add number to involvement
        plt.annotate(f"{involvement:.2f}", (row["defender_x"], row["defender_y"]), fontsize=3, ha="center", va="center", color="black")

        # add defender name
        plt.annotate(row[f"defender_name"], (row["defender_x"], row["defender_y"]-2.25), fontsize=5, ha="center", va="center", color="blue")

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
