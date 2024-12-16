import math

import numpy as np
import streamlit as st

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
    import matplotlib.pyplot as plt

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
        # print(position)


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

    # plt.arrow(x1, y1, x2 - x1, y2 - y1, head_width=2.5, head_length=2.5, color='red', length_includes_head=True, width=arrow_width)

    if not bidirectional:
        arrow = matplotlib.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=25, connectionstyle='arc3,rad=0.2', linewidth=arrow_width, color=arrow_color)
    else:
        arrow = matplotlib.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='<->', mutation_scale=40, connectionstyle='arc3,rad=0.0', linewidth=arrow_width, color=arrow_color)
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
