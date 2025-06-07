import importlib
import os

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mplsoccer

import pandas as pd
import streamlit as st

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import defensive_network.parse.drive
import defensive_network.utility.video
import defensive_network.utility.general
import defensive_network.utility.pitch
import defensive_network.parse.dfb.cdf

importlib.reload(defensive_network.utility.pitch)

importlib.reload(defensive_network.parse.drive)

base_path = os.path.join(os.path.dirname(__file__), "../../../w_raw/preprocessed")


def create_animation(df_tracking, df_events, fpath):
    """
    >>>
    """
    teams = df_tracking["team_id"].unique().tolist()
    ball_team = "BALL"

    folder = os.path.dirname(fpath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    df_passes = df_events[df_events["event_type"] == "pass"]
    df_passes = df_passes[df_passes["frame"].notna()]

    available_sections = sorted(df_tracking["section"].unique().tolist())
    print("available_sections", available_sections)
    for section in available_sections:
        # if section != "second_half":
        #     continue
        print("section", section)

        df_tracking_half = df_tracking[df_tracking["section"] == section]
        df_tracking_half = df_tracking_half.sort_values("frame")
        df_passes_half = df_passes[df_passes["section"] == section]
        df_home = df_tracking_half[df_tracking_half["team_id"] == teams[0]].set_index("frame")
        df_away = df_tracking_half[df_tracking_half["team_id"] == teams[1]].set_index("frame")
        df_ball = df_tracking_half[df_tracking_half["team_id"] == ball_team].set_index("frame")
        if "BALL" not in df_tracking_half["team_id"].unique():
            continue

        pitch = mplsoccer.Pitch(pitch_type='impect', goal_type='line', pitch_width=68, pitch_length=105)
        fig, ax = pitch.draw(figsize=(16 * 0.8, 10.4 * 0.8))
        plt.title("Tracking data")

        # ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
        # away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon
        # home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)  # purple
        ball, = ax.plot([], [], color="black", marker="x", linestyle='None', ms=20)
        away, = ax.plot([], [], color="red", marker="o", linestyle='None', ms=10)
        home, = ax.plot([], [], color="blue", marker="o", linestyle='None', ms=10)

        pass_plot = pitch.arrows(
            0, 0, 0, 0, width=2, headwidth=10, headlength=10, color="black",
            # color='blue' if p4ss["team_id_1"] == "c6ed81e07bcb710e1388ea8865eb5314" else 'red',
            ax=ax,
        )
        print("pass_plot", pass_plot)

        frame2pass_interval = {}
        df_tracking_half = df_tracking_half.sort_values("frame")
        df_passes_half = df_passes_half.sort_values("frame")
        df_passes_half = df_passes_half.set_index("frame")

        # df_passes_half = df_passes_half[df_passes_half.index < 1000]
        # df_tracking_half = df_tracking_half[df_tracking_half["frame"] < 1000]

        current_passing_frame = None
        current_passing_end_frame = None
        next_passing_frame = df_passes_half.index[0]

        all_frames = df_tracking_half.drop_duplicates("frame")["frame"]
        for fr in all_frames:
            if next_passing_frame is not None and fr >= next_passing_frame:
                current_passing_frame = next_passing_frame
                current_passing_end_frame = df_passes_half.loc[current_passing_frame]["frame_rec"]

                df_next = df_passes_half[df_passes_half.index > fr + 1]
                next_passing_frame = df_next.index[0] if len(df_next) > 0 else None

            if current_passing_frame is not None and current_passing_end_frame is not None and \
                    fr >= current_passing_frame and fr <= current_passing_end_frame:
                frame2pass_interval[fr] = df_passes_half.loc[current_passing_frame]
            if current_passing_end_frame is not None and fr > current_passing_end_frame:
                current_passing_frame = None
                current_passing_end_frame = None

        def animate(current_frame):
            seconds = current_frame / 25
            mmss = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

            if current_frame in frame2pass_interval:
                pass_for_interval = frame2pass_interval[current_frame]
                pass_plot.set_offsets([pass_for_interval["x_event"], pass_for_interval["y_event"]])
                pass_plot.set_UVC(pass_for_interval["x_target"] - pass_for_interval["x_event"],
                                  pass_for_interval["y_target"] - pass_for_interval["y_event"])
                xpass = pass_for_interval["xpass"]
            else:
                pass_plot.set_offsets([0, 0])
                pass_plot.set_UVC([0], [0])
                xpass = None

            if xpass is not None:
                xpass = f"{xpass:.1%}"
            ax.set_title(f"Section {section}, Frame {current_frame}, Time: {mmss}, xPass: {xpass}")

            # t1 = time.time()
            df_ball_to_plot = df_ball
            df_home_to_plot = df_home
            df_away_to_plot = df_away
            # print(f"A {time.time() - t1:.2f} seconds.")

            ball_frame = df_ball_to_plot.iloc[current_frame]
            ball.set_data([ball_frame["x_tracking"]], [ball_frame["y_tracking"]])
            # print(f"B {time.time() - t1:.2f} seconds.", ball_frame)
            frame = ball_frame.name
            # print(f"B {time.time() - t1:.2f} seconds.")

            # print("c", df_away_to_plot)

            # away.set_data(df_away_to_plot.loc[df_away["frame"] == frame, 'x_tracking'], df_away_to_plot.loc[df_away["frame"] == frame, 'y_tracking'])
            away.set_data(df_away_to_plot.loc[frame, 'x_tracking'], df_away_to_plot.loc[frame, 'y_tracking'])
            # print(f"C {time.time() - t1:.2f} seconds.")
            home.set_data(df_home_to_plot.loc[frame, 'x_tracking'], df_home_to_plot.loc[frame, 'y_tracking'])
            # print(f"Frame {i} took {time.time() - t1:.2f} seconds.")
            return ball, away, home

        # n_frames = len(df_tracking_half["frame"].unique())
        n_frames = df_tracking_half["frame"].max()
        # print("n_frames")
        # print(n_frames)
        # n_frames = len(df_ball)
        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)

        # export mp4
        basepath, ext = os.path.splitext(fpath)
        target_fpath = f"{basepath}_{section}.mp4"
        anim.save(target_fpath, writer='ffmpeg', fps=25, progress_callback=lambda i, n: print(f'Saving frame {i} of {n} ({i / n:.1%}) to {target_fpath}') if i % 30 == 0 else None)

        plt.close(fig)


def main():
    matplotlib.use('TkAgg')  # make plots show in new window (for animation)

    # folder = "C:/Users/Jonas/Downloads/dfl_test_data/2324/"
    folder = base_path
    if not os.path.exists(folder):
        raise NotADirectoryError(f"Folder {folder} does not exist")

    folder_events = os.path.join(folder, "events")
    folder_tracking = os.path.join(folder, "tracking")
    folder_animation = os.path.join(folder, "animation")

    match_slugified_strings_to_animate = [os.path.splitext(file)[0] for file in os.listdir(folder_tracking)]

    existing_matches = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder("tracking", st_cache=True)]
    # existing_matches = [file.split(".")[0] for file in os.listdir(os.path.join(os.path.dirname(__file__), "../../../w_raw/preprocessed/tracking"))]
    st.write("existing_matches")
    st.write("existing_matches")
    st.write(existing_matches)
    st.write("A")

    for match_slugified_string in existing_matches:  # defensive_network.utility.general.progress_bar(match_slugified_strings_to_animate):
        st.write(match_slugified_string)
        target_fpath = os.path.join(folder_animation, f"{match_slugified_string}.mp4")
        if os.path.exists(target_fpath):
            print(f"File {target_fpath} already exists, skipping")
            continue
        # df_event = pd.read_csv(os.path.join(folder_events, f"{match_slugified_string}.csv"))
        # df_tracking = pd.read_parquet(os.path.join(folder_tracking, f"{match_slugified_string}.parquet"))
        try:
            df_event = defensive_network.parse.drive.download_csv_from_drive(f"events/{match_slugified_string}.csv", st_cache=True)
            # df_event = defensive_network.parse.dfb.cdf.get_events(base_path, match_slugified_string)
        except FileNotFoundError as e:
            # st.write(e)
            continue

        st.write("df_event")
        st.write(df_event)

        df_tracking = defensive_network.parse.drive.download_parquet_from_drive(f"tracking/{match_slugified_string}.parquet")
        st.write(f"{match_slugified_string=}")
        # df_tracking = defensive_network.parse.dfb.cdf.get_tracking(base_path, match_slugified_string)
        # create_animation(df_tracking, df_event, target_fpath)
        df_passes = df_event[df_event["event_type"] == "pass"]

# tracking_time_col], p4ss["rec_time

        #replace 1900-01-01 with None
        st.write(df_passes)
        st.write(df_tracking["datetime_tracking"])
        df_passes["datetime_tracking"] = pd.to_datetime(df_passes["datetime_tracking"].replace("1900-01-01 00:00:00.000000+0000", None).replace("1900-01-01 00:00:00+00:00", None), format="ISO8601")
        st.write(df_passes)
        st.write(df_tracking["datetime_tracking"])

        df_passes["datetime_event"] = pd.to_datetime(df_passes["datetime_event"], format="ISO8601")
        df_passes["datetime_tracking"] = pd.to_datetime(df_passes["datetime_tracking"], format="ISO8601")
        df_passes["datetime_tracking"] = pd.to_datetime(df_passes["datetime_tracking"], format="ISO8601")

        # df_tracking = df_tracking[df_tracking["frame"] < 1000]
        # df_passes = df_passes[df_passes["frame"] < 1000]

        st.write("df_tracking")
        st.write(df_tracking)
        st.write("df_passes")
        st.write(df_passes)

        importlib.reload(defensive_network.utility.video)

        defensive_network.utility.video.pass_video(df_tracking, df_passes, out_fpath=os.path.join(os.path.dirname(__file__), "test.mp4"), overwrite_if_exists=False)

        # TODO check utility.video, add both timestamps (corrected/original)

        break



if __name__ == '__main__':
    main()
