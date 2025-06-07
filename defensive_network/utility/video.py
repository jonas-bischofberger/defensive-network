import gc
import importlib
import os
import cv2
import numpy as np

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

import defensive_network.utility
import defensive_network.utility.pitch


def pass_video(df_tracking, df_passes, out_fpath, event_frame_cols=("frame", "original_frame_id", "matched_frame"), tracking_frame_col="frame", frame_rec_cols=["frame_rec"], event_section_col="section", tracking_section_col="section", fps=25, overwrite_if_exists=True):
    st.write("df_passes")
    st.write(df_passes)

    overwrite_if_exists = True

    img_files = []

    for section, df_passes_section in df_passes.groupby(event_section_col):
        event_frame_col = event_frame_cols[0]
        frame_rec_col = frame_rec_cols[0]

        df_tracking[tracking_frame_col] = df_tracking[tracking_frame_col].astype(int)
        df_tracking_section = df_tracking[df_tracking[tracking_section_col] == section].sort_values(tracking_frame_col)
        df_passes_section[event_frame_col] = df_passes_section[event_frame_col].astype(int)
        df_passes_section = df_passes_section.sort_values(event_frame_col)

        # for event_frame_col in event_frame_cols:
        #     for frame_rec_col in frame_rec_cols:
        st.write(f"Processing section {section}, event_frame_col {event_frame_col}, frame_rec_col {frame_rec_col}")

        pass_frames = [(p4ss[event_frame_col], p4ss[frame_rec_col]) for _, p4ss in df_passes_section.iterrows()]

        st.write("df_tracking_section")
        st.write(df_tracking_section)

        current_pass = None
        for frame, df_tracking_frame in defensive_network.utility.general.progress_bar(df_tracking_section.groupby(tracking_frame_col), total=len(df_tracking[tracking_frame_col].unique())):
            if frame > 150:
                break

            slugified_match_string = df_passes_section["slugified_match_string"].iloc[0]
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{slugified_match_string}_{section}_{frame}.png"))

            if not overwrite_if_exists and os.path.exists(path):
                img_files.append(path)
                continue

            is_during_pass = any([p[0] <= frame <= p[1] for p in pass_frames])

            if frame in df_passes_section[event_frame_col].values:
                current_pass = df_passes_section[df_passes_section[event_frame_col] == frame].iloc[0]

            # st.write(f"Current pass for frame {frame} in section {section}:")
            # st.write(current_pass)

            match_string = df_passes["match_string"].iloc[0]
            try:
                mmss = df_tracking_frame["mmss"].iloc[0]
            except KeyError:
                mmss = "xx:xx"

            df_next_pass = df_passes_section[((df_passes_section[event_frame_col]) >= frame)]
            if not is_during_pass:
                if len(df_next_pass) == 0:
                    next_pass = None
                else:
                    next_pass = df_next_pass.iloc[0]

                frames_away = int(next_pass[event_frame_col] - frame) if next_pass is not None else None

                # st.write(f"Next pass for frame {frame} in section {section} ({frames_away} frames away):")
                # st.write(next_pass)
                pass_to_plot = next_pass
            else:
                # st.info("Frame is during pass")
                pass_to_plot = current_pass


            # if frame != 35 or section != "second_half":
            #     continue


            additional_frame_cols = {"matched_frame": "black", "original_frame_id": "cyan"}
            defensive_network.utility.pitch.plot_pass(pass_to_plot, df_tracking_frame,
                                                      # make_pass_transparent=not is_during_pass,
                                                      additional_x_coordinates={"x_event": "yellow",
                                                                                "x_event_player_1": "orange",
                                                                                "x_tracking_player_1": "red",
                                                                                "x_target": "blue",
                                                                                "x_event_player_2": "violet",
                                                                                "x_tracking_player_2": "green"
                                                                                },
                                                      additional_y_coordinates={"y_event": "yellow",
                                                                                "y_event_player_1": "orange",
                                                                                "y_tracking_player_1": "red",
                                                                                "y_target": "blue",
                                                                                "y_event_player_2": "violet",
                                                                                "y_tracking_player_2": "green"
                                                                                },
                                                      additional_frame_cols=additional_frame_cols,
                                                      )
            # defensive_network.utility.pitch.plot_pass(pass_to_plot, df_tracking_frame, make_pass_transparent=not is_during_pass, additional_x_coordinates={"x_event_player_1": "orange", "x_tracking_player_1": "red", "x_event_player_2": "violet", "x_tracking_player_2": "green"}, additional_y_coordinates={"y_event_player_1": "orange", "y_tracking_player_1": "red", "y_event_player_2": "violet", "y_tracking_player_2": "green"}, pass_x_col="x_event_player_1", pass_y_col="y_event_player_1", pass_end_x_col="x_event_player_2", pass_end_y_col="y_event_player_2")
            subtype_str = f"{pass_to_plot['event_subtype']} " if not pd.isna(pass_to_plot["event_subtype"]) else ""
            outcome = pass_to_plot["outcome"] if "outcome" in pass_to_plot else ""
            xt = pass_to_plot['pass_xt'] if "pass_xt" in pass_to_plot else np.nan
            player1 = pass_to_plot['player_name_1'] if "player_name_1" in pass_to_plot else ""
            player2 = pass_to_plot['player_name_2'] if "player_name_2" in pass_to_plot else ""
            additional_frames_str = [f"{frame_col}:{pass_to_plot[frame_col]}" for frame_col in additional_frame_cols]
            plt.title(f"{match_string} {mmss}\n({section} fr={frame} in {pass_to_plot[event_frame_col]}->{pass_to_plot[frame_rec_col]}\n{additional_frames_str})\n{subtype_str}{outcome} {xt:.3f} xT, {pass_to_plot['xpass']:.1%} xPass {player1} -> {player2}")

            st.write(plt.gcf())
            plt.savefig(path, dpi=300)
            img_files.append(path)

            plt.close()
            gc.collect()

    return _assemble_video(img_files, out_fpath)

    # TODO event timestamps vs tracking unclear
    #
    # # st.write(df_tracking.head(5))
    # # st.write("df_passes", df_passes.shape)
    # # st.write(df_passes)
    # min_time = df_tracking[tracking_time_col].min()
    # max_time = df_tracking[tracking_time_col].max()
    # df_passes = df_passes.sort_values(event_time_col)
    # # first_frame = df_passes[frame_col].iloc[0]
    # # last_frame = df_passes[frame_rec_col].iloc[-1]
    # # first_time = df_tracking[df_tracking[frame_col] == first_frame][tracking_time_col].iloc[0]
    # # last_time = df_tracking[df_tracking[frame_col] == last_frame][tracking_time_col].iloc[0]
    # # first_time = max(min_time, df_passes[event_time_col].iloc[0] - pd.Timedelta(seconds=padding_seconds))
    # # last_time = min(max_time, df_passes[event_time_col].iloc[-1] + pd.Timedelta(seconds=padding_seconds))
    #
    # df_passes["pass_index"] = df_passes.index
    # df_tracking = df_tracking.merge(df_passes[[frame_col, "pass_index"]], on=frame_col, how="left")
    # df_tracking = df_tracking.merge(df_passes[[frame_rec_col, "pass_index"]].rename(columns={"pass_index": "rec_index"}), left_on=frame_col, right_on=frame_rec_col, how="left")
    #
    # # df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]], on=frame_col, how="left")
    # assert frame_col in df_passes.columns
    # assert frame_col in df_tracking.columns
    # # st.write("df_tracking[[frame_col, tracking_time_col]]")
    # # st.write(df_tracking[[frame_col, tracking_time_col]])
    # # df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]].rename(columns={tracking_time_col: "rec_time"}), left_on=frame_rec_col, right_on=frame_col, how="left")
    # df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]].drop_duplicates().rename(columns={tracking_time_col: "rec_time", frame_col: "do_not_keep_me"}), left_on=frame_rec_col, right_on="do_not_keep_me", how="left").drop(columns=["do_not_keep_me"])
    # # st.write("df_passes")
    # # st.write(df_passes)
    # assert frame_col in df_passes.columns
    #
    # # st.write("df_tracking a", df_tracking.shape)
    # # st.write(df_tracking)
    #
    # df_tracking["is_pass_start_or_end"] = df_tracking["pass_index"].notna() | df_tracking["rec_index"].notna()
    # first_time = df_tracking[df_tracking["is_pass_start_or_end"]][tracking_time_col].min()
    # last_time = df_tracking[df_tracking["is_pass_start_or_end"]][tracking_time_col].max()
    # first_time = max(min_time, first_time - pd.Timedelta(seconds=padding_seconds))
    # last_time = min(max_time, last_time + pd.Timedelta(seconds=padding_seconds))
    #
    # # st.write("df_passes")
    # # st.write(df_passes)
    # # st.write(df_passes[[frame_col, frame_rec_col]])
    # pass_frames = [(p4ss[frame_col], p4ss[frame_rec_col]) for _, p4ss in df_passes.iterrows()]
    # # df_passes[tracking_time_col] = pd.to_datetime(df_passes[tracking_time_col])
    #
    # st.write("df_passes")
    # st.write(df_passes)
    #
    # pass_times = [(p4ss[tracking_time_col], p4ss["rec_time"]) for _, p4ss in df_passes.iterrows()]
    #
    # # st.write("first_time", first_time)
    # # st.write("last_time", last_time)
    #
    # df_tracking = df_tracking[df_tracking[tracking_time_col].between(first_time, last_time, inclusive="both")]
    # df_tracking = df_tracking.sort_values(tracking_time_col)
    #
    # # st.write("df_tracking")
    # # st.write(df_tracking[df_tracking[frame_col] == df_passes[frame_col].iloc[1]])
    # # st.write("pass_times")
    # # st.write(pass_times)
    #
    # def is_during_pass(x):
    #     # if "100.0" in x[frame_col]:
    #     #     st.write([p[0] <= x[tracking_time_col] <= p[1] for p in pass_times])
    #     #     st.write([(p[0], x[tracking_time_col], p[1]) for p in pass_times])
    #     # pass_presence = [p[0] <= x[tracking_time_col] <= p[1] for p in pass_times]
    #     # if any(pass_presence):
    #     #     return pass_presence.index(True)
    #     # return None
    #     # return any([p[0] <= x[tracking_time_col] <= p[1] for p in pass_times])
    #     return any([p[0] <= x[frame_col] <= p[1] for p in pass_frames])
    #
    # def closest_pass(x, df_passes, _frame_col, _frame_rec_col, _period_col):
    #     # pass_frames = [(p4ss[_frame_col], p4ss[_frame_rec_col]) for _, p4ss in df_passes.iterrows()]
    #
    #     series_frame_closeness = (df_passes[_frame_col] - x[_frame_col]).abs()
    #     series_rec_closeness = (df_passes[_frame_rec_col] - x[_frame_rec_col]).abs()
    #
    #     # pass_presence = [p[0] <= x[tracking_time_col] <= p[1] for p in pass_times]  # TODO
    #     # st.write(x[tracking_time_col])
    #     # st.write(pass_frames)
    #     # pass_closeness = [min(abs(p[0] - x[tracking_time_col]), abs(p[1] - x[tracking_time_col])) for p in pass_times]
    #     pass_closeness = [min(abs(p4ss[_frame_col] - x[_frame_col]), abs(p4ss[_frame_rec_col] - x[_frame_rec_col])) for _, p4ss in df_passes.iterrows()]
    #     # closest = pass_closeness.index(min(pass_closeness))
    #     return df_passes.iloc[pass_closeness.index(min(pass_closeness))]["pass_index"]
    #     return closest
    #
    # with st.spinner("Calculating is_during_pass"):
    #     df_tracking["is_during_pass"] = df_tracking.apply(lambda x: is_during_pass(x), axis=1)
    #
    # with st.spinner("Calculating closest_pass"):
    #     df_tracking["closest_pass"] = df_tracking.apply(lambda x: closest_pass(x, df_passes, frame_col, frame_rec_col, _period_col="section"), axis=1)
    #
    # df_passes = df_passes.sort_values(event_time_col)
    #
    # # st.write("df_tracking b", df_tracking.shape)
    # # st.write(df_tracking)
    #
    # # current_pass = df_passes.iloc[0]
    # # if len(df_passes) > 1:
    # #     next_pass_index = 1
    # #     # st.write("df_passes", next_pass_index)
    # #     # st.write(df_passes)
    # #     # st.write(df_passes.iloc[next_pass_index])
    # #     next_pass_candidate = df_passes.iloc[next_pass_index]
    # # else:
    # #     next_pass_candidate = None
    # slugified_match_string = df_passes["slugified_match_string"].iloc[0]
    #
    # img_files = []
    # columns = st.columns(3)
    # for frame_nr, (frame, df_tracking_frame) in defensive_network.utility.general.progress_bar(enumerate(df_tracking.groupby(frame_col)), total=len(df_tracking[frame_col].unique())):
    #     time = df_tracking_frame[tracking_time_col].iloc[0]
    #     time_str = str(time).replace("+", "_").replace(":", "_")
    #
    #     # columns[frame_nr % 3].write(plt.gcf())
    #     path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{slugified_match_string}_{time_str}.png"))
    #     if overwrite_if_exists or not os.path.exists(path):
    #         current_pass = df_passes.iloc[df_tracking_frame["closest_pass"].iloc[0]]
    #         is_during_pass = df_tracking_frame["is_during_pass"].iloc[0]
    #
    #         match_string = df_passes["match_string"].iloc[0]
    #         mmss = df_tracking_frame["mmss"].iloc[0]
    #         subtype_str = f"{current_pass['event_subtype']} " if not pd.isna(current_pass["event_subtype"]) else ""
    #
    #         importlib.reload(defensive_network.utility.pitch)
    #         defensive_network.utility.pitch.plot_pass(current_pass, df_tracking_frame, make_pass_transparent=not is_during_pass, additional_x_coordinates={"x_event": "yellow", "x_event_player_1": "orange", "x_event_player_2": "red"}, additional_y_coordinates={"y_event": "yellow", "y_event_player_1": "orange", "y_tracking_player_1": "red", "y_target": "blue", "y_event_player_2": "violet", "y_tracking_player_2": "green"})
    #         plt.title(f"{match_string} {mmss} ()\n{subtype_str}{current_pass['outcome']} {current_pass['pass_xt']:.3f} xT, {current_pass['xpass']:.1%} xPass {current_pass['player_name_1']} -> {current_pass['player_name_2']}")
    #
    #         plt.savefig(path)
    #         # st.write(f"Saved {path}")
    #         plt.close()
    #     else:
    #         pass
    #         # st.write(f"Skipped {path}")
    #     img_files.append(path)
    #
    # _assemble_video(img_files, out_fpath)


def _assemble_video(image_fpaths, video_fpath):
    st.write(f"Creating video from {len(image_fpaths)} images...")
    st.write(image_fpaths)

    first_frame = cv2.imread(image_fpaths[0])
    height, width, layers = first_frame.shape

    video = cv2.VideoWriter(video_fpath, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))

    for image_file in defensive_network.utility.general.progress_bar(image_fpaths, total=len(image_fpaths)):
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()
    st.write(f"Done {video_fpath}")
    return
