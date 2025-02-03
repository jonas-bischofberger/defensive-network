import importlib
import os

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

import defensive_network.utility
from defensive_network.utility.pitch import plot_pass


def pass_video(df_tracking, df_passes, out_fpath, frame_col="full_frame", frame_rec_col="full_frame_rec", event_time_col="datetime_event", tracking_time_col="datetime_tracking", padding_seconds=5, overwrite_if_exists=False):
    # TODO event timestamps vs tracking unclear

    # st.write(df_tracking.head(5))
    # st.write("df_passes", df_passes.shape)
    # st.write(df_passes)
    min_time = df_tracking[tracking_time_col].min()
    max_time = df_tracking[tracking_time_col].max()
    df_passes = df_passes.sort_values(event_time_col)
    # first_frame = df_passes[frame_col].iloc[0]
    # last_frame = df_passes[frame_rec_col].iloc[-1]
    # first_time = df_tracking[df_tracking[frame_col] == first_frame][tracking_time_col].iloc[0]
    # last_time = df_tracking[df_tracking[frame_col] == last_frame][tracking_time_col].iloc[0]
    # first_time = max(min_time, df_passes[event_time_col].iloc[0] - pd.Timedelta(seconds=padding_seconds))
    # last_time = min(max_time, df_passes[event_time_col].iloc[-1] + pd.Timedelta(seconds=padding_seconds))

    df_passes["pass_index"] = df_passes.index
    df_tracking = df_tracking.merge(df_passes[[frame_col, "pass_index"]], on=frame_col, how="left")
    df_tracking = df_tracking.merge(df_passes[[frame_rec_col, "pass_index"]].rename(columns={"pass_index": "rec_index"}), left_on=frame_col, right_on=frame_rec_col, how="left")

    # df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]], on=frame_col, how="left")
    assert frame_col in df_passes.columns
    assert frame_col in df_tracking.columns
    # st.write("df_tracking[[frame_col, tracking_time_col]]")
    # st.write(df_tracking[[frame_col, tracking_time_col]])
    # df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]].rename(columns={tracking_time_col: "rec_time"}), left_on=frame_rec_col, right_on=frame_col, how="left")
    df_passes = df_passes.merge(df_tracking[[frame_col, tracking_time_col]].drop_duplicates().rename(columns={tracking_time_col: "rec_time", frame_col: "do_not_keep_me"}), left_on=frame_rec_col, right_on="do_not_keep_me", how="left").drop(columns=["do_not_keep_me"])
    # st.write("df_passes")
    # st.write(df_passes)
    assert frame_col in df_passes.columns

    # st.write("df_tracking a", df_tracking.shape)
    # st.write(df_tracking)

    df_tracking["is_pass_start_or_end"] = df_tracking["pass_index"].notna() | df_tracking["rec_index"].notna()
    first_time = df_tracking[df_tracking["is_pass_start_or_end"]][tracking_time_col].min()
    last_time = df_tracking[df_tracking["is_pass_start_or_end"]][tracking_time_col].max()
    first_time = max(min_time, first_time - pd.Timedelta(seconds=padding_seconds))
    last_time = min(max_time, last_time + pd.Timedelta(seconds=padding_seconds))

    # st.write("df_passes")
    # st.write(df_passes)
    # st.write(df_passes[[frame_col, frame_rec_col]])
    pass_frames = [(p4ss[frame_col], p4ss[frame_rec_col]) for _, p4ss in df_passes.iterrows()]
    df_passes[tracking_time_col] = pd.to_datetime(df_passes[tracking_time_col])
    pass_times = [(p4ss[tracking_time_col], p4ss["rec_time"]) for _, p4ss in df_passes.iterrows()]

    # st.write("first_time", first_time)
    # st.write("last_time", last_time)

    df_tracking = df_tracking[df_tracking[tracking_time_col].between(first_time, last_time, inclusive="both")]
    df_tracking = df_tracking.sort_values(tracking_time_col)

    # st.write("df_tracking")
    # st.write(df_tracking[df_tracking[frame_col] == df_passes[frame_col].iloc[1]])
    # st.write("pass_times")
    # st.write(pass_times)

    def is_during_pass(x):
        # if "100.0" in x[frame_col]:
        #     st.write([p[0] <= x[tracking_time_col] <= p[1] for p in pass_times])
        #     st.write([(p[0], x[tracking_time_col], p[1]) for p in pass_times])
        # pass_presence = [p[0] <= x[tracking_time_col] <= p[1] for p in pass_times]
        # if any(pass_presence):
        #     return pass_presence.index(True)
        # return None
        return any([p[0] <= x[tracking_time_col] <= p[1] for p in pass_times])

    def closest_pass(x):
        pass_presence = [p[0] <= x[tracking_time_col] <= p[1] for p in pass_times]  # TODO
        pass_closeness = [min(abs(p[0] - x[tracking_time_col]), abs(p[1] - x[tracking_time_col])) for p in pass_times]
        closest = pass_closeness.index(min(pass_closeness))
        return closest

    with st.spinner("Calculating is_during_pass"):
        df_tracking["is_during_pass"] = df_tracking.apply(lambda x: is_during_pass(x), axis=1)

    with st.spinner("Calculating closest_pass"):
        df_tracking["closest_pass"] = df_tracking.apply(lambda x: closest_pass(x), axis=1)

    df_passes = df_passes.sort_values(event_time_col)

    # st.write("df_tracking b", df_tracking.shape)
    # st.write(df_tracking)

    # current_pass = df_passes.iloc[0]
    # if len(df_passes) > 1:
    #     next_pass_index = 1
    #     # st.write("df_passes", next_pass_index)
    #     # st.write(df_passes)
    #     # st.write(df_passes.iloc[next_pass_index])
    #     next_pass_candidate = df_passes.iloc[next_pass_index]
    # else:
    #     next_pass_candidate = None
    slugified_match_string = df_passes["slugified_match_string"].iloc[0]

    img_files = []
    columns = st.columns(3)
    for frame_nr, (frame, df_tracking_frame) in defensive_network.utility.general.progress_bar(enumerate(df_tracking.groupby(frame_col)), total=len(df_tracking[frame_col].unique())):
        time = df_tracking_frame[tracking_time_col].iloc[0]
        time_str = str(time).replace("+", "_").replace(":", "_")

        # columns[frame_nr % 3].write(plt.gcf())
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"{slugified_match_string}_{time_str}.png"))
        if overwrite_if_exists or not os.path.exists(path):
            current_pass = df_passes.iloc[df_tracking_frame["closest_pass"].iloc[0]]
            is_during_pass = df_tracking_frame["is_during_pass"].iloc[0]

            match_string = df_passes["match_string"].iloc[0]
            mmss = df_tracking_frame["mmss"].iloc[0]
            subtype_str = f"{current_pass['event_subtype']} " if not pd.isna(current_pass["event_subtype"]) else ""

            plot_pass(current_pass, df_tracking_frame, make_pass_transparent=not is_during_pass)
            plt.title(f"{match_string} {mmss}\n{subtype_str}{current_pass['outcome']} {current_pass['pass_xt']:.3f} xT, {current_pass['xpass']:.1%} xPass {current_pass['player_name_1']} -> {current_pass['player_name_2']}")

            plt.savefig(path)
            # st.write(f"Saved {path}")
            plt.close()
        else:
            pass
            # st.write(f"Skipped {path}")
        img_files.append(path)

    create_video(sorted(img_files), out_fpath)


def create_video(fpaths, target_fpath):
    import cv2
    first_frame = cv2.imread(fpaths[0])
    height, width, layers = first_frame.shape

    video = cv2.VideoWriter(target_fpath, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))

    for image_file in fpaths:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()
    st.write(f"Done {target_fpath}")
    return
