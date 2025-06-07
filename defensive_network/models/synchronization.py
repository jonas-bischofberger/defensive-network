import importlib

import pandas as pd
import numpy as np
import etsy.sync
import etsy.scoring
import streamlit as st

import collections

importlib.reload(etsy.sync)
importlib.reload(etsy.scoring)

# from defensive_network.tests.data import df_events, df_tracking

SynchronizationResult = collections.namedtuple("SynchronizationResult", ["matched_frames", "scores"])


def synchronize(df_events, df_tracking, fps_tracking=25):
    df_events = df_events[
        (df_events["event_type"] != "referee") &
        (df_events["player_id_1"].notna())
    ].reset_index()

    df_events["period_id"] = df_events["section"].map({"first_half": 1, "second_half": 2})
    df_events["frame_id"] = df_events["frame"]
    df_events["timestamp"] = pd.to_datetime(df_events["datetime_event"]).dt.tz_localize(None)
    df_events["player_id"] = df_events["player_id_1"]#.fillna("dummy")
    df_events["type_name"] = "pass"
    df_events["start_x"] = np.clip(df_events["x_event"].astype(float) + 52.5, 0, 105)
    df_events["start_y"] = np.clip(df_events["y_event"].astype(float) + 34.0, 0, 68)
    df_events["bodypart_id"] = 0

    df_tracking["period_id"] = df_tracking["section"].map({"first_half": 1, "second_half": 2})
    df_tracking["timestamp"] = pd.to_datetime(df_tracking["datetime_tracking"]).dt.tz_localize(None)
    df_tracking["frame"] = df_tracking["frame"]
    df_tracking["ball"] = df_tracking["player_id"] == "BALL"
    assert df_tracking["ball"].any()
    df_tracking["x"] = np.clip(df_tracking["x_tracking"].astype(float) + 52.5, 0, 105)
    df_tracking["y"] = np.clip(df_tracking["y_tracking"].astype(float) + 34.0, 0, 68)
    df_tracking["z"] = 0.0
    df_tracking["acceleration"] = 0.0

    # df_events = df_events[df_events["frame"] < 5000].reset_index(drop=True)
    # df_tracking = df_tracking[df_tracking["frame"] < 5000].reset_index(drop=True)
    df_events = df_events.reset_index(drop=True)
    df_tracking = df_tracking.reset_index(drop=True)

    # Initialize event-tracking synchronizer with given event data (df_events),
    # tracking data (df_tracking), and recording frequency of the tracking data (fps_tracking)
    ETSY = etsy.sync.EventTrackingSynchronizer(df_events, df_tracking, fps=fps_tracking)

    # Run the synchronization
    ETSY.synchronize()

    # Inspect the matched frames and scores
    df_events["matched_frame"] = ETSY.matched_frames
    df_events["scores"] = ETSY.scores

    df_events = df_events.set_index("index")

    return SynchronizationResult(matched_frames=df_events["matched_frame"], scores=df_events["scores"])


if __name__ == '__main__':
    df_events = _get_csv("events/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.csv").reset_index(drop=True)
    # df_tracking = _get_parquet("tracking/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.parquet").reset_index(drop=True)
    df_tracking = _get_local_parquet("Y:/w_raw/preprocessed/tracking/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.parquet").reset_index(drop=True)

    res = synchronize(df_events, df_tracking)
    st.write("res")
    st.write(res.matched_frames)
    st.write(res.scores)
    df_events["matched_frame"] = res.matched_frames
    df_tracking["matched_frame"] = res.matched_frames

    st.write("df_events with matched frames")
    st.write(df_events)
