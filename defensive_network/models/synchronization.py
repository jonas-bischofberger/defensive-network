import collections

import etsy.sync

import defensive_network.tests.data
import defensive_network.utility.dataframes

SynchronizationResult = collections.namedtuple("SynchronizationResult", ["matched_frames", "scores"])


def synchronize(
    df_tracking, df_events, fps_tracking, event_period_col="period_id", tracking_period_col="period_id",
    tracking_frame_col="frame_id", event_timestamp_col="timestamp", event_passer_col="player_id", event_x_col="x",
    event_y_col="y", ball_player_id="ball", tracking_timestamp_col="timestamp", tracking_x_col="x", tracking_y_col="y",
    tracking_player_col="player_id",
):
    """
    Wrapper around ETSY https://github.com/ML-KULeuven/ETSY

    >>> defensive_network.utility.dataframes.prepare_doctest()
    >>> df_tracking, df_events = defensive_network.tests.data.get_minimal_sync_example()
    >>> df_tracking
        period_id  frame_id               timestamp     x     y player_id
    0           1       100 2025-01-01 00:00:00.000  50.0  30.0         A
    1           1       101 2025-01-01 00:00:00.040  51.0  30.5         A
    2           1       102 2025-01-01 00:00:00.080  52.0  31.0         A
    3           1       103 2025-01-01 00:00:00.120  53.0  31.5         A
    4           1       100 2025-01-01 00:00:00.000  50.0  30.0      ball
    5           1       101 2025-01-01 00:00:00.040  51.0  30.5      ball
    6           1       102 2025-01-01 00:00:00.080  52.0  31.0      ball
    7           1       103 2025-01-01 00:00:00.120  53.0  31.5      ball
    8           2       200 2025-01-01 00:00:10.000  59.0  39.5         A
    9           2       201 2025-01-01 00:00:10.040  60.0  40.0         A
    10          2       202 2025-01-01 00:00:10.080  61.0  40.5         A
    11          2       203 2025-01-01 00:00:10.120  62.0  41.0      ball
    >>> df_events
       period_id               timestamp player_id      x     y
    0          1 2025-01-01 00:00:00.080         A  52.00  31.0
    1          1 2025-01-01 00:00:00.081         A  52.01  31.0
    2          2 2025-01-01 00:00:10.040         A  60.00  40.0
    >>> df_events["etsy_frame"], df_events["etsy_score"] = synchronize(df_tracking, df_events, fps_tracking=25)
    >>> df_events
       period_id               timestamp player_id      x     y  etsy_frame  etsy_score
    0          1 2025-01-01 00:00:00.080         A  52.00  31.0       100.0         NaN
    1          1 2025-01-01 00:00:00.081         A  52.01  31.0       102.0   99.994671
    2          2 2025-01-01 00:00:10.040         A  60.00  40.0       200.0         NaN
    """
    if not {1, 2}.issubset(set(df_events[event_period_col].unique())):
        raise ValueError(f"Etsy requires period names 1 and 2 to be present in events. Found: {df_events[event_period_col].unique()}")
    if not {1, 2}.issubset(set(df_tracking[tracking_period_col].unique())):
        raise ValueError(f"Etsy requires period names 1 and 2 to be present in tracking data. Found: {df_tracking[tracking_period_col].unique()}")

    df_events = df_events[[event_period_col, event_timestamp_col, event_passer_col, event_x_col, event_y_col]].rename(columns={
        event_period_col: "period_id", event_timestamp_col: "timestamp", event_passer_col: "player_id",
        event_x_col: "start_x", event_y_col: "start_y"
    }).copy()
    df_tracking = df_tracking[[tracking_period_col, tracking_frame_col, tracking_timestamp_col, tracking_x_col, tracking_y_col, tracking_player_col]].rename(columns={
        tracking_period_col: "period_id", tracking_frame_col: "frame", tracking_timestamp_col: "timestamp",
        tracking_x_col: "x", tracking_y_col: "y", tracking_player_col: "player_id"
    }).copy()
    df_events["type_name"] = "pass"
    df_events["bodypart_id"] = 0

    df_tracking["ball"] = df_tracking["player_id"] == ball_player_id
    df_tracking["acceleration"] = 0.0
    df_tracking["z"] = 0.0

    i_valid_events = df_events[["player_id", "start_x", "start_y"]].notna().all(axis=1)
    i_valid_tracking = df_tracking[["x", "y"]].notna().all(axis=1)
    df_events["index"] = df_events.index  # preserve index
    df_tracking["index"] = df_tracking.index
    df_events = df_events.loc[i_valid_events].reset_index(drop=True)
    df_tracking = df_tracking.loc[i_valid_tracking].reset_index(drop=True)

    ETSY = etsy.sync.EventTrackingSynchronizer(df_events, df_tracking, fps=fps_tracking)
    ETSY.synchronize()

    df_events["etsy_frame"] = ETSY.matched_frames
    df_events["etsy_score"] = ETSY.scores

    df_events = df_events.set_index("index")

    return SynchronizationResult(matched_frames=df_events["etsy_frame"], scores=df_events["etsy_score"])
