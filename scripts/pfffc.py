import functools
import gc
import importlib
import json
import os.path
import re
import time

import slugify.slugify
import kloppy.pff

import accessible_space
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import defensive_network.parse.dfb.cdf
import defensive_network.models.involvement
import defensive_network.models.expected_receiver
import defensive_network.models.formation_v2
import defensive_network.parse.drive
import defensive_network.utility.pitch
import defensive_network.models.responsibility

importlib.reload(defensive_network.models.expected_receiver)
importlib.reload(defensive_network.models.formation_v2)
importlib.reload(defensive_network.parse.dfb.cdf)

def fuzzy_spatial_merge(df_event, df_tracking, max_distance=0.5):
    """
    Merge df_event with df_tracking based on closest x_tracking and y_tracking
    within a given distance threshold, for each player_id.

    Parameters:
    - df_event: DataFrame with columns ["player_id", "x_tracking", "y_tracking"]
    - df_tracking: DataFrame with columns ["player_id", "x_tracking", "y_tracking", "frame"]
    - max_distance: float, max distance allowed for a match

    Returns:
    - merged DataFrame with event and matching tracking info
    """
    merged_rows = []

    # Group by player_id
    for pid, event_group in df_event.groupby("player_id"):
        if pid not in df_tracking["player_id"].values:
            continue
        tracking_group = df_tracking[df_tracking["player_id"] == pid]

        # Build KDTree for tracking group
        tracking_coords = tracking_group[["x_tracking", "y_tracking"]].values
        if tracking_coords.shape[0] == 0:
            continue
        tree = cKDTree(tracking_coords)

        # Query closest tracking point for each event point
        event_coords = event_group[["x_tracking", "y_tracking"]].values
        distances, indices = tree.query(event_coords, distance_upper_bound=max_distance)

        # Filter valid matches
        valid = distances != np.inf
        matched_events = event_group.iloc[valid].copy()
        matched_events["tracking_index"] = indices[valid]

        tracking_reset = tracking_group.reset_index()
        merged = matched_events.merge(tracking_reset, left_on="tracking_index", right_on="index", suffixes=("_event", "_tracking"))
        merged_rows.append(merged)

    # Combine all merged parts
    if merged_rows:
        return pd.concat(merged_rows, ignore_index=True).drop(columns=["tracking_index", "index"])
    else:
        return pd.DataFrame()  # Empty if no matches


# @st.cache_resource
def process_pfffc(event_file):
    file = event_file

    def _get_data(file):

        # file = os.path.join(event_path, file)

        tracking_file = file.replace("Event", "Tracking").replace(".json", ".jsonl.bz2")
        meta_file = file.replace("Event Data", "Metadata")
        roster_file = file.replace("Event Data", "Rosters")

        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
        with open(roster_file, 'r') as f:
            roster_data = json.load(f)

        df_meta = pd.json_normalize(meta_data)
        meta = df_meta.iloc[0]

        df_roster = pd.json_normalize(roster_data)
        player2team = {str(int(player["player.id"])): player["team.id"] for player in
                       df_roster.to_dict(orient="records")}

        # player2team.update({float(player["player.id"]): player["team.id"] for player in df_roster.to_dict(orient="records")})

        # tracking_file = "C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/pff/Tracking Data/3812.jsonl/3812.jsonl"
        with st.spinner("Reading tracking data..."):
            df_tracking2 = pd.read_json(tracking_file, lines=True, compression='bz2')
            st.write("period a", tracking_file, event_file)
            st.write(df_tracking2.columns)
            st.write(df_tracking2["period"].unique())
            st.write(df_tracking2[["videoTimeMs", "frameNum", "period", "periodGameClockTime", "game_event_id"]])
            with open(file, 'r') as f:
                event_data = json.load(f)
            df_event = pd.json_normalize(event_data)  #remove later
            st.write("df_event")
            st.write(df_event)

        event_id_to_frame = df_tracking2[["game_event_id", "frameNum"]].dropna().set_index("game_event_id")[
            "frameNum"].to_dict()

        st.write("df_tracking2.head()")
        st.write(df_tracking2.head())
        st.write(df_tracking2["period"].unique())

        # df_tracking2["game_event_type"] = df_tracking2["game_event"].apply(lambda x: x["game_event_type"] if isinstance(x, dict) else None)
        # df_tracking2["possession_event_type"] = df_tracking2["possession_event"].apply(lambda x: x["possession_event_type"] if isinstance(x, dict) else None)
        #
        # inout = {
        #     "OUT": 0,
        #     "OTB": 1,
        #     "FIRSTKICKOFF": 1,
        #     "SECONDKICKOFF": 1,
        #     "THIRDKICKOFF": 1,
        #     "FOURTHKICKOFF": 1,
        #     "END": 0,
        # }
        # df_tracking2["ball_status"] = df_tracking2["game_event_type"].apply(lambda x: inout.get(x, None)).ffill().bfill()
        # # st.write(df_tracking2[["frameNum", "ball_status"]].drop_duplicates().sort_values("frameNum").head(10000))
        # # frame2ball_status = df_tracking2.set_index("frameNum")["ball_status"].to_dict()
        #
        del df_tracking2
        gc.collect()

        # df_tracking2 = df_tracking2.drop(columns=["balls", "homePlayers", "awayPlayers", "game_event", "possession_event"])
        # df_tracking2 = df_tracking2.explode("homePlayersSmoothed")
        # df_tracking2 = df_tracking2.explode("awayPlayersSmoothed")

        # @st.cache_resource
        def _get_dataa():
            df = kloppy.pff.load_tracking(
                meta_data=meta_file,
                roster_meta_data=roster_file,
                raw_data=tracking_file,

                # Optional Parameters
                coordinates="pff",
                sample_rate=None,
                limit=None,
                only_alive=False,
            ).to_df()
            return df

        df = _get_dataa()
        st.write("period b")
        st.write(df["period_id"].unique())

        x_cols = [col for col in df.columns if col.endswith("_x")]
        y_cols = [col for col in df.columns if col.endswith("_y")]
        players = [col.split("_x")[0] for col in x_cols]
        coordinate_cols = [[x_col, y_col] for x_col, y_col in zip(x_cols, y_cols)]
        df_tracking = accessible_space.per_object_frameify_tracking_data(
            df, frame_col="frame_id", coordinate_cols=coordinate_cols, players=players, player_to_team=player2team
        ).drop(columns=[col for col in df.columns if col.endswith("_d") or col.endswith("_s")])
        df_tracking = df_tracking[df_tracking["x"].notna() & df_tracking["y"].notna()]
        st.write("period c")
        st.write(df_tracking["period_id"].unique())
        # fr = 4226
        # st.write('df_tracking[df_tracking["frame_id"] == fr]', df_tracking[df_tracking["frame_id"] == fr].shape)
        # st.write(df_tracking[df_tracking["frame_id"] == fr])

        # df_tracking["timestamp_str"] = df_tracking["timestamp"].astype(str)

        with open(file, 'r') as f:
            event_data = json.load(f)

        df_event = pd.json_normalize(event_data)
        match_id = int(os.path.basename(file).split(".")[0])
        st.write(f"{match_id=}")
        if match_id in [10510, 10511]:  # Argentina - Netherlands and Croatia - Brazil lack tracking data for extra time
            df_event = df_event[df_event["gameEvents.period"].isin([1, 2])]
        if match_id == 3833:  # Poland - Saudi Arabia somehow has 1 event at period 0 with no tracking data
            df_event = df_event[df_event["gameEvents.period"].isin([1, 2])]
        st.write(df_event)
        return df_event, df_tracking, meta, df_roster, player2team, event_id_to_frame

    df_event, df_tracking, meta, df_roster, player2team, event_id_to_frame = _get_data(file)

    # assert set(df_tracking["player_id"]) == set(df_tracking_reduced["playerId"])

    fps = meta["fps"]

    # df_event["frame"] = df_event["startTime"] * fps  # TODO wrong - loses nice synchro information
    df_event["frame"] = None  # TODO wrong - loses nice synchro information
    # df_event["frame_rec"] = df_event["endTime"] * fps  # TODO wrong - loses nice synchro information
    df_event["frame_rec"] = None  # TODO wrong - loses nice synchro information

    # game_clock

    # st.stop()

    df_event["datetime_event"] = pd.to_datetime(df_event["startTime"], unit='s', utc=True)
    df_event = df_event.rename(columns={
        "gameEvents.playerId": "player_id_1",
        "possessionEvents.targetPlayerId": "player_id_2",
        "gameEvents.period": "section",
    })

    df_event["event_subtype"] = df_event["gameEvents.gameEventType"].apply(lambda x: {
        "END": "final_whistle",
        "FIRSTKICKOFF": "kick_off",
        "SECONDKICKOFF": "kick_off",
        "THIRDKICKOFF": "kick_off",
        "FOURTHKICKOFF": "kick_off",
    }.get(x, None))
    i_challenge = df_event["possessionEvents.challengeWinnerPlayerId"].notna() & df_event["possessionEvents.homeDuelPlayerId"].notna() & df_event["possessionEvents.awayDuelPlayerId"].notna()
    df_event.loc[i_challenge, "event_subtype"] = "tackle"
    df_event.loc[i_challenge, "player_id_1"] = df_event.loc[i_challenge, "possessionEvents.challengeWinnerPlayerId"]
    df_event.loc[i_challenge, "player_id_2"] = df_event.loc[i_challenge].apply(lambda x: x["possessionEvents.homeDuelPlayerId"] if x["possessionEvents.challengeWinnerPlayerId"] == x["possessionEvents.awayDuelPlayerId"] else x["possessionEvents.awayDuelPlayerId"], axis=1)
    assert "tackle" in df_event["event_subtype"].unique(), "Tackle events must be present in the data"

    i_nan = df_event["event_subtype"].isna()
    df_event.loc[i_nan, "event_subtype"] = df_event.loc[i_nan, "possessionEvents.possessionEventType"].apply(
        lambda x: {
            "PA": "pass",
            "SH": "shot",
        }.get(x, None))
    df_event["event_type"] = df_event["event_subtype"]
    df_event["event_outcome"] = df_event["possessionEvents.passOutcomeType"].apply(lambda x: {
        "C": "successfully_completed",
        "B": "unsuccessful",
        "D": "unsuccessful",
        "G": None,
        "I": None,
        "O": "unsuccessful",
        "S": "unsuccessful",
    }.get(x, None))
    i_unsuccessful = (df_event["event_outcome"] == "unsuccessful") & (df_event["event_type"] == "pass")
    df_event.loc[i_unsuccessful, "player_id_2"] = None
    i_intercepted = df_event["possessionEvents.defenderPlayerId"].notna()
    df_event.loc[i_intercepted, "player_id_2"] = df_event.loc[i_intercepted, "possessionEvents.defenderPlayerId"]
    df_event.loc[i_intercepted, "outcome"] = "intercepted"
    df_event["team_id_2"] = df_event["player_id_2"].map(player2team)
    i_pass = df_event["event_type"] == "pass"
    df_event.loc[i_pass, "pass_is_intercepted"] = (df_event.loc[i_pass, "event_outcome"] == "unsuccessful") & ~pd.isna(
        df_event.loc[i_pass, "player_id_2"])
    assert df_event.loc[i_pass, "pass_is_intercepted"].sum() > 0  # todo away
    assert "tackle" in df_event["event_subtype"].unique(), "Tackle events must be present in the data"

    df_tracking["ball_state"] = df_tracking["ball_state"].map({"alive": 1, "dead": 0})

    # df_event["event_subtype"] = df_event["event_subtype"].fillna()

    def foo(x, col="x", player_col="player_id_1"):
        if pd.isna(x[player_col]):
            return pd.NA
        for player in x["homePlayers"] + x["awayPlayers"]:
            if str(player["playerId"]) == str(x[player_col]):
                return player[col]
        return pd.NA
        # lst = [player for player in x["homePlayers"] + x["awayPlayers"] if str(player["playerId"]) == x["player_id_1"]]
        # return lst[0][col]

    # def foo_y(x):
    #     if pd.isna(x['player_id_1']):
    #         return pd.NA
    #     lst = [player for player in x["homePlayers"] + x["awayPlayers"] if player["playerId"] == x["player_id_1"]]
    #     return lst[0][col]

    df_event["player_id_1"] = df_event["player_id_1"].astype(str).str.replace(".0", "")
    df_event["player_id_2"] = df_event["player_id_2"].astype(str).str.replace(".0", "")
    df_event["team_id_1"] = df_event["player_id_1"].map(player2team)
    assert df_event["team_id_1"].notna().any()

    df_event["x_event_player_1"] = df_event.apply(foo, axis=1)
    df_event["y_event_player_1"] = df_event.apply(functools.partial(foo, col="y", player_col="player_id_1"), axis=1)
    df_event["x_event_player_2"] = df_event.apply(functools.partial(foo, col="x", player_col="player_id_2"), axis=1)
    df_event["y_event_player_2"] = df_event.apply(functools.partial(foo, col="y", player_col="player_id_2"), axis=1)

    for col in ["x_event_player_1", "y_event_player_1", "x_event_player_2", "y_event_player_2"]:
        df_event[col.replace("event", "tracking")] = df_event[col]

    start_time = pd.Timestamp("2023-01-01 00:00:00")  # or whatever your base time is
    section_time = pd.Timedelta(hours=4)
    df_tracking["timestamp"] = pd.to_datetime(
        start_time + df_tracking["timestamp"] + df_tracking["period_id"].astype(float) * section_time, utc=True)
    # df_tracking["timestamp"] = pd.to_datetime(start_time + df_tracking["timestamp"], utc=True)
    # st.stop()

    # df_tracking["timestamp"] = pd.to_datetime(df_tracking["timestamp"])
    df_tracking = df_tracking.rename(columns={
        "timestamp": "datetime_tracking", "x": "x_tracking", "y": "y_tracking", "period_id": "section",
        "frame_id": "frame", "ball_owning_team_id": "ball_poss_team_id", "ball_state": "ball_status",
    })
    st.write(df_tracking["section"].unique())

    assert len(df_tracking["section"].unique()) >= 2
    assert len(df_event["section"].unique()) >= 2

    # "possessionEvents":{
    # "possessionEventType":"PA"

    df_roster = df_roster.rename(columns={
        "player.nickname": "short_name",
        "player.id": "player_id",
        "team.name": "team_name",
        "team.id": "team_id",
        "positionGroupType": "position"
    })

    keep_event_cols = ["gameId", "gameEventId", "startTime", "endTime", "duration", "eventTime", "sequence",
                       "gameEvents.startGameClock", "gameEvents.startFormattedGameClock", "section",
                       "gameEvents.teamId", "gameEvents.teamName", "gameEvents.gameEventType"]
    dfs_tracking = []
    for col in ["homePlayers", "awayPlayers", "ball"]:
        df_tracking_reduced = df_event.explode(col)
        df_tracking_reduced = pd.concat(
            [df_tracking_reduced[keep_event_cols], df_tracking_reduced[col].apply(pd.Series)], axis=1)
        df_tracking_reduced["player_team"] = col
        if col == "ball":
            df_tracking_reduced["playerId"] = "BALL"
            df_tracking_reduced["player_team"] = "BALL"
        else:
            df_tracking_reduced["playerId"] = df_tracking_reduced["playerId"].astype(str)

        dfs_tracking.append(df_tracking_reduced)

    df_tracking["player_id"] = df_tracking["player_id"].apply(lambda x: {"ball": "BALL"}.get(x, x))
    assert "BALL" in df_tracking["player_id"].unique()
    df_tracking_reduced = pd.concat(dfs_tracking)
    df_tracking_reduced = df_tracking_reduced.rename(columns={
        "x": "x_tracking",
        "y": "y_tracking",
        "playerId": "player_id"
    })

    df_event["frame"] = df_event["gameEventId"].map(event_id_to_frame)
    df_event["frame_rec"] = (df_event["frame"] + df_event["duration"] * fps).round()

    # df_tracking = df_tracking[
    #     (df_tracking["framed"] % 10 == 0) | (df_tracking["frame"].isin(df_event["frame"].unique()))  # keep only every 10th frame or frames that match an event frame
    # ]

    # df_tracking["x_tracking"] = df_tracking["x_tracking"].round(3)
    # df_tracking["y_tracking"] = df_tracking["y_tracking"].round(3)
    # df_tracking_reduced["x_tracking"] = df_tracking_reduced["x_tracking"].round(3)
    # df_tracking_reduced["y_tracking"] = df_tracking_reduced["y_tracking"].round(3)
    # df_test = df_tracking_reduced[["player_id", "x_tracking", "y_tracking", "gameEventId"]].dropna().merge(df_tracking[["player_id", "x_tracking", "y_tracking", "frame"]].dropna(), on=["player_id", "x_tracking", "y_tracking"], how="left").dropna()
    # # make sure there are no frame-event id duplicates
    # st.stop()

    # for event_id, df_tracking_event in df_tracking_reduced.sort_values("gameEvents.startGameClock").groupby("gameEventId"):
    #     for frame, df_tracking_candidate in df_tracking[df_tracking["x_tracking"].notna()].groupby("frame"):
    #
    #         df_test = fuzzy_spatial_merge(
    #             df_event=df_tracking_event[["gameEventId", "player_id", "x_tracking", "y_tracking"]],
    #             df_tracking=df_tracking[["player_id", "x_tracking", "y_tracking", "frame"]].dropna(),
    #             max_distance=1.0
    #         )
    #
    #
    #
    #         # df_test = df_tracking_event[["gameEventId", "player_id", "x_tracking", "y_tracking"]].merge(df_tracking[["player_id", "x_tracking", "y_tracking", "frame"]], on=["player_id", "x_tracking", "y_tracking"], how="left")
    #         st.stop()
    #
    #         # TODO better solution maybe: replace tracking data with partial tracking data - is the full one really required?

    df_tracking["player_id"] = df_tracking["player_id"].astype(str)

    # df_tracking["player_id"] = "p" + df_tracking["player_id"]
    # # map "pBALL" to "BALL
    # df_tracking["player_id"] = df_tracking["player_id"].replace({"pBALL": "BALL"})
    # assert "BALL" in df_tracking["player_id"].unique(), "BALL must be in player_id"

    df_event["player_id_2"] = df_event["player_id_2"].astype(str).replace({"None": None, "nan": None, "NaN": None})
    df_event["player_id_1"] = df_event["player_id_1"].astype(str).replace({"None": None, "nan": None, "NaN": None})
    df_event["team_id_1"] = df_event["team_id_1"].astype(str).replace({"None": None, "nan": None, "NaN": None})
    df_event["team_id_2"] = df_event["team_id_2"].astype(str).replace({"None": None, "nan": None, "NaN": None})

    df_tracking["team_id"] = df_tracking["team_id"].astype(str).replace({"None": None, "nan": None, "NaN": None})
    # df_event["player_id_1"] = "p" + df_event["player_id_1"].astype(str).str.replace(".0", "").replace({"pNone": None})
    # df_event["player_id_2"] = "p" + df_event["player_id_2"].astype(str).str.replace(".0", "").replace({"pNone": None})
    # df_event["team_id_1"] = "t" + df_event["team_id_1"].astype(str).str.replace(".0", "").replace({"tNone": None})
    # df_event["team_id_2"] = "t" + df_event["team_id_2"].astype(str).str.replace(".0", "").replace({"tNone": None})
    # df_tracking["team_id"] = "t" + df_tracking["team_id"].astype(str).str.replace(".0", "").replace({"tNone": None})
    # df_tracking_reduced["player_id"] = "p" + df_tracking_reduced["player_id"].astype(str)
    # df_tracking_reduced["team_id"] = "t" + df_tracking_reduced["team_id"].astype(str)

    assert df_event["player_id_2"].isna().any()
    df_tracking = df_tracking.drop_duplicates(keep="first", subset=["frame", "player_id"])

    st.write("sections")
    st.write(df_tracking["section"].unique())
    st.write(df_event["section"].unique())

    df_tracking, df_event = defensive_network.parse.dfb.cdf.augment_match_data(
        meta, df_event, df_tracking, df_roster, edit_section=False
    )
    assert df_tracking["vx"].notna().all()
    df_tracking.reset_index().to_parquet("test_tracking.parquet", index=False)
    df_event.reset_index().to_csv("test_event.csv", index=False)

    #### INVOLVEMENT
    df_tracking["team_id"] = df_tracking["team_id"].astype(str).str.replace(".0", "").replace(
        {"None": None, "nan": None, "NaN": None})
    df_event["team_id_1"] = df_event["team_id_1"].astype(str).str.replace(".0", "").replace(
        {"None": None, "nan": None, "NaN": None})
    df_event["team_id_2"] = df_event["team_id_2"].astype(str).str.replace(".0", "").replace(
        {"None": None, "nan": None, "NaN": None})
    df_event["player_id_1"] = df_event["player_id_1"].astype(str).str.replace(".0", "").replace(
        {"None": None, "nan": None, "NaN": None})
    df_event["player_id_2"] = df_event["player_id_2"].astype(str).str.replace(".0", "").replace(
        {"None": None, "nan": None, "NaN": None})

    st.write("Ch E")
    st.write(df_event.shape)
    df_event = df_event.drop_duplicates(subset=["gameEventId"])
    st.write(df_event.shape)

    st.write("Ch T")
    st.write(df_tracking.shape)
    df_tracking = df_tracking.drop_duplicates(subset=["frame", "player_id"])
    st.write(df_tracking.shape)

    df_involvement = defensive_network.models.involvement.get_involvement(df_event, df_tracking,
                                                                          tracking_defender_meta_cols=[
                                                                              "role_category"])

    df_involvement["network_receiver_role_category"] = df_involvement["expected_receiver_role_category"].where(
        df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])
    df_involvement["defender_role_category"] = df_involvement["defender_role_category"]  # .fillna("unknown")
    df_involvement["role_category_1"] = df_involvement["role_category_1"]  # .fillna("unknown")
    df_involvement["network_receiver_role_category"] = df_involvement[
        "network_receiver_role_category"]  # .fillna("unknown")
    df_involvement = df_involvement.dropna(
        subset=["defender_id", "defender_role_category", "role_category_1", "network_receiver_role_category"],
        how="any")
    intrinsic_context_cols = ["defending_team", "role_category_1", "network_receiver_role_category",
                              "defender_role_category"]

    df_involvement = df_involvement.drop_duplicates(["defender_id", "involvement_pass_id"])

    dfg_responsibility = defensive_network.models.responsibility.get_responsibility_model(df_involvement,
                                                                                          responsibility_context_cols=intrinsic_context_cols)
    df_involvement["raw_intrinsic_responsibility"], df_involvement["raw_intrinsic_relative_responsibility"], \
        df_involvement["valued_intrinsic_responsibility"], df_involvement[
        "valued_intrinsic_relative_responsibility"] = defensive_network.models.responsibility.get_responsibility(
        df_involvement, dfg_responsibility_model=dfg_responsibility, context_cols=intrinsic_context_cols)

    defensive_network.utility.pitch.plot_passes_with_involvement(
        df_involvement, df_tracking, responsibility_col="raw_intrinsic_relative_responsibility", n_passes=5
    )
    assert len(df_involvement) > 100

    return df_event, df_tracking, meta, df_roster, df_involvement


def main():
    path = st.text_input("Enter the path to the raw data file:", "Y:/misc/FIFA World Cup 2022/")# os.path.join(os.path.dirname(__file__), "../pff"))
    event_path = os.path.join(path, "Event Data")
    tracking_path = os.path.join(path, "Tracking Data")
    files = [f for f in os.listdir(event_path) if f.endswith("json")]
    st.write("Files found:", files)
    import defensive_network.utility.general

    existing_involvement_files = defensive_network.parse.drive.list_files_in_drive_folder("tracking")

    existing_slugs = [file["name"].split(".")[0] for file in existing_involvement_files]

    overwrite_if_exists = st.checkbox("Overwrite existing files if they exist", value=False)

    @st.cache_resource
    def _get_slug(event_file):
        gc.collect()
        meta_file = event_file.replace("Event Data", "Metadata")
        meta = pd.read_json(meta_file)
        meta["competition.name"] = meta["competition"].apply(lambda x: x["name"] if isinstance(x, dict) else x)
        meta["homeTeam.name"] = meta["homeTeam"].apply(lambda x: x["name"] if isinstance(x, dict) else x)
        meta["awayTeam.name"] = meta["awayTeam"].apply(lambda x: x["name"] if isinstance(x, dict) else x)
        meta = meta.iloc[0]
        match_string = f"{meta['competition.name']} {meta['season']}: {meta['week']}.ST {meta['homeTeam.name']} - {meta['awayTeam.name']}"
        meta["slugified_match_string"] = slugify.slugify(match_string)
        return meta["slugified_match_string"]

    for event_file in defensive_network.utility.general.progress_bar(files, desc="Processing PFF FC data"):
        event_file = os.path.join(event_path, event_file)

        slugified_match_string = _get_slug(event_file)
        # if not overwrite_if_exists and slugified_match_string in existing_slugs:
        #     st.info(f"Skipping {slugified_match_string} as it already exists in the drive.")
            # continue

        try:
            df_event, df_tracking, meta, df_roster, df_involvement = process_pfffc(event_file)
        except (AssertionError, KeyError) as e:
            st.write(e)
            raise e
            continue

        gc.collect()
        time.sleep(5)
        gc.collect()

        ###
        player_group_cols = ["player_id_1", "role_category_1"]
        receiver_group_cols = ["player_id_2", "role_category_2"]
        involvement_group_cols = ["defender_id", "defender_role_category"]
        df_event["event_id"] = df_event["gameEventId"].astype(str)
        dfg_players_interceptions = df_event[df_event["outcome"] == "intercepted"].groupby(receiver_group_cols, observed=False).agg(
            n_interceptions=("event_id", "count"),
        ).fillna(0)
        dfg_players_interceptions["n_interceptions"] = dfg_players_interceptions["n_interceptions"].fillna(0)
        dfg_players_interceptions = dfg_players_interceptions.rename(columns={"player_id_2": "player_id_1"})
        assert len(dfg_players_interceptions) > 0, "No interceptions found in the data"

        ###

        # st.write("df_event")
        # st.write(df_event)
        # st.write(df_event[["player_id_1", "player_id_2", "event_subtype"] + [col for col in df_event.columns if "challenge" in col.lower() or "duel" in col.lower()]])

        i_challenge = df_event["possessionEvents.challengeWinnerPlayerId"].notna() & df_event["possessionEvents.homeDuelPlayerId"].notna() & df_event["possessionEvents.awayDuelPlayerId"].notna()
        # df_event.loc[i_challenge, "event_subtype"] = "tackle"
        # df_event.loc[i_challenge, "player_id_1"] = df_event.loc[i_challenge, "possessionEvents.challengeWinnerPlayerId"]
        # df_event.loc[i_challenge, "player_id_2"] = df_event.loc[i_challenge].apply(lambda x: x["possessionEvents.homeDuelPlayerId"] if x["possessionEvents.challengeWinnerPlayerId"] == x["possessionEvents.awayDuelPlayerId"] else x["possessionEvents.awayDuelPlayerId"], axis=1)

        assert "tackle" in df_event["event_subtype"].unique(), "Tackle events must be present in the data"

        # rename some stuff from the meta series
        meta = meta.rename({
            "competition.id": "competition_id",
            "competition.name": "competition_name",
            "week": "match_day",
            "season": "season_name",
            "awayTeam.id": "guest_team_id",
            "awayTeam.name": "guest_team_name",
            "homeTeam.id": "home_team_id",
            "homeTeam.name": "home_team_name",
            "id": "match_id",
        })
        meta["season_id"] = meta["season_name"]
        meta["host"] = "FIFA"
        meta["event_vendor"] = "pff_fc"
        meta["tracking_vendor"] = "pff_fc"
        meta["match_string"] = f"{meta['competition_name']} {meta['season_name']}: {meta['match_day']}.ST {meta['home_team_name']} - {meta['guest_team_name']}"
        meta["slugified_match_string"] = slugify.slugify(meta["match_string"])

        df_roster["match_id"] = meta["match_id"]
        df_roster["team_role"] = df_roster["team_id"].apply(lambda x: "home" if x == meta["home_team_id"] else "away")
        df_roster = df_roster.rename(columns={
            "shirtNumber": "jersey_number",
            "started": "starting",
        })
        df_roster["first_name"] = df_roster["short_name"].str.split(" ").str[0]
        df_roster["last_name"] = df_roster["short_name"].str.split(" ").str[1:].str.join(" ")
        df_roster["position_group"] = None
        df_roster["captain"] = None

        df_meta = pd.DataFrame([meta])

        df_event["match_id"] = meta["match_id"]
        df_event["slugified_match_string"] = meta["slugified_match_string"]
        df_event["match_string"] = meta["match_string"]
        df_tracking["match_id"] = meta["match_id"]
        df_tracking["slugified_match_string"] = meta["slugified_match_string"]
        df_tracking["match_string"] = meta["match_string"]
        df_involvement["match_id"] = meta["match_id"]
        df_involvement["slugified_match_string"] = meta["slugified_match_string"]
        df_involvement["match_string"] = meta["match_string"]

        defensive_network.parse.drive.append_to_parquet_on_drive(df_meta, "meta.csv", format="csv", key_cols=["match_id"])
        defensive_network.parse.drive.append_to_parquet_on_drive(df_roster, "lineups.csv", format="csv", key_cols=["match_id", "team_id", "player_id", "position"])

        slugified_match_string = meta["slugified_match_string"]
        defensive_network.parse.drive.upload_csv_to_drive(df_involvement, f"involvement/{slugified_match_string}.csv")

        df_tracking.to_parquet(f"Y:/w_raw/finalized/tracking/{slugified_match_string}.parquet", index=False)

        defensive_network.parse.drive.upload_csv_to_drive(df_event, f"events/{slugified_match_string}.csv")

        # df_event["original_full_frame"] = df_event["section"].str.cat(df_event["original_frame_id"].astype(float).astype(str), sep="-")
        # df_event["matched_full_frame"] = df_event["section"].str.cat(df_event["matched_frame_id"].astype(float).astype(str), sep="-")
        df_tracking_reduced = df_tracking[
            df_tracking["full_frame"].isin(df_event["full_frame"])
            # df_tracking["full_frame"].isin(df_event["original_full_frame"]) |
            # df_tracking["full_frame"].isin(df_event["matched_full_frame"])
        ]
        df_tracking_reduced = df_tracking_reduced.drop_duplicates()
        defensive_network.parse.drive.upload_parquet_to_drive(df_tracking_reduced, f"tracking/{slugified_match_string}.parquet")

        st.write("DONE", slugified_match_string)


# def main2():
#     @st.cache_resource
#     def _read_data():
#         df_tracking = pd.read_parquet("test_tracking.parquet")
#         df_event = pd.read_csv("test_event.csv")
#         return df_tracking, df_event
#     df_tracking, df_event = _read_data()
#
#     df_tracking = df_tracking.drop_duplicates(keep="first", subset=["frame", "player_id"])
#
#     assert df_tracking["vx"].notna().all()
#     df_tracking.to_parquet("test_tracking.parquet", index=False)


    # df_tracking = pd.read_parquet("test_tracking.parquet")
    # df_event = pd.read_csv("test_event.csv")

    # replace .0 with empty string in all team cols but let None stay None


if __name__ == '__main__':
    main()
