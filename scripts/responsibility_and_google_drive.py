"""
Preprocess the monolithic DFB data into a more manageable format.

Preprocessed files:
- lineups.csv
- meta.csv
- events/{match_string}.csv
- tracking/{match_string}.parquet
"""

import gc
import importlib
import math
import os

import pandas as pd
import slugify
import streamlit as st

import sys

import defensive_network.utility.dataframes

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import defensive_network.utility.general
import defensive_network.parse.dfb.meta

import defensive_network.parse.dfb.cdf

import defensive_network.parse.drive
import defensive_network.models.formation
import defensive_network.models.involvement
import defensive_network.models.responsibility
import defensive_network.models.synchronization
import defensive_network.utility.pitch

importlib.reload(defensive_network.parse.dfb.meta)
importlib.reload(defensive_network.utility.general)
importlib.reload(defensive_network.parse.drive)


@st.cache_resource
def _read_df(fpath, **kwargs):
    return pd.read_csv(fpath, **kwargs)


def get_dfb_csv_files_in_folder(folder, exclude_files=None):
    tracking_cols = "frame,match_id,player_id,team_id,event_vendor,tracking_vendor,datetime_tracking,section,x_tracking,y_tracking,z_tracking,d_tracking,a_tracking,s_tracking,ball_status,ball_poss_team_id"
    event_cols = [
        "frame,match_id,event_id,event_vendor,tracking_vendor,datetime_event,datetime_tracking,event_type,event_subtype,event_outcome,player_id_1,team_id_1,player_id_2,team_id_2,x_event,y_event,x_tracking_player_1,y_tracking_player_1,x_tracking_player_2,y_tracking_player_2,section,xg,xpass,player_pressure_1,player_pressure_2,assist_action,assist_type,rotation_ball,foot,direction,origin_setup,foul_type,card_color,reason,frame_rec,packing_traditional,packing_horizontal,packing_vertical,packing_attention",
        "frame,match_id,event_id,event_vendor,tracking_vendor,datetime_event,datetime_tracking,event_type,event_subtype,event_outcome,player_id_1,team_id_1,player_id_2,team_id_2,x_event_player_1,y_event_player_1,x_event_player_2,y_event_player_2,x_tracking_player_1,y_tracking_player_1,x_tracking_player_2,y_tracking_player_2,section,xg,xpass,player_pressure_1,player_pressure_2,assist_action,assist_type,rotation_ball,foot,direction,origin_setup,foul_type,card_color,reason,frame_rec,packing_traditional,packing_horizontal,packing_vertical,packing_attention"
    ]
    lineup_cols = "match_id,event_vendor,tracking_vendor,team_id,team_name,team_role,player_id,jersey_number,first_name,last_name,short_name,position_group,position,starting,captain"
    meta_cols = "competition_name,competition_id,host,match_day,season_name,season_id,kickoff_time,match_id,event_vendor,tracking_vendor,match_title,home_team_name,home_team_id,guest_team_name,guest_team_id,result,country,stadium_id,stadium_name,precipitation,pitch_x,pitch_y,total_time_first_half,total_time_second_half,playing_time_first_half,playing_time_second_half,ds_parser_version,xg_tag,xg_sha1,xpass_tag,xpass_sha1,fps"
    metaccccc = "competition_name,competition_id,host,match_day,season_name,season_id,kickoff_time,match_id,event_vendor,tracking_vendor,match_title,home_team_name,home_team_id,guest_team_name,guest_team_id,result,country,stadium_id,stadium_name,precipitation,pitch_x,pitch_y,total_time_first_half,total_time_second_half,playing_time_first_half,playing_time_second_half,ds_parser_version,xg_tag,xg_sha1,xpass_tag,xpass_sha1,fps"

    # csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    csv_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(".csv")]
    header_to_filetype = [
        (tracking_cols, "tracking"),
        (event_cols, "event"),
        (meta_cols, "meta"),
        (lineup_cols, "lineup"),
    ]
    filetype_to_files = {}
    for csv_file in csv_files:
        if csv_file.startswith("._"):
            continue
        full_csv_file = csv_file  # os.path.join(folder, csv_file)
        if exclude_files is not None and full_csv_file in exclude_files:
            continue
        with open(full_csv_file) as f:
            first_line = f.readline().strip()
        for header, ft in header_to_filetype:
            if isinstance(header, list):
                if first_line in header:
                    file_type = ft
                else:
                    continue
            elif isinstance(header, str):
                if first_line == header:
                    file_type = ft
                else:
                    continue
            else:
                raise ValueError(f"Unknown header type: {type(header)}")

            # file_type = header_to_filetype[first_line]  # If not successful, the file is wrong -> check manually
            filetype_to_files[file_type] = filetype_to_files.get(file_type, []) + [full_csv_file]

    return filetype_to_files


def process_meta(files, target_fpath):
    dfs = []
    for file in files:
        df_meta = _read_df(file)
        df_meta["match_string"] = df_meta.apply(lambda row: f"{row['competition_name']} {row['season_name']}: {row['match_day']}.ST {row['match_title'].replace('-', ' - ')}", axis=1)
        df_meta["slugified_match_string"] = df_meta["match_string"].apply(slugify.slugify)
        dfs.append(df_meta)
    df_meta = pd.concat(dfs, axis=0)
    st.write(f"Wrote df_meta to {target_fpath}")
    st.write(df_meta)
    # df_meta.to_csv(target_fpath)
    return defensive_network.parse.drive.upload_csv_to_drive(df_meta, "meta.csv")


def process_lineups(files, target_fpath, overwrite_if_exists=True):
    dfs = []
    for file in files:
        df_lineup = _read_df(file)
        dfs.append(df_lineup)
    df_lineup = pd.concat(dfs, axis=0)
    df_lineup = df_lineup.replace(-9999, None)  # weird DFB convention to use -9999 for missing values
    # df_lineup.to_csv(target_fpath)
    st.write(f"Wrote df_lineup to {target_fpath}")
    # st.write(df_lineup)
    return defensive_network.parse.drive.upload_csv_to_drive(df_lineup, "lineups.csv")


def process_events(raw_event_paths, preprocessed_events_folder, meta_fpath, folder_tracking, overwrite_if_exists=True):
    # if not os.path.exists(preprocessed_events_folder):
    #     os.makedirs(preprocessed_events_folder)

    def get_target_x_y(row, df_tracking_indexed):
        receiver = row["player_id_2"]
        try:
            receiver_frame = df_tracking_indexed.loc[(row["frame_rec"], row["section"], receiver)]
            return receiver_frame["x_tracking"], receiver_frame["y_tracking"]
        except KeyError:
            return None, None

    # dfs = [_read_df(events_fpath) for events_fpath in raw_event_paths]
    dfs = [_read_df(events_fpath) for events_fpath in raw_event_paths]
    df_events = pd.concat(dfs, axis=0)
    df_events = df_events.replace(-9999, None)
    df_meta = _read_df(meta_fpath)
    df_events = df_events.merge(df_meta[["match_id", "slugified_match_string", "match_string"]], on="match_id")
    for match_id, df_events_match in df_events.groupby("match_id"):
        slugified_match_string = df_events_match["slugified_match_string"].iloc[0]
        match_string = df_events_match["match_string"].iloc[0]
        target_fpath = os.path.join(preprocessed_events_folder, f"{slugified_match_string}.csv")
        if not overwrite_if_exists and os.path.exists(target_fpath):
            continue

        try:
            df_tracking = pd.read_parquet(os.path.join(folder_tracking, f"{slugified_match_string}.parquet"))
        except FileNotFoundError:
            df_tracking = None
            st.warning(f"Tracking data not found for match {match_string}")

        if df_tracking is not None:
            df_tracking_indexed = df_tracking.set_index(["frame", "section", "player_id"])
            with st.spinner("Calculating target x, y from tracking data..."):
                keys = df_events_match.apply(lambda row: get_target_x_y(row, df_tracking_indexed), axis=1)

            df_events_match[["x_target", "y_target"]] = pd.DataFrame(keys.tolist(), index=df_events_match.index)
            df_events_match["x_target"] = df_events_match["x_target"].fillna(df_events_match["x_tracking_player_2"])
            df_events_match["y_target"] = df_events_match["y_target"].fillna(df_events_match["y_tracking_player_2"])

        st.write(f"Processing {match_string} with {len(df_events_match)} events to {target_fpath}")
        df_events_match.to_csv(target_fpath, index=False)
        # defensive_network.parse.drive.upload_csv_to_drive(df_events_match, target_fpath)
    st.write(f"Saved events to {preprocessed_events_folder}")


def process_tracking(files, target_folder, df_meta, chunksize=5e6):
    # # TODO overwrite_if_exists not implemented yet - would probably make it much faster, but how to do it?
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    for file in defensive_network.utility.general.progress_bar(files, desc=f"Processing tracking files ({len(files)} found)"):
        st.write(f"Processing tracking file {file} [chunksize={chunksize}]")

        @st.cache_resource
        def _get_n_lines(file):
            return defensive_network.utility.general.get_number_of_lines_in_file(file)

        n_lines = _get_n_lines(file)
        st.write(f"{n_lines=}")

        def _partition_by_match(_df_chunk, _target_folder, df_meta):
            for match_id, df_match_chunk in _df_chunk.groupby("match_id"):
                slugified_match_string = df_meta[df_meta["match_id"] == match_id]["slugified_match_string"].iloc[0]
                fpath_match = os.path.join(_target_folder, f"{slugified_match_string}.parquet")
                st.write("fpath_match", fpath_match)

                ### Assertions that take too much time
                # df_nan = df_match_chunk[df_match_chunk[["frame", "player_id"]].isna().all(axis=1)]
                # if len(df_nan) > 0:
                #     st.write("df_nan")
                #     st.write(df_nan)
                #     raise ValueError("NaN values in frame or player_id")

                # df_partial_duplicates = df_chunk[df_chunk.duplicated(subset=["match_id", "section", "frame", "player_id"], keep=False)]
                # assert len(df_partial_duplicates) == 0

                st.write(f"Processing match {match_id} with {len(df_match_chunk)} rows to {fpath_match}")
                defensive_network.utility.dataframes.append_to_parquet_file(df_match_chunk, fpath_match, key_cols=["match_id", "section", "frame", "player_id"], overwrite_key_cols=True)
                # defensive_network.parse.drive.append_to_parquet_on_drive(df_match_chunk, fpath_match, key_cols=["match_id", "section", "frame", "player_id"], overwrite_key_cols=True)

        with pd.read_csv(file, chunksize=chunksize, delimiter=",") as reader:
            total = math.ceil(n_lines / chunksize)
            for df_chunk in defensive_network.utility.general.progress_bar(reader, total=total, desc="Reading df_tracking in chunks"):
                _partition_by_match(df_chunk, target_folder, df_meta)
                del df_chunk
                gc.collect()


def check_tracking_files(folder):
    for file in os.listdir(folder):
        if not file.endswith(".parquet"):
            continue
        fpath = os.path.join(folder, file)
        df = pd.read_parquet(fpath)
        st.write(fpath)
        for section, df_section in df.groupby("section"):
            if not len(df_section["frame"].unique()) == df_section["frame"].max() - df_section["frame"].min() + 1:
                st.error(f"Missing frames in {section} of {fpath}")
            else:
                st.write(section, df_section["frame"].min(), df_section["frame"].max(), "OK")


def main():
    # overwrite_if_exists = True
    overwrite_if_exists = st.toggle("Overwrite if exists", value=False)
    _process_meta = st.toggle("Process meta", value=False)
    _process_lineups = st.toggle("Process lineups", value=False)
    _process_events = st.toggle("Process events", value=False)
    _process_tracking = st.toggle("Process tracking", value=False)
    _check_tracking_files = st.toggle("Check tracking files", value=False)
    _pp_to_drive = st.toggle("Preprocess and upload to drive", value=False)
    _do_involvement = st.toggle("Process involvements", value=False)
    _calculate_responsibility_model = st.toggle("Calculate responsibility model", value=False)
    _do_create_matchsums = st.toggle("Create matchsums", value=False)
    folder = st.text_input("Folder", "Y:/w_raw")
    if not os.path.exists(folder):
        st.warning(f"Folder {folder} does not exist")
        # return
    fpath_target_meta = st.text_input("Processed meta.csv file", os.path.join(folder, "meta.csv"))
    fpath_target_lineup = st.text_input("Processed lineups.csv file", os.path.join(folder, "lineups.csv"))
    # folder_events = st.text_input("Folder for preprocessed events", os.path.join(folder, "events"))
    # folder_tracking = st.text_input("Folder for preprocessed tracking", os.path.join(folder, "tracking"))

    folder_tracking = "tracking/"
    folder_events = "events/"
    folder_pp_tracking = os.path.join(folder, "preprocessed", folder_tracking)
    folder_pp_events = os.path.join(folder, "preprocessed", folder_events)
    folder_drive_tracking = "tracking"
    folder_drive_events = "events"
    fpath_drive_team_matchsums = "team_matchsums.csv"
    fpath_drive_players_matchsums = "players_matchsums.csv"
    folder_drive_involvement = "involvement"

    # filetype_to_files = get_dfb_csv_files_in_folder(folder, [fpath_target_meta, fpath_target_lineup])
    filetype_to_files = get_dfb_csv_files_in_folder(folder, [])
    st.write(f"Found files in {folder}:", filetype_to_files)

    if _process_meta:
        process_meta(filetype_to_files["meta"], fpath_target_meta)
    if _process_lineups:
        process_lineups(filetype_to_files["lineup"], fpath_target_lineup)
    if _process_tracking:
        chunksize = st.number_input("Rows per chunk of tracking data (more = faster but consumes more RAM)", min_value=1, value=5000000)
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv")
        process_tracking(filetype_to_files["tracking"], folder_pp_tracking, df_meta, chunksize)
    if _process_events:
        # folder_events = os.path.join(folder, folder_events)
        process_events(filetype_to_files["event"], folder_pp_events, fpath_target_meta, folder_tracking, overwrite_if_exists)
    if _check_tracking_files:
        check_tracking_files(folder_tracking)

    if _pp_to_drive:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv")
        # df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-18-st-bayer-leverkusen-eintracht-frankfurt"]

        if not overwrite_if_exists:
            event_files = defensive_network.parse.drive.list_files_in_drive_folder(folder_drive_events)
            event_matches = [f["name"].split(".")[0] for f in event_files]
            tracking_files = defensive_network.parse.drive.list_files_in_drive_folder(folder_drive_tracking)
            tracking_matches = [f["name"].split(".")[0] for f in tracking_files]
            df_meta = df_meta[~df_meta["slugified_match_string"].isin(event_matches) & ~df_meta["slugified_match_string"].isin(tracking_matches)]

        df_lineups = defensive_network.parse.drive.download_csv_from_drive("lineups.csv")
        # df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-12-st-rb-leipzig-1-fc-koln"]
        finalize_events_and_tracking_to_drive(folder_pp_tracking, folder_pp_events, df_meta, df_lineups, folder_drive_events, folder_drive_tracking)

    if _calculate_responsibility_model:
        def calc_involvement_model(folder_drive_involvement, fpath_out="responsibility_model.csv"):
            involvement_files = defensive_network.parse.drive.list_files_in_drive_folder(folder_drive_involvement)

            @st.cache_resource
            def _get_involvement():
                dfs = []
                for file in defensive_network.utility.general.progress_bar(involvement_files, total=len(involvement_files), desc="Involvement concat"):
                    df = defensive_network.parse.drive.download_csv_from_drive(os.path.join(folder_drive_involvement, file["name"]))
                    dfs.append(df)

                df = pd.concat(dfs)
                return df

            df_involvement = _get_involvement()
            st.write("df_involvement", df_involvement.shape)
            st.write(df_involvement.head())

            df_involvement_test = df_involvement[
                (df_involvement["role_category_1"] == "central_defender") &
                (df_involvement["network_receiver_role_category"] == "right_winger") &
                (df_involvement["defender_role_category"] == "right_winger")
            ]
            df_involvement = df_involvement[df_involvement["event_type"] == "pass"]
            df_involvement["network_receiver_role_category2"] = df_involvement["expected_receiver_role_category"].where(df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])

            dfg = defensive_network.models.responsibility.get_responsibility_model(df_involvement)
            st.write("dfg")
            st.write(dfg)


            for match_id, df_involvement_test_match in df_involvement_test.groupby("slugified_match_string"):
                df_tracking_match = defensive_network.parse.drive.download_parquet_from_drive(f"tracking/{match_id}.parquet")
                st.write("df_involvement_test_match")
                st.write(df_involvement_test_match)
                defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement_test_match, df_tracking_match, n_passes=5)

            # df_involvement = df_involvement[~df_involvement["pass_is_intercepted"]]

            defensive_network.parse.drive.upload_csv_to_drive(dfg, fpath_out)

            st.stop()

        calc_involvement_model(folder_drive_involvement)

    if _do_create_matchsums:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv")
        df_lineups = defensive_network.parse.drive.download_csv_from_drive("lineups.csv")
        # df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-12-st-rb-leipzig-1-fc-koln"]
        create_matchsums(folder_drive_tracking, folder_drive_events, df_meta, df_lineups, fpath_drive_team_matchsums, fpath_drive_players_matchsums)

    if _do_involvement:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv")
        # df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-18-st-bayer-leverkusen-eintracht-frankfurt"]
        process_involvements(df_meta, folder_drive_tracking, folder_drive_events, folder_drive_involvement)


def _create_matchsums(df_event, df_tracking, series_meta, df_lineup):
    dfgs = []

    assert "pass_xt" in df_event.columns

    # Minutes
    df_possession = df_tracking.groupby(["section"]).apply(lambda df_section : df_section.groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "nunique"}))
    df_possession = df_possession.reset_index().groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "sum"})
    df_possession["frame"] = df_possession["frame"] / (series_meta["fps"] * 60)
    df_possession = df_possession.rename(columns={"frame": "minutes"}).reset_index().pivot(index="ball_poss_team_id", columns="ball_status", values="minutes").drop(columns=[0])
    df_possession.columns = [f"net_minutes_in_possession"]
    df_possession["net_minutes_opponent_in_possession"] = df_possession["net_minutes_in_possession"].values[::-1]
    df_possession["net_minutes"] = df_possession["net_minutes_opponent_in_possession"] + df_possession["net_minutes_in_possession"]

    total_minutes = (
        df_tracking.groupby("section")["datetime_tracking"]
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
        .sum()
    )
    df_possession["total_minutes"] = total_minutes

    # Points
    teams = df_event["team_id_1"].dropna().unique()
    assert len(teams) == 2
    df_event["team_id_1"] = pd.Categorical(df_event["team_id_1"], teams)
    df_goals = df_event[(df_event["event_type"] == "shot") & (df_event["event_outcome"] == "successful")]
    dfg_result = df_goals.groupby("team_id_1", observed=False).agg(goals=("event_id", "count"))
    dfg_result["goals_against"] = dfg_result["goals"].iloc[::-1].values

    def calc_points(goals, goals_against):
        if goals > goals_against:
            return 3
        elif goals == goals_against:
            return 1
        elif goals < goals_against:
            return 0
        else:
            raise ValueError

    dfg_result["points"] = dfg_result.apply(lambda x: calc_points(x["goals"], x["goals_against"]), axis=1)
    dfgs.append(dfg_result)

    # xG
    dfg_xg = df_event.groupby("team_id_1", observed=False).agg(xg=("xg", "sum"))
    dfg_xg["xg_against"] = dfg_xg["xg"].iloc[::-1].values
    dfgs.append(dfg_xg)

    # Pass xT
    df_passes = df_event[df_event["event_type"] == "pass"]
    dfg_xt_total = df_passes.groupby("team_id_1", observed=False).agg(total_xt=("pass_xt", "sum"))
    dfg_xt_total["total_xt_against"] = dfg_xt_total["total_xt"].iloc[::-1].values
    dfgs.append(dfg_xt_total)

    dfg_xt_total_only_positive = df_passes[df_passes["pass_xt"] > 0].groupby("team_id_1", observed=False).agg(total_xt_only_positive=("pass_xt", "sum"))
    dfg_xt_total_only_positive["total_xt_only_positive_against"] = dfg_xt_total_only_positive["total_xt_only_positive"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_positive)

    dfg_xt_total_only_negative = df_passes[df_passes["pass_xt"] < 0].groupby("team_id_1", observed=False).agg(total_xt_only_negative=("pass_xt", "sum"))
    dfg_xt_total_only_negative["total_xt_only_negative_against"] = dfg_xt_total_only_negative["total_xt_only_negative"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_negative)

    dfg_xt_total_only_successful = df_passes[df_passes["event_outcome"] == "successfully_completed"].groupby("team_id_1", observed=False).agg(total_xt_only_successful=("pass_xt", "sum"))
    dfg_xt_total_only_successful["total_xt_only_successful_against"] = dfg_xt_total_only_successful["total_xt_only_successful"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_successful)

    # Number of passes
    dfg_n_passes = df_passes.groupby("team_id_1", observed=False).agg(passes=("event_id", "count"))
    dfg_n_passes["passes_against"] = dfg_n_passes["passes"].iloc[::-1].values
    dfgs.append(dfg_n_passes)

    # Interceptions
    dfg_interceptions = df_event[df_event["outcome"] == "intercepted"].groupby("team_id_2", observed=False).agg(
        n_interceptions=("event_id", "count")
    )
    dfgs.append(dfg_interceptions)

    # Tackles
    dfg_tackles = df_event[df_event["event_subtype"] == "tackle"].groupby("team_id_1", observed=False).agg(
        n_tackles=("event_id", "count")
    )
    dfgs.append(dfg_tackles)

    dfg_team = pd.concat(dfgs, axis=1).reset_index().rename(columns={"index": "team_id"})
    dfg_team["match_id"] = series_meta["match_id"]
    dfg_team = defensive_network.utility.dataframes.move_column(dfg_team, "match_id", 1)

    dfgs_players = []
    dfg_players_tackles_won = df_event[df_event["event_subtype"] == "tackle"].groupby("player_id_1", observed=False).agg(
        n_tackles_won=("event_id", "count"),
    )
    dfgs_players.append(dfg_players_tackles_won)
    dfg_players_tackles_lost = df_event[df_event["event_subtype"] == "tackle"].groupby("player_id_1", observed=False).agg(
        n_tackles_lost=("event_id", "count"),
    )
    dfgs_players.append(dfg_players_tackles_lost)

    dfg_players = pd.concat(dfgs_players, axis=1)
    dfg_players["n_tackles"] = dfg_players["n_tackles_won"] + dfg_players["n_tackles_lost"]
    dfg_players = dfg_players.reset_index().rename(columns={"player_id_1": "player_id"})
    dfg_players["match_id"] = series_meta["match_id"]
    dfg_players = defensive_network.utility.dataframes.move_column(dfg_players, "match_id", 1)

    st.write("dfg_players", dfg_players.shape)
    st.write(dfg_players)

    return dfg_team, dfg_players


def process_involvements(df_meta, folder_tracking, folder_events, target_folder):
    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Processing involvements"):
        match_string = match["match_string"]
        st.write(f"#### {match_string}")
        slugified_match_string = match["slugified_match_string"]

        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")
        df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
        df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events)

        df_events = df_events[df_events["event_type"] == "pass"]

        for coordinates in ["original", "sync"]:
            with st.expander(f"Plot passes with involvement ({coordinates})"):
                if coordinates == "original":
                    df_events["frame"] = df_events["original_frame_id"]
                else:
                    df_events["frame"] = df_events["matched_frame"]

                df_events["full_frame"] = df_events["section"].str.cat(df_events["frame"].astype(float).astype(str), sep="-")

                df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_defender_meta_cols=["role_category"])
                df_involvement["network_receiver_role_category"] = df_involvement["expected_receiver_role_category"].where(df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])
                dfg_responsibility = defensive_network.models.responsibility.get_responsibility_model(df_involvement, responsibility_context_cols=["defending_team", "role_category_1", "network_receiver_role_category", "defender_role_category"])
                df_involvement["intrinsic_responsibility"], _ = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model=dfg_responsibility)

                # upload
                # target_fpath = os.path.join(target_folder, f"{slugified_match_string}.csv")
                # defensive_network.parse.drive.upload_csv_to_drive(df_involvement, target_fpath)

                st.write("df_involvement")
                st.write(df_involvement)

                defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, responsibility_col="intrinsic_responsibility", n_passes=5)


def create_matchsums(folder_tracking, folder_events, df_meta, df_lineups, target_fpath_team, target_fpath_players):
    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Creating matchsums"):
        df_lineup = df_lineups[df_lineups["match_id"] == match["match_id"]]
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]
        st.write(f"Creating matchsums for {match_string}")
        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")
        st.write(f"fpath_tracking: {fpath_tracking}")
        st.write(f"fpath_events: {fpath_events}")

        # df_tracking = pd.read_parquet(fpath_tracking)
        # df_events = pd.read_csv(fpath_events)
        df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
        df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events)

        dfg_team, dfg_players = _create_matchsums(df_events, df_tracking, match, df_lineup)
        st.write("dfg_team")
        st.write(dfg_team)
        st.write("dfg_players")
        st.write(dfg_players)

        defensive_network.parse.drive.append_to_parquet_on_drive(dfg_team, target_fpath_team, key_cols=["team_id", "match_id"], overwrite_key_cols=True, format="csv")
        defensive_network.parse.drive.append_to_parquet_on_drive(dfg_players, target_fpath_players, key_cols=["player_id", "match_id"], overwrite_key_cols=True, format="csv")

        st.write("df_tracking", df_tracking.shape)
        st.write(df_tracking.head())
        st.write("df_events", df_events.shape)
        st.write(df_events.head())


def finalize_events_and_tracking_to_drive(folder_tracking, folder_events, df_meta, df_lineups, target_folder_events, target_folder_tracking):
    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Finalizing matches"):
        df_lineup = df_lineups[df_lineups["match_id"] == match["match_id"]]
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]
        st.write(f"Finalizing {match_string}")
        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")

        df_tracking = pd.read_parquet(fpath_tracking)
        df_event = pd.read_csv(fpath_events)

        # df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
        # df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events)

        with st.spinner("Augmenting event and tracking data..."):
            df_tracking, df_event = defensive_network.parse.dfb.cdf.augment_match_data(match, df_event, df_tracking, df_lineup)

        # df_events = defensive_network.parse.drive.download_csv_from_drive("events/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.csv").reset_index(drop=True)
        # df_tracking = _get_parquet("tracking/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.parquet").reset_index(drop=True)
        # df_tracking = _get_local_parquet("Y:/w_raw/preprocessed/tracking/bundesliga-2023-2024-22-st-bayer-leverkusen-werder-bremen.parquet").reset_index(
        #     drop=True)

        # with st.spinner("Synchronizing event and tracking data..."):
        #     res = defensive_network.models.synchronization.synchronize(df_event, df_tracking)
        #
        #     df_event["original_frame_id"] = df_event["frame"]
        #     df_event["matched_frame"] = res.matched_frames
        #     df_event["matching_score"] = res.scores
        #     df_event["frame"] = df_event["matched_frame"].fillna(df_event["frame"])

        # df_event["full_frame"] = df_event["section"].str.cat(df_event["frame"].astype(float).astype(str), sep="-")

        # Make tracking data smaller to store in Drive
        # original_full_frames = df_event["original_frame_id"].astype(str).str.cat(df_event["section"], sep="-").values
        # df_tracking = df_tracking[df_tracking["full_frame"].isin(df_event["full_frame"]) | df_tracking["full_frame"].isin(original_full_frames)]

        with st.spinner("Uploading to drive..."):
            drive_path_events = os.path.join(target_folder_events, f"{slugified_match_string}.csv")
            drive_path_tracking = os.path.join(target_folder_tracking, f"{slugified_match_string}.parquet")

            st.write("C")
            st.write(df_event["original_frame_id"] - df_event["frame"])
            st.write(df_event[["original_frame_id", "frame", "full_frame", "full_frame_rec"]])

            defensive_network.parse.drive.upload_csv_to_drive(df_event, drive_path_events)
            defensive_network.parse.drive.upload_parquet_to_drive(df_tracking, drive_path_tracking)

        st.write(f"Finalized events and tracking for {match_string} ({drive_path_events} and {drive_path_tracking})")

        # df_events_downloaded = defensive_network.parse.drive.download_csv_from_drive(drive_path_events)
        # df_tracking_downloaded = defensive_network.parse.drive.download_parquet_from_drive(drive_path_tracking)

        # dfg_team, dfg_players = _create_matchsums(df_events, df_tracking, match, df_lineup)
        #
        # st.write("dfg_team")
        # st.write(dfg_team)
        # st.write("dfg_players")
        # st.write(dfg_players)


def concat_metas_and_lineups():
    for kind, path in [
        ("meta", "C:/Users/Jonas/Downloads/dfl_test_data/2324/meta"),
        ("lineup", "C:/Users/Jonas/Downloads/dfl_test_data/2324/lineup"),
    ]:
        files = os.listdir(path)
        dfs = []
        for file in files:
            fpath = os.path.join(path, file)
            df = pd.read_csv(fpath)
            dfs.append(df)
        df_meta = pd.concat(dfs, axis=0)
        df_meta.to_csv(f"C:/Users/Jonas/Downloads/dfl_test_data/2324/{kind}.csv", index=False)


if __name__ == '__main__':
    main()

    # df = pd.read_csv("C:/Users/j.bischofberger/Downloads/Neuer Ordner (18)/defensive-network-main/w_raw/meta.csv", dtype=meta_schema)
    # write_parquet(df, fpath)
    # concat_metas_and_lineups()
    # main()
