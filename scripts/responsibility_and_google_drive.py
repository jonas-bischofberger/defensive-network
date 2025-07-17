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

import patsy
# import statsmodels.api as sm
from scipy.stats import pearsonr

import numpy as np
import statsmodels.formula.api
import statsmodels.api

import pandas as pd
import slugify
import streamlit as st

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.utility.dataframes

import seaborn as sns
import matplotlib.pyplot as plt
import defensive_network.utility.general
import defensive_network.parse.dfb.meta

import defensive_network.parse.dfb.cdf

import defensive_network.parse.drive
import defensive_network.models.formation
import defensive_network.models.involvement
import defensive_network.models.responsibility
import defensive_network.models.synchronization
import defensive_network.utility.pitch

import defensive_network.utility.video
importlib.reload(defensive_network.utility.video)

importlib.reload(defensive_network.parse.dfb.cdf)
importlib.reload(defensive_network.parse.dfb.meta)
importlib.reload(defensive_network.utility.general)
importlib.reload(defensive_network.parse.drive)
importlib.reload(defensive_network.models.responsibility)
importlib.reload(defensive_network.models.synchronization)
importlib.reload(defensive_network.models.involvement)


# @st.cache_resource
def _read_df(fpath, **kwargs):
    return pd.read_csv(fpath, **kwargs)


# @st.cache_resource
def get_dfb_csv_files_in_folder(folder, exclude_files=None):
    tracking_cols = [
        "frame,match_id,player_id,team_id,event_vendor,tracking_vendor,datetime_tracking,section,x_tracking,y_tracking,z_tracking,d_tracking,a_tracking,s_tracking,ball_status,ball_poss_team_id",
        "frame,match_id,player_id,team_id,event_vendor,tracking_vendor,datetime_tracking,section,x_tracking,y_tracking,z_tracking,d_tracking,a_tracking,s_tracking,ball_status,ball_poss_team_id",
    ]
    event_cols = [
        "frame,match_id,event_id,event_vendor,tracking_vendor,datetime_event,datetime_tracking,event_type,event_subtype,event_outcome,player_id_1,team_id_1,player_id_2,team_id_2,x_event,y_event,x_tracking_player_1,y_tracking_player_1,x_tracking_player_2,y_tracking_player_2,section,xg,xpass,player_pressure_1,player_pressure_2,assist_action,assist_type,rotation_ball,foot,direction,origin_setup,foul_type,card_color,reason,frame_rec,packing_traditional,packing_horizontal,packing_vertical,packing_attention",
        "frame,match_id,event_id,event_vendor,tracking_vendor,datetime_event,datetime_tracking,event_type,event_subtype,event_outcome,player_id_1,team_id_1,player_id_2,team_id_2,x_event_player_1,y_event_player_1,x_event_player_2,y_event_player_2,x_tracking_player_1,y_tracking_player_1,x_tracking_player_2,y_tracking_player_2,section,xg,xpass,player_pressure_1,player_pressure_2,assist_action,assist_type,rotation_ball,foot,direction,origin_setup,foul_type,card_color,reason,frame_rec,packing_traditional,packing_horizontal,packing_vertical,packing_attention",
        "frame,match_id,event_id,event_vendor,tracking_vendor,datetime_event,datetime_tracking,event_type,event_subtype,event_outcome,player_id_1,team_id_1,player_id_2,team_id_2,x_event,y_event,x_tracking_player_1,y_tracking_player_1,x_tracking_player_2,y_tracking_player_2,section,xg,xpass,player_pressure_1,player_pressure_2,assist_action,assist_type,rotation_ball,foot,direction,origin_setup,foul_type,card_color,reason,frame_rec,packing_traditional,packing_horizontal,packing_vertical,packing_attention,slugified_match_string,match_string"
    ]
    lineup_cols = [
        "match_id,event_vendor,tracking_vendor,team_id,team_name,team_role,player_id,jersey_number,first_name,last_name,short_name,position_group,position,starting,captain",
        "Unnamed: 0,match_id,event_vendor,tracking_vendor,team_id,team_name,team_role,player_id,jersey_number,first_name,last_name,short_name,position_group,position,starting,captain",
    ]
    meta_cols = [
        "competition_name,competition_id,host,match_day,season_name,season_id,kickoff_time,match_id,event_vendor,tracking_vendor,match_title,home_team_name,home_team_id,guest_team_name,guest_team_id,result,country,stadium_id,stadium_name,precipitation,pitch_x,pitch_y,total_time_first_half,total_time_second_half,playing_time_first_half,playing_time_second_half,ds_parser_version,xg_tag,xg_sha1,xpass_tag,xpass_sha1,fps",
        "Unnamed: 0,competition_name,competition_id,host,match_day,season_name,season_id,kickoff_time,match_id,event_vendor,tracking_vendor,match_title,home_team_name,home_team_id,guest_team_name,guest_team_id,result,country,stadium_id,stadium_name,precipitation,pitch_x,pitch_y,total_time_first_half,total_time_second_half,playing_time_first_half,playing_time_second_half,ds_parser_version,xg_tag,xg_sha1,xpass_tag,xpass_sha1,fps,match_string,slugified_match_string",
    ]

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
    return defensive_network.parse.drive.append_to_parquet_on_drive(df_meta, "meta.csv", format="csv", key_cols=["match_id"])


def process_lineups(files, target_fpath, overwrite_if_exists=True):
    dfs = []
    for file in files:
        df_lineup = _read_df(file)
        dfs.append(df_lineup)
    df_lineup = pd.concat(dfs, axis=0)
    df_lineup = df_lineup.replace(-9999, None)  # weird DFB convention to use -9999 for missing values
    # df_lineup.to_csv(target_fpath)
    # st.write(f"Wrote df_lineup to {target_fpath}")
    if "Unnamed: 0" in df_lineup.columns:
        df_lineup = df_lineup.drop(columns=["Unnamed: 0"])

    df_lineup = df_lineup.drop_duplicates(subset=["match_id", "team_id", "player_id", "position"])

    # st.write(df_lineup)
    # st.stop()
    # return defensive_network.parse.drive.upload_csv_to_drive(df_lineup, "lineups.csv")
    return defensive_network.parse.drive.append_to_parquet_on_drive(df_lineup, "lineups.csv", format="csv", key_cols=["match_id", "team_id", "player_id", "position"])


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

        # @st.cache_resource
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



# def calculate_icc(df: pd.DataFrame, metric: str, subject_id: str, covariate_col: str) -> float:
#     """
#     Calculates the Intraclass Correlation Coefficient (ICC) for a performance metric
#     across repeated measures for subjects (e.g., players).
#
#     Parameters:
#         df (pd.DataFrame): Input dataframe with repeated measures.
#         metric (str): Column name of the performance metric (e.g., 'goals').
#         subject_id (str): Column name for the subject identifier (e.g., 'player_id').
#
#     Returns:
#         float: ICC value (between 0 and 1).
#     """
#     # Check columns
#     if subject_id not in df.columns or metric not in df.columns:
#         raise ValueError(f"Column '{metric}' or '{subject_id}' not found in data")
#
#     # Drop rows with NaNs in the relevant columns
#     df = df[[subject_id, metric]].dropna()
#
#     if df.empty or df[subject_id].nunique() < 2:
#         raise ValueError("Not enough valid data to calculate ICC")
#
#     df = df[[subject_id, metric]]
#     formula = f"{metric} ~ 1"
#     model = statsmodels.formula.api.mixedlm(formula, data=df, groups=df[subject_id])
#     with st.spinner(f"Fitting mixed model for {metric}..."):
#         result = model.fit()
#
#     var_between = result.cov_re.iloc[0, 0]  # between-subject variance
#     var_within = result.scale  # within-subject (residual) variance
#
#     icc = var_between / (var_between + var_within)
#     return icc


import pandas as pd
import statsmodels.formula.api as smf

# def calculate_icc(df: pd.DataFrame, metric: str, subject_id: str, covariate_col: str = None) -> float:
#     """
#     Calculates the Intraclass Correlation Coefficient (ICC) for a performance metric
#     across repeated measures for subjects (e.g., players), adjusting for a covariate
#     such as position.
#
#     Parameters:
#         df (pd.DataFrame): Input dataframe with repeated measures.
#         metric (str): Column name of the performance metric (e.g., 'goals').
#         subject_id (str): Column name for the subject identifier (e.g., 'player_id').
#         covariate_col (str): Column name of the covariate to adjust for (e.g., 'position').
#
#     Returns:
#         float: ICC value (between 0 and 1).
#     """
#     # Check required columns
#     for col in [subject_id, metric, covariate_col]:
#         if col not in df.columns:
#             raise ValueError(f"Column '{col}' not found in data")
#
#     # Drop missing values
#     df = df[[subject_id, metric, covariate_col]].dropna()
#
#     if df.empty or df[subject_id].nunique() < 2:
#         raise ValueError("Not enough valid data to calculate ICC")
#
#     # Include covariate as a fixed effect
#     formula = f"{metric} ~ C({covariate_col})"  # treat covariate as categorical
#     st.write("df[subject_id]")
#     st.write(df[subject_id])
#     st.write("df[covariate_col]")
#     st.write(df[covariate_col])
#     model = smf.mixedlm(formula, data=df, groups=df[subject_id])
#     result = model.fit()
#
#     # Extract variance components
#     var_between = result.cov_re.iloc[0, 0]  # Between-subject variance (player)
#     var_within = result.scale              # Residual (within-subject) variance
#
#     # Calculate ICC
#     icc = var_between / (var_between + var_within)
#     return icc
#

import pandas as pd
import statsmodels.formula.api as smf

def calculate_icc(df: pd.DataFrame, metric: str, subject_id: str, covariate_col: str = None) -> float:
    """
    Calculates the Intraclass Correlation Coefficient (ICC) for a performance metric
    across repeated measures for subjects (e.g., players), optionally adjusting for
    a covariate such as position.

    Parameters:
        df (pd.DataFrame): Input dataframe with repeated measures.
        metric (str): Column name of the performance metric (e.g., 'goals').
        subject_id (str): Column name for the subject identifier (e.g., 'player_id').
        covariate_col (str or None): Optional column name of a covariate to adjust for (e.g., 'position').

    Returns:
        float: ICC value (between 0 and 1).
    """
    # Check required columns
    if subject_id not in df.columns or metric not in df.columns:
        raise ValueError(f"Column '{subject_id}' or '{metric}' not found in data")

    columns_to_use = [subject_id, metric]
    if covariate_col:
        if covariate_col not in df.columns:
            raise ValueError(f"Column '{covariate_col}' not found in data")
        columns_to_use.append(covariate_col)

    # Drop missing values
    df = df[columns_to_use].dropna()

    # st.write("df[subject_id]")
    # st.write(df[subject_id])
    # drop duplicate cols
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty or len(df[subject_id].unique()) < 2:
        raise ValueError("Not enough valid data to calculate ICC")

    # Construct the formula
    if covariate_col:
        df[covariate_col] = df[covariate_col].astype("category")
        formula = f"{metric} ~ C({covariate_col})"
    else:
        formula = f"{metric} ~ 1"

    # Fit mixed effects model
    try:
        model = smf.mixedlm(formula, data=df, groups=df[subject_id])
    except patsy.PatsyError as e:
        st.write(e)
        return None
    result = model.fit()

    # Extract variance components
    var_between = result.cov_re.iloc[0, 0]  # Between-subject variance
    var_within = result.scale               # Residual (within-subject) variance

    # Calculate ICC
    icc = var_between / (var_between + var_within)
    return icc





def aggregate_matchsums(df_player_matchsums, group_cols=["player_id"]):
    dfg = df_player_matchsums.groupby(group_cols).agg(
        # Classic metrics
        n_interceptions=("n_interceptions", "sum"),
        n_passes=("n_passes", "sum"),
        n_tackles_won=("n_tackles_won", "sum"),
        n_tackles_lost=("n_tackles_lost", "sum"),

        # Involvement
        # total_valued_contribution=("total_valued_contribution", "sum"),
        # total_valued_fault=("total_valued_fault", "sum"),
        # total_valued_involvement=("total_valued_involvement", "sum"),
        # total_raw_contribution=("total_raw_contribution", "sum"),
        # total_raw_fault=("total_raw_fault", "sum"),
        # total_raw_involvement=("total_raw_involvement", "sum"),
        # n_passes_with_contribution=("n_passes_with_contribution", "sum"),
        # n_passes_with_fault=("n_passes_with_fault", "sum"),
        # n_passes_with_involvement=("n_passes_with_involvement", "sum"),

        # Responsibility
        # total_valued_contribution_responsibility=("total_valued_contribution_responsibility", "sum"),
        # total_valued_fault_responsibility=("total_valued_fault_responsibility", "sum"),
        # total_valued_responsibility=("total_valued_responsibility", "sum"),
        # n_passes_with_contribution_responsibility=("n_passes_with_contribution_responsibility", "sum"),
        # n_passes_with_fault_responsibility=("n_passes_with_fault_responsibility", "sum"),
        # n_passes_with_responsibility=("n_passes_with_responsibility", "sum"),
        # TODO valued intrinsic fault/contribution responsibility etc.
        # total_raw_responsibility=("total_raw_responsibility", "sum"),
        # total_raw_contribution_responsibility=("total_raw_contribution_responsibility", "sum"),
        # total_raw_fault_responsibility=("total_raw_fault_responsibility", "sum"),
        # total_intrinsic_responsibility=("total_intrinsic_responsibility", "sum"),
        # total_intrinsic_relative_responsibility=("total_intrinsic_relative_responsibility", "sum"),
        # total_intrnsic_fault_responsibility=("total_intrinsic_fault_responsibility", "sum"),
        # total_intrinsic_contribution_responsibility=("total_intrinsic_contribution_responsibility", "sum"),

 # total_intrinsic_contribution_responsibility_per90

        # Minutes
        minutes_played=("minutes_played", "sum"),
    )
    dfg["n_tackles"] = (dfg["n_tackles_won"] + dfg["n_tackles_lost"])

    minutes_played_total = dfg["minutes_played"].copy()
    dfg = (dfg.div(dfg["minutes_played"], axis=0) * 90).rename(columns=lambda x: f"{x}_per90")
    dfg["minutes_played"] = minutes_played_total
    dfg["tackles_won_share"] = dfg["n_tackles_won_per90"] / dfg["n_tackles_per90"]

    for col in ["short_name", "first_name", "last_name"]:
        df_player_matchsums[col] = df_player_matchsums[col].replace("0", "")

    df_player_matchsums["first_plus_last_name"] = (df_player_matchsums["first_name"].astype(str) + " " + df_player_matchsums["last_name"].astype(str)).str.strip()
    df_player_matchsums["normalized_name"] = df_player_matchsums["short_name"].where(df_player_matchsums["short_name"].notna() & (df_player_matchsums["short_name"] != ""), df_player_matchsums["first_plus_last_name"])

    # Resp per pass
    dfg_per_pass = df_player_matchsums.groupby(group_cols).agg(
        total_valued_contribution_responsibility=("total_valued_contribution_responsibility", "sum"),
        total_valued_fault_responsibility=("total_valued_fault_responsibility", "sum"),
        total_valued_responsibility=("total_valued_responsibility", "sum"),
        total_valued_involvement=("total_valued_involvement", "sum"),
        total_valued_contribution=("total_valued_contribution", "sum"),
        total_valued_fault=("total_valued_fault", "sum"),
        total_raw_contribution_responsibility=("total_raw_contribution_responsibility", "sum"),
        total_raw_fault_responsibility=("total_raw_fault_responsibility", "sum"),
        total_raw_responsibility=("total_raw_responsibility", "sum"),
        total_raw_contribution=("total_raw_contribution", "sum"),
        total_raw_fault=("total_raw_fault", "sum"),
        total_raw_involvement=("total_raw_involvement", "sum"),

        total_relative_raw_responsibility=("total_relative_raw_responsibility", "sum"),
        total_relative_raw_fault_responsibility=("total_relative_raw_fault_responsibility", "sum"),
        total_relative_raw_contribution_responsibility=("total_relative_raw_contribution_responsibility", "sum"),
        total_relative_valued_responsibility=("total_relative_valued_responsibility", "sum"),
        total_relative_valued_fault_responsibility=("total_relative_valued_fault_responsibility", "sum"),
        total_relative_valued_contribution_responsibility=("total_relative_valued_contribution_responsibility", "sum"),

    # total_intrinsic_valued_involvement=("total_intrinsic_valued_involvement", "sum"),
        # total_intrinsic_valued_contribution=("total_intrinsic_valued_contribution_responsibility", "sum"),
        # total_intrinsic_valued_fault=("total_intrinsic_valued_fault_responsibility", "sum"),
        # total_intrinsic_valued_responsibility=("total_intrinsic_valued_responsibility", "sum"),
        # total_intrinsic_valued_contribution_responsibility=("total_intrinsic_valued_contribution_responsibility", "sum"),
        # total_intrinsic_valued_fault_responsibility=("total_intrinsic_valued_fault_responsibility", "sum"),

        # n_passes_with_contribution_responsibility=("n_passes_with_contribution_responsibility", "sum"),
        # n_passes_with_fault_responsibility=("n_passes_with_fault_responsibility", "sum"),
        n_passes_with_responsibility=("n_passes_with_responsibility", "sum"),
        n_passes_with_involvement=("n_passes_with_involvement", "sum"),

        # classic metrics
        # n_interceptions=("n_interceptions", "sum"),
        # n_tackles=("n_tackles", "sum"),
        # n_tackles_won=("n_tackles_won", "sum"),
        # n_tackles_lost=("n_tackles_lost", "sum"),
        n_passes_against=("n_passes_against", "sum"),
    )
    dfg_per_pass["total_valued_contribution_responsibility_per_pass"] = dfg_per_pass["total_valued_contribution_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_valued_fault_responsibility_per_pass"] = dfg_per_pass["total_valued_fault_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_valued_responsibility_per_pass"] = dfg_per_pass["total_valued_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_valued_involvement_per_pass"] = dfg_per_pass["total_valued_involvement"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_valued_contribution_per_pass"] = dfg_per_pass["total_valued_contribution"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_valued_fault_per_pass"] = dfg_per_pass["total_valued_fault"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_raw_contribution_responsibility_per_pass"] = dfg_per_pass["total_raw_contribution_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_raw_fault_responsibility_per_pass"] = dfg_per_pass["total_raw_fault_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_raw_responsibility_per_pass"] = dfg_per_pass["total_raw_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_raw_contribution_per_pass"] = dfg_per_pass["total_raw_contribution"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_raw_fault_per_pass"] = dfg_per_pass["total_raw_fault"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_raw_involvement_per_pass"] = dfg_per_pass["total_raw_involvement"] / dfg_per_pass["n_passes_with_involvement"]
    dfg_per_pass["total_relative_raw_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_relative_raw_fault_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_fault_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_relative_raw_contribution_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_contribution_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_relative_valued_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_relative_valued_fault_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_fault_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    dfg_per_pass["total_relative_valued_contribution_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_contribution_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]

#         total_relative_raw_responsibility=("total_relative_raw_responsibility", "sum"),
    #         total_relative_raw_fault_responsibility=("total_relative_raw_fault_responsibility", "sum"),
    #         total_relative_raw_contribution_responsibility=("total_relative_raw_contribution_responsibility", "sum"),
    #         total_relative_valued_responsibility=("total_relative_valued_responsibility", "sum"),
    #         total_relative_valued_fault_responsibility=("total_relative_valued_fault_responsibility", "sum"),
    #         total_relative_valued_contribution_responsibility=("total_relative_valued_contribution_responsibility", "sum"),

    # dfg_per_pass["total_intrinsic_valued_involvement_per_pass"] = dfg_per_pass["total_intrinsic_valued_involvement"] / dfg_per_pass["n_passes_with_involvement"]
    # dfg_per_pass["total_intrinsic_valued_contribution_per_pass"] = dfg_per_pass["total_intrinsic_valued_contribution"] / dfg_per_pass["n_passes_with_involvement"]
    # dfg_per_pass["total_intrinsic_valued_fault_per_pass"] = dfg_per_pass["total_intrinsic_valued_fault"] / dfg_per_pass["n_passes_with_involvement"]
    # dfg_per_pass["total_intrinsic_valued_responsibility_per_pass"] = dfg_per_pass["total_intrinsic_valued_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    # dfg_per_pass["total_intrinsic_valued_contribution_responsibility_per_pass"] = dfg_per_pass["total_intrinsic_valued_contribution_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]
    # dfg_per_pass["total_intrinsic_valued_fault_responsibility_per_pass"] = dfg_per_pass["total_intrinsic_valued_fault_responsibility"] / dfg_per_pass["n_passes_with_responsibility"]

    # dfg_per_pass["n_interceptions_per_pass"] = dfg_per_pass["n_interceptions"] / dfg_per_pass["n_passes_against"]

    dfg_per_pass = dfg_per_pass.drop(columns=[col for col in [
        # "n_passes_with_contribution_responsibility",
        # "n_passes_with_fault_responsibility",
        "n_passes_with_responsibility",
        "n_passes_with_involvement",
        "total_valued_contribution_responsibility",
        "total_valued_fault_responsibility",
        "total_valued_responsibility",
        "total_valued_involvement",
        "total_valued_contribution",
        "total_valued_fault",
        "total_raw_contribution_responsibility",
        "total_raw_fault_responsibility",
        "total_raw_responsibility",
        "total_raw_contribution",
        "total_raw_fault",
        "total_raw_involvement",
        # "total_intrinsic_valued_involvement",
        "total_intrinsic_valued_contribution",
        "total_intrinsic_valued_fault",
        "total_intrinsic_valued_responsibility",
        "total_intrinsic_valued_contribution_responsibility",
        "total_intrinsic_valued_fault_responsibility",
        "total_relative_raw_responsibility",
        "total_relative_raw_fault_responsibility",
        "total_relative_raw_contribution_responsibility",
        "total_relative_valued_responsibility",
        "total_relative_valued_fault_responsibility",
        "total_relative_valued_contribution_responsibility",
    ] if col in dfg_per_pass.columns])
    dfg_per_pass = dfg_per_pass.rename(columns=lambda x: x.replace("total_", ""))
    dfg = dfg.join(dfg_per_pass, on=group_cols, rsuffix="_per_pass")

    dfg_meta = df_player_matchsums.groupby(group_cols).agg(
        short_name=("short_name", "first"),
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        normalized_name=("normalized_name", "first"),
    )
    dfg = dfg.join(dfg_meta, on=group_cols)

    return dfg.reset_index()



def main():
    # overwrite_if_exists = True
    overwrite_if_exists = st.toggle("Overwrite if exists", value=False)
    _process_meta = st.toggle("Process meta", value=False)
    _process_lineups = st.toggle("Process lineups", value=False)
    _process_events = st.toggle("Process events", value=False)
    _process_tracking = st.toggle("Process tracking", value=False)
    _check_tracking_files = st.toggle("Check tracking files", value=False)
    _pp_to_drive = st.toggle("Preprocess and upload to drive", value=False)
    _do_reduction = st.toggle("Upload reduced tracking data to Drive", False)
    _do_involvement = st.toggle("Process involvements", value=False)
    _calculate_responsibility_model = st.toggle("Calculate responsibility model", value=False)
    _do_create_matchsums = st.toggle("Create matchsums", value=False)
    _do_videos = st.toggle("Create videos", value=False)
    _do_analysis = st.toggle("Do analysis", value=False)

    folder = st.text_input("Folder", "Y:/w_raw")
    # if not os.path.exists(folder):
    #     st.warning(f"Folder {folder} does not exist")
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
    folder_full_tracking = os.path.join(folder, "finalized", folder_tracking)
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
        finalize_events_and_tracking_to_drive(folder_pp_tracking, folder_pp_events, df_meta, df_lineups, folder_drive_events, folder_drive_tracking, folder_full_tracking)

    if _do_reduction:
        # def upload_reduced_tracking_data(df_meta, drive_folder_events, full_tracking_folder, drive_folder_tracking):
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=False)
        upload_reduced_tracking_data(df_meta, folder_drive_events, folder_full_tracking, folder_drive_tracking)

    if _do_involvement:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=False)
        # df_meta = df_meta[(df_meta["match_id"] == "2d4fe74894566dc4b826bd608deaa53c") | (df_meta["slugified_match_string"] == "bundesliga-2023-2024-18-st-bayer-leverkusen-eintracht-frankfurt")].iloc[::-1]
        process_involvements(df_meta, folder_drive_tracking, folder_drive_events, folder_drive_involvement, overwrite_if_exists)

    if _calculate_responsibility_model:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=False)
        st.write("df_meta")
        st.write(df_meta)
        match_id_2_competition_name = df_meta.set_index("match_id")["competition_name"].to_dict()
        st.write("match_id_2_competition_name")
        st.write(match_id_2_competition_name)

        def calc_involvement_model(folder_drive_involvement, fpath_out="responsibility_model.csv", competition_name=None):
            involvement_files = defensive_network.parse.drive.list_files_in_drive_folder(folder_drive_involvement)

            df_meta_filtered = df_meta[df_meta["competition_name"] == competition_name] if competition_name else df_meta
            involvement_files = [file for file in involvement_files if file["name"].split(".")[0] in df_meta_filtered["slugified_match_string"].values]

            # @st.cache_resource
            def _get_involvement(involvement_files):
                dfs = []
                for file in defensive_network.utility.general.progress_bar(involvement_files, total=len(involvement_files), desc="Involvement concat"):
                    df = defensive_network.parse.drive.download_csv_from_drive(os.path.join(folder_drive_involvement, file["name"]))
                    df["match_id"] = df["match_id"].astype(str)
                    df["competition_name"] = df["match_id"].map(match_id_2_competition_name)
                    # st.write("df")
                    # st.write(df)
                    assert "competition_name" in df.columns, "Competition name column missing in involvement data"
                    assert df["competition_name"].notna().all(), "Some match_ids do not have a competition name"
                    assert len(set(df["competition_name"])) == 1, "Multiple competition names found in involvement data"
                    necessary_columns = ["role_category_1", "network_receiver_role_category", "defender_role_category", "expected_receiver_role_category", "event_type"]
                    dfs.append(df)

                df = pd.concat(dfs)
                return df

            df_involvement = _get_involvement(involvement_files)

            df_involvement_test = df_involvement[
                (df_involvement["role_category_1"] == "central_defender") &
                (df_involvement["network_receiver_role_category"] == "right_winger") &
                (df_involvement["defender_role_category"] == "right_winger")
            ]
            df_involvement = df_involvement[df_involvement["event_type"] == "pass"]
            df_involvement["network_receiver_role_category2"] = df_involvement["expected_receiver_role_category"].where(df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])

            if competition_name is not None:
                df_involvement = df_involvement[df_involvement["competition_name"] == competition_name]
                df_involvement_test = df_involvement_test[df_involvement_test["competition_name"] == competition_name]

            dfg = defensive_network.models.responsibility.get_responsibility_model(df_involvement)
            dfg["competition_name"] = competition_name

            for match_id, df_involvement_test_match in df_involvement_test.groupby("slugified_match_string"):
                df_tracking_match = defensive_network.parse.drive.download_parquet_from_drive(f"tracking/{match_id}.parquet")
                defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement_test_match, df_tracking_match, n_passes=5)

            # df_involvement = df_involvement[~df_involvement["pass_is_intercepted"]]

            defensive_network.parse.drive.upload_csv_to_drive(dfg.reset_index(), fpath_out.replace("Men's", "Mens"))
            st.write(f"Uploaded this df to {fpath_out}:")
            st.write(dfg)

        selected_competition_names = st.multiselect("Select competitions for responsibility model", df_meta["competition_name"].unique(), ["FIFA Men's World Cup"])

        for competition_name in selected_competition_names:
            st.write(f"Calculating responsibility model for {competition_name}...")
            calc_involvement_model(folder_drive_involvement, f"responsibility_model_{competition_name}.csv", competition_name)

        if st.toggle("Calculate responsibility model for all competitions", value=False):
            calc_involvement_model(folder_drive_involvement, competition_name=None)

    if _do_create_matchsums:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=False)
        df_lineups = defensive_network.parse.drive.download_csv_from_drive("lineups.csv", st_cache=False)
        # df_meta = df_meta[df_meta["match_id"] == "2d4fe74894566dc4b826bd608deaa53c"]
        create_matchsums(folder_full_tracking, folder_drive_events, df_meta, df_lineups, fpath_drive_team_matchsums, fpath_drive_players_matchsums, overwrite_if_exists=overwrite_if_exists)

    if _do_videos:
        create_videos()

    if _do_analysis:
        DEFAULT_MINUTES = st.number_input("Default minimal minutes played total", min_value=0, value=300)
        DEFAULT_MINUTES_PER_MATCH = st.number_input("Default minimal minutes per match", min_value=0, value=30)

        df_player_matchsums = defensive_network.parse.drive.download_csv_from_drive(fpath_drive_players_matchsums, st_cache=False)
        df_player_matchsums["is_rückrunde"] = df_player_matchsums["kickoff_time"].apply(lambda x: pd.to_datetime(x, errors="coerce").year == 2024)
        df_player_matchsums = df_player_matchsums[df_player_matchsums["role_category"] != "GK"]

        competitions = df_player_matchsums["competition_name"].unique()
        selected_competitions = st.multiselect("Competitions", competitions, ["3. Liga"])

        df_player_matchsums = df_player_matchsums[df_player_matchsums["competition_name"].isin(selected_competitions)]

        position_mapping = {'CB-3': 'CB', 'LB': 'FB', 'LCB-3': 'CB', 'LCB-4': 'CB', 'LDM': 'CM', 'LS': 'CF',
                            'LW': 'Winger', 'LZM': 'CM', 'RB': 'FB', 'RCB-3': 'CB', 'RCB-4': 'CB', 'RDM': 'CM',
                            'RS': 'CF', 'RW': 'Winger', 'RWB': 'FB', 'RZM': 'CM', 'ST': 'CF', 'ZDM': 'CM', 'ZOM': 'CAM',
                            'LWB': 'FB'
                            }
        df_player_matchsums["coarse_position"] = df_player_matchsums["role_category"].map(position_mapping)
        assert df_player_matchsums["coarse_position"].notna().all(), "Some positions are not mapped to coarse positions"
        df_player_matchsums["player_pos_minutes_played"] = df_player_matchsums.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")

        df_agg_match_player_pos = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position", "match_id"])
        # df_agg_match_player_pos["player_minutes_played"] = df_agg_match_player_pos.groupby("player_id")["minutes_played"].transform("sum")
        df_agg_match_player_pos["player_pos_minutes_played"] = df_agg_match_player_pos.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")
        # df_agg_match_player_pos["player_matches_played"] = df_agg_match_player_pos.groupby("player_id")["match_id"].transform("nunique")
        # st.write("df_agg_match_player_pos")
        # st.write(df_agg_match_player_pos[["player_id", "coarse_position", "match_id"] + [col for col in df_agg_match_player_pos.columns if "minutes" in col]])

        df_agg_player_pos = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position"])
        df_agg_player_pos["player_minutes_played"] = df_agg_player_pos.groupby("player_id")["minutes_played"].transform("sum")
        # df_agg_match_player_pos["player_matches_played"] = df_agg_match_player_pos.groupby("player_id")["match_id"].transform("nunique")

        df_agg_team = aggregate_matchsums(df_player_matchsums, group_cols=["team_id", "coarse_position", "player_id"]).reset_index()
        # df_agg_team["full_name"] = df_agg_team["first_name"] + " " + df_agg_team["last_name"]

        df_agg_by_season_half = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position", "is_rückrunde"])
        # df_agg_match_player = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "match_id"])
        # df_agg_match_player["player_minutes_played"] = df_agg_match_player.groupby("player_id")["minutes_played"].transform("sum")
        # df_agg_match_player["player_matches_played"] = df_agg_match_player.groupby("player_id")["match_id"].transform("nunique")

        default_kpis = [
            # Classic metrics
            "n_interceptions_per90",
            "n_tackles_won_per90",
            "n_tackles_per90",
            "tackles_won_share",
            "n_passes_per90",

            "raw_contribution_per_pass",
            "raw_fault_per_pass",
            "raw_involvement_per_pass",
            "raw_contribution_responsibility_per_pass",
            "raw_fault_responsibility_per_pass",
            "raw_responsibility_per_pass",

            "valued_contribution_per_pass",
            "valued_fault_per_pass",
            "valued_involvement_per_pass",
            "valued_contribution_responsibility_per_pass",
            "valued_fault_responsibility_per_pass",
            "valued_responsibility_per_pass",

            "relative_raw_contribution_responsibility_per_pass",
            "relative_raw_fault_responsibility_per_pass",
            "relative_raw_responsibility_per_pass",
            "relative_valued_contribution_responsibility_per_pass",
            "relative_valued_fault_responsibility_per_pass",
            "relative_valued_responsibility_per_pass",

            # Valued Responsibility
            # "total_valued_contribution_responsibility_per90",
            # "total_valued_fault_responsibility_per90",
            # "total_valued_responsibility_per90",
            #
            # "total_intrinsic_contribution_responsibility_per90",
            # "total_intrinsic_fault_responsibility_per90",
            # "total_intrinsic_responsibility_per90",

            # total_raw_contribution=("raw_involvement", "sum"),
            # total_contribution=("involvement", "sum"),
            # total_valued_contribution_responsibility=("valued_responsibility", "sum"),
            # total_contribution_responsibility=("responsibility", "sum"),
            # total_intrinsic_contribution_responsibility=("intrinsic_responsibility", "sum"),
            # total_intrinsic_valued_contribution_responsibility=("intrinsic_valued_responsibility", "sum"),
            # n_passes_with_contribution=("raw_involvement", lambda x: (x != 0).sum()),
            # n_passes_with_contribution_responsibility=("valued_responsibility", lambda x: ((x < 0) & (x != 0)).sum()),


            # "total_responsibility_per90",
            # "total_intrinsic_responsibility_per90",
            # "total_intrinsic_relative_responsibility_per90",

            # Pass counts
            # "n_passes_with_contribution_responsibility_per90",
            # "n_passes_with_fault_responsibility_per90",
            # "n_passes_with_responsibility_per90",
            # "n_passes_with_contribution_per90",
            # "n_passes_with_fault_per90",
            # "n_passes_with_involvement_per90",

            # Involvement
            # "total_involvement_per90",
            # "total_contribution_per90",
            # "total_fault_per90",
            # "total_raw_contribution_per90",
            # "total_raw_fault_per90",
            # "total_raw_involvement_per90",

            # Per pass
            # "total_valued_responsibility_per_pass",
            # "total_valued_contribution_responsibility_per_pass",
            # "total_valued_fault_responsibility_per_pass",
            # "total_valued_involvement_per_pass",
            # "total_valued_contribution_per_pass",
            # "total_valued_fault_per_pass",
            # "total_raw_responsibility_per_pass",
            # "total_raw_contribution_responsibility_per_pass",
            # "total_raw_fault_responsibility_per_pass",
            # "total_raw_involvement_per_pass",
            # "total_raw_contribution_per_pass",
            # "raw_fault_per_pass",
        ]
        kpis = st.multiselect("KPIs", options=sorted(df_agg_player_pos.columns), default=sorted([kpi for kpi in default_kpis if kpi in df_agg_player_pos.columns or "per_pass" in kpi]))
        for kpi in kpis:
            if kpi not in df_agg_match_player_pos.columns:
                st.warning(f"{kpi} not in df_agg_match_player_pos")
        kpis = [kpi for kpi in kpis if kpi in df_agg_match_player_pos.columns]

        if st.toggle("Matchsum Descriptives", False):
            # Position counts
            df_desc1 = df_agg_match_player_pos.groupby(["coarse_position", "player_id"]).agg(
                n_matches=("match_id", "nunique"),
                minutes_played=("minutes_played", "sum"),
                normalized_name=("normalized_name", "first"),
            ).reset_index()
            st.write("df_desc1")
            st.write(df_desc1)

        if st.toggle("Season-by-season correlation", False):
            with st.expander("Season-by-season correlations"):
                # df_agg_player_rr = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "is_rückrunde"])
                # df_agg_player_rr = df_agg_player_rr.reset_index().set_index("player_id")
                df_agg = df_agg_by_season_half.copy().reset_index()
                min_minutes = st.number_input("Minimum minutes played by player for Season-by-season corr.", min_value=0, value=DEFAULT_MINUTES, key="min_minutes_corr")
                df_agg = df_agg[df_agg["minutes_played"] > min_minutes]
                df_hinrunde = df_agg[df_agg["is_rückrunde"] == False]
                df_rückrunde = df_agg[df_agg["is_rückrunde"] == True]
                df_data_new = df_hinrunde.merge(df_rückrunde, on=["player_id", "coarse_position"], suffixes=("_hinrunde", "_rückrunde"))

                data = []
                for kpi_nr, kpi in enumerate(df_agg.columns):
                    # correlate hinrunde with rückrunde
                    if kpi == "player_id" or kpi == "is_rückrunde" or kpi == "coarse_position":
                        continue

                    # correlation plot with trendline
                    # sns.scatterplot(data=df_hinrunde, x=kpi, y=df_rückrunde[kpi], hue="is_rückrunde")
                    # sns.regplot(data=df_agg, x=kpi, y="is_rückrunde", scatter=True, color='blue', label='Trendline')
                    try:
                        # sns.regplot(data=df_data_new, x=f"{kpi}_hinrunde", y=f"{kpi}_rückrunde", scatter=True, color='blue', label='Trendline')
                        sns.lmplot(data=df_data_new, x=f"{kpi}_hinrunde", y=f"{kpi}_rückrunde", hue="coarse_position", scatter=True)

                    except (ValueError, np.exceptions.DTypePromotionError) as e:
                        continue

                    for i, row in df_data_new.iterrows():
                        plt.text(
                            row[f"{kpi}_hinrunde"],  # x position
                            row[f"{kpi}_rückrunde"],  # y position
                            row["short_name_hinrunde"],
                            fontsize=9,
                            ha='right',
                            va='bottom'
                        )

                    try:
                        # correlation_coefficient = df_hinrunde[kpi].corr(df_rückrunde[kpi])
                        correlation_coefficient = float(df_data_new[f"{kpi}_hinrunde"].corr(df_data_new[f"{kpi}_rückrunde"]))

                        x = df_data_new[f"{kpi}_hinrunde"]
                        y = df_data_new[f"{kpi}_rückrunde"]
                        coarse_pos = pd.get_dummies(df_data_new["coarse_position"], drop_first=True).astype(float)  # One-hot encode

                        # Step 1: Regress x and y on the dummies

                        x_model = statsmodels.api.OLS(x, statsmodels.api.add_constant(coarse_pos)).fit()
                        y_model = statsmodels.api.OLS(y, statsmodels.api.add_constant(coarse_pos)).fit()

                        # Step 2: Get residuals
                        x_resid = x_model.resid
                        y_resid = y_model.resid

                        # Step 3: Correlate residuals
                        partial_corr = x_resid.corr(y_resid)

                    except ValueError as e:
                        st.write(e)
                        correlation_coefficient = np.nan
                        partial_corr = np.nan

                    df_data_new_only_CBs = df_data_new[df_data_new["coarse_position"] == "CB"]
                    correlation_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"]))
                    spearman_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"], method='spearman'))

                    columns = st.columns(2)
                    plt.title(f"{kpi} (r={correlation_coefficient:.2f})")
                    columns[0].write(kpi)
                    columns[1].write(f"r={correlation_coefficient:.2f}, partial_r={partial_corr:.2f}, r_only_CBs={correlation_coefficient_only_CBs:.2f}, spearman_only_CBs={spearman_coefficient_only_CBs:.2f}")
                    # st.write(correlation_coefficient)
                    #
                    plt.xlabel("Hinrunde")
                    plt.ylabel("Rückrunde")
                    plt.title(f"Correlation of {kpi} between Hinrunde and Rückrunde")

                    # plot diagonal
                    plt.plot(
                        [df_data_new[f"{kpi}_hinrunde"].min(), df_data_new[f"{kpi}_hinrunde"].max()],
                        [df_data_new[f"{kpi}_rückrunde"].min(), df_data_new[f"{kpi}_rückrunde"].max()],
                        color='black', linestyle='--', label='Diagonal'
                    )
                    columns[0].write(plt.gcf())

                    plt.close()

                    plt.figure()
                    # scatter plot with trendline
                    sns.regplot(data=df_data_new, x=f"{kpi}_hinrunde", y=f"{kpi}_rückrunde", scatter=True, color='blue', label='Trendline')
                    plt.xlabel(f"{kpi} Hinrunde")
                    plt.ylabel(f"{kpi} Rückrunde")
                    plt.title(f"Scatter plot of {kpi} between Hinrunde and Rückrunde")
                    columns[1].write(plt.gcf())
                    plt.close()

                    data.append({
                        "kpi": kpi,
                        "correlation_coefficient": correlation_coefficient,
                        "correlation_coefficient_only_CBs": correlation_coefficient_only_CBs,
                        "abs_correlation_coefficient_only_CBs": abs(correlation_coefficient_only_CBs),
                        "spearman_coefficient_only_CBs": spearman_coefficient_only_CBs,
                        "abs_spearman_coefficient_only_CBs": abs(spearman_coefficient_only_CBs),
                        "partial_correlation_coefficient": partial_corr,
                        "abs_partial_correlation_coefficient": abs(partial_corr),
                        "n_players": df_data_new["player_id"].nunique(),
                        "n_CBs": df_data_new[df_data_new["coarse_position"] == "CB"]["player_id"].nunique(),
                    })

                df_correlations = pd.DataFrame(data)
                st.write("df_correlations")
                st.write(df_correlations.set_index("kpi").sort_values(by="abs_partial_correlation_coefficient", ascending=False))

        data = []
        if st.toggle("ICC", False):
            min_minutes = st.number_input("Minimum minutes played by player for ICC.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH)
            df_agg = df_agg_match_player_pos[df_agg_match_player_pos["minutes_played"] >= min_minutes].copy()
            df_agg["player_pos_minutes_played"] = df_agg.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")
            min_minutes_player = st.number_input("Minimum minutes played by player-pos for ICC.", min_value=0, value=DEFAULT_MINUTES)
            df_agg = df_agg[df_agg["player_pos_minutes_played"] >= min_minutes_player].copy()
            df_agg = df_agg.loc[:, ~df_agg.columns.duplicated()]

            st.write(f"Using {len(df_agg)} match performances of {len(df_agg['player_id'].unique())} players that played at least {min_minutes_player} minutes for ICC calculation.")
            with st.expander("Match-level ICC (how consistent and discriminative are KPIs on the match-level?)"):
                n_cols = 1
                columns = st.columns(n_cols)
                # player_col = "short_name"
                player_col = "player_id"
                for kpi_nr, kpi in enumerate(df_agg.columns):
                    col = columns[kpi_nr % n_cols]
                    try:
                        # deduplicate
                        df_agg = df_agg.loc[:, ~df_agg.columns.duplicated()]
                        icc = calculate_icc(df_agg, kpi, player_col, "coarse_position")
                        data.append({"kpi": kpi, "icc": icc})
                    except ValueError as e:
                        col.write(e)
                        data.append({"kpi": kpi, "icc": None})
                        continue
                    col.write(f"ICC for {kpi}: {icc}")

                    # sort by average
                    df_agg = df_agg.sort_values(by=kpi, ascending=False)
                    assert kpi in df_agg.columns and "short_name" in df_agg.columns, f"Columns {kpi} or short_name not found in df_agg"
                    # plt.figure(figsize=(16, 9))
                    sns.boxplot(data=df_agg, x="short_name", y=kpi, hue="coarse_position", width=1, dodge=False)
                    plt.xticks(rotation=90)
                    plt.xticks(fontsize=4)  # adjust the value as needed
                    plt.title(f"Discriminability of '{kpi}'")
                    col.write(plt.gcf())
                    plt.close()

            df_icc = pd.DataFrame(data)
            st.write("df_icc")
            st.write(df_icc)

        if st.toggle("Bootstrapped Season-Level ICC", False):
            min_minutes_per_match = st.number_input("Minimum minutes played by player for Season-Level ICC.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH)
            df_base_matchsums = df_player_matchsums.copy()
            df_base_matchsums = df_base_matchsums[df_base_matchsums["minutes_played"] >= min_minutes_per_match]
            min_minutes = st.number_input("Minimum minutes played by player-pos for Season-Level ICC.", min_value=0, value=DEFAULT_MINUTES)
            df_base_matchsums = df_base_matchsums[df_base_matchsums["player_pos_minutes_played"] >= min_minutes].copy()
            st.write(f"Using {len(df_base_matchsums)} match performances of {len(df_base_matchsums['player_id'].unique())} players that played at least {min_minutes} minutes for bootstrapped ICC calculation.")

            def bootstrap_aggregate(df, group_cols, n_bootstrap=2000, random_state=None):
                st.write(f"{n_bootstrap=}")
                rng = np.random.default_rng(random_state)
                results = []
                # results_icc = []

                for group_key, group_df in defensive_network.utility.general.progress_bar(df.groupby(group_cols), total=df[group_cols].nunique().prod(), desc="Bootstrapping"):
                    boots = []
                    iccs = []
                    for i in range(n_bootstrap):
                        sample = group_df.sample(n=len(group_df), replace=True, random_state=rng.integers(0, 1e9))
                        # boots.append(transform_func(sample))
                        df_agg = aggregate_matchsums(sample, group_cols)
                        df_agg["bootstrap_nr"] = i

                        # data = []
                        # for kpi in kpis:
                        #     icc = calculate_icc(df_agg, kpi, "player_id", "coarse_position")
                        #     data.append({"kpi": kpi, "icc": icc})
                        # df_icc = pd.DataFrame(data)
                        # iccs.append(df_icc)
                        boots.append(df_agg)

                    df = pd.concat(boots)
                    df["group_key"] = str(group_key)
                    # df_icc = pd.concat(iccs, axis=0)
                    # df_icc["group_key"] = str(group_key)
                    # result = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
                    # result.update({
                    #     'mean': boots.mean(),
                    #     'ci_lower': np.percentile(boots, 2.5),
                    #     'ci_upper': np.percentile(boots, 97.5)
                    # })
                    results.append(df)
                    # results_icc.append(df_icc)
                    # st.write("result")
                    # st.write(result)

                df = pd.concat(results, axis=0)
                df["n_bootstrap"] = n_bootstrap
                st.write("df_bootstrap")
                st.write(df)

                # df_icc = pd.concat(results_icc, axis=0)
                # df_icc["n_bootstrap"] = n_bootstrap
                # st.write("df_icc_bootstrap")
                # st.write(df_icc)

                return df

            st.write("df_base_matchsums")
            st.write(df_base_matchsums)

            df_agg = bootstrap_aggregate(df_base_matchsums, ["player_id", "coarse_position"])
            df_agg["player_and_coarse_position"] = df_agg["player_id"].astype(str) + "_" + df_agg["coarse_position"]

            st.write("df_agg", df_agg.shape)
            st.write(df_agg.head())

            data = []
            for kpi in kpis:
                icc = calculate_icc(df_agg, kpi, "player_id", "coarse_position")
                icc_posplayer = calculate_icc(df_agg, kpi, "player_id", "player_and_coarse_position")
                uncorrected_icc = calculate_icc(df_agg, kpi, "player_id", None)
                icc_posplayer_uncorrected = calculate_icc(df_agg, kpi, "player_and_coarse_position", None)
                data.append({"kpi": kpi, "icc": icc, "uncorrected_icc": uncorrected_icc, "icc_posplayer": icc_posplayer, "icc_posplayer_uncorrected": icc_posplayer_uncorrected})
            df = pd.DataFrame(data)
            st.write("df")
            st.write(df)

            # st.stop()

        if st.toggle("Histograms", False):
            min_minutes_per_match_hist = st.number_input("Minimum minutes played per match for histograms.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key="min_minutes_per_match_histograms_BAdsvs")
            min_minutes_total = st.number_input("Minimum total minutes played for histograms.", min_value=0, value=DEFAULT_MINUTES, key="min_minutes_total_histograms_BAdsvs")
            df_agg_player_pos_hist = df_agg_player_pos[df_agg_player_pos["minutes_played"] > min_minutes_total]
            df_agg_match_player_pos_hist = df_agg_match_player_pos[df_agg_match_player_pos["minutes_played"] > min_minutes_per_match_hist]
            with st.expander("Histograms"):
                for kpi in kpis:
                    columns = st.columns(2)
                    for col, df_agg, label in zip(columns, [
                            df_agg_player_pos_hist,
                            df_agg_match_player_pos_hist
                    ], ["Season-Level", "Match-Level"]):
                        col.write(f"Descriptives for {kpi} ({label})")
                        col.write(df_agg[kpi].describe())
                        # sns.histplot(df_agg[kpi], kde=True, log_scale=(True, False))
                        try:
                            st.write(df_agg[[kpi, "coarse_position"]])
                            sns.histplot(df_agg, x=kpi, hue="coarse_position", kde=True, log_scale=(False, False), multiple="stack")
                        except np.linalg.LinAlgError as e:
                            st.write(e)
                            continue

                        plt.title(f"Distribution of {kpi}")
                        col.write(plt.gcf())
                        plt.close()

                        plt.figure(figsize=(8, 5))
                        sns.boxplot(data=df_agg, x="coarse_position", y=kpi)
                        plt.title(f"Boxplot of {kpi} by Position ({label})")
                        col.pyplot(plt.gcf())
                        plt.close()

        if st.toggle("KPI correlations as heatmap", False):
            min_minutes_per_match_hist = st.number_input("Minimum minutes played per match for histograms.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key="min_minutes_per_match_hist_KPI_heatmpa")
            min_minutes_total = st.number_input("Minimum total minutes played for histograms.", min_value=0, value=DEFAULT_MINUTES)
            df_agg_player_pos_hist = df_agg_player_pos[df_agg_player_pos["minutes_played"] > min_minutes_total]
            df_agg_match_player_pos_hist = df_agg_match_player_pos[df_agg_match_player_pos["minutes_played"] > min_minutes_per_match_hist]
            with st.spinner("Calculating heatmap..."):
                columns = st.columns(2)
                for col, df_agg, label in zip(columns, [
                    df_agg_player_pos_hist,
                    df_agg_match_player_pos_hist
                ], ["Season-Level", "Match-Level"]):
                    st.write(label)
                    # heatmap
                    corr_matrix = df_agg[kpis].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 4})
                    plt.suptitle("Correlation Heatmap", y=1.02)
                    st.write(plt.gcf())
                    plt.close()


                    # do another heatmap with only the first 4 columns but all rows
                    plt.figure()
                    sns.heatmap(corr_matrix.iloc[:, :5], annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 4})
                    plt.suptitle("Correlation Heatmap", y=1.02)
                    st.write(plt.gcf())
                    plt.close()

        if st.toggle("FIFA Correlations", True):
            with st.expander("FIFA", True):
                import thefuzz
                df_fifa_players = defensive_network.parse.drive.download_csv_from_drive("fifa_ratings.csv", st_cache=False)
                files = defensive_network.parse.drive.list_files_in_drive_folder(".")
                st.write("files")
                st.write(files)
                # df_soccerdonna = defensive_network.parse.drive.download_csv_from_drive("market_values.csv", st_cache=False)
                # df_soccerdonna = defensive_network.parse.drive.download_excel_from_drive("market_values.xlsx", st_cache=False)
                df_soccerdonna = pd.read_excel("C:/Users/Jonas/Downloads/market_values.xlsx")  # TODO WHY
                st.write("df_soccerdonna")
                st.write(df_soccerdonna)
                # df_player_matchsums = defensive_network.parse.drive.download_csv_from_drive("players_matchsums.csv")
                # df_agg_player = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "position"])
                # df_agg_player["coarse_position"] = df_agg_player["position"].map(position_mapping)
                df_agg_player = df_agg_player_pos.copy()
                df_agg_player["full_name"] = df_agg_player["first_name"] + " " + df_agg_player["last_name"]
                min_minutes = st.number_input("Minimum minutes played by player for FIFA correlations.", min_value=0, value=DEFAULT_MINUTES)
                df_agg_player = df_agg_player[df_agg_player["minutes_played"] > min_minutes]

                # df_agg_team = aggregate_matchsums(df_player_matchsums, group_cols=["team_id", "position", "player_id"]).reset_index()
                # df_agg_team["full_name"] = df_agg_team["first_name"] + " " + df_agg_team["last_name"]
                # df_agg_team["coarse_position"] = df_agg_team["position"].map(position_mapping)
                df_agg_team = df_agg_team[df_agg_team["minutes_played"] > min_minutes]
                df_agg_team["full_name"] = df_agg_team["first_name"] + " " + df_agg_team["last_name"]

                # Map FIFA names
                fifa_names = df_fifa_players["name"].tolist()
                soccerdonna_names = df_soccerdonna["name"].tolist()

                manual_fifa_mapping = {
                    "Sammy Jerabek": "Samantha Jerabek",
                    "0 Letícia Santos": "Letícia Santos de Oliveira",
                    "Allie Hess": "Alexandria Loy Hess",
                    # 0 Letícia Santos	Letícia Santos de Oliveira
                    # Allie Hess	Alexandria Loy Hess
                }
                fifa_mapping = manual_fifa_mapping.copy()
                soccerdonna_mapping = dict()

                for idx, row in df_agg_player.iterrows():
                    import thefuzz.process
                    full_name = row['full_name']
                    if full_name in manual_fifa_mapping:
                        best_match, score = manual_fifa_mapping[full_name], 200
                    else:
                        best_match, score = thefuzz.process.extractOne(full_name, fifa_names)

                    # st.write(full_name, ",", best_match, score)
                    if score > 90:
                        df_agg_player.at[idx, 'fifa_match'] = best_match
                        df_agg_player.at[idx, 'match_score'] = score
                        fifa_mapping[full_name]  = best_match
                    else:
                        fifa_mapping[full_name] = None

                    # match Soccerdonna name
                    best_match_soccerdonna, score_soccerdonna = thefuzz.process.extractOne(full_name, soccerdonna_names)

                    if score_soccerdonna > 90:
                        df_agg_player.at[idx, 'soccerdonna_name'] = best_match_soccerdonna
                        df_agg_player.at[idx, 'soccerdonna_score'] = score_soccerdonna
                        soccerdonna_mapping[full_name] = best_match_soccerdonna
                    else:
                        soccerdonna_mapping[full_name] = None

                df_agg_team["fifa_name"] = df_agg_team["full_name"].map(fifa_mapping)
                df_agg_team = df_agg_team.merge(df_fifa_players, left_on="fifa_name", right_on="name", how="left")
                df_agg_team["soccerdonna_name"] = df_agg_team["full_name"].map(soccerdonna_mapping)
                df_agg_team = df_agg_team.merge(df_soccerdonna, left_on="soccerdonna_name", right_on="name", how="left")
                st.write("df_agg_team")
                st.write(df_agg_team)

                df_agg_player = df_agg_player.merge(df_fifa_players, left_on="fifa_match", right_on="name", how="left")
                try:
                    df_agg_player = df_agg_player.merge(df_soccerdonna, left_on="soccerdonna_name", right_on="name", how="left")
                except KeyError as e:
                    df_agg_player["market_value"] = np.nan
                    st.write(e)
                st.write("df_agg_player")
                st.write(df_agg_player)
                # df_agg_player = df_agg_player[df_agg_player["minutes_played"] > 600]
                # st.write("df_agg_team")
                # st.write(df_agg_team)

                data = []
                x_variables = kpis
                y_variables = ["def_awareness", "defending", "market_value"]
                # xy_variables = [(x, y) for x in x_variables for y in y_variables]
                for x_variable in x_variables:
                    columns = st.columns(len(y_variables))
                    for y_nr, y_variable in enumerate(y_variables):
                        col = columns[y_nr]
                        try:
                            # sns.regplot(data=df_agg_player, x=x_variable, y=y_variable, hue="position", scatter=True, label='Trendline')
                            sns.lmplot(data=df_agg_player, x=x_variable, y=y_variable, hue="coarse_position", scatter=True)
                            # set lower ylim to 0
                            plt.ylim(bottom=0)

                        except (ValueError, np.exceptions.DTypePromotionError, KeyError) as e:
                            col.write(e)

                        # for i, row in df_data_new.iterrows():
                        #     plt.text(
                        #         row[f"{kpi}_hinrunde"],  # x position
                        #         row[f"{kpi}_rückrunde"],  # y position
                        #         row["short_name_hinrunde"],
                        #         fontsize=9,
                        #         ha='right',
                        #         va='bottom'
                        #     )

                        try:
                            correlation_coefficient = df_agg_player[x_variable].corr(df_agg_player[y_variable])
                        except (ValueError, KeyError) as e:
                            st.write(e)
                            correlation_coefficient = np.nan

                        df_agg_player_only_cb = df_agg_player[df_agg_player["coarse_position"] == "CB"]
                        correlation_coefficient_only_CBs = df_agg_player_only_cb[x_variable].corr(df_agg_player_only_cb[y_variable])

                        plt.title(f"{x_variable}-{y_variable}")
                        col.write(f"{x_variable}-{y_variable}")
                        #
                        plt.xlabel(x_variable)
                        plt.ylabel(y_variable)
                        plt.title(f"Correlation of {x_variable} and {y_variable}")
                        col.write(plt.gcf())
                        plt.close()
                        #
                        # st.write("BLA")
                        # st.write("df_agg_team")
                        # st.write(df_agg_team)

                        # Assume df has columns: x, y, team, position

                        # df_agg_team["position"] = df_agg_team["position_x"].fillna(df_agg_team["position_y"])

                        with st.spinner("Regress"):
                            df_agg_team = df_agg_team.dropna(subset=[x_variable, y_variable, "team_id", "position"])

                            # Regress x on team and position
                            try:
                                model_x = statsmodels.formula.api.ols(f'{x_variable} ~ C(team_id) + C(coarse_position)', data=df_agg_team).fit()
                            except ValueError as e:
                                st.write(e)
                                continue
                            resid_x = model_x.resid

                            # Regress y on team and position
                            model_y = statsmodels.formula.api.ols(f'{y_variable} ~ C(team_id) + C(coarse_position)', data=df_agg_team).fit()
                            resid_y = model_y.resid

                            # Correlation of residuals
                            try:
                                corr, pval = pearsonr(resid_x, resid_y)
                            except ValueError as e:
                                st.write(e)
                                continue

                            # Regress on position only
                            model_x_pos = statsmodels.formula.api.ols(f'{x_variable} ~ C(coarse_position)', data=df_agg_team).fit()
                            resid_x_pos = model_x_pos.resid
                            model_y_pos = statsmodels.formula.api.ols(f'{y_variable} ~ C(coarse_position)', data=df_agg_team).fit()
                            resid_y_pos = model_y_pos.resid
                            corr_pos, pval_pos = pearsonr(resid_x_pos, resid_y_pos)

                            data.append({
                                "r": correlation_coefficient, "abs_r": abs(correlation_coefficient), "x": x_variable,
                                "y": y_variable,
                                # "team_and_position_corrected_correlation": corr, "p": pval,
                                "position_corrected_correlation_pos": corr_pos, "p_pos": pval_pos,
                                "correlation_coefficient_only_CBs": correlation_coefficient_only_CBs,
                                "abs_cc_only_CBs": abs(correlation_coefficient_only_CBs)
                            })

                df = pd.DataFrame(data)
                # df["total_p"] = df["p"] + df["p_pos"]
                # df["both_significant"] = (df["p"] < 0.05) & (df["p_pos"] < 0.05)
                st.write("df")
                st.write(df)


def _create_matchsums(df_event, df_tracking, series_meta, df_lineup, df_involvement):
    dfgs = []

    if "xg" not in df_event.columns:
        df_event["xg"] = None
    if "xpass" not in df_event.columns:
        df_event["xpass"] = None
    if "d_tracking" not in df_tracking.columns:
        df_tracking["d_tracking"] = None

    if "event_id" not in df_event.columns:
        df_event["event_id"] = df_event.index

    assert "pass_xt" in df_event.columns

    match_string = series_meta["match_string"]
    if "World Cup" in match_string:
        df_event["player_id_2"] = df_event["player_id_2"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_event["player_id_1"] = df_event["player_id_1"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_event["team_id_1"] = df_event["team_id_1"].astype(str).replace({"None": None, "nan": None, "NaN": None})
        df_event["team_id_2"] = df_event["team_id_2"].astype(str).replace({"None": None, "nan": None, "NaN": None})
        # df_tracking["player_id"] = df_tracking["player_id"].astype(str).replace({"None": None, "nan": None, "NaN": None})
        # df_tracking["team_id"] = df_tracking["team_id"].astype(str).replace({"None": None, "nan": None, "NaN": None})
        df_involvement["defender_id"] = df_involvement["defender_id"].astype(str).replace({"None": None, "nan": None, "NaN": None})

    # Minutes
    df_possession = df_tracking.groupby(["section"]).apply(lambda df_section : df_section.groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "nunique"}))
    df_possession = df_possession.reset_index().groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "sum"})
    df_possession["frame"] = df_possession["frame"] / (series_meta["fps"] * 60)
    df_possession = df_possession.rename(columns={"frame": "minutes"}).reset_index().pivot(index="ball_poss_team_id", columns="ball_status", values="minutes").drop(columns=[0])
    df_possession.columns = [f"net_minutes_in_possession"]
    df_possession["net_minutes_opponent_in_possession"] = df_possession["net_minutes_in_possession"].values[::-1]
    df_possession["net_minutes"] = df_possession["net_minutes_opponent_in_possession"] + df_possession["net_minutes_in_possession"]

    df_tracking["datetime_tracking"] = pd.to_datetime(df_tracking["datetime_tracking"])

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
    for col, value in series_meta.items():
        dfg_team[col] = value

    dfg_team = defensive_network.utility.dataframes.move_column(dfg_team, "match_id", 1)

    ### Players
    dfgs_players = []
    player_group_cols = ["player_id_1", "role_category_1"]
    receiver_group_cols = ["player_id_2", "role_category_2"]
    involvement_group_cols = ["defender_id", "defender_role_category"]

    # Player name
    dfg_lineup = df_lineup[~df_lineup["player_id"].str.startswith("DFL-OBJ-")].groupby(["player_id"], observed=False).agg(
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        short_name=("short_name", "first"),
        position_group=("position_group", "first"),
        position=("position", "first"),
        team_id=("team_id", "first"),
        team_name=("team_name", "first"),
        team_role=("team_role", "first"),
        starting=("starting", "first"),
        captain=("captain", "first"),
        jersey_number=("jersey_number", "first"),
    ).to_dict()

    # Positions
    df_tracking["role_category"] = df_tracking["role"]
    dfg_tracking = df_tracking.groupby(["player_id", "role_category"]).agg(
        n_frames=("frame", "count"),
        distance_covered=("d_tracking", "sum"),
    )
    dfg_tracking["minutes_played"] = dfg_tracking["n_frames"] / series_meta["fps"] / 60
    dfg_tracking["distance_covered_per_90_minutes"] = dfg_tracking["distance_covered"] / dfg_tracking["minutes_played"] * 90
    # dfg_tracking = df_tracking.groupby(["player_id", "section"]).agg(
    st.write("dfg_trackingdfg_tracking", dfg_tracking.shape)
    st.write(dfg_tracking)
    dfgs_players.append(dfg_tracking)

    # Tackles
    dfg_players_tackles_won = df_event[df_event["event_subtype"] == "tackle"].groupby(player_group_cols, observed=False).agg(
        n_tackles_won=("event_id", "count"),
    ).fillna(0)
    st.write("df_tackles")
    st.write(df_event[df_event["event_subtype"] == "tackle"][["event_id", "player_id_1", "role_category_1"]])
    st.write("dfg_players_tackles_won", dfg_players_tackles_won.shape)
    st.write(dfg_players_tackles_won)
    dfgs_players.append(dfg_players_tackles_won)

    dfg_players_tackles_lost = df_event[df_event["event_subtype"] == "tackle"].groupby(receiver_group_cols, observed=False).agg(
        n_tackles_lost=("event_id", "count"),
    ).reset_index().rename(columns={"player_id_2": "player_id_1", "role_category_2": "role_category_1"}).set_index(["player_id_1", "role_category_1"]).fillna(0)
    st.write("dfg_players_tackles_lost", dfg_players_tackles_lost.shape)
    st.write(dfg_players_tackles_lost)
    dfg_players_tackles_lost["tackles_won_share"] = dfg_players_tackles_won["n_tackles_won"] / (dfg_players_tackles_won["n_tackles_won"] + dfg_players_tackles_lost["n_tackles_lost"])
    st.write("dfg_players_tackles_lost", dfg_players_tackles_lost.shape)
    print(dfg_players_tackles_lost)
    # st.write(dfg_players_tackles_lost)
    dfgs_players.append(dfg_players_tackles_lost)

    # Interceptions
    dfg_players_interceptions = df_event[df_event["outcome"] == "intercepted"].groupby(receiver_group_cols, observed=False).agg(
        n_interceptions=("event_id", "count"),
    ).reset_index().fillna(0).rename(columns={"player_id_2": "player_id_1", "role_category_2": "role_category_1"}).set_index(["player_id_1", "role_category_1"]).fillna(0)
    dfg_players_interceptions["n_interceptions"] = dfg_players_interceptions["n_interceptions"].fillna(0)
    dfg_players_interceptions = dfg_players_interceptions.rename(columns={"player_id_2": "player_id_1"})
    st.write("dfg_players_interceptions", dfg_players_interceptions.shape)
    st.write(dfg_players_interceptions)
    dfgs_players.append(dfg_players_interceptions)

    # Pass xP
    dfg_players_pass = df_passes.groupby(player_group_cols, observed=False).agg(
        n_passes=("event_id", "count"),
        total_xp=("xpass", "sum"),
        total_xt=("pass_xt", "sum"),
    )
    dfg_players_pass["xp_per_pass"] = dfg_players_pass["total_xp"] / dfg_players_pass["n_passes"]
    st.write("dfg_players_pass")
    st.write(dfg_players_pass)
    dfgs_players.append(dfg_players_pass)

    dfg_players_pass_successful = df_passes[df_passes["event_outcome"] == "successfully_completed"].groupby(player_group_cols, observed=False).agg(
        n_passes_successful=("event_id", "count"),
    )
    st.write("dfg_players_pass_successful")
    st.write(dfg_players_pass_successful)
    dfg_players_pass_successful["pass_completion_rate"] = dfg_players_pass_successful["n_passes_successful"] / dfg_players_pass["n_passes"]
    dfgs_players.append(dfg_players_pass_successful)

    # Involvement
    df_involvement = df_involvement.drop_duplicates(["defender_id", "involvement_pass_id"])
    df_fault = df_involvement[df_involvement["pass_xt"] > 0]
    df_contribution = df_involvement[df_involvement["pass_xt"] <= 0]

    dfg_fault = df_fault.groupby(involvement_group_cols, observed=False).agg(
        total_raw_fault=("raw_involvement", "sum"),
        total_valued_fault=("valued_involvement", "sum"),
        total_valued_fault_responsibility=("valued_responsibility", "sum"),
        total_raw_fault_responsibility=("raw_responsibility", "sum"),
        total_relative_raw_fault_responsibility=("relative_raw_responsibility", "sum"),
        total_relative_valued_fault_responsibility=("relative_valued_responsibility", "sum"),

        n_passes_with_contribution_responsibility=("raw_responsibility", lambda x: (x != 0).sum()),

        # total_intrinsic_fault_responsibility=("intrinsic_responsibility", "sum"),
        # total_intrinsic_valued_fault_responsibility=("intrinsic_valued_responsibility", "sum"),
        # n_passes_with_fault=("raw_involvement", lambda x: (x != 0).sum()),
        # n_passes_with_fault_responsibility=("valued_responsibility", lambda x: ((x > 0) & (x != 0)).sum()),
    )
    dfgs_players.append(dfg_fault)
    # st.write("dfg_fault")
    # st.write(dfg_fault)
    dfg_contribution = df_contribution.groupby(involvement_group_cols, observed=False).agg(
        total_raw_contribution=("raw_involvement", "sum"),
        total_valued_contribution=("valued_involvement", "sum"),
        total_valued_contribution_responsibility=("valued_responsibility", "sum"),
        total_raw_contribution_responsibility=("raw_responsibility", "sum"),
        total_relative_raw_contribution_responsibility=("relative_raw_responsibility", "sum"),
        total_relative_valued_contribution_responsibility=("relative_valued_responsibility", "sum"),

        # n_passes_with_fault_responsibility=("valued_responsibility", lambda x: ((x > 0) & (x != 0)).sum()),
        n_passes_with_fault_responsibility=("raw_responsibility", lambda x: (x != 0).sum()),

        # total_intrinsic_contribution_responsibility=("intrinsic_responsibility", "sum"),
        # total_intrinsic_valued_contribution_responsibility=("intrinsic_valued_responsibility", "sum"),
        # n_passes_with_contribution=("raw_involvement", lambda x: (x != 0).sum()),
        # n_passes_with_contribution_responsibility=("valued_responsibility", lambda x: ((x < 0) & (x != 0)).sum()),
    )
    dfgs_players.append(dfg_contribution)

    dfg_involvement = df_involvement.groupby(involvement_group_cols, observed=False).agg(
        # Involvement
        total_raw_involvement=("raw_involvement", "sum"),
        # total_raw_fault=("raw_fault", "sum"),
        # total_raw_contribution=("raw_contribution", "sum"),
        # total_valued_involvement=("valued_involvement", "sum"),  # NOT calulated like this!!!
        # total_fault=("fault", "sum"),
        # total_contribution=("contribution", "sum"),
        n_passes_with_involvement=("raw_involvement", lambda x: (x != 0).sum()),
        n_passes_with_fault=("raw_fault", lambda x: (x != 0).sum()),
        n_passes_with_contribution=("raw_contribution", lambda x: (x != 0).sum()),

        # Responsibility
        # total_intrinsic_responsibility=("intrinsic_responsibility", "sum"),
        # total_intrinsic_valued_responsibility=("intrinsic_valued_responsibility", "sum"),
        # total_intrinsic_relative_responsibility=("intrinsic_relative_responsibility", "sum"),
        total_raw_responsibility=("raw_responsibility", "sum"),
        # total_relative_responsibility=("relative_responsibility", "sum"),  # not meaningful
        total_valued_responsibility=("valued_responsibility", "sum"),
        # total_valued_fault_responsibility=("valued_responsibility", lambda x: (x * (x > 0)).sum()),
        # total_valued_contribution_responsibility=("valued_responsibility", lambda x: (x * (x < 0)).sum()),
        total_relative_raw_responsibility=("relative_raw_responsibility", "sum"),
        total_relative_valued_responsibility=("relative_valued_responsibility", "sum"),

        n_passes_against=("involvement_pass_id", "nunique"),
        n_passes_with_responsibility=("raw_responsibility", lambda x: (x != 0).sum()),
        # n_passes_with_fault_responsibility=("valued_responsibility", lambda x: ((x > 0) & (x != 0)).sum()),
        # n_passes_with_contribution_responsibility=("valued_responsibility", lambda x: ((x < 0) & (x != 0)).sum()),
        # model_radius=("model_radius", "first")
    ).fillna(0)

    # st.write("dfg_involvement")
    # st.write(dfg_involvement)
    dfgs_players.append(dfg_involvement)

    import functools
    dfg_players = pd.concat(dfgs_players, axis=1)

    dfg_players = dfg_players.reset_index().rename(columns={"level_0": "player_id", "level_1": "role_category"})

    for meta_key, meta_value in dfg_lineup.items():
        if meta_key in dfg_players.columns:
            st.warning(f"Column {meta_key} already exists in dfg_players, skipping")
            continue
        dfg_players[meta_key] = dfg_players["player_id"].map(meta_value)

    for col, value in series_meta.items():
        dfg_players[col] = value

    dfg_players = defensive_network.utility.dataframes.move_column(dfg_players, "match_id", 2)
    dfg_team = defensive_network.utility.dataframes.move_column(dfg_team, "match_id", 2)

    dfg_players = dfg_players.fillna(0)
    dfg_team = dfg_team.fillna(0)

    dfg_players["total_valued_involvement"] = dfg_players["total_valued_contribution"] - dfg_players["total_valued_fault"]
    dfg_players["total_valued_responsibility"] = dfg_players["total_valued_contribution_responsibility"] - dfg_players["total_valued_fault_responsibility"]
    dfg_players["total_relative_valued_responsibility"] = dfg_players["total_relative_valued_contribution_responsibility"] - dfg_players["total_relative_valued_fault_responsibility"]

    # st.write("dfg_players")
    # st.write(dfg_players)
    # st.write(dfg_players[["total_valued_involvement", "total_valued_contribution", "total_valued_fault"]])
    # dfg_players["sum_ok"] = np.isclose(dfg_players["total_raw_involvement"], (dfg_players["total_raw_contribution"] + dfg_players["total_raw_fault"]))
    # st.write(dfg_players[["total_raw_involvement", "total_raw_contribution", "total_raw_fault", "sum_ok"]])

    assert (df_involvement["raw_involvement"] == (df_involvement["raw_contribution"] + df_involvement["raw_fault"])).all()
    assert np.allclose(dfg_players["total_raw_involvement"], (dfg_players["total_raw_contribution"] + dfg_players["total_raw_fault"]))
    assert np.allclose(dfg_players["total_raw_responsibility"], (dfg_players["total_raw_contribution_responsibility"] + dfg_players["total_raw_fault_responsibility"]))

    # dfg_players["sum"] = np.isclose(dfg_players["total_valued_involvement"], (dfg_players["total_valued_contribution"] - dfg_players["total_valued_fault"]))
    # st.write(dfg_players[["total_valued_involvement", "total_valued_contribution", "total_valued_fault", "sum"]])
    assert np.allclose(dfg_players["total_valued_involvement"], (dfg_players["total_valued_contribution"] - dfg_players["total_valued_fault"]))
    assert np.allclose(dfg_players["total_valued_responsibility"], (dfg_players["total_valued_contribution_responsibility"] - dfg_players["total_valued_fault_responsibility"]))
    assert np.allclose(dfg_players["total_relative_valued_responsibility"], (dfg_players["total_relative_valued_contribution_responsibility"] - dfg_players["total_relative_valued_fault_responsibility"]))

    assert (dfg_players["n_passes_with_involvement"] == (dfg_players["n_passes_with_contribution"] + dfg_players["n_passes_with_fault"])).all()
    assert (dfg_players["n_passes_with_responsibility"] == (dfg_players["n_passes_with_contribution_responsibility"] + dfg_players["n_passes_with_fault_responsibility"])).all()
    assert (dfg_players["n_passes_with_responsibility"] <= dfg_players["n_passes_against"]).all()
    assert (dfg_players["n_passes_with_involvement"] <= dfg_players["n_passes_against"]).all()

    # st.write("dfg_players")
    # st.write(dfg_players)
    # st.stop()

    return dfg_team, dfg_players


def process_involvements(df_meta, folder_tracking, folder_events, target_folder, overwrite_if_exists=False):
    present_match_ids = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(folder_tracking)]
    df_meta = df_meta[df_meta["slugified_match_string"].isin(present_match_ids)]

    st.write("df_meta")
    st.write(df_meta)

    if not overwrite_if_exists:
        finished_files = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(target_folder)]
        st.write("finished_files")
        st.write(finished_files)
        df_meta = df_meta[~df_meta["slugified_match_string"].isin(finished_files)]

    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Processing involvements"):
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]

        st.write(f"#### {slugified_match_string}")

        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")
        try:
            df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
        except FileNotFoundError:
            continue

        df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events)
        df_events = df_events[df_events["event_type"] == "pass"]
        # st.write("df_events")
        # st.write(df_events[[col for col in df_events.columns if "frame" in col]])
        # st.write(df_events)
        # df_events = df_events[df_events["frame"].isin([69, 70])]
        # st.write("df_tracking")
        # st.write(df_tracking.head(10000))
        # st.write('df_tracking[df_tracking["role_category"].isna()]')
        # st.write(df_tracking[df_tracking["role_category"].isna()])

        df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_defender_meta_cols=["role_category"])

        # # TODO remove
        # df_involvement = df_involvement[df_involvement["involvement_pass_id"] == 0]
        # df_tracking = df_tracking[df_tracking["frame"] == 0]

        # st.write("df_involvement")
        # st.write(df_involvement[["defender_id", "defender_role_category", "role_category_1", "role_category_2", "expected_receiver_role_category"]])
        # st.write(df_involvement[df_involvement["involvement_pass_id"] == 0].shape)
        # st.write(df_involvement[df_involvement["involvement_pass_id"] == 0])
        # #
        # st.write("df_tracking")
        # st.write(df_tracking[df_tracking["frame"] == 0].shape)
        # st.write(df_tracking[df_tracking["frame"] == 0])
        #
        # st.stop()

        df_involvement["network_receiver_role_category"] = df_involvement["expected_receiver_role_category"].where(df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])
        df_involvement["defender_role_category"] = df_involvement["defender_role_category"]#.fillna("unknown")
        df_involvement["role_category_1"] = df_involvement["role_category_1"]#.fillna("unknown")
        df_involvement["network_receiver_role_category"] = df_involvement["network_receiver_role_category"]#.fillna("unknown")
        # st.write("df_involvement a")
        # st.write(df_involvement[["defender_id", "defender_x", "defender_y", "defender_role_category", "role_category_1", "network_receiver_role_category"]])
        df_involvement = df_involvement.dropna(subset=["defender_id", "defender_role_category", "role_category_1", "network_receiver_role_category"], how="any")
        # st.write("df_involvement b")
        # st.write(df_involvement[["defender_id", "defender_x", "defender_y", "defender_role_category", "role_category_1", "network_receiver_role_category"]])
        intrinsic_context_cols = ["defending_team", "role_category_1", "network_receiver_role_category", "defender_role_category"]
        dfg_responsibility = defensive_network.models.responsibility.get_responsibility_model(df_involvement, responsibility_context_cols=intrinsic_context_cols)
        # st.write("dfg_responsibility")
        # st.write(dfg_responsibility)
        df_involvement["raw_intrinsic_responsibility"], df_involvement["raw_intrinsic_relative_responsibility"], df_involvement["valued_intrinsic_responsibility"], df_involvement["valued_intrinsic_relative_responsibility"] = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model=dfg_responsibility, context_cols=intrinsic_context_cols)

        # st.write(df_involvement[df_involvement["involvement_pass_id"] == 0].shape)
        # st.write(df_involvement[df_involvement["involvement_pass_id"] == 0])
        #

# # raw_responsibility", "relative_raw_responsibility", "valued_responsibility", "relative_valued_responsibility
        # st.write("df_involvement")
        # st.write(df_involvement)

        target_fpath = os.path.join(target_folder, f"{slugified_match_string}.csv")
        importlib.reload(defensive_network.utility.pitch)
        # st.write("df_involvement")
        # st.write(df_involvement)
        defensive_network.utility.pitch.plot_passes_with_involvement(
            df_involvement, df_tracking, responsibility_col="raw_intrinsic_relative_responsibility", n_passes=5
        )
        assert len(df_involvement) > 100
        defensive_network.parse.drive.upload_csv_to_drive(df_involvement, target_fpath)
        # st.stop()

        # for coordinates in ["original", "sync"]:
        #     with st.expander(f"Plot passes with involvement ({coordinates})"):
        #         if coordinates == "original":
        #             df_events["frame"] = df_events["original_frame_id"]
        #         else:
        #             df_events["frame"] = df_events["matched_frame"]
        #
        #         df_events["full_frame"] = df_events["section"].str.cat(df_events["frame"].astype(float).astype(str), sep="-")
        #
        #         df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_defender_meta_cols=["role_category"])
        #         df_involvement["network_receiver_role_category"] = df_involvement["expected_receiver_role_category"].where(df_involvement["expected_receiver_role_category"].notna(), df_involvement["role_category_2"])
        #         dfg_responsibility = defensive_network.models.responsibility.get_responsibility_model(df_involvement, responsibility_context_cols=["defending_team", "role_category_1", "network_receiver_role_category", "defender_role_category"])
        #         df_involvement["intrinsic_responsibility"], _ = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model=dfg_responsibility)
        #
        #         # upload
        #         # target_fpath = os.path.join(target_folder, f"{slugified_match_string}.csv")
        #         # defensive_network.parse.drive.upload_csv_to_drive(df_involvement, target_fpath)
        #
        #         st.write("df_involvement")
        #         st.write(df_involvement)
        #
                # defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, responsibility_col="intrinsic_responsibility", n_passes=5)


def create_matchsums(folder_tracking, folder_events, df_meta, df_lineups, target_fpath_team, target_fpath_players, folder_involvement="involvement", overwrite_if_exists=False):
    existing_match_ids = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(folder_events, st_cache=False)]
    df_meta = df_meta[df_meta["slugified_match_string"].isin(existing_match_ids)]

    if not overwrite_if_exists:
        try:
            df_matchsums_player = defensive_network.parse.drive.download_csv_from_drive(target_fpath_players, st_cache=False)
            df_matchsums_player.to_excel("df_matchsums_player.xlsx", index=True)
            df_matchsums_team = defensive_network.parse.drive.download_csv_from_drive(target_fpath_team, st_cache=False)
            match_ids = set(df_matchsums_player["match_id"]).intersection(df_matchsums_team["match_id"])
        except FileNotFoundError:
            match_ids = set()

        df_meta = df_meta[~df_meta["match_id"].isin(match_ids)]

    dfs_player = []
    dfs_team = []
    match_nr = 0
    for _, match in defensive_network.utility.general.progress_bar(df_meta.iloc[::-1].iterrows(), total=len(df_meta), desc="Creating matchsums"):
        match_nr += 1
        competition_name = match["competition_name"]
        # if match_nr <= 104:
        #     continue
        df_lineup = df_lineups[df_lineups["match_id"] == match["match_id"]]
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]
        st.write(f"Creating matchsums for {match_string}")
        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")
        fpath_involvement = os.path.join(folder_involvement, f"{slugified_match_string}.csv")

        # df_tracking = pd.read_parquet(fpath_tracking)
        # df_events = pd.read_csv(fpath_events)
        try:
            # df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
            with st.spinner("Loading data..."):
                df_involvement = defensive_network.parse.drive.download_csv_from_drive(fpath_involvement, st_cache=False)  # TODO no cache

                # @st.cache_resource
                def _get_parquet(fpath):
                    return pd.read_parquet(fpath)

                # df_tracking = pd.read_parquet(fpath_tracking)
                df_tracking = _get_parquet(fpath_tracking)
                df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events, st_cache=False)
                dfg_responsibility_model = defensive_network.parse.drive.download_csv_from_drive(f"responsibility_model_{competition_name.replace('Men\'s', 'Mens')}.csv", st_cache=False).reset_index(drop=True)
                # st.write("dfg_responsibility_model")
                # st.write(dfg_responsibility_model)
            with st.spinner("Calculating Responsibility..."):
                df_involvement["raw_responsibility"], df_involvement["relative_raw_responsibility"], df_involvement["valued_responsibility"], df_involvement["relative_valued_responsibility"] = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model)

        except FileNotFoundError as e:
            st.write(e)
            continue

        with st.spinner("Aggregating matchsums..."):
            dfg_team, dfg_players = _create_matchsums(df_events, df_tracking, match, df_lineup, df_involvement)
        # st.write("dfg_team")
        # st.write(dfg_team)
        # st.write(dfg_team[["team_id", "match_id"]])
        # st.write("dfg_players")
        # st.write(dfg_players)
        # st.write(dfg_players[["player_id", "role_category", "match_id"]])

        dfs_player.append(dfg_players)
        dfs_team.append(dfg_team)

        with st.spinner("Uploading team data..."):
            defensive_network.parse.drive.append_to_parquet_on_drive(dfg_team, target_fpath_team, key_cols=["team_id", "match_id"], overwrite_key_cols=True, format="csv")
        with st.spinner("Uploading players data..."):
            defensive_network.parse.drive.append_to_parquet_on_drive(dfg_players, target_fpath_players, key_cols=["player_id", "role_category", "match_id"], overwrite_key_cols=True, format="csv")
        st.write("Uploaded!")

    # dfg_team = pd.concat(dfs_team)
    # dfg_players = pd.concat(dfs_player)

    # defensive_network.parse.drive.upload_csv_to_drive(dfg_team, target_fpath_team)
    # defensive_network.parse.drive.upload_csv_to_drive(dfg_players, target_fpath_players)


    # st.write("Appended")
    # st.stop()


def finalize_events_and_tracking_to_drive(folder_tracking, folder_events, df_meta, df_lineups, target_folder_events, target_folder_tracking, full_target_folder_tracking, do_not_process_if_synchronized=True):
    existing_match_ids_event = [file.rsplit(".", 1)[0] for file in os.listdir(folder_events)]
    existing_match_ids_tracking = [file.rsplit(".", 1)[0] for file in os.listdir(folder_tracking)]
    existing_match_ids = [match for match in existing_match_ids_event if match in existing_match_ids_tracking]

    df_meta = df_meta[df_meta["slugified_match_string"].isin(existing_match_ids)]

    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Finalizing matches"):
        import gc
        gc.collect()
        df_lineup = df_lineups[df_lineups["match_id"] == match["match_id"]]
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]
        st.write(f"Finalizing {match_string}")
        fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
        fpath_full_tracking = os.path.join(full_target_folder_tracking, f"{slugified_match_string}.parquet")
        fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")

        drive_path_events = os.path.join(target_folder_events, f"{slugified_match_string}.csv")
        drive_path_tracking = os.path.join(target_folder_tracking, f"{slugified_match_string}.parquet")

        if do_not_process_if_synchronized:
            try:
                df_event_existing = defensive_network.parse.drive.download_csv_from_drive(drive_path_events)
                st.write("df_event_existing")
                st.write(df_event_existing)
                if df_event_existing["matched_frame"].notna().any():
                    st.write(f"Skipping {match_string} because it is already synchronized")
                    continue
                pass
            except FileNotFoundError:
                pass

        # @st.cache_resource
        def read_parquet(fpath):
            return pd.read_parquet(fpath)

        # @st.cache_resource
        def read_csv(fpath):
            return pd.read_csv(fpath)

        df_tracking = read_parquet(fpath_tracking)
        df_event = pd.read_csv(fpath_events)

        if "x_event_player_1" not in df_event.columns:
            df_event["x_event_player_1"] = df_event["x_event"]
            df_event["y_event_player_1"] = df_event["y_event"]
        if "x_event_player_2" not in df_event.columns:
            df_event["x_event_player_2"] = df_event["x_tracking_player_2"]
            df_event["y_event_player_2"] = df_event["y_tracking_player_2"]

        # df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
        # df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events)

        with st.spinner("Augmenting event and tracking data..."):
            try:
                df_tracking, df_event = defensive_network.parse.dfb.cdf.augment_match_data(match, df_event, df_tracking, df_lineup)
            except (AssertionError, ValueError) as e:
                raise e
                continue

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

        with st.spinner("Writing full tracking data to disk..."):
            df_tracking.to_parquet(fpath_full_tracking)

        defensive_network.parse.drive.upload_csv_to_drive(df_event, drive_path_events)

        st.write(f"Finalized events and tracking for {match_string} ({drive_path_events} and {fpath_full_tracking}), but didn't upload to drive yet.")

        # df_events_downloaded = defensive_network.parse.drive.download_csv_from_drive(drive_path_events)
        # df_tracking_downloaded = defensive_network.parse.drive.download_parquet_from_drive(drive_path_tracking)

        # dfg_team, dfg_players = _create_matchsums(df_events, df_tracking, match, df_lineup)
        #
        # st.write("dfg_team")
        # st.write(dfg_team)
        # st.write("dfg_players")
        # st.write(dfg_players)


def upload_reduced_tracking_data(df_meta, drive_folder_events, full_tracking_folder, drive_folder_tracking, overwrite_if_exists=False):
    existing_tracking_match_ids = [file.split(".")[0] for file in os.listdir(full_tracking_folder)]
    df_meta = df_meta[df_meta["slugified_match_string"].isin(existing_tracking_match_ids)]

    if not overwrite_if_exists:
        existing_reduced_tracking_match_ids = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(drive_folder_tracking)]
        df_meta = df_meta[~df_meta["slugified_match_string"].isin(existing_reduced_tracking_match_ids)]

    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Reducing tracking data"):
        match_string = match["match_string"]
        slugified_match_string = match["slugified_match_string"]
        st.write(f"Reducing {match_string}")
        fpath_full_tracking = os.path.join(full_tracking_folder, f"{slugified_match_string}.parquet")
        drive_path_events = os.path.join(drive_folder_events, f"{slugified_match_string}.csv")
        drive_path_tracking = os.path.join(drive_folder_tracking, f"{slugified_match_string}.parquet")

        df_events = defensive_network.parse.drive.download_csv_from_drive(drive_path_events)
        df_tracking = pd.read_parquet(fpath_full_tracking)
        st.write("df_events")
        st.write(df_events)
        if "matched_frame" in df_events.columns:
            df_events["matched_frame_id"] = df_events["matched_frame"]

        _upload_reduced_tracking_data(df_events, df_tracking, drive_path_tracking)


def _upload_reduced_tracking_data(df_event, df_tracking, drive_path_tracking):
    # Make tracking data smaller to store in Drive
    df_event["original_full_frame"] = df_event["section"].str.cat(df_event["original_frame_id"].astype(float).astype(str), sep="-")
    df_event["matched_full_frame"] = df_event["section"].str.cat(df_event["matched_frame_id"].astype(float).astype(str), sep="-")
    df_tracking_reduced = df_tracking[
        df_tracking["full_frame"].isin(df_event["full_frame"]) |
        df_tracking["full_frame"].isin(df_event["original_full_frame"]) |
        df_tracking["full_frame"].isin(df_event["matched_full_frame"])
    ]

    with st.spinner("Uploading to drive..."):
        defensive_network.parse.drive.upload_parquet_to_drive(df_tracking_reduced, drive_path_tracking)


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


def create_videos(overwrite_if_exists=False, only_n_frames_per_half=5000):
    import matplotlib
    matplotlib.use('TkAgg')  # make plots show in new window (for animation)

    # folder = "C:/Users/Jonas/Downloads/dfl_test_data/2324/"
    # folder = base_path
    # if not os.path.exists(folder):
    #     raise NotADirectoryError(f"Folder {folder} does not exist")
    #
    # folder_events = os.path.join(folder, "events")
    # folder_tracking = os.path.join(folder, "tracking")
    # folder_animation = os.path.join(folder, "animation")
    # match_slugified_strings_to_animate = [os.path.splitext(file)[0] for file in os.listdir(folder_tracking)]

    existing_matches = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder("tracking", st_cache=False)]
    # existing_matches = [file.split(".")[0] for file in os.listdir(os.path.join(os.path.dirname(__file__), "../../../w_raw/preprocessed/tracking"))]
    # st.write("existing_matches")
    # st.write("existing_matches")
    # st.write(existing_matches)
    # st.write("A")

    for match_slugified_string in existing_matches:  # defensive_network.utility.general.progress_bar(match_slugified_strings_to_animate):
        fname = f"{match_slugified_string}_only_{only_n_frames_per_half}_frames_per_half.mp4"
        folder = "animation"
        drive_fpath = f"{folder}/{fname}"
        if not overwrite_if_exists and fname in defensive_network.parse.drive.list_files_in_drive_folder(folder):
            continue

        st.write(match_slugified_string)
        # target_fpath = os.path.join(folder_animation, f"{match_slugified_string}.mp4")
        # if os.path.exists(target_fpath):
        #     print(f"File {target_fpath} already exists, skipping")
        #     continue
        # df_event = pd.read_csv(os.path.join(folder_events, f"{match_slugified_string}.csv"))
        # df_tracking = pd.read_parquet(os.path.join(folder_tracking, f"{match_slugified_string}.parquet"))
        try:
            df_event = defensive_network.parse.drive.download_csv_from_drive(f"events/{match_slugified_string}.csv", st_cache=False)
            # df_event = defensive_network.parse.dfb.cdf.get_events(base_path, match_slugified_string)
        except FileNotFoundError as e:
            # st.write(e)
            continue

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

        local_fpath = os.path.join(os.path.dirname(__file__), f"{match_slugified_string}.mp4")
        defensive_network.utility.video.pass_video(df_tracking, df_passes, out_fpath=local_fpath, overwrite_if_exists=False,
                                                   only_n_frames_per_half=only_n_frames_per_half)
        defensive_network.parse.drive.upload_file_to_drive(local_fpath, drive_fpath)


if __name__ == '__main__':
    main()

    # df = pd.read_csv("C:/Users/j.bischofberger/Downloads/Neuer Ordner (18)/defensive-network-main/w_raw/meta.csv", dtype=meta_schema)
    # write_parquet(df, fpath)
    # concat_metas_and_lineups()
    # main()
