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
import time

import thefuzz
import thefuzz.process

import patsy
import wfork_streamlit_profiler
# import statsmodels.api as sm
from scipy.stats import pearsonr

import numpy as np
import statsmodels.formula.api
import statsmodels.api
import matplotlib.patheffects

import pandas as pd
import slugify
import streamlit as st

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.utility.dataframes
import thefuzz
import adjustText

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


# pip install pillow requests pycountry
import io, requests, os
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pycountry

# Subdivision flags (England, Scotland, etc.) use special codes on FlagCDN
SUBDIV_MAP = {
    "England": "gb-eng",
    "Scotland": "gb-sct",
    "Wales": "gb-wls",
    "Northern Ireland": "gb-nir",
}

nice_labels = {
    "valued_fault_per90": "Valued Fault per 90",
    "raw_contribution_per_pass": "Raw Contribution per Pass",
    "market_value": "Market value [€]",
    "def_awareness": "Defensive Awareness Rating",
    "n_interceptions_per90": "Interecptions per 90",
}

# Common naming fixes so pycountry resolves them
NAME_FIXES = {
    "USA": "United States",
    "Czech Republic": "Czechia",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo, The Democratic Republic of the",
    "UAE": "United Arab Emirates",
    "Russia": "Russian Federation",
    "Iran": "Iran, Islamic Republic of",
    "Cape Verde": "Cabo Verde",
    "Bolivia": "Bolivia, Plurinational State of",
    "Syria": "Syrian Arab Republic",
    "Moldova": "Moldova, Republic of",
    "Vatican": "Holy See",
}

def country_to_code(name: str) -> str | None:
    """Return a lowercase ISO2 or subdivision code usable on FlagCDN."""
    name = (name or "").strip()
    if not name:
        return None
    if name in SUBDIV_MAP:
        return SUBDIV_MAP[name]
    fixed = NAME_FIXES.get(name, name)
    try:
        return pycountry.countries.lookup(fixed).alpha_2.lower()
    except LookupError:
        return None

# In-memory cache so we fetch each flag only once
_FLAG_IMG_CACHE: dict[str, Image.Image] = {}

def fetch_flag_image(code: str, size_px: int = 160) -> Image.Image | None:
    """
    Get a PNG flag image from FlagCDN (e.g., 'de', 'es', 'qa', 'gb-eng').
    size_px can be 20, 40, 64, 80, 160, etc. We'll use w<size>.
    """
    if not code:
        return None
    key = f"{code}_{size_px}"
    if key in _FLAG_IMG_CACHE:
        return _FLAG_IMG_CACHE[key]

    # https://flagcdn.com/w40/ua.png
    url = f"https://flagcdn.com/w{size_px}/{code}.png"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        st.write(r.status_code, url)
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        _FLAG_IMG_CACHE[key] = img
        return img
    except Exception as e:
        st.write(e)
        return None

def flag_marker_scatter(ax, df, xcol: str, ycol: str, team_col: str = "team_name",
                        size_px: int = 64, zoom: float = 0.16, fallback_s: int = 28):
    """
    Draw flag images at (x, y) instead of point markers.
    - size_px: request size from CDN (40–80 is typical)
    - zoom: scales the image on-figure; tweak to taste
    """
    for _, r in df.iterrows():
        code = country_to_code(str(r[team_col]))
        img  = fetch_flag_image(code, size_px=size_px)
        # st.write(code, img)
        x, y = r[xcol], r[ycol]
        if img is None:
            ax.scatter([x], [y], s=fallback_s)  # fallback if no flag available
            continue
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x, y),
            frameon=False,
            pad=0.0,
            box_alignment=(0.5, 0.5),
            zorder=3
        )
        ax.add_artist(ab)



# @st.cache_resource
def _read_df(fpath, **kwargs):
    return pd.read_csv(fpath, **kwargs)


# @st.cache_resource
def get_dfb_csv_files_in_folder(folder, exclude_files=None):
    tracking_cols = [
        "frame,match_id,player_id,team_id,event_vendor,tracking_vendor,datetime_tracking,section,x_tracking,y_tracking,z_tracking,d_tracking,a_tracking,s_tracking,ball_status,ball_poss_team_id",
        "frame,match_id,player_id,team_id,event_vendor,tracking_vendor,datetime_tracking,section,x_tracking,y_tracking,z_tracking,d_tracking,a_tracking,s_tracking,ball_status,ball_poss_team_id",
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
                if first_line.strip() in header:
                    file_type = ft
                else:
                    continue
            elif isinstance(header, str):
                if first_line.strip() == header:
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
        df[subject_id] = df[subject_id].astype(str)
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
    # st.write("cool")
    # st.write(dfg["n_passes_with_contribution"] / dfg["minutes_played"] * 90)
    # st.write((dfg["n_passes_with_contribution"] / dfg["minutes_played"] * 90).median())
    # st.write((dfg["n_passes_with_contribution"] / dfg["minutes_played"] * 90).mean())
    # st.stop()

    # convert all cols to float
    dfg = dfg.astype(float, errors="ignore")

    dfg["n_tackles"] = (dfg["n_tackles_won"] + dfg["n_tackles_lost"])

    dfg["minutes_played"] = dfg["minutes_played"].astype(float)

    minutes_played_total = dfg["minutes_played"].copy()

    # check all dtypes of dfg
    for col in dfg.columns:
        # st.write(f"{col}: {dfg[col].dtype}")
        dfg[col] = dfg[col].astype(float)

    dfg = (dfg.div(dfg["minutes_played"], axis=0) * 90).rename(columns=lambda x: f"{x}_per90")
    dfg["minutes_played"] = minutes_played_total
    dfg["tackles_won_share"] = dfg["n_tackles_won_per90"].astype(float) / dfg["n_tackles_per90"].astype(float)

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
    dfg_per_pass["total_valued_contribution_responsibility_per_pass"] = dfg_per_pass["total_valued_contribution_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_valued_fault_responsibility_per_pass"] = dfg_per_pass["total_valued_fault_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_valued_responsibility_per_pass"] = dfg_per_pass["total_valued_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_valued_involvement_per_pass"] = dfg_per_pass["total_valued_involvement"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_valued_contribution_per_pass"] = dfg_per_pass["total_valued_contribution"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_valued_fault_per_pass"] = dfg_per_pass["total_valued_fault"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_raw_contribution_responsibility_per_pass"] = dfg_per_pass["total_raw_contribution_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_raw_fault_responsibility_per_pass"] = dfg_per_pass["total_raw_fault_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_raw_responsibility_per_pass"] = dfg_per_pass["total_raw_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_raw_contribution_per_pass"] = dfg_per_pass["total_raw_contribution"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_raw_fault_per_pass"] = dfg_per_pass["total_raw_fault"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_raw_involvement_per_pass"] = dfg_per_pass["total_raw_involvement"].astype(float) / dfg_per_pass["n_passes_with_involvement"].astype(float)
    dfg_per_pass["total_relative_raw_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_relative_raw_fault_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_fault_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_relative_raw_contribution_responsibility_per_pass"] = dfg_per_pass["total_relative_raw_contribution_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_relative_valued_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_relative_valued_fault_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_fault_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)
    dfg_per_pass["total_relative_valued_contribution_responsibility_per_pass"] = dfg_per_pass["total_relative_valued_contribution_responsibility"].astype(float) / dfg_per_pass["n_passes_with_responsibility"].astype(float)

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

    # responsibility - involvement
    # dfg["responsibility_minus_involvement_per_pass"] = dfg["responsibility_per_pass"].astype(float) - dfg["involvement_per_pass"].astype(float)
    dfg["valued_involvement_minus_responsibility_per_pass"] = dfg["valued_involvement_per_pass"].astype(float) - dfg["valued_responsibility_per_pass"].astype(float)
    dfg["raw_involvement_minus_responsibility_per_pass"] = dfg["raw_involvement_per_pass"].astype(float) - dfg["raw_responsibility_per_pass"].astype(float)
    dfg["raw_fault_minus_fault_responsibility_per_pass"] = dfg["raw_fault_per_pass"].astype(float) - dfg["raw_fault_responsibility_per_pass"].astype(float)
    dfg["valued_fault_minus_fault_responsibility_per_pass"] = dfg["valued_fault_per_pass"].astype(float) - dfg["valued_fault_responsibility_per_pass"].astype(float)
    dfg["raw_contribution_minus_contribution_responsibility_per_pass"] = dfg["raw_contribution_per_pass"].astype(float) - dfg["raw_contribution_responsibility_per_pass"].astype(float)
    dfg["valued_contribution_minus_contribution_responsibility_per_pass"] = dfg["valued_contribution_per_pass"].astype(float) - dfg["valued_contribution_responsibility_per_pass"].astype(float)

    # resp fault + fault
    dfg["raw_fault_plus_fault_responsibility_per_pass"] = dfg["raw_fault_per_pass"].astype(float) + dfg["raw_fault_responsibility_per_pass"].astype(float)
    dfg["valued_fault_plus_fault_responsibility_per_pass"] = dfg["valued_fault_per_pass"].astype(float) + dfg["valued_fault_responsibility_per_pass"].astype(float)

#                 "raw_contribution_per_pass",
#                 "raw_fault_per_pass",
#                 "raw_involvement_per_pass",
#                 "raw_contribution_responsibility_per_pass",
#                 "raw_fault_responsibility_per_pass",
#                 "raw_responsibility_per_pass",
#
#                 "valued_contribution_per_pass",
#                 "valued_fault_per_pass",
#                 "valued_involvement_per_pass",
#                 "valued_contribution_responsibility_per_pass",
#                 "valued_fault_responsibility_per_pass",
#                 "valued_responsibility_per_pass",
#

    assert "raw_contribution_minus_contribution_responsibility_per_pass" in dfg.columns
    # all per_pass metrics: create a per90 version
    for col in dfg.columns:
        if col.endswith("_per_pass"):
            dfg[col.replace("_per_pass", "_per90")] = dfg[col].astype(float) * dfg["n_passes_against"].astype(float) / dfg["minutes_played"].astype(float) * 90

    assert "raw_contribution_minus_contribution_responsibility_per90" in dfg.columns

    dfg_meta = df_player_matchsums.groupby(group_cols).agg(
        short_name=("short_name", "first"),
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        normalized_name=("normalized_name", "first"),
    )
    dfg = dfg.join(dfg_meta, on=group_cols)

    dfg["n_interceptions_per_pass"] = dfg["n_interceptions_per90"].astype(float) * dfg["minutes_played"].astype(float) / 90 / dfg["n_passes_against"].astype(float)
    dfg["n_passes_against_per90"] = dfg["n_passes_against"].astype(float) / dfg["minutes_played"].astype(float) * 90
    dfg["n_tackles_per_pass"] = dfg["n_tackles_per90"].astype(float) * dfg["minutes_played"].astype(float) / 90 / dfg["n_passes_against"].astype(float)
    dfg["n_tackles_won_per_pass"] = dfg["n_tackles_won_per90"].astype(float) * dfg["minutes_played"].astype(float) / 90 / dfg["n_passes_against"].astype(float)

    # hochrechnen
    dfg["raw_involvement_combined"] = dfg["raw_contribution_per_pass"] * 250 - dfg["raw_fault_per90"]
    dfg["raw_involvement_combined2"] = dfg["raw_contribution_per_pass"] * 250 + dfg["raw_fault_per90"]
    dfg["raw_responsibility_combined"] = dfg["raw_contribution_responsibility_per_pass"] * 250 - dfg["raw_fault_responsibility_per90"]
    dfg["raw_responsibility_combined2"] = dfg["raw_contribution_responsibility_per_pass"] * 250 + dfg["raw_fault_responsibility_per90"]
    dfg["raw_responsinvolvement_combined"] = dfg["raw_contribution_per_pass"] * 250 + dfg["raw_fault_responsibility_per90"]
    dfg["valued_involvement_combined"] = dfg["valued_contribution_per_pass"] * 250 - dfg["valued_fault_per90"]
    dfg["valued_involvement_combined2"] = dfg["valued_contribution_per_pass"] * 250 + dfg["valued_fault_per90"]
    dfg["valued_responsibility_combined"] = dfg["valued_contribution_responsibility_per_pass"] * 250 - dfg["valued_fault_responsibility_per90"]
    dfg["valued_responsibility_combined2"] = dfg["valued_contribution_responsibility_per_pass"] * 250 + dfg["valued_fault_responsibility_per90"]
    dfg["valued_responsinvolvement_combined"] = dfg["valued_contribution_per_pass"] * 250 + dfg["valued_fault_responsibility_per90"]

    # st.write("dfg")
    # st.write(dfg[[
    #     "raw_contribution_per_pass", "raw_fault_per90", "raw_involvement_combined", "raw_involvement_combined2",
    #     "raw_responsibility_combined", "raw_responsibility_combined2", "minutes_played"
    # ]].mean())
    # i_enough_minutes = dfg["minutes_played"] > 45
    # st.write(dfg.loc[i_enough_minutes, [
    #     "raw_contribution_per_pass", "raw_fault_per90", "raw_involvement_combined", "raw_involvement_combined2",
    #     "raw_responsibility_combined", "raw_responsibility_combined2", "minutes_played",
    # ]])

    assert "raw_contribution_minus_contribution_responsibility_per90" in dfg.columns
    return dfg.reset_index()



def main():
    # profiler = wfork_streamlit_profiler.Profiler()
    # profiler.start()

    # with st.spinner("Uploading..."):
    #     defensive_network.parse.drive.upload_file_to_drive("market_values.xlsx", "market_values.xlsx")
    # with st.spinner("Downloading..."):
    #     df_market_values = defensive_network.parse.drive.download_excel_from_drive("market_values.xlsx", st_cache=True)
    # st.write("df_market_values")
    # st.write(df_market_values)
    # st.stop()

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
    _do_concat_matchsums = st.toggle("Concat matchsums", value=True)
    _do_videos = st.toggle("Create videos", value=False)
    _do_analysis = st.toggle("Do analysis", value=False)
    _do_summary = st.toggle("Analyse summary metrics", False, key=f"summary_table")

    folder = st.text_input("Folder", "Y:/m_raw/2324/")
    # if not os.path.exists(folder):
    #     st.warning(f"Folder {folder} does not exist")
        # return
    fpath_target_meta = st.text_input("Processed meta.csv file", os.path.join(folder, "meta.csv"))
    fpath_target_lineup = st.text_input("Processed lineups.csv file", os.path.join(folder, "lineups.csv"))
    # folder_events = st.text_input("Folder for preprocessed events", os.path.join(folder, "events"))
    # folder_tracking = st.text_input("Folder for preprocessed tracking", os.path.join(folder, "tracking"))
    radius = st.selectbox("Involvement radius (meters)", [10], index=0)
    fpath_all_player_matchsums = f"all_player_matchsums/{radius}.csv"
    fpath_all_team_matchsums = f"all_team_matchsums/{radius}.csv"

    folder_tracking = "tracking/"
    folder_events = "events/"
    folder_pp_tracking = os.path.join(folder, "preprocessed", folder_tracking)
    folder_pp_events = os.path.join(folder, "preprocessed", folder_events)
    folder_drive_tracking = "tracking"
    folder_full_tracking = os.path.join(folder, "finalized", folder_tracking)
    folder_drive_events = "events"
    folder_drive_team_matchsums = f"team_matchsums/{radius}"
    folder_drive_players_matchsums = f"player_matchsums/{radius}"
    folder_drive_involvement = f"involvement/{radius}"

    # filetype_to_files = get_dfb_csv_files_in_folder(folder, [fpath_target_meta, fpath_target_lineup])
    if _process_meta or _process_lineups or _process_tracking or _process_events:
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
            df_meta = df_meta[~df_meta["slugified_match_string"].isin(event_matches) | ~df_meta["slugified_match_string"].isin(tracking_matches)]

        st.write("df_meta")
        st.write(df_meta)

        df_lineups = defensive_network.parse.drive.download_csv_from_drive("lineups.csv")
        # df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-12-st-rb-leipzig-1-fc-koln"]
        finalize_events_and_tracking_to_drive(folder_pp_tracking, folder_pp_events, df_meta, df_lineups, folder_drive_events, folder_drive_tracking, folder_full_tracking)

    if _do_reduction:
        # def upload_reduced_tracking_data(df_meta, drive_folder_events, full_tracking_folder, drive_folder_tracking):
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
        upload_reduced_tracking_data(df_meta, folder_drive_events, folder_full_tracking, folder_drive_tracking)

    if _do_involvement:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
        only_process_wc_final = st.toggle("Only process World Cup final", value=False)
        only_process_a_bundesliga_game = st.toggle("Only process a Bundesliga game", value=False)
        if only_process_wc_final:
            df_meta = df_meta[df_meta["match_string"] == "FIFA Men's World Cup 2022: 8.ST Argentina - France"]
        if only_process_a_bundesliga_game:
            df_meta = df_meta[df_meta["slugified_match_string"] == "bundesliga-2023-2024-4-st-1-fc-nurnberg-1899-hoffenheim"]
            overwrite_if_exists = True
        process_involvements(df_meta, folder_drive_tracking, folder_drive_events, folder_drive_involvement, radius, overwrite_if_exists)

    if _calculate_responsibility_model:
        df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
        match_id_2_competition_name = df_meta.set_index("match_id")["competition_name"].to_dict()

        def calc_involvement_model(folder_drive_involvement, fpath_out="responsibility_model.csv", competition_name=None):
            involvement_files = defensive_network.parse.drive.list_files_in_drive_folder(folder_drive_involvement)

            df_meta_filtered = df_meta[df_meta["competition_name"] == competition_name] if competition_name else df_meta
            involvement_files = [file for file in involvement_files if file["name"].split(".")[0] in df_meta_filtered["slugified_match_string"].values]

            # @st.cache_resource
            def _get_involvement(involvement_files):
                dfs = []
                for file in defensive_network.utility.general.progress_bar(involvement_files, total=len(involvement_files), desc="Involvement concat"):
                    df = defensive_network.parse.drive.download_parquet_from_drive(os.path.join(folder_drive_involvement, file["name"]))
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
            calc_involvement_model(folder_drive_involvement, f"responsibility_model/{competition_name}_{radius}.csv", competition_name)

        if st.toggle("Calculate responsibility model for all competitions", value=False):
            calc_involvement_model(folder_drive_involvement, f"responsibility_model/all_{radius}.csv", competition_name=None)

    if _do_create_matchsums:
        try:
            df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
            df_lineups = defensive_network.parse.drive.download_csv_from_drive("lineups.csv", st_cache=True)
            # df_meta = df_meta[df_meta["match_id"] == "2d4fe74894566dc4b826bd608deaa53c"]
            create_matchsums(folder_full_tracking, folder_drive_events, df_meta, df_lineups, folder_drive_team_matchsums, folder_drive_players_matchsums, folder_drive_involvement, overwrite_if_exists=overwrite_if_exists)
        except Exception as e:
            st.write(e)
            gc.collect()
            st.rerun()

    if _do_concat_matchsums:
        # def _read_all_parquets(drive_folder):
        #     player_matchsum_files = defensive_network.parse.drive.list_files_in_drive_folder(drive_folder, st_cache=True)
        #     dfs = []
        #     for file_nr, file in enumerate(defensive_network.utility.general.progress_bar(player_matchsum_files)):
        #         df = defensive_network.parse.drive.download_parquet_from_drive(os.path.join(drive_folder, file["name"]), st_cache=True)
        #         dfs.append(df)
        #     df = pd.concat(dfs)
        #     return df

        # df_players = _read_all_parquets(folder_drive_players_matchsums)
        df_players = defensive_network.parse.drive.download_all_parquets_from_drive(folder_drive_players_matchsums)
        st.write("df_players")
        st.write(df_players)
        defensive_network.parse.drive.upload_csv_to_drive(df_players, fpath_all_player_matchsums)

        # df_teams = _read_all_parquets(folder_drive_team_matchsums)
        df_teams = defensive_network.parse.drive.download_all_parquets_from_drive(folder_drive_team_matchsums)
        st.write("df_teams")
        st.write(df_teams)
        defensive_network.parse.drive.upload_csv_to_drive(df_teams, fpath_all_team_matchsums)

    if _do_videos:
        create_videos()

    st.write("Before do analysis")
    if _do_analysis:
        expected_matches = {
            "FIFA Men's World Cup": 64,
            "3. Liga": 380,
            "Bundesliga": 132,
        }

        summary_data = []

        _do_best_players = st.toggle("Best players", True, key=f"best_players")
        _do_descriptives = st.toggle("Matchsum Descriptives", True, key=f"matchsum_descriptives")
        _do_seasonal_correlation = st.toggle("Season-by-season correlation", True, key=f"season_by_season_correlation")
        _do_icc = st.toggle("ICC", True, key=f"icc")
        _do_bootstrapped_icc = st.toggle("Bootstrapped Season-Level ICC (TAKES VERY LONG!!!)", True, key=f"bootstrapped_season_level_icc")
        _do_histograms = st.toggle("Histograms", False, key=f"histograms")
        _do_internal_correlations = st.toggle("KPI correlations as heatmap", True, f"kpi_heatmap")
        _do_fifa_correlations = st.toggle("FIFA correlations", True, key=f"fifa_correlations")

        with st.spinner("Loading matchsums..."):
            df_player_matchsums = defensive_network.parse.drive.download_csv_from_drive(fpath_all_player_matchsums, st_cache=True)
            df_player_matchsums["player_id"] = df_player_matchsums["player_id"].astype(str)

            # df_player_matchsums = pd.read_csv("C:/Users/j.bischofberger/OneDrive - VfB Stuttgart 1893 AG/Desktop/code/defensive-network/df_matchsums_player.csv")  # TODO fix

        df_player_matchsums = df_player_matchsums[df_player_matchsums["role_category"] != "GK"]
        # df_player_matchsums["is_rückrunde"] = df_player_matchsums["kickoff_time"].apply(lambda x: pd.to_datetime(x, errors="coerce").year == 2024)
        # df_player_matchsums["is_rückrunde"] = pd.to_datetime(df_player_matchsums["kickoff_time"], errors="coerce").dt.tz_localize(None).year == 2024
        df_player_matchsums["is_rückrunde"] = pd.to_datetime(df_player_matchsums["kickoff_time"].astype(str), errors="coerce").dt.tz_localize(None).dt.year == 2024

        # for each column: write range of matchsum
        # for col in df_player_matchsums.columns:
        #     st.write(col, df_player_matchsums[col].describe())

        competitions = df_player_matchsums["competition_name"].unique()
        # selected_competitions = st.multiselect("Competitions", competitions, ["FIFA Men's World Cup"])
        selected_competitions = st.multiselect("Competitions", competitions, competitions)

        for competition in selected_competitions:
            st.write(f"## {competition}")
            with st.spinner(f"Loading matchsums from... {fpath_all_player_matchsums}"):
                df_player_matchsums = defensive_network.parse.drive.download_csv_from_drive(fpath_all_player_matchsums, st_cache=True)
                df_player_matchsums["player_id"] = df_player_matchsums["player_id"].astype(str)
                df_player_matchsums["match_id"] = df_player_matchsums["match_id"].astype(str)
                # df_player_matchsums = pd.read_csv("C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/df_matchsums_player.csv")  # TODO fix
                # df_player_matchsums = pd.read_csv("C:/Users/j.bischofberger/OneDrive - VfB Stuttgart 1893 AG/Desktop/code/defensive-network/df_matchsums_player.csv")  # TODO fix
                st.write(f"Downloaded df_player_matchsums with shape {df_player_matchsums.shape} from {fpath_all_player_matchsums}")

            df_player_matchsums = df_player_matchsums[df_player_matchsums["role_category"] != "GK"]
            df_player_matchsums["is_rückrunde"] = pd.to_datetime(df_player_matchsums["kickoff_time"].astype(str), errors="coerce").dt.tz_localize(None).dt.year == 2024

            with st.spinner("Getting meta..."):
                df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
            match_id_2_string = df_meta.set_index("match_id")["match_string"].to_dict()
            match_id_2_slugified_string = df_meta.set_index("match_id")["slugified_match_string"].to_dict()
            match_id_2_competition_name = df_meta.set_index("match_id")["competition_name"].to_dict()
            df_meta["expected_matches"] = df_meta["competition_name"].map(expected_matches)

            df_player_matchsums["match_string"] = df_player_matchsums["match_id"].map(match_id_2_string)
            df_player_matchsums["slugified_match_string"] = df_player_matchsums["match_id"].map(match_id_2_slugified_string)
            df_player_matchsums["competition_name"] = df_player_matchsums["match_id"].map(match_id_2_competition_name)
            df_player_matchsums["expected_matches"] = df_player_matchsums["competition_name"].map(expected_matches)
            df_player_matchsums["match_day"] = df_player_matchsums["match_day"].astype(str)

            assert df_player_matchsums["competition_name"].notna().all()
            assert df_player_matchsums["match_string"].notna().all()
            assert df_player_matchsums["slugified_match_string"].notna().all()
            assert df_player_matchsums["expected_matches"].notna().all()

            # Print some statistics
            st.write("Available matches in MATCHSUMS")
            st.write(df_player_matchsums.groupby(["competition_name"]).agg(
                n_matches=("match_id", "nunique"),
                n_match_strings=("match_string", "nunique"),
                n_slugs=("slugified_match_string", "nunique"),
                n_expected_matches=("expected_matches", "first"),
            ))
            st.write(df_player_matchsums.groupby(["competition_name", "match_day"]).agg(
                n_matches=("match_id", "nunique"),
                n_match_strings=("match_string", "nunique"),
                n_slugs=("slugified_match_string", "nunique"),
                n_expected_matches=("expected_matches", "first"),
            ))
            st.write("Available matches in META")
            st.write(df_meta.groupby(["competition_name"]).agg(
                n_matches=("match_id", "nunique"),
                n_match_strings=("match_string", "nunique"),
                n_slugs=("slugified_match_string", "nunique"),
                n_expected_matches=("expected_matches", "first"),
            ))
            st.write(df_meta.groupby(["competition_name", "match_day"]).agg(
                n_matches=("match_id", "nunique"),
                n_match_strings=("match_string", "nunique"),
                n_slugs=("slugified_match_string", "nunique"),
                n_expected_matches=("expected_matches", "first"),
            ))
            event_slugs = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder("events", st_cache=True)]
            # st.write("event_slugs")
            # st.write(event_slugs)
            df_meta["has_event"] = df_meta["slugified_match_string"].isin(event_slugs)
            tracking_slugs = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder("tracking", st_cache=True)]
            # st.write("tracking_slugs")
            # st.write(tracking_slugs)
            df_meta["has_tracking"] = df_meta["slugified_match_string"].isin(tracking_slugs)
            involvement_slugs = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(f"involvement/{radius}", st_cache=True)]
            df_meta["has_involvement"] = df_meta["slugified_match_string"].isin(involvement_slugs)
            matchsum_slugs = df_player_matchsums["slugified_match_string"].tolist()
            df_meta["has_matchsums"] = df_meta["slugified_match_string"].isin(matchsum_slugs)
            st.write("df_meta with availability")
            st.write(df_meta[["match_string"] + [col for col in df_meta.columns if col.startswith("has_")]])

            for comp, df_meta_comp in df_meta.groupby("competition_name"):
                # st.write(df_meta_comp.shape)
                # st.write(df_meta_comp[["match_id", "slugified_match_string", "has_event", "has_tracking", "has_involvement", "has_matchsums"]])
                # st.write(df_meta_comp[["match_id", "slugified_match_string", "has_event", "has_tracking", "has_involvement", "has_matchsums"]].describe())
                # st.write(df_meta_comp[["has_event", "has_tracking", "has_involvement", "has_matchsums"]].mean() * 100)
                cols = ["has_event", "has_tracking", "has_involvement", "has_matchsums"]
                summary = df_meta_comp[cols].agg(["sum", "count"]).T
                summary["Presence (%)"] = (summary["sum"] / summary["count"] * 100).round(2).astype(str) + "%"
                st.write(summary.rename(columns={"sum": "Present", "count": "Total"}))


            assert "match_string" in df_player_matchsums.columns, "match_string column missing in player matchsums"
            # st.write(df_player_matchsums[["match_id", "match_string", "slugified_match_string"]].drop_duplicates())
            # st.write(len(df_player_matchsums[["match_id", "match_string", "slugified_match_string"]].drop_duplicates()))

            st.write(f"#### Analysis for {competition} ###")
            DEFAULT_MINUTES = st.number_input("Default minimal minutes played total", min_value=0, value=300 if competition == "3. Liga" else 150, key=f"{competition}_default_minutes")
            DEFAULT_MINUTES_PER_MATCH = st.number_input("Default minimal minutes per match", min_value=0, value=30, key=f"{competition}_default_minutes_per_match")

            df_player_matchsums = df_player_matchsums[df_player_matchsums["competition_name"].isin([competition])]

            if len(df_player_matchsums) == 0:
                st.warning(f"No matchsums found for competition {competition}, skipping analysis.")
                continue

            position_mapping = {'CB-3': 'CB', 'LB': 'FB', 'LCB-3': 'CB', 'LCB-4': 'CB', 'LDM': 'CM', 'LS': 'CF',
                                'LW': 'Winger', 'LZM': 'CM', 'RB': 'FB', 'RCB-3': 'CB', 'RCB-4': 'CB', 'RDM': 'CM',
                                'RS': 'CF', 'RW': 'Winger', 'RWB': 'FB', 'RZM': 'CM', 'ST': 'CF', 'ZDM': 'CM',
                                'ZOM': 'CAM', 'LWB': 'FB'
                                }
            df_player_matchsums["coarse_position"] = df_player_matchsums["role_category"].map(position_mapping)
            # st.write(df_player_matchsums[["role_category", "coarse_position"]].drop_duplicates())
            assert df_player_matchsums["coarse_position"].notna().all(), "Some positions are not mapped to coarse positions"
            df_player_matchsums["minutes_played"] = df_player_matchsums["minutes_played"].astype(float)
            df_player_matchsums["player_pos_minutes_played"] = df_player_matchsums.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")

            df_agg_match_player_pos = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position", "match_id"])
            # df_agg_match_player_pos["player_minutes_played"] = df_agg_match_player_pos.groupby("player_id")["minutes_played"].transform("sum")
            df_agg_match_player_pos["player_pos_minutes_played"] = df_agg_match_player_pos.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")
            # df_agg_match_player_pos["player_matches_played"] = df_agg_match_player_pos.groupby("player_id")["match_id"].transform("nunique")
            # st.write("df_agg_match_player_pos")
            # st.write(df_agg_match_player_pos[["player_id", "coarse_position", "match_id"] + [col for col in df_agg_match_player_pos.columns if "minutes" in col]])
            df_agg_match_player_pos = df_agg_match_player_pos.merge(df_meta[["match_id", "match_string"]].drop_duplicates(), on="match_id", how="left")

            df_agg_player_pos = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position"])
            df_agg_player_pos_team = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "team_name", "coarse_position"])
            assert df_agg_player_pos_team[["player_id", "team_name", "coarse_position"]].drop_duplicates().shape[0] == df_agg_player_pos_team.shape[0], "Duplicates in df_agg_player_pos_team"
            # st.stop()

            assert "raw_contribution_minus_contribution_responsibility_per90" in df_agg_player_pos.columns
            df_agg_player_pos["player_minutes_played"] = df_agg_player_pos.groupby("player_id")["minutes_played"].transform("sum")
            # df_agg_match_player_pos["player_matches_played"] = df_agg_match_player_pos.groupby("player_id")["match_id"].transform("nunique")

            df_agg_player_pos.to_excel("test.xlsx")

            df_agg_team = aggregate_matchsums(df_player_matchsums, group_cols=["team_id", "coarse_position", "player_id"]).reset_index()
            # df_agg_team["full_name"] = df_agg_team["first_name"] + " " + df_agg_team["last_name"]

            df_agg_by_season_half = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "coarse_position", "is_rückrunde"])
            # df_agg_match_player = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "match_id"])
            # df_agg_match_player["player_minutes_played"] = df_agg_match_player.groupby("player_id")["minutes_played"].transform("sum")
            # df_agg_match_player["player_matches_played"] = df_agg_match_player.groupby("player_id")["match_id"].transform("nunique")
            st.write("e")

            default_kpis = [
                # Classic metrics
                "n_interceptions_per90",
                "n_interceptions_per_pass",
                "n_passes_against_per90",
                "n_tackles_won_per90",
                "n_tackles_per90",
                "tackles_won_share",
                "n_passes_per90",

                "raw_responsibility_combined",
                "raw_responsibility_combined2",
                "raw_involvement_combined",
                "raw_involvement_combined2",
                "valued_involvement_combined",
                "valued_involvement_combined2",
                "valued_responsibility_combined",
                "valued_responsibility_combined2",
                "raw_responsinvolvement_combined",
                "valued_responsinvolvement_combined",
                #     dfg["raw_involvement_combined"] = dfg["raw_contribution_per_pass"] * 250 - dfg["raw_fault_per90"]
                #     dfg["raw_involvement_combined2"] = dfg["raw_contribution_per_pass"] * 250 + dfg["raw_fault_per90"]
                #     dfg["raw_responsibility_combined"] = dfg["raw_contribution_responsibility_per_pass"] * 250 - dfg["raw_fault_responsibility_per90"]
                #     dfg["raw_responsibility_combined2"] = dfg["raw_contribution_responsibility_per_pass"] * 250 + dfg["raw_fault_responsibility_per90"]

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
                # "total_raw_contribution_responsibility_per90",
                # "total_raw_fault_responsibility_per90",
                # "total_raw_responsibility_per90",

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

#     dfg["valued_responsibility_minus_involvement_per_pass"] = dfg["valued_responsibility_per_pass"].astype(float) - dfg["valued_involvement_per_pass"].astype(float)
                #     dfg["raw_responsibility_minus_involvement_per_pass"] = dfg["raw_responsibility_per_pass"].astype(float) - dfg["raw_involvement_per_pass"].astype(float)
                #     dfg["raw_fault_responsibility_minus_fault_per_pass"] = dfg["raw_fault_responsibility_per_pass"].astype(float) - dfg["raw_fault_per_pass"].astype(float)
                #     dfg["valued_fault_responsibility_minus_fault_per_pass"] = dfg["valued_fault_responsibility_per_pass"].astype(float) - dfg["valued_fault_per_pass"].astype(float)
                #     dfg["raw_contribution_responsibility_minus_contribution_per_pass"] = dfg["raw_contribution_responsibility_per_pass"].astype(float) - dfg["raw_contribution_per_pass"].astype(float)
                #     dfg["valued_contribution_responsibility_minus_contribution_per_pass"] = dfg["valued_contribution_responsibility_per_pass"].astype(float) - dfg["valued_contribution_per_pass"].astype(float)
                "valued_involvement_minus_responsibility_per_pass",
                "raw_involvement_minus_responsibility_per_pass",
                "raw_fault_minus_fault_responsibility_per_pass",
                "valued_fault_minus_fault_responsibility_per_pass",
                "raw_contribution_minus_contribution_responsibility_per_pass",
                "valued_contribution_minus_contribution_responsibility_per_pass",

                "raw_fault_plus_fault_responsibility_per_pass",
                "valued_fault_plus_fault_responsibility_per_pass",

                "n_tackles_lost_per90",
            ]
            default_kpis += default_kpis + [kpi.replace("_per_pass", "_per90") for kpi in default_kpis if "_per_pass" in kpi]
            assert "valued_contribution_minus_contribution_responsibility_per90" in default_kpis
            kpis = st.multiselect("KPIs", options=sorted(df_agg_player_pos.columns), default=sorted([kpi for kpi in default_kpis if kpi in df_agg_player_pos.columns or "per_pass" in kpi or "per90" in kpi]), key=f"{competition}_kpis")
            for kpi in kpis:
                if kpi not in df_agg_match_player_pos.columns:
                    st.warning(f"{kpi} not in df_agg_match_player_pos")
            kpis = [kpi for kpi in kpis if kpi in df_agg_match_player_pos.columns]
            kpis = sorted(list(set((kpis))))

            assert "raw_contribution_minus_contribution_responsibility_per90" in default_kpis
            assert "raw_contribution_minus_contribution_responsibility_per90" in kpis

            if _do_best_players:
                for kpi1, kpi2 in [
                    ("valued_fault_per90", "raw_contribution_per_pass"),
                    ("valued_responsibility_combined", "raw_involvement_combined"),
                    ("valued_fault_per90", "valued_fault_responsibility_per90"),
                ]:
                    with st.expander(f"Scatter {kpi1} vs. {kpi2}", False):
                        min_minutes = st.number_input(f"Minimum minutes played by player for scatter {kpi1} vs. {kpi2}.", min_value=0, value=DEFAULT_MINUTES, key=f"{competition}_min_minutes_scatter_{kpi1}_{kpi2}")

                        assert df_agg_player_pos_team[["player_id", "team_name", "coarse_position"]].drop_duplicates().shape[0] == df_agg_player_pos_team.shape[0], "Duplicates in df_agg_player_pos_team"

                        for pos in df_agg_player_pos_team["coarse_position"].unique():
                            df_agg_player_pos_scatter = df_agg_player_pos_team[
                                (df_agg_player_pos_team["minutes_played"] > min_minutes)
                                & (df_agg_player_pos_team["coarse_position"] == pos)
                            ]

                            plt.figure(figsize=(10, 10))

                            flip_x = True if ("fault" in kpi1 or "minus" in kpi1 or "lost" in kpi1) else False
                            flip_y = True if ("fault" in kpi2 or "minus" in kpi2 or "lost" in kpi2) else False

                            plt.xlabel(nice_labels.get(kpi1, kpi1))
                            plt.ylabel(nice_labels.get(kpi2, kpi2))

                            if flip_x:
                                plt.gca().invert_xaxis()
                            if flip_y:
                                plt.gca().invert_yaxis()

                            # grid
                            # plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
                            plt.gca().axhline(df_agg_player_pos_scatter[kpi2].mean(), color='grey', lw=1)
                            plt.gca().axvline(df_agg_player_pos_scatter[kpi1].mean(), color='grey', lw=1)

                            plt.scatter(df_agg_player_pos_scatter[kpi1], df_agg_player_pos_scatter[kpi2])
                            flag_marker_scatter(
                                plt.gca(),
                                df_agg_player_pos_scatter,
                                xcol=kpi1,
                                ycol=kpi2,
                                team_col="team_name",  # contains "Spain", "Qatar", "Germany", ...
                                size_px=40,  # tries https://flagcdn.com/w64/<code>.png
                                zoom=0.6,  # tweak for on-plot size
                            )

                            DY = 0.011

                            df_agg_player_pos_scatter["name"] = (df_agg_player_pos_scatter["first_name"].astype(str) + ". " + df_agg_player_pos_scatter["last_name"]).str.strip()
                            df_agg_player_pos_scatter["name"] = (df_agg_player_pos_scatter["name"].apply(lambda x: x if len(x.split(" ")) == 1 else x.split(" ")[0][0] + ". " + " ".join(x.split(" ")[1:]))).str.strip(".")
                            df_agg_player_pos_scatter["name"] = df_agg_player_pos_scatter["name"].apply(lambda x: {"M. Jae Kim": "Min-jae Kim", "Y. Kim": "Young-gwon Kim"}.get(x, x))
                            texts = []
                            top_names = ["C. Montes", "K. Miller", "N. Aké", "R. Varane", "I. Konaté", "P. Hincapie", "J. Stones", "M. Yoshida",
                                         "T. Ream", "R. Saïss", "S. Vitoria", "C. Romero"]
                            right_shift = {
                                "J. Timber": 0.075, "J. Stones": 0.05, "L. Martinez": 0.04, "J. Rodon": 0.09, "P. Hincapie": 0.04,
                                "R. Saïss": 0.07, "T. Ream": -0.025, "H. Souttar": 0, "M. Akanji": 0.0, "K. Koulibaly": 0.06,
                                "M. Talbi": -0.02, "Y. Meriah": 0.0, "T. Alderweired": -0.075, "C. Romero": -0.0125, "B. Koukhi": 0.005
                            }
                            bottom_shift = {
                                "M. Akanji": 0.0025,
                                "T. Ream": 0.0025
                            }
                            df_agg_player_pos_scatter["annotate_bottom"] = True
                            df_agg_player_pos_scatter.loc[df_agg_player_pos_scatter["name"].isin(top_names), "annotate_bottom"] = False
                            df_agg_player_pos_scatter["right_shift"] = 0
                            df_agg_player_pos_scatter.loc[df_agg_player_pos_scatter["name"].isin(right_shift.keys()), "right_shift"] = df_agg_player_pos_scatter["name"].map(right_shift)
                            df_agg_player_pos_scatter["bottom_shift"] = 0
                            df_agg_player_pos_scatter.loc[df_agg_player_pos_scatter["name"].isin(bottom_shift.keys()), "bottom_shift"] = df_agg_player_pos_scatter["name"].map(bottom_shift)

                            for i, row in df_agg_player_pos_scatter.iterrows():
                                dy = -DY if row["annotate_bottom"] else DY
                                dx = row["right_shift"]
                                dy -= row["bottom_shift"]

                                # if row[kpi1] > df_agg_player_pos_scatter[kpi1].quantile(0.65) or row[kpi2] > df_agg_player_pos_scatter[kpi2].quantile(0.65) or row[kpi1] < df_agg_player_pos_scatter[kpi1].quantile(0.35) or row[kpi2] < df_agg_player_pos_scatter[kpi2].quantile(0.35) \
                                #         or row[kpi1] > df_agg_player_pos_scatter[kpi1].quantile(0.75) and row[kpi2] > df_agg_player_pos_scatter[kpi2].quantile(0.75) \
                                #         or row[kpi1] < df_agg_player_pos_scatter[kpi1].quantile(0.4) and row[kpi2] < df_agg_player_pos_scatter[kpi2].quantile(0.4):
                                if True:
                                    text = plt.text(row[kpi1]-dx, row[kpi2]+dy, row["name"], va="center", ha="center")
                                    texts.append(text)
                            # st.write(plt.gcf())
                            # adjustText.adjust_text(texts)
                            st.write("df_agg_player_pos_scatter " + pos)
                            # st.write(df_agg_player_pos_scatter)
                            st.write(plt.gcf())

                for kpi in kpis:
                    with st.expander(f"Best players by {kpi}", False):
                        playerid2mainpos = df_agg_player_pos_team.groupby("player_id").apply(lambda x: x.loc[x["minutes_played"].idxmax()]["coarse_position"]).to_dict()
                        df_agg_player_pos_team["player_main_position"] = df_agg_player_pos_team["player_id"].map(playerid2mainpos)
                        for pos in df_agg_player_pos_team["coarse_position"].unique():
                            st.write(f"### Best players in {pos} ###")
                            min_minutes = 150 #st.number_input(f"Minimum minutes played by player for best player listing in {pos}.", min_value=0, value=90, key=f"{competition}_{pos}_min_minutes_best_players")
                            df_agg_player_pos_pos2 = df_agg_player_pos_team[
                                (df_agg_player_pos_team["player_main_position"] == pos)
                                & (df_agg_player_pos_team["coarse_position"] == pos)
                            ]
                            df_agg_player_pos_pos2 = df_agg_player_pos_pos2[df_agg_player_pos_pos2["minutes_played"] > min_minutes]
                            st.write(f"#### Best players in {pos} by {kpi} ####")
                            ascending = True if ("fault" in kpi or "minus" in kpi or "lost" in kpi) else False
                            df_sorted = df_agg_player_pos_pos2.sort_values(by=kpi, ascending=ascending)
                            df_sorted["name"] = (df_sorted["first_name"] + " " + df_sorted["last_name"]).str.strip()
                            st.write(df_sorted[["name", "team_name", kpi, "coarse_position", "minutes_played"]].head(10))
                            st.write("...")
                            st.write(df_sorted[["name", "team_name", kpi, "coarse_position", "minutes_played"]].tail(10))
                # st.stop()

            ######### duplicate from remote??
            if _do_fifa_correlations:
                with st.expander("FIFA", True):
                    df_fifa_players = defensive_network.parse.drive.download_csv_from_drive("fifa_ratings.csv", st_cache=True)
                    # df_fifa_players = pd.read_csv("C:/Users/j.bischofberger/Downloads/fifa_ratings.csv")
                    # files = defensive_network.parse.drive.list_files_in_drive_folder(".", st_cache=True)
                    # st.write("files")
                    # st.write(files)
                    # df_soccerdonna = defensive_network.parse.drive.download_csv_from_drive("market_values.csv", st_cache=True)
                    df_soccerdonna = defensive_network.parse.drive.download_excel_from_drive("market_values.xlsx", st_cache=True)
                    # df_soccerdonna = pd.read_excel("C:/Users/j.bischofberger/Downloads/Neuer Ordner (7)/market_values.xlsx")  # TODO WHY
                    # df_soccerdonna = pd.read_excel("C:/Users/Jonas/Downloads/market_values.xlsx")  # TODO WHY
                    # df_soccerdonna = pd.read_excel("C:/Users/j.bischofberger/Downloads/Neuer Ordner (7)/market_values.xlsx")  # TODO WHY
                    # df_player_matchsums = defensive_network.parse.drive.download_csv_from_drive("players_matchsums.csv")
                    # df_agg_player = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "position"])
                    # df_agg_player["coarse_position"] = df_agg_player["position"].map(position_mapping)
                    df_agg_player = df_agg_player_pos.copy()
                    # st.write("df_agg_player 1")
                    # st.write(df_agg_player)
                    # assert player pos is unique
                    assert df_agg_player[["player_id", "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    df_agg_player["full_name"] = (df_agg_player["first_name"] + " " + df_agg_player["last_name"]).str.strip()
                    min_minutes = st.number_input("Minimum minutes played by player for FIFA correlations.", min_value=0, value=DEFAULT_MINUTES, key=f"{competition}_min_minutes_fifa")
                    st.write("df_agg_player")
                    st.write(df_agg_player)
                    df_agg_player = df_agg_player[df_agg_player["minutes_played"] > min_minutes]
                    st.write("df_agg_player")
                    st.write(df_agg_player)
                    assert df_agg_player[["player_id", "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    assert len(df_agg_player) > 0

                    # df_agg_team = aggregate_matchsums(df_player_matchsums, group_cols=["team_id", "position", "player_id"]).reset_index()
                    # df_agg_team["full_name"] = df_agg_team["first_name"] + " " + df_agg_team["last_name"]
                    # df_agg_team["coarse_position"] = df_agg_team["position"].map(position_mapping)
                    df_agg_team = df_agg_team[df_agg_team["minutes_played"] > min_minutes]
                    df_agg_team["full_name"] = (df_agg_team["first_name"] + " " + df_agg_team["last_name"]).str.strip()

                    # Map FIFA names
                    df_fifa_players["name"] = df_fifa_players["name"].str.strip()
                    df_soccerdonna["name"] = df_soccerdonna["name"].str.strip()
                    fifa_names = df_fifa_players["name"].tolist()
                    soccerdonna_names = df_soccerdonna["name"].tolist()

                    manual_fifa_mapping = {
                        "Sammy Jerabek": "Samantha Jerabek",
                        "0 Letícia Santos": "Letícia Santos de Oliveira",
                        "Allie Hess": "Alexandria Loy Hess",
                        "João Cancelo": "João Pedro Cavaco Cancelo",
                        "Steven Vitoria": "Steven de Sousa Vitória",
                        "Pedri": "Pedro González López",
                        "Dani Olmo": "Daniel Olmo Carvajal",
                        "Rodri": "Rodrigo Hernández Cascante",
                        "Gavi": "Pablo Martín Páez Gavira",
                        "Ruben Dias": "Rúben Santos Gato Alves Dias",
                        "Thiago Silva": "Thiago Emiliano da Silva",
                        "Jose Gimenez": "José María Giménez",
                        "Casemiro": "Carlos Henrique Venancio Casimiro",
                        "Pepe": "Képler Laveran Lima Ferreira",
                        "Raphinha": "Raphael Dias Belloli",
                        "Marquinhos": "Marcos Aoás Corrêa",
                        # 0 Letícia Santos	Letícia Santos de Oliveira
                        # Allie Hess	Alexandria Loy Hess
                    }
                    fifa_mapping = manual_fifa_mapping.copy()
                    soccerdonna_mapping = dict()
                    assert df_agg_player[["player_id", "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"

                    # st.write("FIFA names")
                    # st.write(df_fifa_players)

                    for idx, row in df_agg_player.iterrows():
                        full_name = row['full_name']
                        if full_name in manual_fifa_mapping:
                            assert manual_fifa_mapping[
                                       full_name] in fifa_names, f"Manual mapping for {full_name} not in FIFA names"
                            best_match, score = manual_fifa_mapping[full_name], 200
                        else:
                            best_match, score = thefuzz.process.extractOne(full_name, fifa_names)

                        if score > 90:
                            df_agg_player.at[idx, 'fifa_match'] = best_match
                            df_agg_player.at[idx, 'match_score'] = score
                            fifa_mapping[full_name] = best_match
                        else:
                            df_agg_player.at[idx, 'fifa_match'] = None  # best_match
                            df_agg_player.at[idx, 'match_score'] = None  # score
                            fifa_mapping[full_name] = None

                        # match Soccerdonna name
                        best_match_soccerdonna, score_soccerdonna = thefuzz.process.extractOne(full_name, soccerdonna_names)

                        if score_soccerdonna > 90:
                            df_agg_player.at[idx, 'soccerdonna_name'] = best_match_soccerdonna
                            df_agg_player.at[idx, 'soccerdonna_score'] = score_soccerdonna
                            soccerdonna_mapping[full_name] = best_match_soccerdonna
                        else:
                            soccerdonna_mapping[full_name] = None

                    assert df_agg_player[["player_id",
                                          "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    df_agg_team["fifa_name"] = df_agg_team["full_name"].map(fifa_mapping)
                    df_agg_team = df_agg_team.merge(df_fifa_players, left_on="fifa_name", right_on="name", how="left")
                    df_agg_team["soccerdonna_name"] = df_agg_team["full_name"].map(soccerdonna_mapping)
                    df_agg_team = df_agg_team.merge(df_soccerdonna, left_on="soccerdonna_name", right_on="name",
                                                    how="left")
                    # st.write("df_agg_team")
                    # st.write(df_agg_team)
                    # st.write("df_agg_player before")
                    # st.write(df_agg_player)

                    assert df_agg_player[["player_id",
                                          "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    df_fifa_players = df_fifa_players.drop_duplicates(subset=["name"])
                    # st.write("df_fifa_players")
                    # st.write(df_fifa_players)

                    df_agg_player = df_agg_player.merge(df_fifa_players, left_on="fifa_match", right_on="name",
                                                        how="left")
                    assert df_agg_player[["player_id",
                                          "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    try:
                        df_agg_player = df_agg_player.merge(df_soccerdonna, left_on="soccerdonna_name", right_on="name",
                                                            how="left")
                    except KeyError as e:
                        df_agg_player["market_value"] = np.nan
                        st.write(e)
                    assert df_agg_player[["player_id", "coarse_position"]].duplicated().sum() == 0, "Player position is not unique"
                    # df_agg_player = df_agg_player.dropna(subset=["defending"])
                    st.write("df_agg_player")
                    st.write(df_agg_player)
                    # df_agg_player = df_agg_player[df_agg_player["minutes_played"] > 600]
                    # st.write("df_agg_team")
                    # st.write(df_agg_team)

                    # rank players by position group after market value
                    for posgroup, df_posgroup in df_agg_player.groupby("coarse_position"):
                        df_posgroup = df_posgroup.sort_values(by="market_value", ascending=False)
                        df_posgroup["market_value_rank"] = range(1, len(df_posgroup) + 1)
                        df_agg_player.loc[df_posgroup.index, "market_value_rank"] = df_posgroup["market_value_rank"]
                        st.write(f"Top market values in {posgroup}")
                        st.write(df_posgroup[["full_name", "market_value", "market_value_rank", "defending", "def_awareness"]])

                    data = []
                    x_variables = kpis
                    assert "valued_fault_plus_fault_responsibility_per90" in x_variables
                    assert "n_tackles_lost_per90" in x_variables
                    assert "n_interceptions_per_pass" in x_variables
                    assert "n_passes_against_per90" in x_variables
                    y_variables = ["def_awareness", "defending", "market_value"]
                    # xy_variables = [(x, y) for x in x_variables for y in y_variables]
                    for x_variable in x_variables:
                        df_agg_player[x_variable] = pd.to_numeric(df_agg_player[x_variable], errors="coerce")
                        columns = st.columns(len(y_variables))
                        for y_nr, y_variable in enumerate(y_variables):
                            df_agg_player[y_variable] = pd.to_numeric(df_agg_player[y_variable], errors="coerce")
                            col = columns[y_nr]
                            df = df_agg_player.copy().dropna(subset=[x_variable, y_variable, "coarse_position"])

                            try:
                                # sns.regplot(data=df_agg_player, x=x_variable, y=y_variable, hue="position", scatter=True, label='Trendline')
                                plot = False
                                if plot:
                                    fig = plt.figure()
                                    sns.lmplot(data=df.rename(columns={"coarse_position": "Position group"}), x=x_variable, y=y_variable, hue="Position group", scatter=True)
                                    # set lower ylim to 0
                                    plt.ylim(bottom=0)
                                    # set upper limit to max plus 10% of max
                                    max_y = df[y_variable].max()
                                    plt.ylim(top=1.1 * max_y)

                                    # xlabel -> nice name
                                    plt.xlabel(nice_labels.get(x_variable, x_variable))
                                    plt.ylabel(nice_labels.get(y_variable, y_variable))

                                    col.write(plt.gcf())

                                    # st.stop()

                                    # make also a logarithmic plot
                                    # fig_log = plt.figure()
                                    # sns.lmplot(data=df, x=x_variable, y=y_variable, hue="coarse_position", scatter=True,
                                    #            logx=True)
                                    # plt.ylim(bottom=0)
                                    # if not df[y_variable].isna().all():
                                    #     # max_y = df[y_variable].max()
                                    #     plt.ylim(top=1.1 * max_y)
                                    # col.write(plt.gcf())

                                    # plt.close()
                                else:
                                    col.write("-")

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

                            def bootstrapped_correlation_coefficient(x, y, n_bootstraps=1000):
                                x = np.asarray(x)
                                y = np.asarray(y)

                                if x.shape != y.shape:
                                    raise ValueError("x and y must have the same shape.")

                                n = len(x)
                                bootstrapped_corrs = np.empty(n_bootstraps)

                                i = 0
                                while True:
                                    indices = np.random.randint(0, n, n)
                                    x_sample = x[indices]
                                    y_sample = y[indices]
                                    corr = np.corrcoef(x_sample, y_sample)[0, 1]
                                    bootstrapped_corrs[i] = corr
                                    i += 1
                                    if np.isnan(corr):
                                        st.write(x_sample)
                                        st.write(y_sample)
                                        st.write(corr)
                                        st.write("NaN correlation, skipping")
                                        raise ValueError("NaN correlation, skipping")
                                        continue
                                    if i >= n_bootstraps:
                                        break

                                upper = np.percentile(bootstrapped_corrs, 97.5)
                                lower = np.percentile(bootstrapped_corrs, 2.5)
                                estimate = np.mean(bootstrapped_corrs)
                                # st.write(f"Bootstrapped correlation coefficient: {estimate:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                                return estimate, lower, upper

                            try:
                                correlation_coefficient = df[x_variable].corr(df[y_variable])
                            except (ValueError, KeyError) as e:
                                st.write(e)
                                correlation_coefficient = np.nan
                            summary_data.append({"type": "pearson_correlation", "competition_name": competition, "kpi": x_variable, "other_variable": y_variable, "value": correlation_coefficient, "min_minutes_to_include_player": min_minutes})

                            df_agg_player_only_cb = df[df["coarse_position"] == "CB"]
                            assert len(df_agg_player_only_cb) > 0, "No CBs in df_agg_player_only_cb"
                            correlation_coefficient_only_CBs = df_agg_player_only_cb[x_variable].corr(df_agg_player_only_cb[y_variable])
                            summary_data.append({"type": "pearson_correlation_only_cbs", "competition_name": competition, "kpi": x_variable, "other_variable": y_variable, "value": correlation_coefficient_only_CBs, "min_minutes_to_include_player": min_minutes})
                            spearman_correlation_only_cbs = df_agg_player_only_cb[x_variable].corr(df_agg_player_only_cb[y_variable], method="spearman")
                            summary_data.append({"type": "spearman_correlation_only_cbs", "competition_name": competition, "kpi": x_variable, "other_variable": y_variable, "value": spearman_correlation_only_cbs, "min_minutes_to_include_player": min_minutes})
                            correlation_coefficient_only_CBs_bs, cc_lower_only_CBs, cc_upper_only_CBs = bootstrapped_correlation_coefficient(
                                df_agg_player_only_cb[x_variable], df_agg_player_only_cb[y_variable]
                            )

                            #                     df_data_new_only_CBs = df_data_new[df_data_new["coarse_position"] == "CB"]
                            #                     correlation_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"]))
                            #                     spearman_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"], method='spearman'))
                            summary_data.append({
                                "type": "bootstrapped_correlation_only_cbs",
                                "value": correlation_coefficient_only_CBs_bs,
                                "ci_upper": cc_upper_only_CBs,
                                "ci_lower": cc_lower_only_CBs,
                                "kpi": x_variable,
                                "other_variable": y_variable,
                                "competition_name": competition,
                                "alhpa": 0.05,
                                "min_minutes_to_include_player": min_minutes,
                            })

                            df_summary = pd.DataFrame(summary_data)
                            #
                            # plt.title(f"{x_variable}-{y_variable}")
                            # col.write(f"{x_variable}-{y_variable}")
                            # #
                            # plt.xlabel(x_variable)
                            # plt.ylabel(y_variable)
                            # plt.title(f"Correlation of {x_variable} and {y_variable}")
                            # # col.write(fig)
                            # # col.write(fig_log)
                            # plt.close()
                            #
                            # st.write("BLA")
                            # st.write("df_agg_team")
                            # st.write(df_agg_team)

                            # Assume df has columns: x, y, team, position

                            # df_agg_team["position"] = df_agg_team["position_x"].fillna(df_agg_team["position_y"])

                            with st.spinner("Regress"):
                                df_agg_team = df_agg_team.dropna(subset=[x_variable, y_variable, "team_id", "position"])
                                df_agg_team['team_id'] = df_agg_team['team_id'].astype(str)
                                df_agg_team['coarse_position'] = df_agg_team['coarse_position'].astype(str)

                                # Regress x on team and position
                                try:
                                    # st.write("A")
                                    # st.write(x_variable)
                                    # st.write(df_agg_team)
                                    # st.write(df_agg_team[x_variable].describe())
                                    # st.write(df_agg_team["team_id"].describe())
                                    # st.write(df_agg_team["coarse_position"].describe())
                                    model_x = statsmodels.formula.api.ols(
                                        f'{x_variable} ~ C(team_id) + C(coarse_position)', data=df_agg_team).fit()
                                except ValueError as e:
                                    st.write(e)
                                    # st.write(x_variable)
                                    # st.write(df_agg_team[x_variable].describe())
                                    # st.write(df_agg_team["team_id"].describe())
                                    # st.write(df_agg_team["coarse_position"].describe())
                                    continue
                                resid_x = model_x.resid

                                # Regress y on team and position
                                model_y = statsmodels.formula.api.ols(f'{y_variable} ~ C(team_id) + C(coarse_position)',
                                                                      data=df_agg_team).fit()
                                resid_y = model_y.resid

                                # Correlation of residuals
                                try:
                                    corr, pval = pearsonr(resid_x, resid_y)
                                except ValueError as e:
                                    st.write(e)
                                    continue

                                # Regress on position only
                                model_x_pos = statsmodels.formula.api.ols(f'{x_variable} ~ C(coarse_position)',
                                                                          data=df_agg_team).fit()
                                resid_x_pos = model_x_pos.resid
                                model_y_pos = statsmodels.formula.api.ols(f'{y_variable} ~ C(coarse_position)',
                                                                          data=df_agg_team).fit()
                                resid_y_pos = model_y_pos.resid
                                corr_pos, pval_pos = pearsonr(resid_x_pos, resid_y_pos)

                                data.append({
                                    "r": correlation_coefficient, "abs_r": abs(correlation_coefficient),
                                    "x": x_variable,
                                    "y": y_variable,
                                    # "team_and_position_corrected_correlation": corr, "p": pval,
                                    "position_corrected_correlation_pos": corr_pos, "p_pos": pval_pos,
                                    "correlation_coefficient_only_CBs": correlation_coefficient_only_CBs,
                                    "spearman_correlation_only_cbs": spearman_correlation_only_cbs,
                                    "abs_cc_only_CBs": abs(correlation_coefficient_only_CBs),

                                    "correlation_coefficient_only_CBs_bs": correlation_coefficient_only_CBs_bs,
                                    "abs_correlation_coefficient_only_CBs_bs": abs(correlation_coefficient_only_CBs_bs),
                                    "lower_level_025_only_CBs": cc_lower_only_CBs,
                                    "upper_level_975_only_CBs": cc_upper_only_CBs,
                                })

                    df = pd.DataFrame(data)
                    # df["total_p"] = df["p"] + df["p_pos"]
                    # df["both_significant"] = (df["p"] < 0.05) & (df["p_pos"] < 0.05)
                    st.write("df")
                    st.write(df)

                    df_summary = pd.DataFrame(summary_data)
                    st.write("df_summary")
                    st.write(df_summary)

            #########

            if _do_descriptives:
                with st.expander("Descriptives"):
                    # Position counts
                    df_desc1 = df_agg_match_player_pos.groupby(["coarse_position", "player_id"]).agg(
                        n_matches=("match_id", "nunique"),
                        minutes_played=("minutes_played", "sum"),
                        normalized_name=("normalized_name", "first"),
                    ).reset_index()
                    st.write("df_desc1")
                    st.write(df_desc1)
                    st.write(df_agg_match_player_pos)

                    df_desc2 = df_agg_match_player_pos.groupby(["match_id"]).agg(
                        n_players=("player_id", "nunique"),
                        minutes_played=("minutes_played", "sum"),
                        n_positions=("coarse_position", "nunique"),
                        # n_passes=("n_passes", "sum"),
                    ).reset_index()
                    st.write("df_desc2")
                    st.write(df_desc2)

                    # profiler.stop()
                    # st.stop()

            if _do_seasonal_correlation:
                if df_player_matchsums["is_rückrunde"].all() or not df_player_matchsums["is_rückrunde"].all():
                    st.warning("No rückrunde info.")
                with st.expander("Season-by-season correlations"):
                    # df_agg_player_rr = aggregate_matchsums(df_player_matchsums, group_cols=["player_id", "is_rückrunde"])
                    # df_agg_player_rr = df_agg_player_rr.reset_index().set_index("player_id")
                    df_agg = df_agg_by_season_half.copy().reset_index()
                    # st.write("df_agg", df_agg.shape)
                    # st.write(df_agg)
                    min_minutes = st.number_input("Minimum minutes played by player for Season-by-season corr.", min_value=0, value=DEFAULT_MINUTES, key=f"min_minutes_corr_{competition}")
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
                        plot = False
                        if plot:
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
                        try:
                            correlation_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"]))
                            spearman_coefficient_only_CBs = float(df_data_new_only_CBs[f"{kpi}_hinrunde"].corr(df_data_new_only_CBs[f"{kpi}_rückrunde"], method='spearman'))
                        except ValueError as e:
                            correlation_coefficient_only_CBs = np.nan
                            spearman_coefficient_only_CBs = np.nan

                        columns = st.columns(2)
                        plot = False
                        if plot:
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
                        else:
                            columns[0].write("-")
                            columns[1].write("-")

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
                        for corr_coeff, label in [
                            (correlation_coefficient, "seasonal_autocorrelation (Person)"),
                            (partial_corr, "seasonal_autocorrelation (Partial)"),
                            (correlation_coefficient_only_CBs, "seasonal_autocorrelation_only_CBs (Person)"),
                            # (spearman_coefficient, "seasonal_autocorrelation (Spearman)"),
                            (spearman_coefficient_only_CBs, "seasonal_autocorrelation_only_CBs (Spearman)"),
                        ]:
                            summary_data.append({
                                "type": label,
                                "competition_name": competition,
                                "kpi": kpi,
                                "value": corr_coeff,
                                "n_players": df_data_new["player_id"].nunique(),
                                "n_CBs": df_data_new[df_data_new["coarse_position"] == "CB"]["player_id"].nunique(),
                            })

                        # break  # TODO remove

                    df_correlations = pd.DataFrame(data)
                    # st.write("df_correlations")
                    # st.write(df_correlations.set_index("kpi").sort_values(by="abs_partial_correlation_coefficient", ascending=False))
                    df_summary = pd.DataFrame(summary_data)
                    # st.write("df_summary season-by-season")
                    # st.write(df_summary)

            data = []
            if _do_icc:
                min_minutes = st.number_input("Minimum minutes played by player for ICC.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key=f"{competition}_min_minutes")
                df_agg = df_agg_match_player_pos[df_agg_match_player_pos["minutes_played"] >= min_minutes].copy()
                df_agg["player_pos_minutes_played"] = df_agg.groupby(["player_id", "coarse_position"])["minutes_played"].transform("sum")
                min_minutes_player = st.number_input("Minimum minutes played by player-pos for ICC.", min_value=0, value=DEFAULT_MINUTES, key=f"{competition}_min_minutes_player")
                df_agg = df_agg[df_agg["player_pos_minutes_played"] >= min_minutes_player].copy()
                df_agg = df_agg.loc[:, ~df_agg.columns.duplicated()]

                st.write(f"Using {len(df_agg)} match performances of {len(df_agg['player_id'].unique())} players that played at least {min_minutes_player} minutes for ICC calculation.")
                with st.expander("Match-level ICC (how consistent and discriminative are KPIs on the match-level?)"):
                    n_cols = 1
                    columns = st.columns(n_cols)
                    # player_col = "short_name"
                    player_col = "player_id"
                    df_agg[player_col] = df_agg[player_col].astype(str)  # make categorical for later in ICC calc statsmodels fixes bug
                    for kpi_nr, kpi in enumerate(df_agg.columns):
                        col = columns[kpi_nr % n_cols]
                        try:
                            # deduplicate
                            df_agg = df_agg.loc[:, ~df_agg.columns.duplicated()]
                            icc = calculate_icc(df_agg, kpi, player_col, "coarse_position")
                            data.append({"kpi": kpi, "icc": icc})
                            summary_data.append({
                                "type": "match_level_icc",
                                "competition_name": competition,
                                # "x_variable": None,
                                "kpi": kpi,
                                "value": icc,
                                # "alpha": 0.05,
                                "min_minutes_to_include_match_performance": min_minutes,
                                "min_minutes_to_include_player": min_minutes_player,
                            })
                        except ValueError as e:
                            col.write(e)
                            data.append({"kpi": kpi, "icc": None})
                            continue
                        col.write(f"ICC for {kpi}: {icc}")

                        # sort by average
                        df_agg = df_agg.sort_values(by=kpi, ascending=False)
                        assert kpi in df_agg.columns and "short_name" in df_agg.columns, f"Columns {kpi} or short_name not found in df_agg"
                        # plt.figure(figsize=(16, 9))
                        plot = False
                        if plot:
                            sns.boxplot(data=df_agg, x="short_name", y=kpi, hue="coarse_position", width=1, dodge=False)
                            plt.xticks(rotation=90)
                            plt.xticks(fontsize=4)  # adjust the value as needed
                            plt.title(f"Discriminability of '{kpi}'")
                            col.write(plt.gcf())
                            plt.close()
                        else:
                            col.write("-")

                        # break  # TODO remove

                df_icc = pd.DataFrame(data)
                # st.write("df_icc")
                # st.write(df_icc)
                df_summary = pd.DataFrame(summary_data)
                # st.write("df_summary ICC")
                # st.write(df_summary)

            if _do_bootstrapped_icc:
                # min_minutes_per_match = st.number_input("Minimum minutes played by player for Season-Level ICC.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key=f"{competition}_min_minutes_per_match_bootstrap")
                df_base_matchsums = df_player_matchsums.copy()
                # df_base_matchsums = df_base_matchsums[df_base_matchsums["minutes_played"] >= min_minutes_per_match]
                min_minutes = st.number_input("Minimum minutes played by player-pos for Season-Level ICC.", min_value=0, value=DEFAULT_MINUTES, key=f"{competition}_min_minutes_bootstrap")
                df_base_matchsums = df_base_matchsums[df_base_matchsums["player_pos_minutes_played"] >= min_minutes].copy()
                st.write(f"Using {len(df_base_matchsums)} match performances of {len(df_base_matchsums['player_id'].unique())} players that played at least {min_minutes} minutes for bootstrapped ICC calculation.")

                def bootstrap_aggregate(df, group_cols, n_bootstrap=200, random_state=None):
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

                # st.write("df_base_matchsums")
                # st.write(df_base_matchsums)

                df_agg = bootstrap_aggregate(df_base_matchsums, ["player_id", "coarse_position"])
                df_agg["player_and_coarse_position"] = df_agg["player_id"].astype(str) + "_" + df_agg["coarse_position"]

                # st.write("df_agg", df_agg.shape)
                # st.write(df_agg.head())

                data = []
                for kpi in kpis:
                    icc = calculate_icc(df_agg, kpi, "player_id", "coarse_position")
                    summary_data.append({
                        "type": "bootstrapped_season_level_icc (corrected by position)",
                        "competition_name": competition,
                        "kpi": kpi,
                        "other_variable": None,
                        "value": icc,
                        "n_bootstrap": df_agg["n_bootstrap"].unique()[0],
                    })
                    icc_posplayer = calculate_icc(df_agg, kpi, "player_id", "player_and_coarse_position")
                    summary_data.append({
                        "type": "bootstrapped_season_level_icc (corrected by position and player)",
                        "competition_name": competition,
                        "kpi": kpi,
                        "other_variable": None,
                        "value": icc_posplayer,
                        "n_bootstrap": df_agg["n_bootstrap"].unique()[0],
                    })
                    uncorrected_icc = calculate_icc(df_agg, kpi, "player_id", None)
                    summary_data.append({
                        "type": "bootstrapped_season_level_icc (raw)",
                        "competition_name": competition,
                        "kpi": kpi,
                        "other_variable": None,
                        "value": uncorrected_icc,
                        "n_bootstrap": df_agg["n_bootstrap"].unique()[0],
                    })
                    icc_posplayer_uncorrected = calculate_icc(df_agg, kpi, "player_and_coarse_position", None)
                    summary_data.append({
                        "type": "bootstrapped_season_level_icc (raw, corrected by position and player)",
                        "competition_name": competition,
                        "kpi": kpi,
                        "other_variable": None,
                        "value": icc_posplayer_uncorrected,
                        "n_bootstrap": df_agg["n_bootstrap"].unique()[0],
                    })
                    data.append({"kpi": kpi, "icc": icc, "uncorrected_icc": uncorrected_icc, "icc_posplayer": icc_posplayer, "icc_posplayer_uncorrected": icc_posplayer_uncorrected})

                    # break # TODO remove

                df = pd.DataFrame(data)
                # st.write("df")
                # st.write(df)

                # st.stop()

            if _do_histograms:
                min_minutes_per_match_hist = st.number_input("Minimum minutes played per match for histograms.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key=f"min_minutes_per_match_histograms_BAdsvs_{competition}")
                min_minutes_total = st.number_input("Minimum total minutes played for histograms.", min_value=0, value=DEFAULT_MINUTES, key=f"min_minutes_total_histograms_BAdsvs_{competition}")
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
                                # st.write(df_agg[[kpi, "coarse_position"]])
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

            if _do_internal_correlations:
                min_minutes_per_match_hist = st.number_input("Minimum minutes played per match for histograms.", min_value=0, value=DEFAULT_MINUTES_PER_MATCH, key=f"min_minutes_per_match_hist_KPI_heatmpa{competition}")
                min_minutes_total = st.number_input("Minimum total minutes played for histograms.", min_value=0, value=DEFAULT_MINUTES, key=f"min_minutes_total_hist_KPI_heatmap{competition}")
                df_agg_player_pos_hist = df_agg_player_pos[df_agg_player_pos["minutes_played"] > min_minutes_total]
                df_agg_match_player_pos_hist = df_agg_match_player_pos[df_agg_match_player_pos["minutes_played"] > min_minutes_per_match_hist]
                with st.spinner("Calculating heatmap..."):
                    columns = st.columns(2)
                    for col, df_agg, label in zip(columns, [
                        df_agg_player_pos_hist,
                        df_agg_match_player_pos_hist
                    ], ["Season-Level", "Match-Level"]):
                        # heatmap

                        kpis = sorted(list(set((kpis))))
                        corr_matrix = df_agg[kpis].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 4})
                        plt.suptitle("Correlation Heatmap", y=1.02)
                        st.write(plt.gcf())
                        plt.close()

                        # add correlations to summary data
                        for kpi1 in kpis:
                            for kpi2 in kpis:
                                if kpi1 == kpi2:
                                    continue
                                import pandas.errors
                                correlation = corr_matrix.at[kpi1, kpi2]
                                assert pd.api.types.is_scalar(correlation), f"Expected scalar correlation, got {type(correlation)} for ({kpi1}, {kpi2})"

                                # st.write(f"{kpi1=}, {kpi2=}")
                                # print("corr_matrix")
                                # print(corr_matrix)
                                # print("correlation")
                                # print(correlation)
                                # st.write("correlation")
                                # st.write(correlation)
                                summary_data.append({
                                    "type": f"internal_correlation (Pearson, {label})",
                                    "competition_name": competition,
                                    "kpi": kpi1, "other_variable": kpi2, "value": correlation,
                                    "min_minutes_to_include_player": min_minutes_per_match_hist if label == "Season-Level" else None,
                                    "min_minutes_to_include_match_performance": min_minutes_per_match_hist if label == "Match-Level" else None,
                                })

                        # do another heatmap with only the first 4 columns but all rows
                        plt.figure()
                        sns.heatmap(corr_matrix.iloc[:, :5], annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 4})
                        plt.suptitle("Correlation Heatmap", y=1.02)
                        st.write(plt.gcf())
                        plt.close()

                        # do another heatmap with only CBs
                        df_agg_only_cbs = df_agg[df_agg["coarse_position"] == "CB"]
                        corr_matrix_only_cbs = df_agg_only_cbs[kpis].corr()
                        sns.heatmap(corr_matrix_only_cbs, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 4})
                        plt.suptitle("Correlation Heatmap (only CBs)", y=1.02)
                        st.write(plt.gcf())
                        plt.close()

                        # add correlations to summary data
                        for kpi1 in kpis:
                            for kpi2 in kpis:
                                correlation = corr_matrix_only_cbs.loc[kpi1, kpi2]
                                summary_data.append({
                                    "type": f"internal_correlation_only_CB (Pearson, {label})",
                                    "competition_name": competition,
                                    "kpi": kpi1, "other_variable": kpi2, "value": correlation,
                                    "min_minutes_to_include_player": min_minutes_per_match_hist if label == "Season-Level" else None,
                                    "min_minutes_to_include_match_performance": min_minutes_per_match_hist if label == "Match-Level" else None,
                                })

                            # break  # TODO remove
                        # break  # TODO remove

        df_summary = pd.DataFrame(summary_data)
        # df_summary.to_excel("C:/Users/j.bischofberger/Downloads/df_summary_total.xlsx")
        df_summary.to_excel(os.path.join(os.path.dirname(__file__), "df_summary_total.xlsx"))
        st.write("Final summary")
        st.write(df_summary)
        if len(df_summary) == 0:
            st.warning("No summary data to save.")
        else:
            st.write("All present types:", df_summary["type"].unique())
            with st.spinner("Saving summary to Drive..."):
                defensive_network.parse.drive.append_to_parquet_on_drive(df_summary, f"summary2.csv", key_cols=["type", "competition_name", "kpi", "other_variable"], format="csv")
            st.write(f"Saved summary to Drive!")

    if _do_summary:
        dfs = []
        #             "": 380,
        #             "Bundesliga": 132,
        for competitions in [
            ["FIFA Men's World Cup"],
            ["3. Liga"],
            ["Bundesliga"],
            ["FIFA Men's World Cup", "3. Liga", "Bundesliga"],
        ]:
            st.write("###" + ", ".join(competitions))
            df_summary = defensive_network.parse.drive.download_csv_from_drive(f"summary2.csv", st_cache=True)
            # df_summary = pd.read_csv(f"C:/Users/j.bischofberger/Downloads/df_summary.csv")
            # df_summary_original = pd.read_csv(f"C:/Users/j.bischofberger/Downloads/df_summary_2025_09_05_14_53.csv")
            # df_summary2 = pd.read_csv(f"C:/Users/j.bischofberger/Downloads/df_bootstrap_15.csv")
            # st.write("df_summary2")
            # st.write(df_summary2)
            # df_summary = pd.concat([df_summary1, df_summary2])
            # df_summary = df_summary_original.copy()


            # df_summary = df_summary_original.copy()
            df_summary_original = df_summary.copy()
            df_summary = df_summary[df_summary["competition_name"].isin(competitions)]

            def get_df_pivot(df_summary):

                # df_summar1 = pd.read_csv("C:/Users/j.bischofberger/Downloads/df_bl_3liga.csv")
                # df_summar2 = pd.read_csv("C:/Users/j.bischofberger/Downloads/df_summary_world_cup.csv")
                # df_summary = pd.concat([df_summar1, df_summar2])
                # df_summary1 = pd.read_csv("C:/Users/j.bischofberger/Downloads/df_summary_2025_08_29.csv")
                # df_summary2 = pd.read_csv("C:/Users/j.bischofberger/Downloads/df_summary_2025_08_29_internal.csv")
                # df_summary3 = pd.read_csv("C:/Users/j.bischofberger/Downloads/df_summary_2025_08_29_icc.csv")
                # df_summary = pd.concat([df_summary1,
                                        # df_summary2,
                                        # df_summary3])
                i_invalid = (df_summary["competition_name"] == "FIFA Men's World Cup") & (df_summary["type"].str.contains("seasonal_"))
                df_summary = df_summary.loc[~i_invalid]

                i = df_summary["type"].str.contains("internal")
                df_summary = df_summary.loc[~i]
                df_summary["other_variable"] = df_summary["other_variable"].fillna("")

                # i = df_summary["type"].str.contains("seasonal_") | df_summary["type"].str.contains("match_level_icc")
                # df_summary.loc[i, "value"] = df_summary.loc[i, "other_variable"]
                # df_summary.loc[i, "other_variable"] = ""

                # drop internal correlations
                # df_summary = df_summary.drop(columns=[col for col in df_summary.columns if "internal_" in col])
                def foo(x):
                    try:
                        return float(x)
                    except ValueError:
                        return np.nan
                df_summary["value"] = df_summary["value"].apply(foo)

                df_pivot = df_summary.pivot_table(index=["competition_name", "kpi"], columns=["type", "other_variable"], values="value")#.groupby("kpi").mean()
                df_pivot = df_pivot.groupby("kpi").mean()
                df_pivot_A = df_pivot.drop(columns=[col for col in df_pivot.columns if col[0].startswith("internal_correlation") or col[0].startswith("spearman_") or "bootstrap" in col[0] or "only_cbs" not in col[0]]).copy()
                df_pivot_A.columns = ['_'.join(filter(None, col)).strip() for col in df_pivot_A.columns.values]

                for col in df_pivot_A.columns:
                    abs_col = f"abs_{col}"
                    df_pivot_A[abs_col] = df_pivot_A[col].abs()

                df_pivot.columns = ['_'.join(filter(None, col)).strip() for col in df_pivot.columns.values]

                df_pivot_B = df_pivot[[col for col in df_pivot.columns if "seasonal" in col or "match_level_icc" in col or "bootstrapped_season_level_icc (corrected by position)" in col]]

                assert not df_pivot_B.empty

                df_pivot_final = df_pivot_A.join(df_pivot_B).reset_index()

                df_pivot_final["expected_direction"] = (df_pivot_final["kpi"].str.contains("fault") | df_pivot_final["kpi"].str.contains("n_passes_against_")).map({False: 1, True: -1})
                import scipy.stats
                df_pivot_final["pearson_correlation_only_cbs_def_awareness_directed"] = df_pivot_final["pearson_correlation_only_cbs_def_awareness"] * df_pivot_final["expected_direction"]
                df_pivot_final["pearson_correlation_only_cbs_defending_directed"] = df_pivot_final["pearson_correlation_only_cbs_defending"] * df_pivot_final["expected_direction"]
                df_pivot_final["pearson_correlation_only_cbs_market_value_directed"] = df_pivot_final["pearson_correlation_only_cbs_market_value"] * df_pivot_final["expected_direction"]

                return df_pivot_final

            df_pivot_final = get_df_pivot(df_summary)

            df_pivot_original = get_df_pivot(df_summary_original)

            def metrics_scatter(df_full):
                # df_full = df_full[~df_full["kpi"].str.endswith("_per_pass") & ~df_full["kpi"].str.contains("relative_")]
                df_full = df_full[
                    ~df_full["kpi"].str.contains("relative_") &
                    ~df_full["kpi"].str.contains("combined2") &
                    ~df_full["kpi"].str.contains("_plus_") &
                    ~df_full["kpi"].str.contains("_minus_") &
                    ~df_full["kpi"].str.contains("_lost") &
                    ~df_full["kpi"].str.contains("n_passes_per90") &
                    ~df_full["kpi"].str.contains("responsinvolvement")
                ]

                # Decide color per KPI
                def get_color(kpi):
                    if "fault" in kpi:
                        return "#C5162A"
                    elif "contribution" in kpi:
                        return "#009E73"
                    elif any(w in kpi for w in ("interception", "tackle", "passes")):
                        return "#E69F00"
                    else:
                        return "#0072B2"

                df_full["color"] = df_full["kpi"].apply(get_color)
                colors = df_full["color"].tolist()

                # Plot Validity Score vs Robustness, label by KPI
                f_scale = 0.7
                plt.figure(figsize=(12*f_scale, 10*f_scale))

                import matplotlib.font_manager

                import defensive_network.utility.fonts
                defensive_network.utility.fonts.add_fonts()
                # st.write("Available fonts:")
                # for font in sorted(set(f.name for f in matplotlib.font_manager.fontManager.ttflist)):
                #     st.write(" -", font)

                plt.rcParams = plt.rcParamsDefault.copy()  # reset to default first

                # get available fonts
                available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
                # st.write("Available fonts:", available_fonts)

                font = "Roboto Slab"

                assert font in available_fonts, f"Font '{font}' not found. Please install it."
                font_prop = matplotlib.font_manager.FontProperties(family=font)

                plt.rcParams.update({
                    "font.family": font,  # install STIX Two Text on your OS
                    # "mathtext.fontset": "stix",  # matches the text face nicely
                    "font.size": 10,  # tweak to your journal’s spec
                    "axes.titlesize": 11,
                    "axes.labelsize": 10,
                    "pdf.fonttype": 42,  # embed TrueType for selectable text
                    "ps.fonttype": 42,
                    "svg.fonttype": "none",
                })
                for ticklabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                    ticklabel.set_fontproperties(font_prop)

                x = "Robustness Score"
                y = "Validity Score"

                plt.scatter(df_full[x], df_full[y], c=colors, alpha=0.7)

                df_full["nice_kpi"] = df_full["kpi"].str.replace("_combined", "_fused").str.replace("_", " ").str.title().str.replace("N ", "").str.replace("Per90", "per 90").str.replace("Per Pass", "per pass").str.replace("Involvement Fused", "Fused Involvement").str.replace("Responsibility Fused", "Fused Responsibility")
                df_full["label_on_top"] = False

                st.write("df_full")
                st.write(df_full)

                # Annotate points with KPI names, offset labels slightly to reduce overlap
                DY = 4.5
                texts = []
                for i, row in df_full.dropna(subset=[x, y]).iterrows():
                    dy_signed = DY if row["label_on_top"] else -DY
                    text = plt.text(
                        row[x], row[y], row["nice_kpi"], fontsize=7, path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground="white")], color=row["color"], ha='center', va='center'
                    )
                    # text = plt.annotate(
                    #     row["nice_kpi"], (row[x], row[y]), textcoords="offset points", ha="center",
                    #     xytext=(0, dy_signed),
                    #     fontsize=7, path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground="white")],
                    #     va='bottom' if row["label_on_top"] else 'top', color=row["color"]
                    # )
                    texts.append(text)

                adjustText.adjust_text(texts, expand=(0.7, 0.5))

                plt.xlabel(x, fontproperties=font_prop)
                plt.ylabel(y, fontproperties=font_prop)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.axhline(y=0, color='grey', linestyle='-')
                plt.axvline(x=0, color='grey', linestyle='-')
                # plt.gca().set_facecolor("#F9FAFB")

                # add legend explaining the colors
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='Fault', markerfacecolor="#C5162A", markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Contribution', markerfacecolor="#009E73", markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Combined', markerfacecolor="#0072B2", markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Traditional', markerfacecolor="#E69F00", markersize=8),
                ]
                plt.legend(handles=legend_elements, loc='best', fontsize=8, frameon=True, framealpha=0.5, prop=font_prop)

                # if len(competitions) == 1:
                #     plt.xlim(-3.2, 3.2)
                #     plt.ylim(-3.2, 3.2)

                competition_str = "_".join(competitions)
                st.write(f"{competition_str=}")

                plt.savefig(f"C:/Users/j.bischofberger/Downloads/xy_{competition_str}.png", dpi=300, bbox_inches='tight')

                st.write(plt.gcf())
                # plt.show()

            for score_col in [
                "pearson_correlation_only_cbs_def_awareness_directed",
                "pearson_correlation_only_cbs_defending_directed",
                "pearson_correlation_only_cbs_market_value_directed",
                "match_level_icc",
                "bootstrapped_season_level_icc (corrected by position)",
                "seasonal_autocorrelation (Partial)",
                "seasonal_autocorrelation_only_CBs (Person)",
            ]:
                try:
                    mu, sigma = df_pivot_original[score_col].mean(skipna=True), df_pivot_original[score_col].std(ddof=0, skipna=True)
                    # df_pivot_final[f"z_{score_col}"] = scipy.stats.zscore(df_pivot_final[score_col], nan_policy="omit")
                    df_pivot_final[f"z_{score_col}"] = (df_pivot_final[score_col] - mu) / sigma
                except KeyError:
                    df_pivot_final[score_col] = np.nan
                    df_pivot_final[f"z_{score_col}"] = np.nan

            # df_pivot_final["Validity Score"] = (
            #     df_pivot_final["z_pearson_correlation_only_cbs_def_awareness_directed"] +
            #     df_pivot_final["pearson_correlation_only_cbs_defending_directed"] +
            #     2 * df_pivot_final["z_pearson_correlation_only_cbs_market_value_directed"]
            # ) / 4
            # df_pivot_final["Validity Score"] = df_pivot_final
            # df_pivot_final["Robustness (and Discrimination) Score"] = (
            #     df_pivot_final["z_match_level_icc"] * 3 +
            #     df_pivot_final["z_seasonal_autocorrelation (Partial)"] +
            #     df_pivot_final["z_seasonal_autocorrelation_only_CBs (Person)"]
            # ) / 6

            df_pivot_final["validity_enumerator"] = df_pivot_final["z_pearson_correlation_only_cbs_def_awareness_directed"].fillna(0) + df_pivot_final["z_pearson_correlation_only_cbs_market_value_directed"].fillna(0)
            df_pivot_final["validity_denominator"] = (~df_pivot_final["pearson_correlation_only_cbs_def_awareness_directed"].isna()).astype(int) + (~df_pivot_final["pearson_correlation_only_cbs_market_value_directed"].isna()).astype(int)
            # df_pivot_final["validity_enumerator"] = df_pivot_final["z_pearson_correlation_only_cbs_def_awareness_directed"].fillna(0) + \
            #                                          df_pivot_final["z_pearson_correlation_only_cbs_defending_directed"].fillna(0) + \
            #                                          2 * df_pivot_final["z_pearson_correlation_only_cbs_market_value_directed"].fillna(0)
            # df_pivot_final["validity_denominator"] = (~df_pivot_final["pearson_correlation_only_cbs_def_awareness_directed"].isna()).astype(int) + \
            #                                            (~df_pivot_final["pearson_correlation_only_cbs_defending_directed"].isna()).astype(int) + \
            #                                            2 * (~df_pivot_final["pearson_correlation_only_cbs_market_value_directed"].isna()).astype(int)
            df_pivot_final["Validity Score"] = df_pivot_final["validity_enumerator"] / df_pivot_final["validity_denominator"].replace(0, np.nan)

            df_pivot_final["robustness_enumerator"] = df_pivot_final["z_match_level_icc"].fillna(0) + \
                                                       df_pivot_final["z_bootstrapped_season_level_icc (corrected by position)"].fillna(0) + \
                                                       df_pivot_final["z_seasonal_autocorrelation (Partial)"].fillna(0)
            df_pivot_final["robustness_denominator"] = (~df_pivot_final["match_level_icc"].isna()).astype(int) + \
                                                         (~df_pivot_final["bootstrapped_season_level_icc (corrected by position)"].isna()).astype(int) + \
                                                         (~df_pivot_final["seasonal_autocorrelation (Partial)"].isna()).astype(int)
            # df_pivot_final["robustness_enumerator"] = df_pivot_final["z_match_level_icc"].fillna(0) * 3 + \
            #                                            df_pivot_final["z_bootstrapped_season_level_icc (corrected by position)"].fillna(0) + \
            #                                            df_pivot_final["z_seasonal_autocorrelation (Partial)"].fillna(0) + \
            #                                            df_pivot_final["z_seasonal_autocorrelation_only_CBs (Person)"].fillna(0)
            # df_pivot_final["robustness_denominator"] = 3 * (~df_pivot_final["match_level_icc"].isna()).astype(int) + \
            #                                              (~df_pivot_final["bootstrapped_season_level_icc (corrected by position)"].isna()).astype(int) + \
            #                                              (~df_pivot_final["seasonal_autocorrelation (Partial)"].isna()).astype(int) + \
            #                                              (~df_pivot_final["seasonal_autocorrelation_only_CBs (Person)"].isna()).astype(int)
            df_pivot_final["Robustness Score"] = df_pivot_final["robustness_enumerator"] / df_pivot_final["robustness_denominator"].replace(0, np.nan)

            # df_pivot_final["z_bootstrapped_season_level_icc (corrected by position)"] +

            # df_pivot_final["Validity Score"] = (df_pivot_final["pearson_correlation_only_cbs_def_awareness"] * df_pivot_final["expected_direction"] +
            #                                     df_pivot_final["pearson_correlation_only_cbs_defending"] * df_pivot_final["expected_direction"] +
            #                                     2 * df_pivot_final["pearson_correlation_only_cbs_market_value"] * df_pivot_final["expected_direction"])
            # df_pivot_final["Robustness (and Discrimination) Score"] = (df_pivot_final["match_level_icc"] * 3 +
            #                                                            df_pivot_final["bootstrapped_season_level_icc (corrected by position)"] +
            #                                                            df_pivot_final["seasonal_autocorrelation (Partial)"] +
            #                                                            df_pivot_final["seasonal_autocorrelation_only_CBs (Person)"])

            # st.write("df_pivot_final")
            # st.write(df_pivot_final)
            df_pivot_final["competitions"] = ", ".join(competitions)
            df_pivot_final["n_competitions"] = len(competitions)

            dfs.append(df_pivot_final.copy())

            metrics_scatter(df_pivot_final)

        df_all = pd.concat(dfs, axis=0)
        st.write("df_all")
        st.write(df_all)
        df_all.to_csv("C:/Users/j.bischofberger/Downloads/df_all.csv", index=False)

# =[@[abs_pearson_correlation_only_cbs_def_awareness]]+[@[abs_pearson_correlation_only_cbs_defending]]+2*[@[abs_pearson_correlation_only_cbs_market_value]]

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
    dfgs_players.append(dfg_tracking)

    # Tackles
    dfg_players_tackles_won = df_event[df_event["event_subtype"] == "tackle"].groupby(player_group_cols, observed=False).agg(
        n_tackles_won=("event_id", "count"),
    ).fillna(0)
    dfgs_players.append(dfg_players_tackles_won)

    with st.spinner("Calculating tackles lost..."):
        dfg_players_tackles_lost = df_event[df_event["event_subtype"] == "tackle"].groupby(receiver_group_cols, observed=False).agg(
            n_tackles_lost=("event_id", "count"),
        ).reset_index().rename(columns={"player_id_2": "player_id_1", "role_category_2": "role_category_1"}).set_index(["player_id_1", "role_category_1"]).fillna(0)
        dfg_players_tackles_lost["tackles_won_share"] = dfg_players_tackles_won["n_tackles_won"] / (dfg_players_tackles_won["n_tackles_won"] + dfg_players_tackles_lost["n_tackles_lost"])
        dfgs_players.append(dfg_players_tackles_lost)

    # Interceptions
    dfg_players_interceptions = df_event[df_event["outcome"] == "intercepted"].groupby(receiver_group_cols, observed=False).agg(
        n_interceptions=("event_id", "count"),
    ).reset_index().fillna(0).rename(columns={"player_id_2": "player_id_1", "role_category_2": "role_category_1"}).set_index(["player_id_1", "role_category_1"]).fillna(0)
    dfg_players_interceptions["n_interceptions"] = dfg_players_interceptions["n_interceptions"].fillna(0)
    dfg_players_interceptions = dfg_players_interceptions.rename(columns={"player_id_2": "player_id_1"})
    dfgs_players.append(dfg_players_interceptions)

    # Pass xP
    dfg_players_pass = df_passes.groupby(player_group_cols, observed=False).agg(
        n_passes=("event_id", "count"),
        total_xp=("xpass", "sum"),
        total_xt=("pass_xt", "sum"),
    )
    dfg_players_pass["xp_per_pass"] = dfg_players_pass["total_xp"] / dfg_players_pass["n_passes"]
    dfgs_players.append(dfg_players_pass)

    dfg_players_pass_successful = df_passes[df_passes["event_outcome"] == "successfully_completed"].groupby(player_group_cols, observed=False).agg(
        n_passes_successful=("event_id", "count"),
    )
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

    # for dfg_players

    return dfg_team, dfg_players


def process_involvements(df_meta, folder_tracking, folder_events, target_folder, radius, overwrite_if_exists=False):
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

        # cast team col to string
        df_events["team_id_1"] = df_events["team_id_1"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_events["team_id_2"] = df_events["team_id_2"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_tracking["team_id"] = df_tracking["team_id"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_tracking["player_id"] = df_tracking["player_id"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_events["player_id_1"] = df_events["player_id_1"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_events["player_id_2"] = df_events["player_id_2"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})
        df_events["expected_receiver"] = df_events["expected_receiver"].astype(str).str.replace(".0", "").replace({"None": None, "nan": None, "NaN": None})

        # st.write("df_events")
        # st.write(df_events[[col for col in df_events.columns if "frame" in col]])
        # st.write(df_events)
        # df_events = df_events[df_events["frame"].isin([69, 70])]
        # st.write("df_tracking")
        # st.write(df_tracking.head(10000))
        # st.write('df_tracking[df_tracking["role_category"].isna()]')
        # st.write(df_tracking[df_tracking["role_category"].isna()])

        df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_defender_meta_cols=["role_category"], model_radius=radius)

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

        importlib.reload(defensive_network.utility.pitch)
        # st.write("df_involvement")
        # st.write(df_involvement)<
        # print tracking teams

        # shuffle involvement and tracking
        # df_involvement = df_involvement.sample(frac=1, random_state=42).reset_index(drop=True)
        # df_tracking = df_tracking.sample(frac=1, random_state=42).reset_index(drop=True)

        defensive_network.utility.pitch.plot_passes_with_involvement(
            df_involvement, df_tracking, n_passes=5,
            # responsibility_col="raw_intrinsic_relative_responsibility",
            responsibility_col=None,
        )
        plt.close()
        assert len(df_involvement) > 100
        # defensive_network.parse.drive.upload_csv_to_drive(df_involvement, os.path.join(target_folder, f"{slugified_match_string}.csv"))
        defensive_network.parse.drive.upload_parquet_to_drive(df_involvement, os.path.join(target_folder, f"{slugified_match_string}.parquet"))

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


def create_matchsums(folder_tracking, folder_events, df_meta, df_lineups, target_folder_team, target_folder_players, folder_involvement, overwrite_if_exists=False):
    # @st.cache_resource
    def __create_matchsums(folder_tracking, folder_events, df_meta, df_lineups, target_folder_team, target_folder_players, folder_involvement, overwrite_if_exists):
        existing_match_ids = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(folder_events)]
        existing_match_ids_involvement = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder(folder_involvement)]
        existing_match_ids = set(existing_match_ids).intersection(existing_match_ids_involvement)
        df_meta = df_meta[df_meta["slugified_match_string"].isin(existing_match_ids)]
        st.write(f"{overwrite_if_exists=}")

        if not overwrite_if_exists:
            st.write("1")
            try:
                st.write("1")
                player_matchsum_files = defensive_network.parse.drive.list_files_in_drive_folder(target_folder_players)
                team_matchsum_files = defensive_network.parse.drive.list_files_in_drive_folder(target_folder_team)
                match_ids = [file["name"].split(".")[0] for file in player_matchsum_files]
                match_ids_team = [file["name"].split(".")[0] for file in team_matchsum_files]
                match_ids = set(match_ids).intersection(set(match_ids_team))
                # st.write("1")
                #
                # df_matchsums_player = defensive_network.parse.drive.download_csv_from_drive(target_fpath_players, st_cache=True)
                # match_ids = set(df_matchsums_player["match_id"])
                # df_matchsums_player = pd.read_csv("C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/df_matchsums_player.csv")
                # st.write("1")
                # df_matchsums_player.to_excel("df_matchsums_player.xlsx", index=True)
                # st.write("1")
                # df_matchsums_team = defensive_network.parse.drive.download_csv_from_drive(target_fpath_team, st_cache=True)
                # st.write("1")
                # match_ids = set(df_matchsums_player["match_id"]).intersection(df_matchsums_team["match_id"])
                # st.write("1")
            except FileNotFoundError:
                st.write("1")
                match_ids = set()
            st.write("1")

            df_meta = df_meta[~df_meta["slugified_match_string"].isin(match_ids)]

        dfs_player = []
        dfs_team = []
        match_nr = 0
        match_strings = [match["match_string"] for _, match in df_meta.iterrows()]
        st.write("match_strings")
        st.write(match_strings)

        # df_meta = df_meta[df_meta["slugified_match_string"] == "3-liga-2023-2024-1-st-erzgebirge-aue-fc-ingolstadt-04"]
        t1 = time.time()
        for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Creating matchsums"):
            match_nr += 1
            competition_name = match["competition_name"]
            # if match_nr <= 104:
            #     continue
            df_lineup = df_lineups[df_lineups["match_id"] == match["match_id"]]
            match_string = match["match_string"]
            slugified_match_string = match["slugified_match_string"]
            t2 = time.time()
            st.write(f"Creating matchsums for {match_string} in {t2 - t1:.2f} seconds ({match_nr}/{len(df_meta)})")
            t1 = time.time()
            fpath_tracking = os.path.join(folder_tracking, f"{slugified_match_string}.parquet")
            fpath_events = os.path.join(folder_events, f"{slugified_match_string}.csv")
            fpath_involvement = os.path.join(folder_involvement, f"{slugified_match_string}.parquet")

            # df_tracking = pd.read_parquet(fpath_tracking)
            # df_events = pd.read_csv(fpath_events)
            try:
                # df_tracking = defensive_network.parse.drive.download_parquet_from_drive(fpath_tracking)
                with st.spinner("Loading data..."):
                    df_involvement = defensive_network.parse.drive.download_parquet_from_drive(fpath_involvement, st_cache=True)  # TODO no cache

                    # @st.cache_resource
                    def _get_parquet(fpath):
                        return pd.read_parquet(fpath)

                    # df_tracking = pd.read_parquet(fpath_tracking)
                    df_tracking = _get_parquet(fpath_tracking)
                    df_events = defensive_network.parse.drive.download_csv_from_drive(fpath_events, st_cache=True)
                    cn=competition_name.replace("Men's", "Mens")
                    dfg_responsibility_model = defensive_network.parse.drive.download_csv_from_drive(f"responsibility_model_{cn}.csv", st_cache=True).reset_index(drop=True)
                    # st.write("dfg_responsibility_model")
                    # st.write(dfg_responsibility_model)
                with st.spinner("Calculating Responsibility..."):
                    df_involvement["raw_responsibility"], df_involvement["relative_raw_responsibility"], df_involvement["valued_responsibility"], df_involvement["relative_valued_responsibility"] = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model)

            except FileNotFoundError as e:
                st.write(e)
                continue

            with st.spinner("Aggregating matchsums..."):
                try:
                    dfg_team, dfg_players = _create_matchsums(df_events, df_tracking, match, df_lineup, df_involvement)
                except Exception as e:
                    st.write(e)
                    st.rerun()  # OR use st_autorefresh

            # st.write("dfg_team")
            # st.write(dfg_team)
            # st.write(dfg_team[["team_id", "match_id"]])
            # st.write("dfg_players")
            # st.write(dfg_players)
            # st.write(dfg_players[["player_id", "role_category", "match_id"]])

            # dfs_player.append(dfg_players)
            # dfs_team.append(dfg_team)
            target_fpath_players = os.path.join(target_folder_players, f"{match['slugified_match_string']}.parquet")
            target_fpath_team = os.path.join(target_folder_team, f"{match['slugified_match_string']}.parquet")
            for col in ["first_name", "short_name", "last_name", "position_group", "position", "team_id", "team_name", "team_role"]:
                dfg_players[col] = dfg_players[col].astype(str)  # fix mixed type (0)
            # dfg_players["first_name"] = dfg_players["first_name"].astype(str)  # fix mixed type (0)
            # dfg_players["short_name"] = dfg_players["short_name"].astype(str)  # fix mixed type (0)
            # dfg_players["last_name"] = dfg_players["last_name"].astype(str)  # fix mixed type (0)
            # dfg_players["position_group"] = dfg_players["position_group"].astype(str)  # fix mixed type (0)
            # dfg_players["position"] = dfg_players["position"].astype(str)  # fix mixed type (0)
            # st.write("dfg_players")
            # st.write(dfg_players)
            defensive_network.parse.drive.upload_parquet_to_drive(dfg_players, target_fpath_players)
            defensive_network.parse.drive.upload_parquet_to_drive(dfg_team, target_fpath_team)

        return dfs_player, dfs_team

    dfs_player, dfs_team = __create_matchsums(folder_tracking, folder_events, df_meta, df_lineups, target_folder_team, target_folder_players, folder_involvement, overwrite_if_exists)

    # dfg_players = pd.concat(dfs_player)
    # dfg_team = pd.concat(dfs_team)

    # with st.spinner("Uploading team data..."):
    #     defensive_network.parse.drive.append_to_parquet_on_drive(dfg_team, target_fpath_team, key_cols=["team_id", "match_id"], overwrite_key_cols=True, format="csv")
    # with st.spinner("Uploading players data..."):
    #     # defensive_network.parse.drive.convert_to_parquet_and_append_to_parquet_on_drive(dfg_players, target_fpath_players, key_cols=["player_id", "role_category", "match_id"], overwrite_key_cols=True, format="csv")
    #     defensive_network.parse.drive.append_to_parquet_on_drive(dfg_players, target_fpath_players, key_cols=["player_id", "role_category", "match_id"], overwrite_key_cols=True, format="csv")
    st.write("Done!")

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

    st.write("finalize_events_and_tracking_to_drive df_meta before filter")
    st.write(df_meta)
    st.write("tracking", existing_match_ids_tracking)
    st.write(df_meta[df_meta["slugified_match_string"].isin(existing_match_ids_tracking)])

    # df_meta = df_meta[df_meta["slugified_match_string"].isin(existing_match_ids)]
    df_meta = df_meta[df_meta["slugified_match_string"] == "3-liga-2023-2024-1-st-erzgebirge-aue-fc-ingolstadt-04"]
    st.write("finalize_events_and_tracking_to_drive df_meta after filter")
    st.write(df_meta)

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
        do_not_process_if_synchronized = False
        if do_not_process_if_synchronized:
            try:
                df_event_existing = defensive_network.parse.drive.download_csv_from_drive(drive_path_events)
                st.write("df_event_existing")
                st.write(df_event_existing)
                if df_event_existing["matched_frame"].notna().any():
                    st.write(f"Skipping {match_string} because it is already synchronized")
                    continue
                pass
            except FileNotFoundError as e:
                st.write(e)
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

    existing_matches = [file["name"].split(".")[0] for file in defensive_network.parse.drive.list_files_in_drive_folder("tracking", st_cache=True)]
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
            df_event = defensive_network.parse.drive.download_csv_from_drive(f"events/{match_slugified_string}.csv", st_cache=True)
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
