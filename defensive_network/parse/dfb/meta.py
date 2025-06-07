import pandera.pandas as pa
import pandas as pd

from pandera.typing import Series


class CDFMetaSchema(pa.DataFrameModel):
    competition_name: Series[str]
    competition_id: Series[str]
    host: Series[str]
    match_day: Series[int]
    season_name: Series[str]
    season_id: Series[str]
    kickoff_time: Series[pd.Timestamp]
    match_id: Series[str]
    event_vendor: Series[str]
    tracking_vendor: Series[str]
    match_title: Series[str]
    home_team_name: Series[str]
    home_team_id: Series[str]
    guest_team_name: Series[str]
    guest_team_id: Series[str]
    result: Series[str]
    country: Series[str]
    stadium_id: Series[str]
    stadium_name: Series[str]
    precipitation: Series[str]
    pitch_x: Series[float] = pa.Field(ge=0, le=120)  # typical soccer pitch length in meters
    pitch_y: Series[float] = pa.Field(ge=0, le=90)   # typical soccer pitch width in meters
    total_time_first_half: Series[float] = pa.Field(ge=0)
    total_time_second_half: Series[float] = pa.Field(ge=0)
    playing_time_first_half: Series[float] = pa.Field(ge=0)
    playing_time_second_half: Series[float] = pa.Field(ge=0)
    ds_parser_version: Series[str]
    xg_tag: Series[str]
    xg_sha1: Series[str]
    xpass_tag: Series[str]
    xpass_sha1: Series[str]
    fps: Series[int] = pa.Field(ge=0.001, le=1000)  # frames per second, reasonable range

    class Config:
        strict = True  # error on unexpected columns


import os
import slugify.slugify
import streamlit as st


@st.cache_resource
def _read_df(fpath, **kwargs):
    return pd.read_csv(fpath, **kwargs)


def write_csv(df, fpath, **kwargs):
    pass


def process_meta(raw_meta_files, target_fpath, overwrite_if_exists=True):
    if not overwrite_if_exists and os.path.exists(target_fpath):
        return
    dfs = []
    for file in raw_meta_files:
        df_meta = _read_df(file)
        df_meta["match_string"] = df_meta.apply(lambda row: f"{row['competition_name']} {row['season_name']}: {row['match_day']}.ST {row['match_title'].replace('-', ' - ')}", axis=1)
        df_meta["slugified_match_string"] = df_meta["match_string"].apply(slugify.slugify)
        dfs.append(df_meta)
    df_meta = pd.concat(dfs, axis=0)
    st.write(f"Wrote df_meta to {target_fpath}")
    st.write(df_meta)
    df_meta.to_csv(target_fpath)
