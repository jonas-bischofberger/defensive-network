import gc
import json
import time

import kloppy.pff

import bz2
import sys
import os

from tldextract.cache import DiskCache

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import importlib

import pandas as pd
import streamlit as st

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import defensive_network.utility.general

def _process_event_file(event_file, add_tracking=False):
    with open(event_file, "r") as f:
        data = json.loads(f.read())

    df_events = pd.json_normalize(data)
    if not add_tracking:
        return df_events

    df_events_tracking = df_events.copy()

    dfs_tracking = []
    nested_cols = ["homePlayers", "awayPlayers", "ball"]  # contain lists of dictionaries
    for nested_col in nested_cols:
        nested_data = df_events_tracking[nested_col].reset_index().to_dict(orient='records')  # convert to a long list of dictionaries

        df_tracking = pd.json_normalize(  # flatten the nested dictionaries
            nested_data,
            record_path=nested_col,  # transform the player/ball positions into rows
            meta=['index'],  # keep index so we can merge to the original event data later
        ).add_prefix("tracking.")

        df_tracking["tracking.team"] = {"homePlayers": "home", "awayPlayers": "away", "ball": "ball"}[nested_col]
        dfs_tracking.append(df_tracking)
        df_events_tracking = df_events_tracking.drop(columns=nested_col)  # remove original nested columns

    df_tracking = pd.concat(dfs_tracking, axis=0)
    df_event_tracking = df_events_tracking.merge(df_tracking, left_index=True, right_on="tracking.index", how="left")
    return df_event_tracking


def read_jsonl_bz_streamed(tracking_file):
    """ Read jsonl.bz2 file more memory-efficiently by streaming it line by line """
    def _stream_json_bzw(tracking_file):
        i = 0
        with bz2.open(tracking_file, "rt") as f:
            for line in f:
                i+= 1
                yield json.loads(line)
                # if i >= 10000:
                #     break
    rows = []
    for i, row in enumerate(_stream_json_bzw(tracking_file)):
        rows.append(row)
    return pd.DataFrame(rows)


def _process_tracking_file(tracking_file, long_format=True, use_smoothed_coordinates=True, flatten_event_data=True):
    """
    long_format: True = one row per froame AND player/ball, False = one row per frame with nested lists of players/balls
    use_smoothed_coordinates: True = use smoothed coordinates from PFF, False = use raw coordinates from PFF
    add_event_data: True = adds new event-related columns like event type, time etc.
    """
    # df_tracking = pd.read_json(tracking_file, lines=True, compression="bz2")
    # df_tracking = parse.from_s3.read_jsonl_bz2_s3(tracking_file, st_cache=False)
    df_tracking = read_jsonl_bz_streamed(tracking_file)

    unsmoothed_nested_cols = ["homePlayers", "awayPlayers", "balls"]
    smoothed_nested_cols = ["homePlayersSmoothed", "awayPlayersSmoothed", "ballsSmoothed"]
    nested_cols = unsmoothed_nested_cols if not use_smoothed_coordinates else smoothed_nested_cols

    if use_smoothed_coordinates:
        t = time.time()
        # df_tracking["ballsSmoothed"] = df_tracking["ballsSmoothed"].apply(lambda x: [x])  # this column needs to be converted from dictionary to list of dictionaries to match the others
        # df_tracking["ballsSmoothed"] = [[x] for x in df_tracking["ballsSmoothed"].values]
        df_tracking["ballsSmoothed"] = df_tracking["ballsSmoothed"].map(lambda x: [x])
        st.write(time.time() - t)
        print(time.time() - t)
        df_tracking = df_tracking.drop(columns=unsmoothed_nested_cols)
    else:
        df_tracking = df_tracking.drop(columns=smoothed_nested_cols)

    if not long_format:
        return df_tracking

    dfs_nested = []
    for nested_col in defensive_network.utility.general.progress_bar(nested_cols, desc="Flattening nested columns", total=len(nested_cols)):
        nested_data = df_tracking[nested_col].dropna().reset_index().to_dict(orient='records')  # convert to a long list of dictionaries
        df_tracking = df_tracking.drop(columns=nested_col)  # remove original nested columns

        nested_data = [
            d for d in nested_data
            if any(isinstance(x, dict) for x in d[nested_col])
        ]

        df_nested = pd.json_normalize(  # flatten the nested dictionaries
            nested_data,
            record_path=nested_col,  # transform the player/ball positions into rows
            meta=['index'],  # keep index so we can merge to the original event data later
            record_prefix=f"{nested_col}."
        )
        df_nested["tracking.team"] = {"homePlayers": "home", "awayPlayers": "away", "ball": "ball", "homePlayersSmoothed": "home", "awayPlayersSmoothed": "away", "ballsSmoothed": "ball"}[nested_col]
        dfs_nested.append(df_nested)

        del df_nested
        gc.collect()

    df_nested = pd.concat(dfs_nested, axis=0)
    del dfs_nested
    gc.collect()
    df_tracking = df_tracking.merge(df_nested, left_index=True, right_on="index", how="left")
    del df_nested
    gc.collect()

    if flatten_event_data:
        with st.spinner("Flattening game_event and possession_event..."):
            df_game_event = pd.json_normalize(df_tracking['game_event']).drop(columns=["game_id"]).rename(columns=lambda x: f"game_event.{x}")
            df_tracking = df_tracking.drop(columns=["game_event"]).join(df_game_event)
            del df_game_event
            gc.collect()
            df_possession_event = pd.json_normalize(df_tracking['possession_event']).drop(columns=["game_id", "game_event_id"]).rename(columns=lambda x: f"possession_event.{x}")
            df_tracking = df_tracking.drop(columns=["possession_event"]).join(df_possession_event)
            del df_possession_event
            gc.collect()

    return df_tracking


def _process_tracking_file2(tracking_file, long_format=True, use_smoothed_coordinates=True, flatten_event_data=True):
    ds = kloppy.pff.load_tracking(
        meta_data=tracking_file.replace("Tracking Data", "Metadata").replace(".jsonl.bz2", ".json"),
        roster_meta_data=tracking_file.replace("Tracking Data", "Rosters").replace(".jsonl.bz2", ".json"),
        raw_data=tracking_file,
    )
    df = ds.to_pandas()
    return df
    # datas = []
    # nested_cols = ["homePlayers", "awayPlayers", "balls"] if use_smoothed_coordinates else ["homePlayersSmoothed", "awayPlayersSmoothed", "ballsSmoothed"]
    # event_cols = ["game_event", "possession_event"]
    # with bz2.open(tracking_file, 'rt') as file:  # open in text mode ('rt')
    #     for line in file:
    #         data = json.loads(line)
    #         # Now you can process the JSON object `data`
    #         st.write(data)
    #
    #         break


def _process_meta_files(meta_files):
    metas = []
    for meta_file in utility.general.progress_bar(meta_files, desc="Processing meta files", total=len(meta_files)):
        with open(meta_file, "r") as f:
            meta_data = json.loads(f.read())
        meta_data = pd.json_normalize(meta_data)
        metas.append(meta_data)
    df_meta = pd.concat(metas)
    return df_meta


def _process_lineup_files(lineup_files):
    lineups = []
    for lineup_file in utility.general.progress_bar(lineup_files, desc="Processing lineup files", total=len(lineup_files)):
        match_id = os.path.basename(lineup_file).split(".")[0]
        with open(lineup_file, "r") as f:
            lineup_data = json.loads(f.read())
        df_lineup = pd.json_normalize(lineup_data)
        df_lineup["match_id"] = match_id
        lineups.append(df_lineup)
    df_lineups = pd.concat(lineups)
    return df_lineups


def main():
    event_file = "C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/pff/Event Data/3812.json"
    tracking_file = "C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/pff/Tracking Data/3812.jsonl.bz2"
    meta_dir = "C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/pff/Metadata/"
    lineup_dir = "C:/Users/Jonas/Desktop/Neuer Ordner/neu/phd-2324/defensive-network/pff/Rosters/"

    meta_files = [os.path.join(meta_dir, file) for file in os.listdir(meta_dir) if file.endswith(".json")]
    lineup_files = [os.path.join(lineup_dir, file) for file in os.listdir(lineup_dir) if file.endswith(".json")]

    df_meta = _process_meta_files(meta_files)
    st.write("df_meta")
    st.write(df_meta)
    df_lineups = _process_lineup_files(lineup_files)
    st.write("df_lineups")
    st.write(df_lineups)
    df_events = _process_event_file(event_file)
    st.write("df_events", df_events.shape)
    st.write(df_events)
    df_events_tracking = _process_event_file(event_file, add_tracking=True)
    st.write("df_events_tracking", df_events_tracking.shape)
    st.write(df_events_tracking)
    with st.spinner("Processing tracking file..."):
        df_tracking = _process_tracking_file(tracking_file, long_format=True)
        st.write("df_tracking")
        st.write(df_tracking.head())
        # st.write
        # parse_jsonl_bz2_to_parquet2(tracking_file, "out.parquet", limit=1000)
        # dft = pd.read_parquet("out.parquet")
        # st.write(dft.head())

    # st.write("df_tracking", df_tracking.shape)
    # st.write(df_tracking)

    # st.write(df_tracking.columns)


import json
import dask.bag as db
import dask.dataframe as dd
from tqdm.auto import tqdm

def parse_jsonl_bz2_to_parquet2(input_path, output_path, compression="snappy", use_smoothed=True, show_progress=True, limit=None):
    """
    Parses a compressed JSONL.bz2 file into a flattened player-frame Dask DataFrame
    and writes the result to Parquet.

    Args:
        input_path (str): Path to the input .jsonl.bz2 file.
        output_path (str): Output directory for the Parquet files.
        compression (str): Compression type for Parquet (e.g., 'snappy', 'gzip').
        use_smoothed (bool): Whether to use smoothed positions (True) or raw positions (False).
        show_progress (bool): Whether to display a progress bar.

    Returns:
        dask.dataframe.DataFrame: The resulting Dask DataFrame.
    """

    def parse_frame(line):
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            return []

        base_fields = {
            "frameNum": frame.get("frameNum"),
            "period": frame.get("period"),
            "videoTimeMs": frame.get("videoTimeMs"),
            "periodElapsedTime": frame.get("periodElapsedTime"),
            "periodGameClockTime": frame.get("periodGameClockTime"),
            "gameRefId": frame.get("gameRefId"),
        }

        def flatten_player(p, team, role):
            return {
                **base_fields,
                "team": team,
                "jerseyNum": p.get("jerseyNum"),
                "confidence": p.get("confidence"),
                "visibility": p.get("visibility"),
                "x": p.get("x"),
                "y": p.get("y"),
                "z": p.get("z", None),
                "role": role,
            }

        rows = []

        # Choose source depending on use_smoothed
        hp_key = "homePlayersSmoothed" if use_smoothed else "homePlayers"
        ap_key = "awayPlayersSmoothed" if use_smoothed else "awayPlayers"
        ball_key = "ballsSmoothed" if use_smoothed else "balls"

        # These can be None if explicitly present as null
        for p in frame.get(hp_key) or []:
            rows.append(flatten_player(p, "home", role="player"))

        for p in frame.get(ap_key) or []:
            rows.append(flatten_player(p, "away", role="player"))

        # Balls can be missing, None, a list, or a dict
        balls = frame.get(ball_key)
        if isinstance(balls, list):
            for b in balls:
                rows.append(flatten_player(b, "ball", role="ball"))
        elif isinstance(balls, dict):
            rows.append(flatten_player(balls, "ball", role="ball"))

        # Add expanded event info
        game_event = frame.get("game_event") or {}
        possession_event = frame.get("possession_event") or {}

        for r in rows:
            for k, v in game_event.items():
                r[f"game_event_{k}"] = v
            for k, v in possession_event.items():
                r[f"possession_event_{k}"] = v

        return rows

    # Load lines lazily
    # bag = db.read_text(input_path)

    import bz2
    import dask.bag as db

    if limit is not None:
        with bz2.open(input_path, "rt") as f:
            lines = [f.readline() for _ in range(limit)]
        bag = db.from_sequence(lines, npartitions=2)
    else:
        bag = db.read_text(input_path)


    # Optional progress bar using tqdm
    if show_progress:
        total = None
        try:
            import bz2
            with bz2.open(input_path, 'rt') as f:
                total = sum(1 for _ in f)
        except:
            pass  # fallback if counting fails

        tqdm.pandas(desc="Parsing lines", total=total)

        # Use progress bar with map_partitions
        def with_progress(part):
            return list(tqdm(map(parse_frame, part), desc="Partition", leave=False))

        bag = bag.map_partitions(with_progress).flatten()

    else:
        bag = bag.map(parse_frame).flatten()

    # Convert to Dask DataFrame
    ddf = bag.to_dataframe()

    import numpy as np
    nullable_int_cols = [
        "game_event_game_id",
        "possession_event_possession_id",
    ]

    # def clean_and_cast(df):
    #     for col in nullable_int_cols:
    #         if col in df.columns:
    #             # Replace inf/-inf with pd.NA
    #             df[col] = df[col].replace([np.inf, -np.inf], pd.NA)
    #             # Replace any other NaNs with pd.NA (nullable integer missing)
    #             df[col] = df[col].where(pd.notna(df[col]), pd.NA)
    #             # Now convert to nullable Int64
    #             df[col] = df[col].astype("Int64")
    #     return df

    def replace_invalids(df):
        cols = ['game_event_game_id', 'possession_event_possession_id']
        for col in cols:
            if col in df.columns:
                # Replace infinite values with pd.NA (nullable)
                df[col] = df[col].replace([np.inf, -np.inf], pd.NA)
                # Replace NaN (np.nan) with pd.NA
                df[col] = df[col].where(pd.notna(df[col]), pd.NA)
                # At this point dtype still float or object, no cast yet
        return df

    ddf = ddf.map_partitions(replace_invalids)

    def cast_nullable_int(df):
        cols = ['game_event_game_id', 'possession_event_possession_id']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype('Int64')  # pandas nullable integer dtype
        return df
    ddf = ddf.map_partitions(cast_nullable_int)

    # Optional: reduce memory use
    # ddf = ddf.categorize()

    # Write to Parquet
    ddf.to_parquet(output_path, compression=compression, write_index=False)

    return ddf



if __name__ == '__main__':
    import defensive_network.utility.general
    defensive_network.utility.general.start_streamlit_profiler()
    main()
    defensive_network.utility.general.stop_streamlit_profiler()
