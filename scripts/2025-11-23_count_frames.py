import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

import defensive_network.parse.drive
import defensive_network.utility.general


def main():
    event_files = [f["name"] for f in defensive_network.parse.drive.list_files_in_drive_folder("events", st_cache=True)]
    st.write("event_files")
    st.write(event_files)

    df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)

    @st.cache_data
    def _process_event_file(event_file):
        df_events = defensive_network.parse.drive.download_csv_from_drive(f"events/{event_file}")

        match = df_meta[df_meta['slugified_match_string'] == event_file.replace(".csv", "")].iloc[0]

        n_passes = len(df_events[df_events['event_type'] == "pass"])

        return {
            "n_passes": n_passes,
            "match_string": match['match_string'],
            "competition_name": match['competition_name'],
        }

    @st.cache_data
    def _process_tracking_file(event_file):
        df_tracking = pd.read_parquet(f"Y:/m_raw/2324/finalized/tracking/{event_file}.parquet")

        match = df_meta[df_meta['slugified_match_string'] == event_file.replace(".csv", "")].iloc[0]

        n_frames = df_tracking["datetime_tracking"].nunique()

        st.write(event_file, n_frames)

        return {
            "n_frames": n_frames,
            "match_string": match['match_string'],
            "competition_name": match['competition_name'],
        }

    st.write("df_meta", df_meta.shape)
    st.write(df_meta)
    data = []
    data2 = []
    i = 0
    for event_file in defensive_network.utility.general.progress_bar(event_files):
        i += 1
        # result = _process_event_file(event_file)
        # data.append(result)
        result2 = _process_tracking_file(event_file.replace(".csv", ""))
        data2.append(result2)
        # if i >= 10:
        #     break

    # df = pd.DataFrame(data)
    # st.write("df")
    # st.write(df)

    df2 = pd.DataFrame(data2)
    st.write("df2")
    st.write(df2)

    # dfg = df.groupby("competition_name").agg(
    #     n_matches=pd.NamedAgg(column="match_string", aggfunc="count"),
    #     n_passes=pd.NamedAgg(column="n_passes", aggfunc="sum"),
    # ).reset_index().sort_values("n_passes", ascending=False)
    # st.write("dfg")
    # st.write(dfg)

    dfg2 = df2.groupby("competition_name").agg(
        n_matches=pd.NamedAgg(column="match_string", aggfunc="count"),
        n_frames=pd.NamedAgg(column="n_frames", aggfunc="sum"),
    ).reset_index().sort_values("n_frames", ascending=False)
    st.write("dfg2")
    st.write(dfg2)

    for event_file in [
        "3-liga-2023-2024-17-st-sc-verl-dynamo-dresden",
        "bundesliga-2023-2024-1-st-1-fc-nurnberg-werder-bremen",
        "fifa-men-s-world-cup-2022-3-st-south-korea-portugal",
    ]:
        st.write(event_file)
        df_tracking = pd.read_parquet(f"Y:/m_raw/2324/finalized/tracking/{event_file.replace('.csv', '')}.parquet")
        st.write(sorted(df_tracking["datetime_tracking"].unique()))


if __name__ == '__main__':
    main()
