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
        df_tracking = defensive_network.parse.drive.download_csv_from_drive(f"tracking/{event_file}")

        match = df_meta[df_meta['slugified_match_string'] == event_file.replace(".csv", "")].iloc[0]

        st.write(df_tracking.head())
        st.stop()

        n_passes = len(df_events[df_events['event_type'] == "pass"])

        return {
            "n_passes": n_passes,
            "match_string": match['match_string'],
            "competition_name": match['competition_name'],
        }

    st.write("df_meta", df_meta.shape)
    st.write(df_meta)
    data = []
    data2 = []
    for event_file in defensive_network.utility.general.progress_bar(event_files):
        result = _process_event_file(event_file)
        data.append(result)
        # result2 = _process_tracking_file(event_file)
        # data2.append(result2)

    df = pd.DataFrame(data)
    st.write("df")
    st.write(df)

    dfg = df.groupby("competition_name").agg(
        n_matches=pd.NamedAgg(column="match_string", aggfunc="count"),
        n_passes=pd.NamedAgg(column="n_passes", aggfunc="sum"),
    ).reset_index().sort_values("n_passes", ascending=False)
    st.write("dfg")
    st.write(dfg)


if __name__ == '__main__':
    main()
