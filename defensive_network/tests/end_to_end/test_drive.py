import defensive_network.parse.drive

import pandas as pd
import streamlit as st


def _get_test_df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })


def test_drive_basic_read_write_csv():
    df_source = _get_test_df()
    defensive_network.parse.drive.delete_folder_by_path("test/")
    defensive_network.parse.drive.upload_csv_to_drive(df_source, "test/test.csv")
    df = defensive_network.parse.drive.download_csv_from_drive("test/test.csv")
    pd.testing.assert_frame_equal(df, df_source)
    defensive_network.parse.drive.delete_folder_by_path("test/")


def test_drive_basic_read_write_parquet():
    df_source = _get_test_df()
    defensive_network.parse.drive.delete_folder_by_path("test/")
    defensive_network.parse.drive.upload_parquet_to_drive(df_source, "test/test.parquet")
    df = defensive_network.parse.drive.download_parquet_from_drive("test/test.parquet")
    pd.testing.assert_frame_equal(df, df_source)
    defensive_network.parse.drive.delete_folder_by_path("test/")
