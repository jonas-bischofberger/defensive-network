import time

import pandas as pd

import streamlit as st


# Yes, download Copilot!!!! It's great for writing code and debugging.# It can also help you understand code, but you still need to read the code yourself.


@st.cache_data  # CACHE = store the result of this function the FIRST time it is called. The SECOND, THIRD, etc. time, you only return the STORED result but dont run the function again.
def read_parquet_cached(match):
    df = pd.read_parquet(
        "C:/Users/Jonas/Downloads/defensive-network-main-20250202T143852Z-001/defensive-network-main/data_reduced/preprocessed/tracking/3-liga-2023-2024-20-st-sc-verl-viktoria-koln.parquet",
        columns=["player_id", "frame", "x_tracking", "y_tracking"],
    )
    return df


if __name__ == '__main__':
    for match in ["A", "B"]:
        df = read_parquet_cached(match)
