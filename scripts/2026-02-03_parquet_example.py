import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import defensive_network.parse.drive
import streamlit as st

files = defensive_network.parse.drive.list_files_in_drive_folder("involvement/10/")
file_names = [file["name"] for file in files]
st.write(file_names)

for file_name in file_names:
    full_path = "involvement/10/" + file_name
    if "fifa-men-s-world-cup-2022" not in file_name:
        continue
    st.write(full_path)

    df = defensive_network.parse.drive.download_parquet_from_drive(full_path)
    st.write(df)
    break

