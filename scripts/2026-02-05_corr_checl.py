import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import defensive_network.parse.drive
import streamlit as st

match_id = "fifa-men-s-world-cup-2022-3-st-costa-rica-germany"

st.write("match_id")
st.write(match_id)

df = defensive_network.parse.drive.download_parquet_from_drive(f"team_matchsums/10/{match_id}.parquet")
st.write(df)

df = defensive_network.parse.drive.download_csv_from_drive(f"events/{match_id}.csv")
st.write(df[df["possessionEvents.shotOutcomeType"] == "G"])
