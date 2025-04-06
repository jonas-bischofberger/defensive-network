""" Streamlit app to explore StatsBomb data """

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import statsbombpy.sb
import streamlit as st
import pandas as pd


@st.cache_resource
def get_competitions():
    df_competitions = statsbombpy.sb.competitions()
    df_competitions["iteration_name"] = df_competitions["competition_name"] + " " + df_competitions["season_name"]
    df_competitions = df_competitions[["iteration_name"] + [col for col in df_competitions.columns if col != "iteration_name"]]
    df_competitions = df_competitions.sort_values("iteration_name")
    return df_competitions


@st.cache_resource
def get_events(match_id):
    df_events = statsbombpy.sb.events(match_id=match_id)
    player_id_2_team_id = {player["player_id"]: player["team_id"] for player in df_events[["player_id", "team_id"]].to_dict(orient="records")}
    team_id_2_team = {team["team_id"]: team["team"] for team in df_events[["team_id", "team"]].to_dict(orient="records")}
    df_events["pass_recipient_team_id"] = df_events["pass_recipient_id"].apply(lambda x: player_id_2_team_id.get(x, None))

    df_events["timestamp"] = df_events["timestamp"].apply(lambda x: pd.to_datetime(x, format="%H:%M:%S.%f"))
    df_events["mm_ss"] = df_events[["period", "timestamp"]].apply(lambda x: x[1] if x[0] == 1 else x[1] + pd.Timedelta(minutes=45), axis=1)
    df_events["mm_ss"] = df_events["mm_ss"].apply(lambda x: f"{x.hour * 60 + x.minute:02d}:{x.second:02d}")

    df_events["event_string"] = df_events["mm_ss"].fillna("") + " " + df_events["type"].fillna("") + " - " + df_events["player"].fillna("") + " (" + df_events["team"].fillna("") + ") -> " + df_events["pass_recipient"].fillna("") + " (" + df_events["pass_recipient_team_id"].apply(lambda x: team_id_2_team.get(x, "Unknown")).fillna("") + ")"
    df_events = df_events[["event_string"] + [col for col in df_events.columns if col != "event_string"]]
    df_events = df_events.sort_values("event_string")

    return df_events


@st.cache_resource
def get_matches(competition_ids, season_ids):
    dfs_matches = []
    for competition_id, season_id in zip(competition_ids, season_ids):
        df_matches = statsbombpy.sb.matches(competition_id=competition_id, season_id=season_id)
        df_matches["match_string"] = df_matches.apply(match_to_string, axis=1)
        df_matches = df_matches[["match_string"] + [col for col in df_matches.columns if col != "match_string"]]
        df_matches = df_matches.sort_values("match_string")
        dfs_matches.append(df_matches)

    df_matches = pd.concat(dfs_matches)
    return df_matches


@st.cache_resource
def get_360_data(match_id):
    df_360 = statsbombpy.sb.frames(match_id=match_id, fmt='dataframe')
    df_events = get_events(match_id)
    df_360 = df_360.merge(df_events[[col for col in df_events.columns if col not in df_360.columns or col == "id"]], on=["id"])
    df_360 = df_360[["event_string"] + [col for col in df_360.columns if col != "event_string"]]
    return df_360


def match_to_string(match):
    return f"{match['match_date']} {match['home_team']} vs {match['away_team']}"


def main():
    st.write("## StatsBomb data explorer")
    st.write("This app allows you to explore StatsBomb open data")

    df_competitions = get_competitions()

    with st.expander("All competitions"):
        st.write(df_competitions)

    df_competitions = df_competitions[df_competitions["match_available_360"].notna()]

    with st.expander("Competitions with 360 data"):
        st.write(df_competitions)

    selected_iteration_names = st.multiselect('Select competitions', df_competitions['iteration_name'].unique())

    df_matches = get_matches(competition_ids=df_competitions[df_competitions["iteration_name"].isin(selected_iteration_names)]["competition_id"].tolist(), season_ids=df_competitions[df_competitions["iteration_name"].isin(selected_iteration_names)]["season_id"].tolist())

    with st.expander("Matches"):
        st.write(df_matches)

    selected_matches = st.multiselect('Select matches', df_matches['match_string'].unique())
    df_selected_matches = df_matches[df_matches['match_string'].isin(selected_matches)]

    for _, match in df_selected_matches.iterrows():
        st.write(match["match_string"])
        
        df_events = get_events(match['match_id'])

        with st.expander("events"):
            st.write(df_events)

        df_360 = get_360_data(match['match_id'])

        with st.expander("360 data"):
            st.dataframe(df_360.head(1000))


if __name__ == '__main__':
    main()
