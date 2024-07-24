""" Streamlit app to explore Metrica data """

import sys
import os

import matplotlib.cm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import kloppy.metrica

import streamlit_profiler

import warnings
warnings.filterwarnings("ignore")

import importlib

import utility.pitch
import utility.general
import model.passing_network

importlib.reload(model.passing_network)
importlib.reload(utility.pitch)
importlib.reload(utility.general)


@st.cache_resource
def load_raw_tracking_data_single_match(match_id):
    if match_id == 1:
        return kloppy.metrica.load_open_data(match_id=match_id).to_pandas()
    elif match_id == 2:
        return kloppy.metrica.load_open_data(match_id=match_id).to_pandas()
    elif match_id == 3:
        return kloppy.metrica.load_open_data(match_id=match_id).to_pandas()
    raise ValueError(f"Unknown match_id {match_id}")

def preprocess_tracking_and_event_data(df_tracking, df_events):
    ### Determine attacking direction
    period2team2x = {}
    for period, df_tracking_period in df_tracking.groupby("period_id"):
        period2team2x[period] = {}
        for team in ["home", "away"]:
            team_x_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_x_new")]
            team_y_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_y_new")]
            # st.write(df_tracking_period[team_x_cols + team_y_cols].head(50))
            x_avg = df_tracking_period[team_x_cols].mean(axis=1).mean()
            y_avg = df_tracking_period[team_y_cols].mean(axis=1).mean()
            period2team2x[period][team] = x_avg

    period2team2attacking_direction = {}
    for period in period2team2x:
        period2team2attacking_direction[period] = {}
        for team in period2team2x[period]:
            other_team = [t for t in period2team2x[period] if t != team][0]
            attacking_direction = 1 if period2team2x[period][team] < period2team2x[period][other_team] else -1
            period2team2attacking_direction[period][team] = attacking_direction

    ### Event
    df_events = df_events.sort_values("Start Time [s]")

    # team2player = df_events[["Team", "From"]].drop_duplicates().set_index("From").to_dict()["Team"]
    player2team = df_events[["From", "Team"]].drop_duplicates().set_index("From").to_dict()["Team"]
    player2player_tracking_id = {player: f"{player2team[player].lower()}_{utility.general.extract_numbers_from_string_as_ints(player)[-1]}" for player in player2team}
    df_events["from_player_tracking_id"] = df_events["From"].map(player2player_tracking_id)

    # Positions
    team2formation = {"Home": "4-4-2", "Away": "5-3-2"}
    df_events["live_formation"] = df_events["Team"].map(team2formation)

    positions = {
        "Player11": "GK",
        "Player1": "LB",
        "Player2": "LCB",
        "Player3": "RCB",
        "Player4": "RB",
        "Player5": "LW",
        "Player6": "LCM",
        "Player7": "RCM",
        "Player8": "RW",
        "Player9": "RS",
        "Player10": "LS",

        "Player25": "GK",
        "Player17": "RCB",
        "Player16": "CB",
        "Player15": "LCB",
        "Player22": "RWB",
        "Player18": "LWB",
        "Player20": "DM",
        "Player21": "RCM",
        "Player19": "LCM",
        "Player23": "RS",
        "Player24": "LS",
    }
    # 1. Auswechslung 12 für 1 (Verletzung) 7 -> LB, 12 -> RCM, Frame 40000

    substitutions = [
        {
            "team": "Home",
            "frame": 40000,
            "formation_change": None,
            "switches": [
                ("Player12", "RCM"),
                ("Player1", None),
                ("Player7", "LB"),
            ]
        },
        {
            "team": "Home",
            "frame": 112000,
            "formation_change": None,
            "switches": [
                ("Player13", "LW"),
                ("Player6", None),
                ("Player5", "LCM"),
            ]
        },
        {
            "team": "Home",
            "frame": 121000,
            "formation_change": None,
            "switches": [
                ("Player14", "LW"),
                ("Player10", None),
                ("Player13", "LS"),
            ]
        },

        ###

        {
            "team": "Away",
            "frame": 60500,
            "formation_change": "4-3-1-2",
            "switches": [
                ("Player17", "LB"),
                ("Player15", "LCB"),
                ("Player16", "RCB"),
                ("Player22", "RB"),
                ("Player19", "LCM"),
                ("Player20", "DM"),
                ("Player21", "RCM"),
                ("Player18", "AM"),
                ("Player23", "LS"),
                ("Player24", "RS"),
            ]
        },
        {
            "team": "Away",
            "frame": 71500,
            "formation_change": "5-3-2",
            "switches": [
                ("Player17", "LWB"),
                ("Player15", "LCB"),
                ("Player16", "CB"),
                ("Player22", "RWB"),
                ("Player19", "LCM"),
                ("Player20", "RCB"),
                ("Player21", "DM"),
                ("Player18", "RCM"),
                ("Player23", "RS"),
                ("Player24", "LS"),
            ]
        },
        {
            "team": "Away",
            "frame": 107000,
            "formation_change": None,
            "switches": [
                ("Player22", None),
                ("Player27", "RWB"),
                ("Player24", None),
                ("Player26", "LS"),
            ]
        },
        {
            "team": "Away",
            "frame": 119500,
            "formation_change": None,
            "switches": [
                ("Player28", "RCM"),
                ("Player19", None),
                ("Player18", "LCM"),
            ]
        }
    ]

    df_events["from_position"] = df_events["From"].map(positions)
    df_events["to_position"] = df_events["To"].map(positions)
    # st.stop()

    for substitution in substitutions:
        i_frames = df_events["Start Frame"] > substitution["frame"]
        for player, new_position in substitution["switches"]:
            i_from = df_events["From"] == player
            i_to = df_events["To"] == player
            df_events.loc[i_frames & i_from, "from_position"] = new_position
            df_events.loc[i_frames & i_to, "to_position"] = new_position

            if substitution["formation_change"] is not None:
                i_team = df_events["Team"] == "Home"
                df_events.loc[i_frames, "live_formation"] = substitution["formation_change"]

    # Coordinates
    df_events["x"] = (df_events["Start X"] - 0.5) * 105
    df_events["y"] = (df_events["Start Y"] - 0.5) * 68
    df_events["x_end"] = (df_events["End X"] - 0.5) * 105
    df_events["y_end"] = (df_events["End Y"] - 0.5) * 68
    df_events["attacking_direction"] = df_events.apply(lambda x: period2team2attacking_direction[x["Period"]][x["Team"].lower()], axis=1)
    df_events["attacking_direction_str"] = df_events["attacking_direction"].apply(lambda x: "left-to-right" if x == 1 else "right-to-left")
    df_events["x_norm"] = df_events["x"] * df_events["attacking_direction"]
    df_events["y_norm"] = df_events["y"] * df_events["attacking_direction"]
    df_events["x_end_norm"] = df_events["x_end"] * df_events["attacking_direction"]
    df_events["y_end_norm"] = df_events["y_end"] * df_events["attacking_direction"]

    # Timestamps
    def _period_and_seconds_to_mm_ss_ms(period, seconds_since_half_kickoff):
        minutes = seconds_since_half_kickoff // 60
        seconds = seconds_since_half_kickoff % 60
        milliseconds = (seconds_since_half_kickoff % 1) * 1000

        if minutes >= 45:
            extra_minutes = minutes % 45
            extratime_str = f"+{int(extra_minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"
            base_minutes_str = 90 if period == 2 else 45
            return f"{base_minutes_str}{extratime_str}"
        else:
            minutes = minutes if period == 1 else minutes + 45
            return f"{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"

    kickoff_starttimes = df_events.groupby("Period")["Start Time [s]"].min()
    df_events["kickoff_starttime"] = df_events["Period"].apply(lambda x: kickoff_starttimes[x])
    df_events["seconds_since_halfstart"] = df_events["Start Time [s]"] - df_events["kickoff_starttime"]
    df_events["matchtime_str"] = df_events.apply(lambda x: _period_and_seconds_to_mm_ss_ms(x["Period"], x["seconds_since_halfstart"]), axis=1)

    # Nice event_string
    df_events["event_string"] = df_events["matchtime_str"] + " - " + df_events["Subtype"].where(df_events["Subtype"].notnull(), df_events["Type"]).astype(str) + " " + df_events["From"].astype(str) + (" -> " + df_events["To"]).where(df_events["To"].notna(), "")
    df_events = df_events[["event_string"] + [col for col in df_events.columns if col != "event_string"]]

    ### Tracking
    # Frame
    df_tracking["custom_frame_id"] = df_tracking["frame_id"] - 1

    x_cols = [col for col in df_tracking.columns if col.endswith("_x")]
    new_x_cols = [f"{col}_new" for col in x_cols]
    y_cols = [col for col in df_tracking.columns if col.endswith("_y")]
    new_y_cols = [f"{col}_new" for col in y_cols]
    df_tracking[new_x_cols] = (df_tracking[x_cols] - 0.5) * 105
    df_tracking[new_y_cols] = (df_tracking[y_cols] - 0.5) * 68

    for period, df_tracking_period in df_tracking.groupby("period_id"):
        period2team2x[period] = {}
        for team in ["home", "away"]:
            team_x_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_x_new")]
            team_y_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_y_new")]

            new_team_x_cols = [f"{col}_norm" for col in team_x_cols]
            new_team_y_cols = [f"{col}_norm" for col in team_y_cols]

            i_period = df_tracking["period_id"] == period

            df_tracking.loc[i_period, new_team_x_cols] = df_tracking.loc[i_period, team_x_cols].values * period2team2attacking_direction[period][team]

            df_tracking.loc[i_period, new_team_y_cols] = df_tracking.loc[i_period, team_y_cols].values * period2team2attacking_direction[period][team]

    df_tracking = df_tracking.drop(columns=x_cols + y_cols).rename(columns={col: col.replace("_new", "") for col in df_tracking.columns})
    df_tracking = df_tracking.drop(columns=["frame_id"]).rename(columns={"custom_frame_id": "frame_id"})
    df_tracking = df_tracking.dropna(axis=1, how="all")

    return df_tracking, df_events


@st.cache_resource
def load_event_data_single_match(match_id):
    if match_id == "1" or match_id == 1:
        return pd.read_csv("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv")
    elif match_id == "2" or match_id == 2:
        return pd.read_csv("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_2/Sample_Game_2_RawEventsData.csv")
    elif match_id == "3" or match_id == 3:
        return kloppy.metrica.load_event(
            event_data="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_events.json",
            meta_data="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
        ).to_pandas()
    raise ValueError(f"Unknown match_id {match_id}")


def main():
    if st.button("Cache leeren"):
        st.cache_resource.clear()

    profiler = streamlit_profiler.Profiler()
    profiler.start()

    st.write("## Metrica data explorer")
    st.write("This app allows you to explore Metrica open data")

    ###

    import matplotlib.pyplot as plt
    import mplsoccer.pitch

    df_tracking = load_raw_tracking_data_single_match(1)
    df_events = load_event_data_single_match(1)

    df_tracking, df_events = preprocess_tracking_and_event_data(df_tracking, df_events)

    st.write("Event data")
    st.write(df_events)
    st.write("Tracking data (first 50 rows)")
    st.write(df_tracking.head(50))

    df_passes = df_events[df_events["Type"] == "PASS"]

    for team, df_passes_team in df_passes.groupby(["Team"]):
        st.write(f"### {team}")

        most_common_formation = df_passes_team["live_formation"].value_counts().idxmax()

        df_passes_team_filtered = df_passes_team[
            df_passes_team["from_position"].notna() & df_passes_team["to_position"].notna() & (df_passes_team["live_formation"] == most_common_formation)
        ]

        df_nodes, df_edges = model.passing_network.get_passing_network_df(
            df_passes_team_filtered,
            x_col="x_norm",  # column with x position of the pass
            y_col="y_norm",  # column with y position of the pass
            from_col="from_position",  # column with unique (!) ID or name of the player/position/... who passes the ball
            to_col="to_position",  # column with unique (!) ID or name of the player/position/... who receives the ball
            x_to_col="x_end_norm",  # column with x position of the pass target
            y_to_col="y_end_norm",  # column with y position of the pass target
        )

        # df_nodes["other_value"] = 0

        # fig = model.passing_network.plottt(df_nodes, df_edges)
        fig, ax = model.passing_network.plot_passing_network(
            df_nodes=df_nodes,
            df_edges=df_edges,
            show_colorbar=False,
            node_size_multiplier=30,
            colormap=matplotlib.cm.get_cmap("viridis"),
        )

        st.write(fig)

        st.write("Nodes")
        st.write(df_nodes)
        st.write("Edges")
        st.write(df_edges)

    selected_passes = st.multiselect("Select pass", df_passes.index, format_func=lambda x: df_passes.loc[x, "event_string"])

    if selected_passes == []:
        selected_passes = df_passes.index

    for i_pass in selected_passes:
        p4ss = df_passes.loc[i_pass]
        st.write(p4ss["event_string"])

        # Plot the pass as an arrow
        # st.write("p4ss")
        # st.write(p4ss)

        columns = st.columns(2)

        for is_target in [False, True]:
            pitch = mplsoccer.Pitch(pitch_type="impect", pitch_width=68, pitch_length=105, axis=True)
            fig, ax = pitch.draw()

            pitch.arrows(p4ss["x"], p4ss["y"], p4ss["x_end"], p4ss["y_end"], width=2, headwidth=10, headlength=10, color='blue' if p4ss["Team"] == "Away" else 'red', ax=ax)

            # Plot positions
            if not is_target:
                # i_tr = df_tracking["timestamp"] == p4ss["Start Time [s]"]
                i_tr = df_tracking["frame_id"] == p4ss["Start Frame"]
            else:
                # i_tr = df_tracking["timestamp"] == p4ss["End Time [s]"]
                i_tr = df_tracking["frame_id"] == p4ss["End Frame"]

            for team in ["home", "away"]:
                x_cols = [col for col in df_tracking.columns if col.endswith("_x") and col.startswith(team)]
                y_cols = [col for col in df_tracking.columns if col.endswith("_y") and col.startswith(team)]
                player_names = [f"{col.rsplit('_', -1)[1]}" for col in x_cols]
                x_pos = df_tracking.loc[i_tr, x_cols].values[0]
                y_pos = -df_tracking.loc[i_tr, y_cols].values[0]
                pitch.scatter(x_pos, y_pos+0., color="red" if team == "home" else "blue", ax=ax)

                # plot names
                for i, _ in enumerate(x_pos):
                    plt.gca().annotate(player_names[i], (x_pos[i], y_pos[i]-3.5), ha="center", va="bottom")

            pitch.scatter(df_tracking.loc[i_tr, "ball_x"], -df_tracking.loc[i_tr, "ball_y"], color="black", ax=ax, marker="x", s=200)

            # Display the plot
            plt.show()
            columns[int(is_target)].write(fig)

    profiler.stop()


if __name__ == '__main__':
    main()
