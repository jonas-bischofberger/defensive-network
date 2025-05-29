""" Streamlit app to explore Metrica data """

import sys
import os

import matplotlib.cm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import numpy as np
import kloppy.metrica

import streamlit_profiler

import warnings
warnings.filterwarnings("ignore")

import defensive_network.utility.pitch
import defensive_network.utility.general
import defensive_network.models.passing_network


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
            team_x_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_x")]
            team_y_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_y")]
            # st.write(df_tracking_period[team_x_cols + team_y_cols].head(50))
            x_avg = df_tracking_period[team_x_cols].mean(skipna=True, axis=1).mean(skipna=True)

            y_avg = df_tracking_period[team_y_cols].dropna().mean(axis=1).mean()
            period2team2x[period][team] = x_avg

            assert not np.isnan(x_avg)

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
    team2players = df_events.groupby("Team")["From"].unique().to_dict()
    player2player_tracking_id = {player: f"{player2team[player].lower()}_{defensive_network.utility.general.extract_numbers_from_string_as_ints(player)[-1]}" for player in player2team}
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

    for substitution in substitutions:
        i_frames = df_events["Start Frame"] > substitution["frame"]
        for player, new_position in substitution["switches"]:
            i_from = df_events["From"] == player
            i_to = df_events["To"] == player
            df_events.loc[i_frames & i_from, "from_position"] = new_position
            df_events.loc[i_frames & i_to, "to_position"] = new_position

        if substitution["formation_change"] is not None:
            i_team = df_events["Team"] == substitution["team"]
            df_events.loc[i_frames & i_team, "live_formation"] = substitution["formation_change"]

    # assert that every combination of live_formation and player has a unique position

    # Coordinates
    df_events["x"] = (df_events["Start X"] - 0.5) * 105
    # cri
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

    tracking_player_ids = [col.rsplit("_", 1)[0] for col in df_tracking.columns if col.endswith("_x")]
    tracking_player_id2name = {player: f"Player{player.split('_')[-1]}" for player in tracking_player_ids if player != "ball"}

    def transform_tracking():
        exploded_rows = []
        # progress_text = st.empty()
        # progress_bar = st.progress(0)
        for _, row in df_tracking.iterrows():
            # progress_bar.progress(_ / len(df_tracking))
            # progress_text.text(f"Processing row {_} of {len(df_tracking)}")
            # For each column A, B, C, append corresponding values

            # drop all x or y rows
            reduced_row = row.drop([col for col in row.index if "_x" in col or "_y" in col or "_s" in col or "_d" in col or "_z" in col])

            for tracking_player_id in tracking_player_ids:
                player = "ball" if tracking_player_id == "ball" else tracking_player_id2name[tracking_player_id]
                team = player2team.get(player, "ball")
                exploded_rows.append(reduced_row.to_dict() | {
                    'player': player,
                    'team': team,
                    'x': row[f'{tracking_player_id}_x'],
                    'y': row[f'{tracking_player_id}_y']
                })
        exploded_df = pd.DataFrame(exploded_rows)
        return exploded_df

    df_tracking = transform_tracking()

    # x_cols = [col for col in df_tracking.columns if col.endswith("_x")]
    # new_x_cols = [f"{col}_new" for col in x_cols]
    # y_cols = [col for col in df_tracking.columns if col.endswith("_y")]
    # new_y_cols = [f"{col}_new" for col in y_cols]
    df_tracking["x"] = (df_tracking["x"] - 0.5) * 105
    df_tracking["y"] = (df_tracking["y"] - 0.5) * 68

    for period, df_tracking_period in df_tracking.groupby("period_id"):
        period2team2x[period] = {}
        for team in team2formation:
            # team_x_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_x_new")]
            # team_y_cols = [col for col in df_tracking.columns if col.startswith(f"{team}_") and col.endswith("_y_new")]
            # new_team_x_cols = [f"{col}_norm" for col in team_x_cols]
            # new_team_y_cols = [f"{col}_norm" for col in team_y_cols]

            i_period = df_tracking["period_id"] == period
            i_team = df_tracking["team"] == team

            # df_tracking.loc[i_period & i_team, "x_norm"] = df_tracking.loc[i_period & i_team, "x"] * period2team2attacking_direction[period][team.lower()]
            # df_tracking.loc[i_period & i_team, "y_norm"] = df_tracking.loc[i_period & i_team, "y"] * period2team2attacking_direction[period][team.lower()]

            # df_tracking.loc[i_period, new_team_x_cols] = df_tracking.loc[i_period, team_x_cols].values * period2team2attacking_direction[period][team]
            #
            # df_tracking.loc[i_period, new_team_y_cols] = df_tracking.loc[i_period, team_y_cols].values * period2team2attacking_direction[period][team]

    # df_tracking = df_tracking.drop(columns=x_cols + y_cols).rename(columns={col: col.replace("_new", "") for col in df_tracking.columns})
    df_tracking = df_tracking.drop(columns=["frame_id"]).rename(columns={"custom_frame_id": "frame_id"})
    df_tracking = df_tracking.dropna(axis=1, how="all")

    return df_tracking, df_events, player2team, team2players


@st.cache_resource
def get_preprocessed_tracking_and_event_data():
    df_tracking = load_raw_tracking_data_single_match(1)
    df_events = load_event_data_single_match(1)

    df_tracking, df_events, player2team, team2players = preprocess_tracking_and_event_data(df_tracking, df_events)

    return df_tracking, df_events, player2team, team2players


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

    # df_tracking = load_raw_tracking_data_single_match(1)
    # df_events = load_event_data_single_match(1)

    df_tracking, df_events, player2team, team2players = get_preprocessed_tracking_and_event_data()
    # team2players = df_tracking.groupby("team")["player"].unique().to_dict()
    player2position = df_events.set_index("From")["from_position"].to_dict() | df_events.set_index("To")["to_position"].to_dict()
    team2color = {"Home": "red", "Away": "blue"}

    with st.expander("View data"):
        st.write("Event data")
        st.write(df_events)
        st.write("Tracking data (first 500 rows)")
        st.write(df_tracking.head(500))

    df_passes = df_events[df_events["Type"] == "PASS"]

    def distance_to_goal(x_norm, y_norm, which):
        x_goal = -52.5 if which == "own" else 52.5

        return np.linalg.norm([x_norm - x_goal, y_norm], axis=0)

        i_goal_direct_line = np.abs(y_norm) < 7.32 / 2
        i_above_goal = y_norm > 7.32 / 2
        i_below_goal = y_norm < -7.32 / 2
        return np.where(i_goal_direct_line, np.abs(x_norm - x_goal),
                        np.where(i_above_goal,
                                 np.linalg.norm([x_norm - x_goal], y_norm - 7.32/2),
                                 np.linalg.norm([x_norm - x_goal], y_norm + 7.32 / 2)))

        # return np.where(y_norm < -3.66, np.linalg.norm([x_norm - x_goal, y_norm + 3.66], axis=0), np.where(y_norm > 3.66, np.linalg.norm([x_norm - x_goal, y_norm - 3.66], axis=0), x_norm - x_goal))

        # if abs(y_norm) <= 3.66:
        #     return x_norm - x_goal
        # else:
        #     if y_norm > 3.66:
        #         return np.linalg.norm([x_norm - x_goal, y_norm - 3.66])
        #     else:
        #         return np.linalg.norm([x_norm - x_goal, y_norm + 3.66])

    # df_passes["distance_to_own_goal"] = df_passes.apply(lambda x: distance_to_goal(x["x_norm"], x["y_norm"], "own"), axis=1)
    # df_passes["distance_to_opponent_goal"] = df_passes.apply(lambda x: distance_to_goal(x["x_norm"], x["y_norm"], "opponent"), axis=1)
    # df_passes["pass_distance_to_own_goal"] = df_passes.apply(lambda x: distance_to_goal(x["x_norm"], x["y_norm"], "own"), axis=1)
    # df_passes["pass_end_distance_to_own_goal"] = df_passes.apply(lambda x: distance_to_goal(x["x_end_norm"], x["y_end_norm"], "own"), axis=1)
    df_passes["pass_distance_to_attacked_goal"] = distance_to_goal(df_passes["x_norm"], df_passes["y_norm"], "opp")
    df_passes["pass_end_distance_to_attacked_goal"] = distance_to_goal(df_passes["x_end_norm"], df_passes["y_end_norm"], "opp")
    # df_passes["pass_distance_to_defended_goal"] = distance_to_goal(df_passes["x_norm"], df_passes["y_norm"], "opp")
    # df_passes["pass_end_distance_to_defended_goal"] = distance_to_goal(df_passes["x_end_norm"], df_passes["y_end_norm"], "opp")

    df_tracking_passes = df_tracking[df_tracking["frame_id"].isin(df_passes["Start Frame"]) | df_tracking["frame_id"].isin(df_passes["End Frame"])]

    fr2adir = df_passes.set_index("Start Frame")["attacking_direction"].to_dict()
    fr2adir.update(df_passes.set_index("End Frame")["attacking_direction"].to_dict())

    df_tracking_passes["attacking_direction"] = df_tracking_passes["frame_id"].apply(lambda x: fr2adir.get(x, None))
    df_tracking_passes["x_norm"] = df_tracking_passes["x"] * df_tracking_passes["attacking_direction"]
    df_tracking_passes["y_norm"] = df_tracking_passes["y"] * df_tracking_passes["attacking_direction"]

    i_notna = df_tracking_passes["x_norm"].notna()
    # df_tracking_passes.loc[i_notna, "distance_to_attacked_goal"] = distance_to_goal(df_tracking_passes.loc[i_notna, "x_norm"], df_tracking_passes.loc[i_notna, "y_norm"], "own")
    df_tracking_passes.loc[i_notna, "distance_to_defended_goal"] = distance_to_goal(df_tracking_passes.loc[i_notna, "x_norm"], df_tracking_passes.loc[i_notna, "y_norm"], "opp")

    ### offensive xT metrics
    for pass_index, p4ss in df_passes.iterrows():
        i_p4ss = df_tracking_passes["frame_id"] == p4ss["Start Frame"]
        i_p4ss_end = df_tracking_passes["frame_id"] == p4ss["End Frame"]

        assert len(df_tracking_passes[i_p4ss]) == len(df_tracking_passes[i_p4ss_end])

        defending_team = "Home" if p4ss["Team"] == "Away" else "Away"
        i_defenders = df_tracking_passes["team"] == defending_team

        # ball_x = p4ss["x_norm"]
        # ball_y = p4ss["y_norm"]
        # ball_x_end = p4ss["x_end_norm"]
        # ball_y_end = p4ss["y_end_norm"]

        ball_distance_to_attacked_goal = p4ss["pass_distance_to_attacked_goal"]
        ball_end_distance_to_attacked_goal = p4ss["pass_end_distance_to_attacked_goal"]

        df_tracking_passes.loc[i_p4ss & i_defenders, "is_closer_to_own_goal_than_ball"] = df_tracking_passes.loc[i_p4ss & i_defenders, "distance_to_defended_goal"] < ball_distance_to_attacked_goal
        df_tracking_passes.loc[i_p4ss_end & i_defenders, "is_closer_to_own_goal_than_ball"] = df_tracking_passes.loc[i_p4ss_end & i_defenders, "distance_to_defended_goal"] < ball_end_distance_to_attacked_goal

        df_tracking_passes.loc[i_p4ss & i_defenders, "is_outplayed"] = (df_tracking_passes.loc[i_p4ss & i_defenders, "is_closer_to_own_goal_than_ball"].values & (~df_tracking_passes.loc[i_p4ss_end & i_defenders, "is_closer_to_own_goal_than_ball"].values)).astype(bool)
        df_tracking_passes.loc[i_p4ss & i_defenders, "is_inplayed"] = ((~df_tracking_passes.loc[i_p4ss & i_defenders, "is_closer_to_own_goal_than_ball"].values) & df_tracking_passes.loc[i_p4ss_end & i_defenders, "is_closer_to_own_goal_than_ball"].values).astype(bool)

        # st.write("df_tracking_passes.loc[i_p4ss & i_defenders]")
        # st.write(df_tracking_passes.loc[i_p4ss & i_defenders])
        # st.write("df_tracking_passes.loc[i_p4ss_end & i_defenders]")
        # st.write(df_tracking_passes.loc[i_p4ss_end & i_defenders])

        n_outplayed_defenders_old = df_tracking_passes.loc[i_p4ss_end & i_defenders,
        "is_closer_to_own_goal_than_ball"].sum() - df_tracking_passes.loc[i_p4ss & i_defenders,
        "is_closer_to_own_goal_than_ball"].sum()
        n_outplayed_defenders = df_tracking_passes.loc[i_p4ss & i_defenders, "is_outplayed"].sum()
        n_inplayed_defenders = df_tracking_passes.loc[i_p4ss & i_defenders, "is_inplayed"].sum()

        df_passes.loc[pass_index, "n_outplayed_defenders_old"] = n_outplayed_defenders_old
        df_passes.loc[pass_index, "n_outplayed_defenders"] = n_outplayed_defenders
        df_passes.loc[pass_index, "n_inplayed_defenders"] = n_inplayed_defenders

    ### Offensive Passing networks
    with st.expander("Offensive passing networks"):
        xt_metric_positive = st.selectbox("xT metric positive", [None, "n_outplayed_defenders"], format_func=lambda x: {None: "0", "n_outplayed_defenders": "Outplayed Defenders"}[x], index=1)
        xt_metric_negative = st.selectbox("xT metric negative", [None, "n_inplayed_defenders"], format_func=lambda x: {None: "0", "n_inplayed_defenders": "Inplayed Defenders"}[x], index=0)

        # negative_weight_mode = st.selectbox("Adjust NEGATIVE OFFENSIVE weights", ["keep", "remove", "clip"], format_func=lambda x: {
        #     "keep": "Keep negative weights", "remove": "Remove passes with negative weight", "clip": "Set negative weights to 0"
        # }[x], index=0)
        # if xt_metric is not None:
        #     if negative_weight_mode == "remove":
        #         df_passes = df_passes[df_passes[xt_metric] >= 0]
        #     elif negative_weight_mode == "clip":
        #         df_passes[xt_metric] = df_passes[xt_metric].clip(lower=0)
        #

        if xt_metric_positive is None and xt_metric_negative is None:
            df_passes["xt_metric"] = 0
        elif xt_metric_positive is not None and xt_metric_negative is None:
            df_passes["xt_metric"] = df_passes[xt_metric_positive]
        elif xt_metric_positive is None and xt_metric_negative is not None:
            df_passes["xt_metric"] = -df_passes[xt_metric_negative]
        else:
            df_passes["xt_metric"] = df_passes[xt_metric_positive] - df_passes[xt_metric_negative]

        for team, df_passes_team in df_passes.groupby(["Team"]):
            team = team[0]
            st.write(f"### {team}")

            for formation, df_passes_team_formation in df_passes_team.groupby("live_formation"):
                most_common_formation = df_passes_team["live_formation"].value_counts().idxmax()

                if formation != most_common_formation:
                    continue

                st.write(f"#### {formation}")

                # df_passes_team_filtered = df_passes_team[
                #     df_passes_team["from_position"].notna() & df_passes_team["to_position"].notna() & (df_passes_team["live_formation"] == most_common_formation)
                # ]

                df_nodes, df_edges = defensive_network.models.passing_network.get_passing_network(
                    df_passes_team_formation,
                    x_col="x_norm",  # column with x position of the pass
                    y_col="y_norm",  # column with y position of the pass
                    from_col="from_position",  # column with unique (!) ID or name of the player/position/... who passes the ball
                    to_col="to_position",  # column with unique (!) ID or name of the player/position/... who receives the ball
                    x_to_col="x_end_norm",  # column with x position of the pass target
                    y_to_col="y_end_norm",  # column with y position of the pass
                    value_col="xt_metric",  # column with the value of the pass (e.g. expected threat)
                )

                # df_nodes["other_value"] = 0

                # fig = defensive_network.models.passing_network.plottt(df_nodes, df_edges)
                fig, ax = defensive_network.models.passing_network.plot_passing_network(
                    df_nodes=df_nodes,
                    df_edges=df_edges,
                    show_colorbar=False,
                    node_size_multiplier=30,
                    max_color_value_edges=40,
                    max_color_value_nodes=80,
                    label_col="xt_metric",
                    label_format_string="{:.0f}",
                    arrow_color_col="value_passes",
                    node_color_col="value_passes",

                    # colormap=matplotlib.cm.get_cmap("viridis"),
                )

                st.write(fig)

                st.write("Nodes")
                st.write(df_nodes)
                st.write("Edges")
                st.write(df_edges)

    with st.expander("Defensive passing networks"):
        # defensive_weight_mode = st.selectbox("DEFENSIVE weights method", ["outplayed", "outplayed_inplayed"], format_func=lambda x: {"None": "0", "outplayed": "Being outplayed: Only blame", "outplayed_inplayed": "Being outplayed and inplayed: Blame and credit"}[x], index=0)
        defensive_weight_mode_blame = st.selectbox("DEFENSIVE weights bad defending", [None, "outplayed"], format_func=lambda x: {None: "0", "outplayed": "Being out-played"}[x], index=1)
        defensive_weight_mode_reward = st.selectbox("Defensive weights good defending", [None, "inplayed"], format_func=lambda x: {None: "0", "inplayed": "Being in-played"}[x], index=1)

        for team, df_passes_team in df_passes.groupby(["Team"]):
            if defensive_weight_mode_blame is None and defensive_weight_mode_reward is None:
                break
            # if team == "Home":
            #     continue
            team = team[0]
            st.write("### ", team)
            other_team = "Home" if team == "Away" else "Away"
            most_common_formation = df_passes_team["live_formation"].value_counts().idxmax()
            df_passes_team_filtered = df_passes_team[
                df_passes_team["from_position"].notna() & df_passes_team["to_position"].notna() &
                (df_passes_team["live_formation"] == most_common_formation)
            ]

            for defender in team2players[other_team]:
                defender_with_pos = f"{defender} ({player2position.get(defender, 'unknown position')})"

                # st.write(f"#### {defender_with_pos}")

                df_tracking_defender_passes = df_tracking_passes[(df_tracking_passes["player"] == defender) & (df_tracking_passes["frame_id"].isin(df_passes_team_filtered["Start Frame"]))]
                df_passes_team_filtered_defender = df_passes_team_filtered[df_passes_team_filtered["Start Frame"].isin(df_tracking_defender_passes["frame_id"])].copy()
                df_passes_team_filtered_defender = df_passes_team_filtered_defender.drop_duplicates(subset=["Start Frame"])

                # st.write("df_tracking_defender_passes", df_tracking_defender_passes.shape)
                # st.write(df_tracking_defender_passes)
                # st.write("df_passes_team_filtered_defender", df_passes_team_filtered_defender.shape)
                # st.write(df_passes_team_filtered_defender)

                assert len(df_tracking_defender_passes) == len(df_passes_team_filtered_defender)

                df_passes_team_filtered_defender["is_outplayed"] = df_tracking_defender_passes["is_outplayed"].values.astype(int)
                df_passes_team_filtered_defender["is_inplayed"] = df_tracking_defender_passes["is_inplayed"].values.astype(int)

                df_passes_team_filtered_defender["score"] = 0
                if defensive_weight_mode_blame == "outplayed":
                    df_passes_team_filtered_defender["score"] += df_passes_team_filtered_defender["is_outplayed"]
                if defensive_weight_mode_reward == "inplayed":
                    df_passes_team_filtered_defender["score"] -= df_passes_team_filtered_defender["is_inplayed"]

                # st.write("df_passes_team_filtered_defender")
                # st.write(df_passes_team_filtered_defender)

                df_nodes, df_edges = defensive_network.models.passing_network.get_passing_network(
                    df_passes_team_filtered_defender,
                    x_col="x_norm",  # column with x position of the pass
                    y_col="y_norm",  # column with y position of the pass
                    from_col="from_position",  # column with unique (!) ID or name of the player/position/... who passes the ball
                    to_col="to_position",  # column with unique (!) ID or name of the player/position/... who receives the ball
                    x_to_col="x_end_norm",  # column with x position of the pass target
                    y_to_col="y_end_norm",  # column with y position of the pass
                    value_col="score",  # column with the value of the pass (e.g. expected threat)
                )

                # st.write(df_nodes)
                # st.write(df_edges)
                df_edges = df_edges[df_edges["value_passes"] != 0]
                fig, ax = defensive_network.models.passing_network.plot_passing_network(
                    df_nodes=df_nodes,
                    df_edges=df_edges,
                    show_colorbar=False,
                    node_size_multiplier=30,
                    max_color_value_edges=5,
                    min_color_value_edges=-5,
                    max_color_value_nodes=5,
                    min_color_value_nodes=-5,
                    node_color_col="value_passes",
                    arrow_color_col="value_passes",
                    label_col="value_passes",
                    label_format_string="{:.0f}",
                    annotate_top_n_edges=5,
                    # colormap=matplotlib.cm.get_cmap("PuBuGn"),
                    colormap=matplotlib.colormaps.get_cmap("coolwarm"),
                )
                plt.title(defender_with_pos)

                st.write(fig)

        # else:
        #     raise NotImplementedError()

    ### Pass plotting
    with st.expander("Plot Passes"):
        sort_by = st.selectbox("Sort by", ["Start Time [s]", "n_outplayed_defenders", "n_inplayed_defenders"], index=1, format_func=lambda x: {"Start Time [s]": "Start Time", "n_outplayed_defenders": "Outplayed Defenders", "n_inplayed_defenders": "Inplayed Defenders"}[x])
        ascending = True if sort_by == "Start Time [s]" else False
        show_top_n = st.slider("Show top N passes", 1, len(df_passes), 10)
        df_passes = df_passes.sort_values(sort_by, ascending=ascending)

        # selected_passes = st.multiselect("Select pass", df_passes.index, format_func=lambda x: df_passes.loc[x, "event_string"], default=df_passes.index[0:10].tolist())
        selected_passes = []

        if selected_passes == []:
            selected_passes = df_passes.index

        pass_nr = 0
        for i_pass in selected_passes:
            if pass_nr >= show_top_n:
                break
            pass_nr += 1

            p4ss = df_passes.loc[i_pass]
            st.write(p4ss["event_string"])

            st.write("Out-played Defenders", int(p4ss["n_outplayed_defenders"]) if not np.isnan(p4ss["n_outplayed_defenders"]) else 0, ", In-played Defenders", int(p4ss["n_inplayed_defenders"]) if not np.isnan(p4ss["n_inplayed_defenders"]) else 0)

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

                for team in team2color:
                    # x_cols = [col for col in df_tracking.columns if col.endswith("_x") and col.startswith(team)]
                    # y_cols = [col for col in df_tracking.columns if col.endswith("_y") and col.startswith(team)]
                    # player_names = [f"{col.rsplit('_', -1)[1]}" for col in x_cols]
                    # x_pos = df_tracking.loc[i_tr, x_cols].values[0]
                    # y_pos = -df_tracking.loc[i_tr, y_cols].values[0]

                    df_fr_team = df_tracking.loc[i_tr & (df_tracking["team"] == team)]
                    x_pos = df_fr_team["x"].values
                    y_pos = -df_fr_team["y"].values
                    player_names = df_fr_team["player"].values

                    pitch.scatter(x_pos, y_pos+0., color=team2color[team], ax=ax)

                    # plot names
                    for i, _ in enumerate(x_pos):
                        plt.gca().annotate(player_names[i], (x_pos[i], y_pos[i]-3.5), ha="center", va="bottom")

                i_ball = df_tracking["player"] == "ball"
                pitch.scatter(df_tracking.loc[i_tr & i_ball, "x"], -df_tracking.loc[i_tr & i_ball, "y"], color="black", ax=ax, marker="x", s=200)

                # Display the plot
                plt.show()
                columns[int(is_target)].write(fig)

    profiler.stop()


if __name__ == '__main__':
    main()
