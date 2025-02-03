import importlib
import os

import accessible_space
import pandas as pd
import streamlit as st

import defensive_network.models
from ..utility import general
import defensive_network.models.framerate


importlib.reload(defensive_network.models)


def get_meta_fpath(base_path):
    return os.path.abspath(os.path.join(base_path, "meta.csv"))


def get_lineup_fpath(base_path):
    return os.path.abspath(os.path.join(base_path, "lineup.csv"))


def get_event_fpath(base_path, slugified_match_string):
    return os.path.abspath(os.path.join(base_path, "events", f"{slugified_match_string}.csv"))


def get_tracking_fpath(base_path, slugified_match_string):
    return os.path.abspath(os.path.join(base_path, "tracking", f"{slugified_match_string}.parquet"))


def get_preprocessed_tracking_fpath(base_path, slugified_match_string):
    path = os.path.abspath(os.path.join(base_path, "preprocessed/tracking", f"{slugified_match_string}.parquet"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


@st.cache_resource
def get_all_meta(base_path):
    return pd.read_csv(get_meta_fpath(base_path))


@st.cache_resource
def get_all_lineups(base_path):
    return pd.read_csv(get_lineup_fpath(base_path))


@st.cache_resource
def get_match_data(base_path, slugified_match_string, xt_model="ma2024", expected_receiver_model="power2017", formation_model="average_pos"):
    event_fpath = get_event_fpath(base_path, slugified_match_string)
    tracking_fpath = get_tracking_fpath(base_path, slugified_match_string)

    df_meta = get_all_meta(base_path)
    match_id = df_meta[df_meta["slugified_match_string"] == slugified_match_string]["match_id"].iloc[0]
    df_lineup = get_all_lineups(base_path)
    df_lineup_match = df_lineup[df_lineup["match_id"] == match_id]
    playerid2name = df_lineup_match[['player_id', 'short_name']].set_index('player_id').to_dict()['short_name']
    team2name = df_lineup_match.loc[df_lineup_match["team_name"].notna(), ['team_id', 'team_name']].set_index('team_id').to_dict()['team_name']
    
    df_event = pd.read_csv(event_fpath)
    assert df_event["datetime_event"].notna().all()
    df_event["datetime_event"] = pd.to_datetime(df_event["datetime_event"])
    df_event = df_event.sort_values("datetime_event")

    import importlib
    importlib.reload(defensive_network.models.framerate)
    estimated_fps = defensive_network.models.framerate.estimate_framerate_by_linear_slope(df_event)
    assert estimated_fps == 25

    # replace 1900-01-01 with NaN
    df_event["player_name_1"] = df_event["player_id_1"].map(playerid2name)
    df_event["player_name_2"] = df_event["player_id_2"].map(playerid2name)
    df_event["team_name_1"] = df_event["team_id_1"].map(team2name)
    df_event["team_name_2"] = df_event["team_id_2"].map(team2name)

    sorted_sections = list(general.uniquify_keep_order(df_event["section"].dropna()))
    i_whistle = df_event["event_subtype"] == "final_whistle"
    assert len(sorted_sections) == len(df_event.loc[i_whistle]), f"len(sorted_sections)={len(sorted_sections)} != len(df_event.loc[i_whistle])={len(df_event.loc[i_whistle])}"
    df_event.loc[i_whistle, "section"] = sorted_sections

    whistle_gap_seconds = 300
    assert len(sorted_sections) == 2

    whistle_first_half = df_event.loc[i_whistle, "datetime_event"].iloc[0]
    border_seconds = whistle_first_half + pd.Timedelta(seconds=whistle_gap_seconds)
    i_first_half = df_event["datetime_event"] < border_seconds
    df_event.loc[i_first_half, "section"] = sorted_sections[0]
    df_event.loc[~i_first_half, "section"] = sorted_sections[1]

    # Fill missing frames
    df_kickoffs = df_event.loc[df_event["event_subtype"] == "kick_off", :]
    section_times = df_kickoffs.groupby("section")["datetime_event"].agg("min")
    i_min_section_time = df_event.apply(lambda x: x["datetime_event"] == section_times.get(x["section"], None), axis=1)
    df_event.loc[i_min_section_time, "frame"] = df_event.loc[i_min_section_time, "frame"].fillna(0)  # first kickoff in a section is 0 (if not otherwise given), TODO shouldnt be necessary as frame is filled by interpolation
    df_earliest_kickoffs = df_kickoffs.loc[i_min_section_time]

    section_start_frames = df_event.loc[i_min_section_time, "frame"].values
    assert (section_start_frames == 0).all(), f"section_start_frames={section_start_frames}, need to adapt to that"

    i_frame_nan = df_event["frame"].isna()
    df_event.loc[i_frame_nan, "frame"] = df_event.loc[i_frame_nan].apply(lambda x: (x["datetime_event"] - section_times[x["section"]]).total_seconds() * estimated_fps, axis=1).round().astype(int)
    assert df_event["frame"].notna().all()

    section_end_times = df_event.loc[i_whistle].groupby("section")["datetime_event"].agg("max")
    # Fill section info
    for section_nr, (section, section_time) in enumerate(section_times.items()):
        section_end_time = section_end_times[section]
        i_section = df_event["datetime_event"].between(section_time, section_end_time, inclusive="both") | (df_event["section"] == section)
        df_event.loc[i_section, "section"] = section
        df_event.loc[i_section, "seconds_since_section_start"] = (df_event.loc[i_section, "datetime_event"] - section_time).dt.total_seconds()
        df_event.loc[i_section, "mmss"] = df_event.loc[i_section, "seconds_since_section_start"].apply(lambda x: defensive_network.utility.general.seconds_since_period_start_to_mmss(x, section_nr))

    df_event["full_frame"] = df_event["section"].str.cat(df_event["frame"].astype(float).astype(str), sep="-")
    df_event["full_frame_rec"] = df_event["section"].str.cat(df_event["frame_rec"].astype(str), sep="-")

    @st.cache_resource
    def _get_parquet(fpath):
        return pd.read_parquet(fpath)

    df_tracking = _get_parquet(tracking_fpath)  # pd.read_parquet(tracking_fpath)

    # interpolate missing tracking values
    df_tracking["x_tracking"] = df_tracking["x_tracking"].interpolate()  # TODO do properly
    df_tracking["y_tracking"] = df_tracking["y_tracking"].interpolate()

    df_tracking["full_frame"] = df_tracking["section"].str.cat(df_tracking["frame"].astype(float).astype(str), sep="-")
    # i_event_frames_not_in_tracking_data = ~df_event["full_frame"].isin(df_tracking["full_frame"])
    # df_event.loc[i_event_frames_not_in_tracking_data, "frame"] = None

    df_event["event_string"] = df_event["mmss"].astype(str) + ": " + df_event["event_subtype"].astype(str).where(df_event["event_subtype"].notna(), df_event["event_type"]) + " " + df_event["player_name_1"].astype(str) + " (" + df_event["team_name_1"].astype(str) + ") -> " + df_event["player_name_2"].astype(str) + " (" + df_event["team_name_2"].astype(str) + ") (" + df_event["event_outcome"].astype(str) + ")"

    i_pass = df_event["event_type"] == "pass"
    i_out = df_event["player_id_2"].isna()

    df_tracking["player_name"] = df_tracking["player_id"].map(playerid2name)

    default_pass_frames = 0.8 / estimated_fps

    df_event["frame_rec"] = df_event["frame_rec"].fillna(df_event["frame"] + default_pass_frames).round()
    assert df_event["frame"].notna().all()
    assert (df_event["frame"] + default_pass_frames).notna().all()
    assert df_event["frame_rec"].notna().all()

    def get_target_x_y(row, df_tracking_indexed, receiver):
        if pd.isna(receiver):
            receiver = "BALL"
        try:
            receiver_frame = df_tracking_indexed.loc[(row["frame_rec"], row["section"], receiver)]
            return receiver_frame["x_tracking"], receiver_frame["y_tracking"]
        except KeyError as e:
            receiver = "BALL"
            try:
                receiver_frame = df_tracking_indexed.loc[(row["frame_rec"], row["section"], receiver)]
                return receiver_frame["x_tracking"], receiver_frame["y_tracking"]
            except KeyError as e:
                # get closest frame
                df_tracking_player = df_tracking_indexed.loc[(slice(None), row["section"], receiver), :]
                closest_frame = df_tracking_player.reset_index()["frame"].sub(row["frame_rec"]).abs().idxmin()
                x_tracking = df_tracking_player.loc[closest_frame, "x_tracking"].iloc[0]
                y_tracking = df_tracking_player.loc[closest_frame, "y_tracking"].iloc[0]
                return x_tracking, y_tracking

        #     return get_target_x_y(row, df_tracking_indexed, "BALL")

    df_tracking_indexed = df_tracking.set_index(["frame", "section", "player_id"])

    with st.spinner("Calculating x, y from tracking data..."):
        keys = df_event.apply(lambda row: get_target_x_y(row, df_tracking_indexed, row["player_id_1"]), axis=1)
    df_event[["x_event", "y_event"]] = pd.DataFrame(keys.tolist(), index=df_event.index)
    df_event["x_event"] = df_event["x_event"].fillna(df_event["x_tracking_player_1"])
    df_event["y_event"] = df_event["y_event"].fillna(df_event["y_tracking_player_1"])

    with st.spinner("Calculating target x, y from tracking data..."):
        keys_target = df_event.apply(lambda row: get_target_x_y(row, df_tracking_indexed, row["player_id_2"]), axis=1)
    df_event[["x_target", "y_target"]] = pd.DataFrame(keys_target.tolist(), index=df_event.index)
    df_event["x_target"] = df_event["x_target"].fillna(df_event["x_tracking_player_2"])
    df_event["y_target"] = df_event["y_target"].fillna(df_event["y_tracking_player_2"])
    assert df_event["x_target"].notna().all()

    import importlib
    import accessible_space.interface
    importlib.reload(accessible_space.interface)
    df_tracking["playing_direction"] = accessible_space.infer_playing_direction(
        df_tracking, team_col="team_id", period_col="section", team_in_possession_col="ball_poss_team_id", x_col="x_tracking"
    )
    df_tracking["x_norm"] = df_tracking["x_tracking"] * df_tracking["playing_direction"]
    df_tracking["y_norm"] = df_tracking["y_tracking"] * df_tracking["playing_direction"]
    team2section2playing_direction = df_tracking.groupby(["ball_poss_team_id", "section"])["playing_direction"].first()

    df_event["playing_direction"] = df_event[['team_id_1', 'section']].apply(lambda x: team2section2playing_direction.loc[x['team_id_1'], x['section']] if not pd.isna(x['team_id_1']) else None, axis=1)
    df_event["x_norm"] = df_event["x_event"] * df_event["playing_direction"]
    df_event["y_norm"] = df_event["y_event"] * df_event["playing_direction"]

    df_event["x_target_norm"] = df_event["x_target"] * df_event["playing_direction"]  # TODO move x_target calculation here
    df_event["y_target_norm"] = df_event["y_target"] * df_event["playing_direction"]

    df_event["is_successful"] = df_event["event_outcome"] == "successfully_completed"

    xt_res = defensive_network.models.get_expected_threat(df_event, xt_model=xt_model)
    df_event["pass_xt"] = xt_res.delta_xt

    i_pass = df_event["event_type"] == "pass"
    df_event.loc[i_pass, "pass_is_intercepted"] = (df_event.loc[i_pass, "event_outcome"] == "unsuccessful") & ~pd.isna(df_event.loc[i_pass, "player_id_2"])
    df_event.loc[i_pass, "pass_is_out"] = (df_event.loc[i_pass, "event_outcome"] == "unsuccessful") & pd.isna(df_event.loc[i_pass, "player_id_2"])
    df_event.loc[i_pass, "pass_is_successful"] = df_event.loc[i_pass, "event_outcome"] == "successfully_completed"

    # assert all three exist
    assert df_event.loc[i_pass, "pass_is_intercepted"].sum() > 0
    assert df_event.loc[i_pass, "pass_is_out"].sum() > 0
    assert df_event.loc[i_pass, "pass_is_successful"].sum() > 0

    df_event.loc[i_pass, "outcome"] = df_event.loc[i_pass].apply(lambda x: "successful" if x["pass_is_successful"] else ("intercepted" if x["pass_is_intercepted"] else "out"), axis=1)
    xr_result = defensive_network.models.get_expected_receiver(df_event.loc[i_pass, :], df_tracking)
    df_event.loc[i_pass, "expected_receiver"] = xr_result.expected_receiver
    df_event.loc[i_pass, "expected_receiver_name"] = df_event.loc[i_pass, "expected_receiver"].map(playerid2name)
    df_event.loc[i_pass, "is_intercepted"] = df_event.loc[i_pass, "outcome"] == "intercepted"

    df_tracking = df_tracking[df_tracking["team_id"] != "referee"]

    df_tracking = defensive_network.models.add_velocity(df_tracking)

    with st.spinner("Inferring formation"):
        res = defensive_network.models.detect_formation(df_tracking, model=formation_model)
        df_tracking["role"] = res.role
        df_tracking["role_name"] = res.role_name
        df_tracking["formation_instance"] = res.formation_instance

    ball_player = "BALL"
    i_ball = df_tracking["player_id"] == ball_player
    df_tracking.loc[i_ball, "role"] = ball_player
    df_tracking.loc[i_ball, "role_name"] = ball_player

    fpath_preprocessed_tracking = get_preprocessed_tracking_fpath(base_path, slugified_match_string)
    df_tracking.to_parquet(fpath_preprocessed_tracking, index=False)

    # Add role to events
    frameplayerrole = df_tracking.groupby(["frame", "player_id"])["role"].first().to_dict()
    role2name = df_tracking.groupby("role")["role_name"].first().to_dict()
    role2name = {role: role_name.replace("def", "off") for role, role_name in role2name.items()}
    frameplayerrole = {k: v.replace("def", "off") for k, v in frameplayerrole.items() if v is not None}

    df_tracking["role"] = df_tracking["role"].apply(lambda x: str(x).replace("def", "off") if not pd.isna(x) else x)
    df_tracking["role_name"] = df_tracking.apply(lambda x: role2name.get(x["role"], x["role"]), axis=1)

    frameplayerrole = df_tracking.groupby(["frame", "player_id"])["role"].first().to_dict()
    role2name = df_tracking.groupby("role")["role_name"].first().to_dict()
    role2name = {role: role_name.replace("def", "off") for role, role_name in role2name.items()}
    frameplayerrole = {k: v.replace("def", "off") for k, v in frameplayerrole.items() if v is not None}

    with st.spinner("role_1"):
        df_event["role_1"] = df_event[["frame", "player_id_1"]].apply(lambda x: frameplayerrole.get((x["frame"], x["player_id_1"]), None), axis=1)
    with st.spinner("role_2"):
        df_event["role_2"] = df_event[["frame", "player_id_2"]].apply(lambda x: frameplayerrole.get((x["frame"], x["player_id_2"]), None), axis=1)
    df_event["role_name_1"] = df_event["role_1"].map(role2name)
    df_event["role_name_2"] = df_event["role_2"].map(role2name)
    df_event["expected_receiver_role"] = df_event[["frame", "expected_receiver"]].apply(lambda x: frameplayerrole.get((x["frame"], x["expected_receiver"]), None), axis=1)
    df_event["expected_receiver_role_name"] = df_event["expected_receiver_role"].map(role2name)

    return df_tracking, df_event
