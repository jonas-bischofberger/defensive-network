import datetime
import gc
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import seaborn as sns

import importlib
import accessible_space.interface

import traceback
import inspect

import streamlit as st
import memory_profiler

def custom_write(*args):
    print(*args)

# st.write = custom_write

import defensive_network.parse.dfb.cdf
import defensive_network.utility.general
import defensive_network.models.formation
import defensive_network.models.involvement
import defensive_network.models.passing_network
import defensive_network.models.average_position
import defensive_network.models.expected_receiver
import defensive_network.utility.stats

pd.options.mode.chained_assignment = None

importlib.reload(defensive_network.parse.dfb.cdf)

def get_metrics(df_event, df_tracking, series_meta):
    dfgs = []

    # Minutes
    df_possession = df_tracking.groupby(["section"]).apply(lambda df_section : df_section.groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "nunique"}))
    df_possession = df_possession.reset_index().groupby(["ball_poss_team_id", "ball_status"]).agg({"frame": "sum"})
    df_possession["frame"] = df_possession["frame"] / (series_meta["fps"].iloc[0] * 60)
    df_possession = df_possession.rename(columns={"frame": "minutes"}).reset_index().pivot(index="ball_poss_team_id", columns="ball_status", values="minutes").drop(columns=[0])
    df_possession.columns = [f"net_minutes_in_possession"]
    df_possession["net_minutes_opponent_in_possession"] = df_possession["net_minutes_in_possession"].values[::-1]
    df_possession["net_minutes"] = df_possession["net_minutes_opponent_in_possession"] + df_possession["net_minutes_in_possession"]
    total_minutes = (
        df_tracking.groupby("section")["datetime_tracking"]
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
        .sum()
    )
    df_possession["total_minutes"] = total_minutes

    # Points
    teams = df_event["team_id_1"].dropna().unique()
    assert len(teams) == 2
    df_event["team_id_1"] = pd.Categorical(df_event["team_id_1"], teams)
    df_goals = df_event[(df_event["event_type"] == "shot") & (df_event["event_outcome"] == "successful")]
    dfg_result = df_goals.groupby("team_id_1", observed=False).agg(goals=("event_id", "count"))
    dfg_result["goals_against"] = dfg_result["goals"].iloc[::-1].values

    def calc_points(goals, goals_against):
        if goals > goals_against:
            return 3
        elif goals == goals_against:
            return 1
        elif goals < goals_against:
            return 0
        else:
            raise ValueError

    dfg_result["points"] = dfg_result.apply(lambda x: calc_points(x["goals"], x["goals_against"]), axis=1)
    dfgs.append(dfg_result)

    # xG
    dfg_xg = df_event.groupby("team_id_1", observed=False).agg(xg=("xg", "sum"))
    dfg_xg["xg_against"] = dfg_xg["xg"].iloc[::-1].values
    dfgs.append(dfg_xg)

    # Pass xT
    df_passes = df_event[df_event["event_type"] == "pass"]
    dfg_xt_total = df_passes.groupby("team_id_1", observed=False).agg(total_xt=("pass_xt", "sum"))
    dfg_xt_total["total_xt_against"] = dfg_xt_total["total_xt"].iloc[::-1].values
    dfgs.append(dfg_xt_total)

    dfg_xt_total_only_positive = df_passes[df_passes["pass_xt"] > 0].groupby("team_id_1", observed=False).agg(total_xt_only_positive=("pass_xt", "sum"))
    dfg_xt_total_only_positive["total_xt_only_positive_against"] = dfg_xt_total_only_positive["total_xt_only_positive"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_positive)

    dfg_xt_total_only_negative = df_passes[df_passes["pass_xt"] < 0].groupby("team_id_1", observed=False).agg(total_xt_only_negative=("pass_xt", "sum"))
    dfg_xt_total_only_negative["total_xt_only_negative_against"] = dfg_xt_total_only_negative["total_xt_only_negative"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_negative)

    dfg_xt_total_only_successful = df_passes[df_passes["event_outcome"] == "successfully_completed"].groupby("team_id_1", observed=False).agg(total_xt_only_successful=("pass_xt", "sum"))
    dfg_xt_total_only_successful["total_xt_only_successful_against"] = dfg_xt_total_only_successful["total_xt_only_successful"].iloc[::-1].values
    dfgs.append(dfg_xt_total_only_successful)

    # Number of passes
    dfg_n_passes = df_passes.groupby("team_id_1", observed=False).agg(passes=("event_id", "count"))
    dfg_n_passes["passes_against"] = dfg_n_passes["passes"].iloc[::-1].values
    dfgs.append(dfg_n_passes)

    # Interceptions
    dfg_interceptions = df_event[df_event["outcome"] == "intercepted"].groupby("team_id_2", observed=False).agg(
        n_interceptions=("event_id", "count")
    )
    dfgs.append(dfg_interceptions)

    # Tackles
    dfg_tackles = df_event[df_event["event_subtype"] == "tackle"].groupby("team_id_1", observed=False).agg(
        n_tackles=("event_id", "count")
    )
    dfgs.append(dfg_tackles)

    dfgs_players = []
    dfg_players_tackles_won = df_event[df_event["event_subtype"] == "tackle"].groupby("player_id_1", observed=False).agg(
        n_tackles_won=("event_id", "count"),
    )
    dfgs_players.append(dfg_players_tackles_won)
    dfg_players_tackles_lost = df_event[df_event["event_subtype"] == "tackle"].groupby("player_id_2", observed=False).agg(
        n_tackles_lost=("event_id", "count"),
    )
    dfgs_players.append(dfg_players_tackles_lost)

    dfg_players = pd.concat(dfgs_players)
    dfg_players["n_tackles"] = dfg_players["n_tackles_won"] + dfg_players["n_tackles_lost"]
    st.write("dfg_players", dfg_players.shape)
    st.write(dfg_players)

    return dfgs


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def _unpack_args(func, arguments):
    parameters = inspect.signature(func).parameters
    parameter_names = [param for param in parameters]
    unpacked_args = []
    default_args = _get_default_args(func)
    if isinstance(arguments, tuple):
        unpacked_args = arguments
    elif isinstance(arguments, dict):
        unpacked_args = tuple(arguments.get(param, default_args.get(param, -123)) for param in parameter_names)
    return unpacked_args
def _process_return_wrapper(func, args, queue, disable_stdout=False):
    try:
        result = func(*args)
        queue.put(result)
        return result
    except Exception as e:
        queue.put("Fail " + str(e))
        msg = repr(e)
        traceback.print_exc()
        return msg


def _f(x, y):
    print(x)
    return y


def run_as_subprocess(func, arguments):
    """
    >>> run_as_subprocess(_f, {"x": 1, "y": 2})
    2
    """
    multiprocessing.set_start_method("spawn", force=True)

    q = multiprocessing.SimpleQueue()

    unpacked_args = _unpack_args(func, arguments)

    p = multiprocessing.Process(target=_process_return_wrapper, args=(func, unpacked_args, q))
    p.start()
    result = q.get()
    p.join()
    return result


# import memory_profiler
# @memory_profiler.profile
def _get_data(all_slugified_match_strings, base_path, xt_model, expected_receiver_model, formation_model, selected_tracking_player_col, selected_tracking_player_name_col, use_tracking_average_position, selected_value_col, selected_player_col, selected_receiver_col, selected_expected_receiver_col, selected_expected_receiver_name_col, selected_receiver_name_col, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius, selected_player_name_col, defender_col, defender_name_col, remove_passes_with_zero_involvement, get_instant_analysis):
    importlib.reload(defensive_network.models.formation)

    overwrite_if_exists = False

    df_meta = defensive_network.parse.dfb.cdf.get_all_meta(base_path)
    st.write("df_meta")
    st.write(df_meta)

    stichtag = None  # datetime.datetime(year=2025, month=2, day=27, hour=0, minute=0, second=0)

    datas = []
    fpaths = {sms: defensive_network.parse.dfb.cdf.get_team_level_analysis_fpath(base_path, sms) for sms in all_slugified_match_strings}
    # all_slugified_match_strings = [sms for sms in all_slugified_match_strings if not os.path.exists(fpaths[sms]) or overwrite_if_exists]
    for match_nr, slugified_match_string in defensive_network.utility.general.progress_bar(enumerate(all_slugified_match_strings), total=len(all_slugified_match_strings), desc="Team level analysis", unit="match"):
        series_meta = df_meta[df_meta["slugified_match_string"] == slugified_match_string]

        fpath = fpaths[slugified_match_string]  # defensive_network.parse.cdf.get_team_level_analysis_fpath(base_path, slugified_match_string)
        # if not os.path.exists(fpath):  # todo remove
        #     continue

        # if slugified_match_string in ["3-liga-2023-2024-12-st-msv-duisburg-arminia-bielefeld", "3-liga-2023-2024-13-st-msv-duisburg-rot-weiss-essen", "3-liga-2023-2024-16-st-fc-ingolstadt-04-rot-weiss-essen", "3-liga-2023-2024-16-st-hallescher-fc-1-fc-saarbrucken"]:
        #     continue

        # if slugified_match_string == "3-liga-2023-2024-16-st-sv-sandhausen-msv-duisburg":
        #     continue

        # st.write(match_nr, "slugified_match_string", slugified_match_string)
        if stichtag is not None and os.path.exists(fpath):
            last_modified = os.path.getmtime(fpath)
            if last_modified > stichtag.timestamp():
                df = pd.read_csv(fpath)
                datas.append(df)
                continue
        elif os.path.exists(fpath):
            df = pd.read_csv(fpath)
            datas.append(df)
            continue

        ### Comment in to get instant analysis
        if get_instant_analysis:
            break

        use_subprocess = False
        # df_tracking, df_event = run_as_subprocess(get_match_data, {"base_path": base_path, "slugified_match_string": slugified_match_string, "xt_model": xt_model, "expected_receiver_model": expected_receiver_model, "formation_model": formation_model, "plot_formation": False})  # subprocess to avoid memory issues
        try:
            if use_subprocess:
                df_tracking, df_event = run_as_subprocess(defensive_network.parse.dfb.cdf.get_match_data, {"base_path": base_path, "slugified_match_string": slugified_match_string, "xt_model": xt_model, "expected_receiver_model": expected_receiver_model, "formation_model": formation_model, "plot_formation": False, "overwrite_if_exists": False})  # subprocess to avoid memory issues
            else:
                df_tracking, df_event = defensive_network.parse.dfb.cdf.get_match_data(
                    base_path, slugified_match_string, xt_model=xt_model,
                    expected_receiver_model=expected_receiver_model, formation_model=formation_model, plot_formation=False,
                )
        except AssertionError as e:
            st.error(f"Assertion error: {e}")
            continue
        if match_nr == 0:
            defensive_network.utility.general.start_streamlit_profiler()

        st.write("df_event")
        st.write(df_event)

        def validate(df_tracking, df_event):
            assert df_tracking["ball_poss_team_id"].nunique() == 2
            assert df_tracking["team_id"].nunique() == 3, df_tracking["team_id"].unique()
            assert "BALL" in df_tracking["team_id"].unique(), df_tracking["team_id"].unique()

        validate(df_tracking, df_event)

        dfgs = get_metrics(df_event, df_tracking, series_meta)

# 1. Bundesliga
        # 2. Bundesliga
        # 3. Liga
        # Ligue 1
        # Ligue 2
        # Eredivisie
        # Superligaen
        #
        #
        # 19%| | 164/869 [00:17<01:06, 10.54it/s]

        # test if xt is always ok
        df_metrics = pd.concat(dfgs, axis=1)
        st.write("df_metrics")
        st.write(df_metrics)
        del dfgs
        gc.collect()

        # Filter data
        df_passes = df_event[df_event["event_type"] == "pass"].dropna(subset=["frame"])
        df_passes = df_passes[df_passes["frame"].isin(df_tracking["frame"])]
        # df_tracking = df_tracking.loc[df_tracking[selected_tracking_player_col].notna()]  # RAM !!!!!!!!!!!!!!!!!!!!!!!!!!
        # df_tracking = df_tracking.dropna(subset=[selected_tracking_player_col]) # RAM !!!!!!!!!!!!!!!!!!!!!!!!!!
        df_tracking.dropna(subset=[selected_tracking_player_col], inplace=True) # RAM !!!!!!!!!!!!!!!!!!!!!!!!!!
        df_passes = df_passes[df_passes["frame"].isin(df_tracking["frame"])]

        # Assert that filtering has no effect
        # dfgs_filtered = get_metrics(df_passes, df_tracking)
        # df_metrics_filtered = pd.concat(dfgs_filtered, axis=1)
        # for col in ["passes", "passes_against", "total_xt", "total_xt_against", "total_xt_only_positive", "total_xt_only_positive_against", "total_xt_only_successful", "total_xt_only_successful_against", "total_xt_only_negative", "total_xt_only_negative_against"]:
        #     # assert df_metrics[col].sum() == df_metrics_filtered[col].sum(), (col, df_metrics[col].sum(), df_metrics_filtered[col].sum())  # rare
        #     pass

        # total_minutes = 0
        # for period, df_period in df_tracking.groupby("section"):
        #     period_minutes = (df_period["datetime_tracking"].max() - df_period["datetime_tracking"].min()).total_seconds() / 60
        #     total_minutes += period_minutes
        total_minutes = (
            df_tracking.groupby("section")["datetime_tracking"]
            .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
            .sum()
        )

        average_positions = None
        if use_tracking_average_position:
            average_positions = defensive_network.models.average_position.get_average_tracking_positions_off_def(df_tracking)["off"]

        # @st.cache_resource
        def _get_involvement(
            value_col, player_col, receiver_col, tracking_player_col, tracking_player_name_col, slugified_match_string, involvement_model_success_pos_value,
            involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted,
            model_radius, overwrite_if_exists=False,
        ):
            # fpath = defensive_network.parse.cdf.get_involvement_fpath(base_path, slugified_match_string + involvement_model_success_pos_value + involvement_model_success_neg_value + involvement_model_out + involvement_model_intercepted + str(model_radius))
            # if not overwrite_if_exists and os.path.exists(fpath):
            #     return pd.read_csv(fpath)
            df_involvement = defensive_network.models.involvement.get_involvement(
                df_passes, df_tracking, event_receiver_col=receiver_col,
                tracking_player_col=tracking_player_col, involvement_model_success_pos_value=involvement_model_success_pos_value,
                involvement_model_success_neg_value=involvement_model_success_neg_value,
                involvement_model_out=involvement_model_out, involvement_model_intercepted=involvement_model_intercepted,
                model_radius=model_radius, event_value_col=value_col, tracking_player_name_col=tracking_player_name_col,
            )
            # df_involvement.to_csv(fpath, index=False)
            return df_involvement

        # total_involvement_value = df_passes.groupby("team_id_1").agg({selected_value_col: "sum"})[selected_value_col].to_dict()
        # total_fault_value = df_passes.loc[df_passes[selected_value_col] > 0].groupby("team_id_1").agg({selected_value_col: "sum"})[selected_value_col].to_dict()
        # total_contribution_value = df_passes.loc[df_passes[selected_value_col] < 0].groupby("team_id_1").agg({selected_value_col: "sum"})[selected_value_col].to_dict()

        df_involvement = _get_involvement(selected_value_col, selected_player_col, selected_receiver_col, selected_tracking_player_col, selected_tracking_player_name_col, slugified_match_string, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius)

        # involvement statistics
        dfg = df_involvement.groupby(["event_id"]).agg(
            n_defenders=("defender_id", "nunique"),
            n_defenders_unique=("defender_id", "unique"),
            n_defenders_unique_names=("defender_name", "unique"),
            pass_xt=("pass_xt", "first"),
            n_pass_xt_unique_values=("pass_xt", "nunique"),
            pass_xt_unique_values=("pass_xt", "unique"),
            involvement_type=("involvement_type", "first"),
            n_involvement_type_unique_values=("involvement_type", "nunique"),
            involvement_type_unique_values=("involvement_type", "unique"),
            n_involvement_model_unique_values=("involvement_model", "nunique"),
            involvement_model_unique_values=("involvement_model", "unique"),
            raw_involvement=("raw_involvement", "sum"),
            n_raw_involvement_unique_values=("raw_involvement", "nunique"),
            raw_contribution=("raw_contribution", "sum"),
            raw_fault=("raw_fault", "sum"),
            involvement=("involvement", "sum"),
            contribution=("contribution", "sum"),
            fault=("fault", "sum"),
        )
        assert (dfg["n_pass_xt_unique_values"] == 1).all()
        assert (dfg["n_defenders"] == 11).all()

        # TODO add stats

        team_datas = []
        for team, df_involvement_team in df_involvement.groupby("team_id_1"):
            df_involvement_team["network_receiver"] = df_involvement_team[selected_expected_receiver_col].where(df_involvement_team[selected_expected_receiver_col].notna(), df_involvement_team[selected_receiver_col])
            df_involvement_team["network_receiver_name"] = df_involvement_team[selected_expected_receiver_name_col].where(df_involvement_team[selected_expected_receiver_name_col].notna(), df_involvement_team[selected_receiver_name_col])
            for involvement_type_col in ["involvement", "contribution", "fault"]:
                for xt_rule in ["", "_only_successful_passes"]:
                    # if xt_rule == "only_positive":
                    #     df_involvement_rule = df_involvement_team[df_involvement_team[selected_value_col] > 0].copy()
                    if xt_rule == "_only_successful_passes":
                        df_involvement_rule = df_involvement_team[df_involvement_team["event_outcome"] == "successfully_completed"].copy()
                    elif xt_rule == "":
                        df_involvement_rule = df_involvement_team.copy()
                    else:
                        raise ValueError(xt_rule)

                    st.write("df_involvement_rule")
                    st.write(df_involvement_rule)

                    importlib.reload(defensive_network.models.passing_network)
                    networks = defensive_network.models.passing_network.get_defensive_networks(
                        df_involvement_rule, value_col=selected_value_col, involvement_type_col=involvement_type_col,
                        player_col=selected_player_col, player_name_col=selected_player_name_col,
                        receiver_col="network_receiver", receiver_name_col="network_receiver_name",
                        defender_id_col=defender_col, defender_name_col=defender_name_col,
                        total_minutes=total_minutes, average_positions=average_positions,
                        remove_passes_with_zero_involvement=remove_passes_with_zero_involvement,
                    )
                    metrics = defensive_network.models.passing_network.analyse_defensive_networks(networks)

                    total_off_involvement = metrics.off_involvement_type_network.team["Total Degree"]

                    # df = metrics.def_network_team_sums
                    df = metrics.def_network_team_means
                    df["team"] = team
                    df["involvement_type"] = involvement_type_col
                    df["xt_rule"] = xt_rule
                    # df["value_col"] = selected_value_col
                    df["total_off_network_degree"] = total_off_involvement
                    # df["total_inv_type_value"] = abs(df[f"total_{involvement_type_col}_value"])

                    team_datas.append(df)

        df = pd.concat(team_datas, axis=1).T

        df_pivot = df.pivot(index="team", columns=["involvement_type", "xt_rule"])
        df_pivot.columns = [f"{col[0]}_{col[1]}{col[2]}" for col in df_pivot.columns]

        df_pivot["value_col"] = selected_value_col

        df = df_pivot.merge(df_metrics, left_index=True, right_index=True)
        # st.stop()

        df.to_csv(fpath, index=False)
        print(f"Saved to {fpath}")
        datas.append(df)

        del df_tracking, df_event, df_passes, df_involvement, networks, metrics, df, team_datas, df_pivot
        gc.collect()
        if match_nr == 0:
            defensive_network.utility.general.stop_streamlit_profiler()

        gc.collect()
        # break  # TODO remove

        # if match_nr >= 2:
        #     break

    df = pd.concat(datas, axis=0)

    del datas
    gc.collect()

    return df


default_fpath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../2324"))


# @memory_profiler.profile
def team_level_analysis_dashboard():
    with st.expander("Settings"):
        base_path = st.text_input("Base path", default_fpath)
        df_meta = defensive_network.parse.dfb.cdf.get_all_meta(base_path)
        df_lineups = defensive_network.parse.dfb.cdf.get_all_lineups(base_path)
        all_tracking_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.cdf.get_tracking_fpath(base_path, "")))
        all_event_files = os.listdir(os.path.dirname(defensive_network.parse.dfb.cdf.get_event_fpath(base_path, "")))

        all_files = [file for file in all_tracking_files if file.replace("parquet", "csv") in all_event_files]
        all_slugified_match_strings = [os.path.splitext(file)[0] for file in all_files]

        involvement_types = ["intercepted", "out", "success_and_neg_value", "success_and_pos_value"]

        # Select options
        xt_model = st.selectbox("Select xT model", ["ma2024", "the_athletic"])
        expected_receiver_model = st.selectbox("Select expected receiver model", ["power2017"])
        formation_model = st.selectbox("Select formation model", ["average_pos"])
        all_models = ["circle_circle_rectangle", "circle_passer", "circle_receiver", "intercepter"]
        involvement_model_success_pos_value = st.selectbox("Select involvement model for successful passes with positive value", all_models, index=0)
        involvement_model_success_neg_value = st.selectbox("Select involvement model for successful passes with negative value", all_models, index=1)
        involvement_model_out = st.selectbox("Select involvement model for passes that are interrupted by referee (e.g. goes out of the pitch)", all_models, index=1)
        involvement_model_intercepted = st.selectbox("Select involvement model for passes that are intercepted", all_models, index=3)
        model_radius = st.number_input("Circle model radius", min_value=0, value=5)
        selected_player_mode = st.selectbox("Select player column for networks", [("player_id_1", "player_name_1", "player_id_2", "player_name_2", "expected_receiver", "expected_receiver_name", "player_id", "player_name", "Player"), ("role_1", "role_name_1", "role_2", "role_name_2", "expected_receiver_role", "expected_receiver_role_name", "role", "role_name", "Role")], format_func=lambda x: x[-1], index=1)
        selected_player_col, selected_player_name_col, selected_receiver_col, selected_receiver_name_col, selected_expected_receiver_col, selected_expected_receiver_name_col, selected_tracking_player_col, selected_tracking_player_name_col = selected_player_mode[0], selected_player_mode[1], selected_player_mode[2], selected_player_mode[3], selected_player_mode[4], selected_player_mode[5], selected_player_mode[6], selected_player_mode[7]
        use_tracking_average_position = st.toggle("Use average player position in tracking data for passing networks")
        selected_value_col = st.selectbox("Value column for networks", ["pass_xt", None], format_func=lambda x: str(x))
        plot_involvement_examples = st.multiselect("Plot involvement examples", involvement_types, default=involvement_types)
        n_examples_per_type = st.number_input("Number of examples", min_value=0, value=3)
        show_def_full_metrics = st.toggle("Show individual offensive metrics for each defender", value=False)
        remove_passes_with_zero_involvement = st.toggle("Remove passes with 0 involvement from networks", value=True)
        defender_col = st.selectbox("Defender for defensive networks", ["defender_id"], format_func=lambda x: {"defender_id": "Defender (player)"}[x])
        defender_name_col = {"defender_id": "defender_name"}[defender_col]

    get_instant_analysis = st.toggle("Get instant analysis", value=True)

    df = _get_data(all_slugified_match_strings, base_path, xt_model, expected_receiver_model, formation_model, selected_tracking_player_col, selected_tracking_player_name_col, use_tracking_average_position, selected_value_col, selected_player_col, selected_receiver_col, selected_expected_receiver_col, selected_expected_receiver_name_col, selected_receiver_name_col, involvement_model_success_pos_value, involvement_model_success_neg_value, involvement_model_out, involvement_model_intercepted, model_radius, selected_player_name_col, defender_col, defender_name_col, remove_passes_with_zero_involvement, get_instant_analysis)
    gc.collect()

    # for involvement_type in df["involvement_type"].unique():
    #     i_involvement_type = df["involvement_type"] == involvement_type
    #     df.loc[~i_involvement_type, f"total_{involvement_type}_value"] = None
    #     df.loc[i_involvement_type, "total_inv_type_value"] = df.loc[i_involvement_type, f"total_{involvement_type}_value"].abs()
    # df["average_involvement"]
    df["uninvolved_xT"] = df["total_xt_only_successful"] - df[f"total_off_network_degree_involvement_only_successful_passes"]
    # assert (df["uninvolved_xT"] > 0).all()
    df["uninvolved_xT_%"] = df["uninvolved_xT"] / df[f"total_xt_only_successful"]
    # df["uninvolved_xT_%2"] = df["uninvolved_xT"] / df[f"total_xt_only_successful"]

    df["xg_diff"] = df["xg"] - df["xg_against"]
    df["goal_diff"] = df["goals"] - df["goals_against"]

    # Weighted Density makes no sense (should be the same as the normal total
    # df = df[[col for col in df.columns if "Weighted Density" not in col]]

    df = df[df.columns.sort_values()]
    st.write("df")
    st.write(sorted(df.columns.tolist()))
    st.write(df)

    df_numeric = df[[col for col in df.columns if col != "value_col"]]

    target_col = st.selectbox("y variable", ["goals_against", "xg_against", "xg_diff", "goal_diff"])

    # def partial_correlations(df):
    #     import statsmodels
    importlib.reload(defensive_network.utility.stats)
    covariate_cols = ["total_xt_only_successful_against", "total_xt_only_positive_against", "total_xt_against",
                      "total_xt", "total_xt_only_positive", "total_xt_only_successful"
                      ]

    excluded_cols = ["goals_against", "goal_diff", "points", "passes_against", "passes", ]
    df_numeric = df_numeric[[col for col in df_numeric.columns if col not in ["n_interceptions", "n_tackles"]]]

    df_pcorr = defensive_network.utility.stats.partial_correlation_matrix(df_numeric[[col for col in df_numeric.columns if col not in excluded_cols or col in covariate_cols or col == target_col]], covariate_cols=covariate_cols, y_col=target_col)

    st.write("df_pcorr")
    st.write(df_pcorr)

    # import pingouin as pg
    for is_partial in [True, False]:
        plt.figure(figsize=(16, 12))
        st.write(f"{is_partial=}")
        if is_partial:
            df_corr = df_numeric.pcorr()
        else:
            df_corr = df_numeric.corr()
        st.write("df_corr")
        st.write(df_corr)
        sns.heatmap(df_corr[[target_col]], annot=True, cmap="coolwarm", fmt=".2f")
        st.write(plt.gcf())
        plt.close()

        plt.figure(figsize=(16, 12))
        sns.heatmap(df_corr, annot=False, cmap="coolwarm", fmt=".2f")
        st.write(plt.gcf())
        plt.close()

    n_columns = 2
    columns = st.columns(n_columns)
    ### analyse correlations
    for i, col in enumerate(df.columns):
        # plt.figure()
        # plt.scatter(df[col], df["xg_against"])
        # plt.xlabel(f"{col}")
        # plt.ylabel("xg_against")
        # columns[i % n_columns].write(plt.gcf())
        # plt.close()
        # Scatter plot with trendline
        import numpy.exceptions
        import numpy._core._exceptions
        st.write(f"#### {col}")
        try:
            sns.regplot(x=col, y="xg_against", data=df)
        except (numpy.exceptions.DTypePromotionError, numpy._core._exceptions._UFuncInputCastingError, numpy._core._exceptions.UFuncTypeError) as e:
            st.write(e)
            continue
        columns[i % n_columns].write(plt.gcf())
        plt.close()
        plt.show()


#     for involvement_type, df_by_involvement in df.groupby("involvement_type"):
#         for req_involvement_type, x_col, y_col in [
#             (None, "Unweighted Density", "xg_against"),
#             (None, "Weighted Density", "xg_against"),
#             (None, "Team reciprocity", "xg_against"),
#             (None, "Total Degree", "xg_against"),
#             (None, "uninvolved_xT", "xg_against"),
#             (None, "uninvolved_xT_%", "xg_against"),
# #                ("", "Total Degree", "xg_against"),
#         ]:
#             plt.scatter(df_by_involvement[x_col], df_by_involvement[y_col])
#             plt.xlabel(f"{involvement_type.capitalize()} {x_col}")
#             plt.ylabel(y_col)
#             st.write(plt.gcf())
#             plt.close()


if __name__ == '__main__':
    team_level_analysis_dashboard()

