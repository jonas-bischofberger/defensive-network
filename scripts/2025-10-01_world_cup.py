import importlib

import streamlit as st
import sys
import pandas as pd

import numpy as np

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import defensive_network.parse.drive
import defensive_network.models.responsibility
import defensive_network.models.passing_network
import defensive_network.models.formation_v2
import defensive_network.utility.general

importlib.reload(defensive_network.models.passing_network)


def main():
    df_meta = defensive_network.parse.drive.download_csv_from_drive("meta.csv", st_cache=True)
    df_meta = df_meta[df_meta["competition_name"] == "FIFA Men's World Cup"]
    st.write("df_meta")
    st.write(df_meta)

    teams_reached_knockout = [
        "Netherlands", "Senegal", "England", "USA", "Argentina", "Poland", "France", "Australia",
        "Japan", "Spain", "Morocco", "Croatia", "Brazil", "Switzerland", "Portugal", "South Korea"
    ]

    dfg_responsibility = defensive_network.parse.drive.download_csv_from_drive("responsibility_model_Bundesliga.csv", st_cache=True)

    # st.write("dfg_responsibility")
    # st.write(dfg_responsibility)

    df_team_matchsums = defensive_network.parse.drive.download_csv_from_drive("team_matchsums.csv", st_cache=True)
    st.write("df_team_matchsums")
    st.write(df_team_matchsums)

    @st.cache_data
    def _analyze_match(match_string):
        match = df_meta[df_meta["match_string"] == match_string].iloc[0]
        df_involvement = defensive_network.parse.drive.download_csv_from_drive(f"involvement/{match['slugified_match_string']}.csv", st_cache=False)

        _dfs = []

        # st.write("df_involvement")
        # st.write(df_involvement)
        # st.write(df_involvement["defender_id"].unique())
        # st.write(df_involvement["defender_name"].unique())
        # st.stop()

        teams = df_involvement["gameEvents.teamName"].unique().tolist()
        df_involvement["other_team_name"] = df_involvement["gameEvents.teamName"].apply(lambda x: teams[0] if x ==
                                                                                                              teams[
                                                                                                                  1] else
        teams[1])
        df_involvement["raw_responsibility"], _, df_involvement[
            "valued_responsibility"], _ = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility_model=dfg_responsibility, context_cols=[
            "role_category_1", "network_receiver_role_category", "defender_role_category"])
        player2name = \
        df_involvement[["player_id_1", "player_name_1"]].drop_duplicates("player_id_1").set_index("player_id_1")[
            "player_name_1"].to_dict()
        dfg_role_frequencies = df_involvement.groupby(["role_category_1", "team_name_1",
                                                       "player_name_1"]).size().reset_index(name="count")
        dfg_role_frequencies["total_occurrences"] = dfg_role_frequencies.groupby(["role_category_1", "team_name_1"])[
            "count"].transform("sum")
        dfg_role_frequencies["frequency"] = dfg_role_frequencies["count"] / dfg_role_frequencies["total_occurrences"]
        dfg_role_frequencies = dfg_role_frequencies.sort_values(by=["role_category_1", "team_name_1",
                                                                    "frequency"], ascending=[True, True, False])
        dfg_role_frequencies["cumulative_frequency"] = dfg_role_frequencies.groupby(["role_category_1", "team_name_1"])[
            "frequency"].cumsum()
        dfg_role_frequencies["previous_cum_frequency_exceeds_0.8"] = \
        dfg_role_frequencies.groupby(["role_category_1", "team_name_1"])[
            "cumulative_frequency"].shift(1).fillna(0) > 0.6
        dfg_role_frequencies = dfg_role_frequencies[~dfg_role_frequencies["previous_cum_frequency_exceeds_0.8"]]
        dfg_role_frequencies = dfg_role_frequencies

        def _normalize_name(name):
            belong_to_last_name = ["van", "de", "Van", "De"]

            parts = name.split()
            if len(parts) == 1:
                return name
            first_name = parts[0]
            last_name = parts[-1]
            if parts[-2] in belong_to_last_name:
                last_name = parts[-2] + " " + last_name
            return f"{last_name}"

        dfg_role_strings = dfg_role_frequencies.groupby(["role_category_1", "team_name_1"]).agg(
            role_string=("player_name_1", lambda x: "/".join([_normalize_name(name) for name in x])),
        ).reset_index()

        role_string_mapping = dfg_role_strings.set_index(["role_category_1", "team_name_1"])["role_string"].to_dict()
        role_string_mapping = dfg_role_strings.set_index(["role_category_1", "team_name_1"])["role_string"]
        # st.write("role_string_mapping")
        # st.write(role_string_mapping)

        player2team = \
        df_involvement[["player_id_1", "team_name_1"]].drop_duplicates("player_id_1").set_index("player_id_1")[
            "team_name_1"].to_dict()
        df_involvement["team_name_2"] = df_involvement["possessionEvents.receiverPlayerId"].map(player2team)
        df_involvement = df_involvement[
            df_involvement["role_category_1"].isin(dfg_role_strings["role_category_1"].unique())]
        df_involvement["role_string"] = df_involvement.set_index(["role_category_1", "team_name_1"]).index.map(
            dfg_role_strings.set_index(["role_category_1", "team_name_1"])["role_string"])
        x2 = \
        dfg_role_strings.rename(columns={"role_category_1": "network_receiver_role_category", "team_name_1": "team_name_2"}).set_index([
                                                                                                                                           "network_receiver_role_category",
                                                                                                                                           "team_name_2"])[
            "role_string"]

        # st.write(df_involvement[[col for col in df_involvement.columns if "team" in col.lower() or "receiver" in col.lower()]])
        # st.write(df_involvement[["network_receiver_role_category", "other_team_name"]])

        df_involvement["network_receiver_role_string"] = df_involvement.set_index(["network_receiver_role_category",
                                                                                   "team_name_2"]).index.map(x2)

        df_involvement["defender_role_string"] = df_involvement.set_index(["defender_role_category",
                                                                           "other_team_name"]).index.map(
            dfg_role_strings.set_index(["role_category_1", "team_name_1"])["role_string"])

        # st.write("df_involvement")
        # st.write(df_involvement)

        # st.write(df_involvement[["role_category_1", "team_name_1", "role_string", "network_receiver_role_category", "network_receiver_role_string", "defender_role_category", "defender_role_string", "valued_responsibility", "valued_involvement"]])
        # st.write(df_involvement.loc[df_involvement["defender_role_category"] == "GK",["role_category_1", "team_name_1", "role_string", "network_receiver_role_category", "network_receiver_role_string", "defender_role_category", "defender_role_string", "valued_responsibility", "valued_involvement"]])

        # dfg = df_involvement.groupby("other_team_name").agg(
        #     total_raw_fault=("raw_fault", "sum"),
        #     total_valued_fault=("valued_fault", "sum"),
        #     total_raw_involvement=("raw_involvement", "sum"),
        #     total_valued_involvement=("valued_involvement", "sum"),
        #     total_raw_contribution=("raw_contribution", "sum"),
        #     total_valued_contribution=("valued_contribution", "sum"),
        #     total_raw_responsibility=("raw_responsibility", "sum"),
        #     total_valued_responsibility=("valued_responsibility", "sum"),
        # ).reset_index()
        # dfg["team_reached_knockout"] = dfg["other_team_name"].apply(lambda x: x in teams_reached_knockout)

        match_string = match['match_string']

        for team, df_involvement_team in df_involvement.groupby("team_name_1"):
            # with st.expander(f"{team} ({match_string})"):
            # st.write("df_involvement_team")
            # st.write(df_involvement_team)
            # st.write(df_involvement_team["defender_role_string"])
            df_involvement_team["defender_role_string2"] = df_involvement_team["defender_role_string"]
            # st.write("team")
            # st.write(team)
            # st.write(df_involvement_team[["role_category_1", "role_string", "player_name_1", "team_name_1", "network_receiver_role_category", "network_receiver_role_string", "defender_id", "defender_name"]].drop_duplicates().sort_values(by=["role_category_1", "role_string"]))
            # pn = defensive_network.models.passing_network.get_defensive_networks(df_involvement, receiver_col="network_receiver_role_category", player_col="role_category_1", involvement_type_col="valued_involvement")

            for inv_type in ["valued_fault", "raw_contribution"]:
                pn = defensive_network.models.passing_network.get_defensive_networks(
                    df_involvement_team, receiver_col="network_receiver_role_string",
                    player_col="role_string", involvement_type_col=inv_type, x_col="x_norm",
                    y_col="y_norm", x_to_col="x_target_norm", y_to_col="y_target_norm", defender_id_col="defender_role_string",
                    defender_name_col="defender_role_string2",
                )

                network = pn.off_involvement_type_network
                # plot = defensive_network.models.passing_network.plot_passing_network(
                #     network.df_nodes, network.df_edges, node_size_multiplier=300, arrow_width_multiplier=3,
                # )
                # st.write(f"Invtype Network", inv_type)
                # st.write(plot)

                metrics = defensive_network.models.passing_network.analyse_network(network.df_nodes, network.df_edges)
                # st.write(f"Metrics for {inv_type} network for {team} ({match['slugified_match_string']})")
                # st.write(metrics.team)

                df = metrics.team.copy()

                df["match_string"] = match["match_string"]
                df["competition_name"] = match["competition_name"]
                df["team_name"] = team
                df["involvement_type"] = inv_type

                _dfs.append(df)

            # for player, network in pn.def_networks.items():
            #     player_name = player2name.get(player, "Unknown Player")
            #     import matplotlib.pyplot as plt
            #     plt.figure()
            #     plot = defensive_network.models.passing_network.plot_passing_network(
            #         network.df_nodes, network.df_edges, node_size_multiplier=500, arrow_width_multiplier=3,
            #     )
            #     st.write(f"Defensive Network for {player_name} ({network.defender_name}) ({match['slugified_match_string']})")
            #     st.write(plot)
            #     plt.close()

        # st.stop()

        # st.write("pn")
        # st.write(pn)

        # dfg = defensive_network.models.passing_network.analyse_defensive_networks(networks=pn)
        # st.write("dfg")
        # st.write(dfg.def_network_sums)
        #
        # dfs.append(dfg.def_network_sums)

        df = pd.concat(_dfs, ignore_index=True, axis=1).T
        return df

    dfs = []
    i = 0
    for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Processing World Cup matches"):
        df = _analyze_match(match["match_string"])
        st.write("df")
        st.write(df)
        dfs.append(df)
        i += 1
        # if i > 5:
        #     break

    # for df in dfs:
    #     st.write("df")
    #     st.write(df.to_frame().T)

    df = pd.concat([df for df in dfs], ignore_index=True, axis=0)

    df_meta["home_team_id"] = df_meta["home_team_id"].astype(str).str.replace(".0", "", regex=False)
    df_team_matchsums["team_id"] = df_team_matchsums["team_id"].astype(str).str.replace(".0", "", regex=False)
    teamid2name = df_meta.set_index("home_team_id")["home_team_name"].to_dict()

    df_team_matchsums["team_name"] = df_team_matchsums["team_id"].map(teamid2name)
    st.write("df_team_matchsums")
    st.write(df_team_matchsums[["match_string", "team_id", "team_name", "points", "xg_against"]])
    df_team_matchsums = df_team_matchsums[df_team_matchsums["team_name"].notna()]
    df_team_matchsums["is_win"] = df_team_matchsums["points"] > 1
    df_team_matchsums["is_loss"] = df_team_matchsums["points"] == 0
    st.write("df_team_matchsums")
    st.write(df_team_matchsums[["match_string", "team_id", "team_name", "is_win", "is_loss", "points", "xg_against"]])
    st.write(df_team_matchsums)

    df = df.merge(df_team_matchsums[["match_string", "team_name", "is_win", "is_loss", "points", "xg_against"]], on=["match_string", "team_name"], how="left")

    st.write('df_team_matchsums[["match_string", "team_id", "team_name", "is_win", "is_loss", "points", "xg_against"]]')
    st.write(df_team_matchsums[["match_string", "team_id", "team_name", "is_win", "is_loss", "points", "xg_against"]])

    st.write("df")
    st.write(df)

    @st.cache_data
    def _process_events():
        dfs = []
        for _, match in defensive_network.utility.general.progress_bar(df_meta.iterrows(), total=len(df_meta), desc="Processing World Cup matches for event data"):
            df_events = defensive_network.parse.drive.download_csv_from_drive(f"events/{match['slugified_match_string']}.csv", st_cache=False)
            # st.write("df_events")
            # st.write(df_events)
            # st.write(df_events[[col for col in df_events.columns if "outcometype" in col.lower()]])
            # st.write(df_events[[col for col in df_events.columns if "shot" in col.lower()]])
            # st.write(df_events[[col for col in df_events.columns if "xg" in col.lower()]])
            df_events["is_goal"] = df_events["possessionEvents.shotOutcomeType"] == "G"
            df_events["is_shot"] = df_events["possessionEvents.shotOutcomeType"].notna()
            dfg = df_events.groupby("gameEvents.teamName").agg(
                n_goals=pd.NamedAgg(column="is_goal", aggfunc="sum"),
                n_shots=pd.NamedAgg(column="is_shot", aggfunc="sum"),
            ).reset_index()
            dfg["match_string"] = match["match_string"]
            dfg["n_goals_against"] = dfg["n_goals"].sum() - dfg["n_goals"]
            dfg["n_shots_against"] = dfg["n_shots"].sum() - dfg["n_shots"]



            # st.write("dfg")
            # st.write(dfg)
            dfs.append(dfg)

        dfg = pd.concat(dfs, ignore_index=True, axis=0)
        st.write("dfg")
        st.write(dfg)
        return dfg

    dfg = _process_events()

    df = df.merge(dfg, left_on=["match_string", "team_name"], right_on=["match_string", "gameEvents.teamName"], how="left")
    st.write("df")
    st.write(df)

    df.to_csv("world_cup_defensive_networks.csv", index=False)

    df_fault = df[df["involvement_type"] == "valued_fault"].copy()
    df_contribution = df[df["involvement_type"] == "raw_contribution"].copy()

    for df, label in [
        (df_fault, "Fault"), (df_contribution, "Contribution")
    ]:
        st.write(f"Analyzing {label} data")
        corr_cols = ["Unweighted Density", "Weighted Density", "Team reciprocity", "Total Degree"]

        # correlate with xg against
        dfg_corr = df[corr_cols + ["n_goals_against", "n_shots_against"]].corr()

        # st.write("dfg_corr")
        # st.write(dfg_corr)

        df["goal_diff"] = df["n_goals"] - df["n_goals_against"]
        df["shot_diff"] = df["n_shots"] - df["n_shots_against"]
        target_cols = ["n_goals_against", "n_shots_against", "n_goals", "n_shots"]

        import scipy.stats

        vars_all = corr_cols + target_cols
        df = df[df[vars_all].notna().all(axis=1)]
        rho, p = scipy.stats.spearmanr(df[vars_all], nan_policy="omit")  # returns matrices
        rho_df = pd.DataFrame(rho, index=vars_all, columns=vars_all)
        p_df = pd.DataFrame(p, index=vars_all, columns=vars_all)

        # st.write("Spearman correlation (ρ)")
        # st.dataframe(rho_df.round(3))
        # st.write("P-values")
        # st.dataframe(p_df.applymap(lambda x: round(x, 4)))

        vars_all = corr_cols + target_cols

        # drop rows with any NaNs in the variables of interest (same as your code)
        df_ = df[df[vars_all].notna().all(axis=1)].copy()

        # ---------- Spearman (matrix in one call) ----------
        rho, p = scipy.stats.spearmanr(df_[vars_all], nan_policy="omit")
        rho_df = pd.DataFrame(rho, index=vars_all, columns=vars_all)
        p_df = pd.DataFrame(p, index=vars_all, columns=vars_all)

        # st.write("Spearman correlation (ρ)")
        # st.dataframe(rho_df.round(3))
        # st.write("Spearman p-values")
        # st.dataframe(p_df.round(4))

        # ---------- Pearson (pairwise: r + p) ----------
        def pearson_matrix_with_p(dfin: pd.DataFrame, columns: list[str]):
            k = len(columns)
            r_mat = pd.DataFrame(np.eye(k), index=columns, columns=columns, dtype=float)
            p_mat = pd.DataFrame(np.zeros((k, k)), index=columns, columns=columns, dtype=float)

            for i, a in enumerate(columns):
                xa = dfin[a].to_numpy()
                sa = xa.std(ddof=1)
                for j in range(i, k):
                    b = columns[j]
                    if i == j:
                        r, pval = 1.0, 0.0
                    else:
                        xb = dfin[b].to_numpy()
                        sb = xb.std(ddof=1)
                        if sa == 0 or sb == 0:  # constant column -> undefined correlation
                            r, pval = np.nan, np.nan
                        else:
                            r, pval = scipy.stats.pearsonr(xa, xb)
                    r_mat.iat[i, j] = r_mat.iat[j, i] = r
                    p_mat.iat[i, j] = p_mat.iat[j, i] = pval
            return r_mat, p_mat

        pearson_r_df, pearson_p_df = pearson_matrix_with_p(df_, vars_all)

        st.write("Pearson correlation (r)")
        st.dataframe(pearson_r_df.round(3))
        st.write("Pearson p-values")
        st.dataframe(pearson_p_df.round(4))

        # for _, match in df_team_matchsums.iterrows():
        #     for team in [match[]]


if __name__ == '__main__':
    main()
