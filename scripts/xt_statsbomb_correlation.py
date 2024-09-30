import difflib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.utils
import streamlit as st
from statsbombpy import sb


@st.cache_resource
def get_stastbomb_competitions():
    df_competitions = sb.competitions()
    return df_competitions


@st.cache_resource
def get_statsbomb_matches_world_cup_2022():
    df_competitions = get_stastbomb_competitions()
    df_competitions = df_competitions[(df_competitions["competition_id"] == 43) & (df_competitions["season_id"] == 106)]  # World Cup 2022 IDs

    dfs_matches = []
    for _, iteration in df_competitions.iterrows():
        df_matches = sb.matches(competition_id=iteration["competition_id"], season_id=iteration["season_id"])
        dfs_matches.append(df_matches)
    df_matches = pd.concat(dfs_matches)

    return df_matches


@st.cache_resource
def get_statsbomb_events(match_id):
    df_events = sb.events(match_id=match_id)
    return df_events


def get_sofascore_data():
    df_sofa = pd.read_csv(os.path.join(os.path.dirname(__file__), "../assets/metrics_sofa.csv"))
    df_sofa = df_sofa[df_sofa["match_id"].notnull()]
    df_sofa["match_id"] = df_sofa["match_id"].astype(int)
    eve_cols = [col for col in df_sofa.columns if col.endswith("_xt") or col.endswith("_base")]
    df_sofa = df_sofa.rename(columns={col: f"{col} (Eve)" for col in eve_cols})
    return df_sofa


@st.cache_resource
def calculate_degree_centrality(df_schedule, xt_file, event_types=("Pass",), exclude_negative_xt=True):
    """ Calculate the (valued and un-valued) degree centralities for the matches in df_schedule using the xT model from xt_file. """

    df_sofa = get_sofascore_data()

    # Get xT transition matrix
    df_xt = pd.read_excel(xt_file, header=None)
    num_x_cells = len(df_xt.columns)
    num_y_cells = len(df_xt.index)
    dx_cell = 105 / num_x_cells
    dy_cell = 68 / num_y_cells

    # Start iterating over the matches and calculate the degree centralities
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    dfs = []
    for match_nr, (_, match) in enumerate(df_schedule.iterrows()):
        progress_bar.progress(match_nr / len(df_schedule))
        progress_bar_text.write(match["match_id"])
        match_id_statsbomb = match["match_id"]
        match_id_sofa = match_id_statsbomb  # Statsbomb and Sofascore match IDs are the same!

        df_sofa_match = df_sofa[df_sofa["match_id"] == match_id_sofa]

        df_events = get_statsbomb_events(match_id_statsbomb)

        goalkeepers = df_events[df_events["position"] == "Goalkeeper"]["player"].unique()

        ## Calculate xT
        # Transform Statsbomb coordinates to x in [-52.5, 52.5] and y in [-34, 34]
        df_events["x"] = (df_events["location"].astype(str).str.replace("[", "").str.replace("]", "")).apply(lambda x: float(x.split(",")[0]) if x != "nan" else np.nan)
        df_events["y"] = (df_events["location"].astype(str).str.replace("[", "").str.replace("]", "")).apply(lambda x: float(x.split(",")[1]) if x != "nan" else np.nan)
        df_events["x"] = (df_events["x"] / 120) * 105 - 52.5
        df_events["y"] = (df_events["y"] / 80) * 68 - 34

        df_events["pass_end_x"] = (df_events["pass_end_location"].astype(str).str.replace("[", "").str.replace("]", "")).apply(lambda x: float(x.split(",")[0]) if x != "nan" else np.nan)
        df_events["pass_end_y"] = (df_events["pass_end_location"].astype(str).str.replace("[", "").str.replace("]", "")).apply(lambda x: float(x.split(",")[1]) if x != "nan" else np.nan)
        df_events["pass_end_x"] = (df_events["pass_end_x"] / 120) * 105 - 52.5
        df_events["pass_end_y"] = (df_events["pass_end_y"] / 80) * 68 - 34

        # Get cell index from x and y coordinates
        df_events["x_cell_index"] = np.clip(((df_events["x"] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
        df_events["y_cell_index"] = np.clip(((df_events["y"] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)
        df_events["x_cell_index_after"] = np.clip(((df_events["pass_end_x"] + 52.5) / dx_cell).apply(np.floor), 0, num_x_cells - 1)
        df_events["y_cell_index_after"] = np.clip(((df_events["pass_end_y"] + 34) / dy_cell).apply(np.floor), 0, num_y_cells - 1)

        # assign xT values based on cell index and compute xT of passes
        df_events["xt_before"] = 0
        df_events["xt_after"] = 0
        i_valid_before = df_events["x_cell_index"].notnull() & df_events["y_cell_index"].notnull()  # sometimes we have no cell index because x and y coordinates are missing!
        df_events.loc[i_valid_before, "xt_before"] = df_events.loc[i_valid_before, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index"]), int(x["x_cell_index"])], axis=1)
        i_valid_end = df_events["x_cell_index_after"].notnull() & df_events["y_cell_index_after"].notnull()
        df_events.loc[i_valid_end, "xt_after"] = df_events.loc[i_valid_end, :].apply(lambda x: df_xt.iloc[int(x["y_cell_index_after"]), int(x["x_cell_index_after"])], axis=1)

        # Important: xT after an unsuccessful pass is 0!
        df_events.loc[df_events["pass_outcome"].isin(["Unknown", "Out", "Incomplete"]), "xt_after"] = 0

        df_events["pass_xt"] = df_events["xt_after"] - df_events["xt_before"]

        # Match Statsbomb player names and Sofascore player names so we can merge the two datasets later
        all_statsbomb_players = df_events["player"].dropna().unique()
        all_sofa_players = df_sofa_match["player"].dropna().unique()
        player2sofa = {}
        for player in all_statsbomb_players:
            closest_sofascore_matches = difflib.get_close_matches(player, all_sofa_players, n=1, cutoff=0.5)
            assert len(closest_sofascore_matches) <= 1
            player2sofa[player] = closest_sofascore_matches[0] if len(closest_sofascore_matches) > 0 else None

        df_events["sofascore_player_name"] = df_events["player"].apply(lambda x: player2sofa[x] if x in player2sofa else None)
        df_events["sofascore_receiver_name"] = df_events["pass_recipient"].apply(lambda x: player2sofa[x] if x in player2sofa else None)

        # Only include events of a certain type (e.g. only passes or passes and carries)
        df_events = df_events[df_events["type"].isin(event_types)]

        # Exclude events with negative xT if selected by the user
        if exclude_negative_xt:
            df_events = df_events[(df_events["pass_xt"] > 0)]

        # xT degree centrality
        dfg_xt_as_passer = df_events.groupby("player").agg({"pass_xt": "sum"}).sort_values("pass_xt", ascending=False)
        dfg_xt_as_receiver = df_events.groupby("pass_recipient").agg({"pass_xt": "sum"}).sort_values("pass_xt", ascending=False)
        dfg_xt_total = dfg_xt_as_passer["pass_xt"].fillna(0).add(dfg_xt_as_receiver["pass_xt"].fillna(0), fill_value=0)

        # Classic degree centrality
        dfg_total_passes = df_events.groupby("player").size()
        dfg_total_passes_receiver = df_events.groupby("pass_recipient").size()
        dfg_total_passes_total = dfg_total_passes.fillna(0).add(dfg_total_passes_receiver.fillna(0), fill_value=0)

        # Put everything together and add some metadata
        dfg_overall = pd.concat([dfg_xt_as_passer, dfg_xt_as_receiver, dfg_xt_total, dfg_total_passes, dfg_total_passes_receiver, dfg_total_passes_total], axis=1)
        dfg_overall.columns = ["Out_Degree_xT", "In_Degree_xT", "Degree_Centrality_xT", "Out_Degree_Classic", "In_Degree_Classic", "Degree_Centrality_Classic"]

        dfg_overall["sofascore_match_id"] = match_id_sofa
        dfg_overall["sofascore_player"] = dfg_overall.index.map(player2sofa)  # to merge with SofaScore data later
        dfg_overall["is_goalkeeper"] = dfg_overall.index.isin(goalkeepers)  # to exclude goalkeepers later

        dfs.append(dfg_overall)

    df_overall = pd.concat(dfs)

    return df_overall


def bootstrap_pearson(x, y, n_iterations=10000):
    """ Calculated a bootstraped CI of the Pearson correlation coefficient for the correlation between x and y. """
    corrs = []
    for _ in range(n_iterations):
        x_resample, y_resample = sklearn.utils.resample(x, y)
        corr = np.corrcoef(x_resample, y_resample)[0, 1]
        corrs.append(corr)
    return np.percentile(corrs, [2.5, 97.5])


def main():
    st.write("## Statsbomb xT Network vs Sofascore in World Cup 2022")

    # 1. Let the user select the xT model to use and some additional options
    xt_files = os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/xt_weights")))
    full_xt_files = [os.path.join(os.path.dirname(__file__), "../assets/xt_weights", file) for file in xt_files]

    selected_xt_file = st.selectbox("xT weights", full_xt_files, format_func=lambda x: os.path.basename(x))
    description = {"ma2024.xlsx": "Eve's xT model, weights copied from the  WCPAS 2024 presentation in London", "the_athletic.xlsx": "Weights copied from here https://www.nytimes.com/athletic/2751525/2021/08/06/introducing-expected-threat-or-xt-the-new-metric-on-the-block/"}
    st.write(f"Using xT weights from **{os.path.basename(selected_xt_file)}**: {description[os.path.basename(selected_xt_file)]}")

    exclude_negative_xt = st.checkbox("Exclude passes (and/or dribbles) with negative xT", value=True)
    event_types = st.multiselect("Event types", ["Pass"], default=["Pass"])

    # 2. Calculate network metrics for WC 2022 (currently only degree centrality)
    df_schedule = get_statsbomb_matches_world_cup_2022()
    df_overall = calculate_degree_centrality(df_schedule, xt_file=selected_xt_file, event_types=event_types, exclude_negative_xt=exclude_negative_xt)

    # 3. Exclude goalkeepers if selected by the user
    exclude_goalkeepers = st.checkbox("Exclude goalkeepers")
    if exclude_goalkeepers:
        df_overall = df_overall[~df_overall["is_goalkeeper"]]

    # 4. Merge with SofaScore data
    df_sofa = get_sofascore_data()
    df_overall = df_overall.merge(df_sofa, left_on=["sofascore_match_id", "sofascore_player"], right_on=["match_id", "player"], how="left")

    # 5. Select variables for correlation matrix
    available_cols = [col for col in df_overall.columns if "match_id" not in col and not col.startswith("is_") and col not in ["team", "player", "sofascore_player", "match_string", "Sofascore Rating"]]
    cols_to_analyze = ["Sofascore Rating"] + st.multiselect("Variables for correlation matrix", available_cols, default=available_cols)

    # 6. Calculate Correlation matrix
    correlation_method = st.selectbox("correlation_method", ["pearson", "spearman", "kendall"], format_func=lambda x: {"pearson": "Pearson", "spearman": "Spearman (Rank)", "kendall": "Kendall (Rank)"}[x])
    corr = df_overall[cols_to_analyze].corr(method=correlation_method)
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, annot=True, fmt=".3f", xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title(f"Correlation matrix\nxt_weights={os.path.basename(selected_xt_file)}\nCorrelation method={correlation_method}\nexclude_negative_weights={exclude_negative_xt}")
    plt.yticks(rotation=0)
    fig = plt.gcf()
    fig.savefig("corr.png", dpi=300, bbox_inches="tight")
    st.write(fig)

    # 7. Print table
    st.write(df_overall)

    # 8. Show scatter plots
    if st.toggle("Show scatter/correlation plots"):
        for col in cols_to_analyze:
            if col == "Sofascore Rating":
                continue  # Don't correlate Sofascore rating with itself
            st.write(f"## {col}")
            plt.figure(figsize=(10, 10))
            try:
                sns.regplot(x=col, y="Sofascore Rating", data=df_overall)
            except Exception:
                continue
            fig = plt.gcf()
            fig.savefig("regplot.png", dpi=300, bbox_inches="tight")
            plt.title(f"Correlation {col} vs SofaScore Rating\nxt_weights={os.path.basename(selected_xt_file)}\nCorrelation method={correlation_method}\nexclude_negative_weights={exclude_negative_xt}")
            st.write(fig)

    # 9. Bootstrap Pearson correlation confidence intervals
    st.write("## Bootstrapped Pearson Correlation Confidence Intervals")

    col2confidence_interval = {}
    for col in cols_to_analyze:
        if col == "Sofascore Rating":
            continue

        i_notna = df_overall[col].notnull() & df_overall["Sofascore Rating"].notnull()

        x = df_overall.loc[i_notna, col].values
        y = df_overall.loc[i_notna, "Sofascore Rating"].values

        try:
            ci_lower, ci_upper = bootstrap_pearson(x, y, n_iterations=1000)
        except Exception:
            continue

        col2confidence_interval[col] = (ci_lower, ci_upper)

    # Plot confidence intervals
    means = [(v[1] + v[0]) / 2 for v in col2confidence_interval.values()]  # Mean of each CI
    errors = [(v[1] - (v[1] + v[0]) / 2) for v in col2confidence_interval.values()]  # Error = Upper CI - Mean
    plt.figure()
    plt.errorbar(means, list(col2confidence_interval.keys()), xerr=errors, fmt='o', capsize=5, capthick=2, elinewidth=2, marker='s', markersize=7)
    plt.axvline(0, color="black", linestyle="--")  # Line at 0 for reference
    plt.xlabel("Pearson Correlation (95% CI)")
    plt.title("95% Confidence Intervals for Pearson Correlation between Variables and SofaScore Rating")
    fig = plt.gcf()
    fig.savefig("cis_error_bars.png", dpi=300, bbox_inches="tight")
    st.write(fig)

    for col in col2confidence_interval:
        ci_lower, ci_upper = col2confidence_interval[col]
        st.write(f"Bootstrap 95% CI for Pearson correlation between **{col}** and **Sofascore Rating**: [{ci_lower:.3f}, {ci_upper:.3f}]")


if __name__ == '__main__':
    main()
