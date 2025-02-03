import scipy.optimize
import pandas as pd
import numpy as np
import streamlit as st

from defensive_network.utility.dataframes import get_unused_column_name


def estimate_framerate_by_linear_slope(df_event, datetime_col="datetime_event", frame_col="frame", period_col="section"):
    """
    Estimate the framerate of a data set by fitting a linear model to datetime and frame.

    >>> df = pd.DataFrame({"datetime_event": ["2023-01-01 12:00:00", "2023-01-01 12:00:01", "2023-01-01 12:00:02", "2023-01-01 12:00:03"], "frame": [0, 30, None, 90], "section": ["A", "A", "A", "A"]})
    >>> df["datetime_event"] = pd.to_datetime(df["datetime_event"])
    >>> df
           datetime_event  frame section
    0 2023-01-01 12:00:00    0.0       A
    1 2023-01-01 12:00:01   30.0       A
    2 2023-01-01 12:00:02    NaN       A
    3 2023-01-01 12:00:03   90.0       A
    >>> estimate_framerate_by_linear_slope(df)
    30
    """
    section_to_estimated_framerate = {}
    seconds_col = get_unused_column_name(df_event.columns, "dt_seconds")
    df_event[seconds_col] = df_event[datetime_col].astype('int64') // 10**9
    df_event = df_event.replace([np.inf, -np.inf], np.nan)
    i_frame_nan = df_event[frame_col].isna() | df_event[seconds_col].isna()

    def linear(x, a, b):
        return a * x + b

    for section, df_section in df_event.groupby(period_col):
        highest_fr = df_section[frame_col].max()
        median_fr = df_section[frame_col].median()
        lowest_fr = df_section[frame_col].min()
        dfr = (highest_fr - median_fr) * 2
        dsec = (df_section[seconds_col].max() - df_section[seconds_col].median()) * 2
        section_framerate = dfr / dsec
        section_to_estimated_framerate[section] = round(section_framerate)

        # dx = df_section.loc[~i_frame_nan, seconds_col].diff()
        # dy = df_section.loc[~i_frame_nan, frame_col].diff()
        # st.write(df_section)
        # st.write(df_section.loc[:, [frame_col, seconds_col]])
        # st.write("dx")
        # st.write(dx)
        # st.write("dy")
        # st.write(dy)
        # m = dy / dx
        # estimated_framerate = m.dropna().mean()
        # st.write(m)
        # section_to_estimated_framerate[section] = round(estimated_framerate)
        #
        # # Fit a linear model of seconds vs frame number
        # x = df_section.loc[~i_frame_nan, seconds_col]
        # y = df_section.loc[~i_frame_nan, frame_col]
        # x = x - x.min()  # let seconds (x) start at 0
        # popt, _ = scipy.optimize.curve_fit(linear, x, y)
        #
        # # Slope of the linear fit (a) = estimated framerate
        # estimated_framerate = popt[0]
        # section_to_estimated_framerate[section] = round(estimated_framerate)

    # Check if every section has the same estimated frame rate. If not, this method may not be valid or there is a problem with the data
    if len(set(section_to_estimated_framerate.values())) != 1:
        raise ValueError(f"Estimation of framerate failed! Got multiple different framerates for different periods: {section_to_estimated_framerate}")

    return list(section_to_estimated_framerate.values())[0]
