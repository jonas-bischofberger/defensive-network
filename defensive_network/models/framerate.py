import scipy.optimize
import pandas as pd


def estimate_framerate(df_event, datetime_col="datetime_event", frame_col="frame", period_col="section"):
    """
    >>> df = pd.DataFrame({"datetime_event": ["2023-01-01 12:00:00", "2023-01-01 12:00:01", "2023-01-01 12:00:02", "2023-01-01 12:00:03"], "frame": [0, 30, None, 90], "section": ["A", "A", "A", "A"]})
    >>> df["datetime_event"] = pd.to_datetime(df["datetime_event"])
    >>> df
           datetime_event  frame section
    0 2023-01-01 12:00:00    0.0       A
    1 2023-01-01 12:00:01   30.0       A
    2 2023-01-01 12:00:02    NaN       A
    3 2023-01-01 12:00:03   90.0       A
    >>> estimate_framerate(df)
    30
    """
    estimated_framerates = {}
    i_frame_nan = df_event[frame_col].isna()
    df_event["dt_seconds"] = df_event[datetime_col].astype('int64') // 10**9

    for section, df_section in df_event.groupby(period_col):
        x = df_section.loc[~i_frame_nan, "dt_seconds"]
        y = df_section.loc[~i_frame_nan, frame_col]

        x = x - x.min()

        def linear(x, a, b):
            return a * x + b

        # Fit the linear model using curve_fit
        popt, _ = scipy.optimize.curve_fit(linear, x, y)

        # Extract the slope (a) as the estimated framerate
        estimated_framerate = popt[0]
        estimated_framerates[section] = round(estimated_framerate)



    # assert all equal
    assert len(set(estimated_framerates.values())) == 1

    return list(estimated_framerates.values())[0]

