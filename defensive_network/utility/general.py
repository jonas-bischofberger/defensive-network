import tqdm
import streamlit as st
import pandas as pd


class _Sentinel:
    def __eq__(self, other):
        return isinstance(other, _Sentinel)


_unset = _Sentinel()  # To explicitly differentiate between a default None and a user-set None


def get_number_of_lines_in_file(file_path):
    with open(file_path) as f:
        return sum(1 for _ in f)


def progress_bar(iterable, total=None, desc=None, **kwargs):
    """
    >>> for i in progress_bar([1, 2, 3, 4, 5]):
    ...     pass
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if desc is None:
        try:
            desc = kwargs["desc"]
        except KeyError:
            desc = None

    progress_bar_text = st.empty()
    _progress_bar = st.progress(0)
    i = 0
    for item in tqdm.tqdm(iterable, total=total, **kwargs):
        progress_bar_text.text(f"{desc}: {i}/{total}")
        _progress_bar.progress(i / total)
        i += 1
        yield item


def get_new_unused_column_name(df, prefix):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> get_new_unused_column_name(df, "new")
    'new'
    >>> get_new_unused_column_name(df, "a")
    'a_1'
    >>>
    """
    column_names = df.columns
    i = 1
    new_column_name = f"{prefix}"

    while new_column_name in column_names:
        new_column_name = f"{prefix}_{i}"
        i += 1

    return new_column_name


def extract_numbers_from_string_as_ints(text):
    """
    >>> extract_numbers_from_string_as_ints("There are 3.1 apples, 4 bananas, and 12 oranges.")
    [3, 1, 4, 12]
    """
    import re
    numbers_as_strings = re.findall(r'\d+', text)
    return [int(num) for num in numbers_as_strings]


def uniquify_keep_order(lst):
    """
    >>> uniquify_keep_order([1, 2, 3, 1, 2, 4])
    [1, 2, 3, 4]
    """
    return list(dict.fromkeys(lst))


def seconds_since_period_start_to_mmss(seconds, period_nr):
    """
    >>> seconds_since_period_start_to_mmss(100, 0)
    '01:40'
    >>> seconds_since_period_start_to_mmss(45*60+123, 0)
    '45+2:03'
    >>> seconds_since_period_start_to_mmss(45*60+123, 1)
    '90+2:03'
    >>> seconds_since_period_start_to_mmss(-66, 1)
    '45-1:06'
    """
    assert period_nr in {0, 1}, f"period_nr={period_nr} not in {{0, 1}}"

    mins = int(seconds // 60)
    if mins >= 45:
        mins = 45
        extra_min_string = f"+{int((seconds - 45 * 60) // 60)}"
    elif mins < 0:
        mins = 0
        seconds = -seconds
        extra_min_string = f"-{int(seconds // 60)}"
    else:
        extra_min_string = ""

    if period_nr == 1:
        mins += 45

    return f"{mins:02d}{extra_min_string}:{int(seconds % 60):02d}"


def append_to_parquet(df_to_append, fpath, key_cols, overwrite_key_cols=True):
    """
    >>> import pandas as pd
    >>> import os
    >>> fpath = "test.parquet"
    >>> if os.path.exists(fpath):
    ...     os.remove(fpath)
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    >>> append_to_parquet(df_to_append, fpath, key_cols)
    >>> df_read = pd.read_parquet(fpath)

    >>> append_to_parquet(df_to_append, fpath, key_cols)
    >>> df_read = pd.read_parquet(fpath)
    >>> assert df_read.equals(df_to_append)
    """
    assert overwrite_key_cols, "'overwrite_key_cols' not implemented - we may not need it"

    def assert_no_duplicate_columns(_df):
        duplicate_columns = _df.columns[_df.columns.duplicated()]
        assert len(duplicate_columns) == 0, f"Duplicate columns: {duplicate_columns}"

    def assert_no_duplicate_keys(_df, _key_cols):
        duplicate_keys = _df.duplicated(_key_cols)
        assert not duplicate_keys.any(), f"Duplicate keys: {_df[duplicate_keys]}"

    # assert_no_duplicate_keys(df_to_append, key_cols)
    # assert_no_duplicate_columns(df_to_append)

    try:
        df_existing = pd.read_parquet(fpath)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=df_to_append.columns)

    # assert_no_duplicate_keys(df_existing, key_cols)
    # assert_no_duplicate_columns(df_existing)

    if overwrite_key_cols:
        df_combined = pd.concat([df_existing, df_to_append], axis=0)
        df_combined = df_combined[~df_combined.duplicated(key_cols)]

        # assert_no_duplicate_columns(df_combined)
        # assert_no_duplicate_keys(df_combined, key_cols)
        # assert_no_duplicate_columns(df_to_append)
        # assert_no_duplicate_keys(df_to_append, key_cols)

        df_combined.to_parquet(fpath, index=False)  # TODO


def check_presence_of_required_columns(df, str_data, column_names, column_values, additional_message=None):
    missing_tracking_cols = [(col_name, col_value) for (col_name, col_value) in zip(column_names, column_values) if col_value not in df.columns]
    if len(missing_tracking_cols) > 0:
        raise KeyError(f"""Missing column{'s' if len(missing_tracking_cols) > 1 else ''} in {str_data}: {', '.join(['='.join([str(parameter_name), "'" + str(col) + "'"]) for (parameter_name, col) in missing_tracking_cols])}.{' ' + additional_message if additional_message is not None else ''}""")


def get_unused_column_name(existing_columns, prefix):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Team": [1, 2], "Player": [3, 4]})
    >>> get_unused_column_name(df.columns, "Stadium")
    'Stadium'
    >>> get_unused_column_name(df.columns, "Team")
    'Team_1'
    """
    i = 1
    new_column_name = prefix
    while new_column_name in existing_columns:
        new_column_name = f"{prefix}_{i}"
        i += 1
    return new_column_name


def move_column(df, column_name, new_index):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    >>> move_column(df, "b", 0)
       b  a  c
    0  3  1  5
    1  4  2  6
    >>> move_column(df, "a", -1)
       b  c  a
    0  3  5  1
    1  4  6  2
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if new_index < 0:
        new_index = len(df.columns) + new_index

    col = df.pop(column_name)
    df.insert(new_index, column_name, col)

    return df
