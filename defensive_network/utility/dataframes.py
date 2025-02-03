import pandas as pd


def prepare_doctest():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def get_new_unused_column_name(df, prefix):
    """
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


def assert_no_duplicate_columns(_df):
    duplicate_columns = _df.columns[_df.columns.duplicated()]
    assert len(duplicate_columns) == 0, f"Duplicate columns: {duplicate_columns}"


def assert_no_duplicate_keys(_df, _key_cols):
    duplicate_keys = _df.duplicated(_key_cols)
    assert not duplicate_keys.any(), f"Duplicate keys: {_df[duplicate_keys]}"


def append_to_parquet_file(df_to_append, fpath, key_cols, overwrite_key_cols=True):
    """
    >>> import os
    >>> fpath = "test.parquet"
    >>> if os.path.exists(fpath):
    ...     os.remove(fpath)
    >>> append_to_parquet_file(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), fpath, ["a"])
    >>> pd.read_parquet(fpath)
       a  b
    0  1  3
    1  2  4
    >>> append_to_parquet_file(pd.DataFrame({"a": [2, 3], "b": [5, 6]}), fpath, ["a"])
    >>> pd.read_parquet(fpath)
       a  b
    0  1  3
    1  2  5
    2  3  6
    >>> append_to_parquet_file(pd.DataFrame({"a": [3, 4], "b": [7, 8]}), fpath, ["a"], overwrite_key_cols=False)
    >>> pd.read_parquet(fpath)
       a  b
    0  1  3
    1  2  5
    2  3  6
    3  4  8
    """
    assert_no_duplicate_keys(df_to_append, key_cols)
    assert_no_duplicate_columns(df_to_append)

    try:
        df_existing = pd.read_parquet(fpath)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=df_to_append.columns)

    assert_no_duplicate_keys(df_existing, key_cols)
    assert_no_duplicate_columns(df_existing)

    df_combined = pd.concat([df_existing, df_to_append], axis=0)
    df_combined = df_combined[~df_combined.duplicated(key_cols, keep="last" if overwrite_key_cols else "first")]

    assert_no_duplicate_columns(df_combined)
    assert_no_duplicate_keys(df_combined, key_cols)
    assert_no_duplicate_columns(df_to_append)
    assert_no_duplicate_keys(df_to_append, key_cols)

    df_combined.to_parquet(fpath, index=False)  # TODO


def check_presence_of_required_columns(df, str_data, column_names, column_values, additional_message=None):
    missing_tracking_cols = [(col_name, col_value) for (col_name, col_value) in zip(column_names, column_values) if col_value not in df.columns]
    if len(missing_tracking_cols) > 0:
        raise KeyError(f"""Missing column{'s' if len(missing_tracking_cols) > 1 else ''} in {str_data}: {', '.join(['='.join([str(parameter_name), "'" + str(col) + "'"]) for (parameter_name, col) in missing_tracking_cols])}.{' ' + additional_message if additional_message is not None else ''}""")


def get_unused_column_name(existing_columns, prefix):
    """
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
