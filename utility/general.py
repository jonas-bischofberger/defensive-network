import tqdm
import streamlit as st
import pandas as pd


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
