def get_new_unused_column_name(df, prefix):
    """ Get a new column name that doesn't exist yet in df and that starts with prefix """
    column_names = df.columns
    i = 1
    new_column_name = f"{prefix}_{i}"

    while new_column_name in column_names:
        i += 1
        new_column_name = f"{prefix}_{i}"

    return new_column_name

def extract_numbers_from_string_as_ints(text):
    """
    >>> extract_numbers_from_string_as_ints("There are 3.1 apples, 4 bananas, and 12 oranges.")
    [3, 1, 4, 12]
    """
    import re
    numbers_as_strings = re.findall(r'\d+', text)
    return [int(num) for num in numbers_as_strings]
