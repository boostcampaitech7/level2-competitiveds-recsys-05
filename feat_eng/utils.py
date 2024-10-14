import pandas as pd
import numpy as np


def reduce_mem_usage(df, exclude_columns=None):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage while maintaining numeric precision.
    Parameters:
    df (DataFrame): The dataframe to optimize.
    exclude_columns (list): List of column names to exclude from type conversion.
    """
    if exclude_columns is None:
        exclude_columns = []

    start_mem = df.memory_usage().sum() / 1024**2  # Memory usage before optimization
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    def get_decimal_places(column):
        """
        Get the maximum number of decimal places for a numeric column.
        """
        decimals = column.astype(str).apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
        return decimals.max()

    for col in df.columns:
        if col in exclude_columns:
            continue

        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Get the maximum number of decimal places in the column
                max_decimals = get_decimal_places(df[col])
                if max_decimals > 5:
                    df[col] = df[col].astype(np.float32)  # Use float32 for higher precision
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)  # Use float16 if possible
                else:
                    df[col] = df[col].astype(np.float32)  # Use float32 if float16 is insufficient
        else:
            continue

    end_mem = df.memory_usage().sum() / 1024**2  # Memory usage after optimization
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df