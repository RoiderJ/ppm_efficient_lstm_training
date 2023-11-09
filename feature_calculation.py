from typing import Optional, Any
import numpy as np
import pandas as pd


def calculate_unix_timestamp(df: pd.DataFrame,
                             timestamp_column: str,
                             new_column_name: str):
    """
    Calculates the unix timestamp for a given column in a pandas DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        The pandas dataframe which contains a columns for which the unix timestamp is needed.
    timestamp_column: str
        The column for which a unix timestamp should be calculated.
    new_column_name: str
        This column will be added and contains the unix timestamps.

    Returns
    -------
    pd.DataFrame#
        DataFrame with an additional column which contains the unix timestamps.

    """
    try:
        df[new_column_name] = (df[timestamp_column] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
    except TypeError:
        df[new_column_name] = (df[timestamp_column] - pd.Timestamp(
            "1970-01-01").tz_localize(df[timestamp_column].dt.tz)) / pd.Timedelta('1s')
    return df


def day_of_week(df: pd.DataFrame,
                timestamp_column: Optional[str] = None,
                new_column_name: str = 'DayOfWeek',
                nan_fill_value: Any = 7.):
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)
    df[new_column_name] = df[timestamp_column].dt.dayofweek
    df[new_column_name] = df[new_column_name].fillna(nan_fill_value)
    return df


def next_activity(df: pd.DataFrame,
                  new_column_name: str,
                  case_id_column: str,
                  activity_column: str):
    df[new_column_name] = df.groupby(by=case_id_column)[activity_column].shift(-1)
    df[new_column_name] = df[new_column_name].fillna('!')
    return df


def next_activity_time(df: pd.DataFrame,
                       timestamp_column: str,
                       case_id_column: str,
                       new_column_name: str):
    df = calculate_unix_timestamp(df=df,
                                  new_column_name='__UnixTimestamp__',
                                  timestamp_column=timestamp_column)

    df['__Timeshifted__'] = df.groupby(by=case_id_column)['__UnixTimestamp__'].shift()
    df['__Timeshifted__'] = np.where(df['__Timeshifted__'].isna(), df['__UnixTimestamp__'],
                                     df['__Timeshifted__'])
    df['__TimeToPreviousActivity__'] = df['__UnixTimestamp__'] - df['__Timeshifted__']

    # Time to next activity
    df[new_column_name] = df.groupby(by=case_id_column)[
        '__TimeToPreviousActivity__'].shift(-1)
    df[new_column_name] = df[new_column_name].fillna(0)

    # Remove intermediate columns
    df = df.drop('__UnixTimestamp__', axis=1)
    df = df.drop('__Timeshifted__', axis=1)
    df = df.drop('__TimeToPreviousActivity__', axis=1)

    return df


def remaining_case_time(df: pd.DataFrame,
                        timestamp_column: str,
                        case_id_column: str,
                        new_column_name: str):
    df = calculate_unix_timestamp(df=df,
                                  new_column_name='__UnixTimestamp__',
                                  timestamp_column=timestamp_column)

    df['__Timeshifted__'] = df.groupby(by=case_id_column)['__UnixTimestamp__'].shift()
    df['__Timeshifted__'] = np.where(df['__Timeshifted__'].isna(),
                                     df['__UnixTimestamp__'],
                                     df['__Timeshifted__'])
    df['__TimeToPreviousActivity__'] = df['__UnixTimestamp__'] - df[
        '__Timeshifted__']

    df['__NextActivityTimeLabel__'] = df.groupby(by=case_id_column)[
        '__TimeToPreviousActivity__'].shift(-1)
    df['__NextActivityTimeLabel__'] = df['__NextActivityTimeLabel__'].fillna(0)

    new_dfs = []
    for index, group in df.groupby(by=case_id_column):
        group[new_column_name] = group.loc[::-1, '__NextActivityTimeLabel__'].cumsum()[::-1]
        new_dfs.append(group)
    df = pd.concat(new_dfs)

    # Remove intermediate columns
    df = df.drop('__UnixTimestamp__', axis=1)
    df = df.drop('__Timeshifted__', axis=1)
    df = df.drop('__TimeToPreviousActivity__', axis=1)
    df = df.drop('__NextActivityTimeLabel__', axis=1)

    return df


def time_since_midnight(df: pd.DataFrame,
                        timestamp_column: str,
                        new_column_name: str):
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)
    df[new_column_name] = (df[timestamp_column] - df[timestamp_column].dt.normalize()) / pd.Timedelta('1 second')
    return df


def time_since_sunday(df: pd.DataFrame,
                      timestamp_column: str,
                      new_column_name: str):
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)

    df['__TIME_PAYLOAD_DayOfWeek__'] = df[timestamp_column].dt.dayofweek

    df['__TIME_PAYLOAD_TimeSinceMidnight__'] = (df[timestamp_column] - df[
        timestamp_column].dt.normalize()) / pd.Timedelta(
        '1 second')

    df[new_column_name] = df['__TIME_PAYLOAD_TimeSinceMidnight__'] + (
            df['__TIME_PAYLOAD_DayOfWeek__'] * 3600 * 24)

    # Clean columns
    df = df.drop(['__TIME_PAYLOAD_TimeSinceMidnight__',
                  '__TIME_PAYLOAD_DayOfWeek__'], axis=1)

    return df


def time_since_last_event(df: pd.DataFrame,
                          timestamp_column: str,
                          new_column_name: str,
                          case_id_column: str):
    df = calculate_unix_timestamp(df=df,
                                  new_column_name='__UnixTimestamp__',
                                  timestamp_column=timestamp_column)

    df['__Timeshifted__'] = df.groupby(by=case_id_column)[
        '__UnixTimestamp__'].shift()
    df['__Timeshifted__'] = np.where(df['__Timeshifted__'].isna(),
                                     df['__UnixTimestamp__'],
                                     df['__Timeshifted__'])
    df[new_column_name] = df['__UnixTimestamp__'] - df[
        '__Timeshifted__']

    df = df.drop('__UnixTimestamp__', axis=1)
    df = df.drop('__Timeshifted__', axis=1)

    return df


def time_since_process_start(df: pd.DataFrame,
                             timestamp_column: str,
                             new_column_name: str,
                             case_id_column: str):
    dfs = []
    for case_id, group_df in df.groupby(by=case_id_column):
        group_df = calculate_unix_timestamp(df=group_df,
                                            new_column_name='__UnixTimestamp__',
                                            timestamp_column=timestamp_column)
        group_df[new_column_name] = group_df['__UnixTimestamp__'] - group_df['__UnixTimestamp__'].min()
        dfs.append(group_df)
    return pd.concat(dfs)
