import pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import dataset_config
import feature_calculation
from copy import deepcopy
from tqdm import tqdm
from tensorflow.keras.preprocessing import sequence
from itertools import chain
import h5py
from pathlib import Path
import os
import joblib

def buildOHE(index: int,
             n: int):
    """
    Get one hot encoding.

    Parameters
    ----------
    index: int
        The index to be encoded as 1.
    n: int
        Total number of values present. Determines the length of the encoded vector.

    Returns
    -------
    list
        One hot encoding as a list.

    """
    L=[0]*n
    L[index]=1
    return L

def load_dataset(name: str,
                 hdf5_filepath_train: str,
                 hdf5_filepath_test: str,
                 scale_label: bool,
                 one_hot_encoding: bool,
                 prefix_based: bool,
                 static_data_as_sequence: bool,
                 data_storage: str):
    """
    Initiate preparation of dataset.

    Parameters
    ----------
    name: str
        The name of the dataset as indicated in dataset_config.py
    hdf5_filepath_train: str
        The folder to which the training and validation features are saved to.
    hdf5_filepath_test: str
        The folder to which the test features are saved to.
    scale_label: bool
        Indicates whether the target labels should be scaled.
    one_hot_encoding: bool
        Indicates whether one hot encoding is applied or integer encoding.
    prefix_based: bool
        Indicates whether data is prepared for prefix-based approaches (many-to-one)
        or trace_based approaches (many-to-many)
    static_data_as_sequence: bool
        Indicates whether case attributes are duplicated for each timestep.
    data_storage: str
        Location where the event log can be found.

    Returns
    -------
    int
       Integer indicating the maximum label.

    """
    return _load_dataset_name(data_storage + "/" + name + ".csv",
                              getattr(dataset_config, name),
                              prefix_based=prefix_based,
                              hdf5_filepath_train=hdf5_filepath_train,
                              hdf5_filepath_test=hdf5_filepath_test,
                              scale_label=scale_label,
                              one_hot_encoding=one_hot_encoding,
                              static_data_as_sequence=static_data_as_sequence
                              )


def check_static_values(group_df,
                        cols_dict):
    """
    Check whether columns indicated as case attributes contain really only one value per case.

    Parameters
    ----------
    group_df: pd.DataFrame
        Dataframe containing events only for one case.
    cols_dict: dict
        Must contain the keys 'CATEGORICAL_STATIC_COLUMNS' and 'NUMERICAL_STATIC_COLUMNS'.
        The values to these keys are lists consisting of strings that indicate columns which are case attributes.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a static column has more than one value for a case.

    """
    for col in cols_dict['CATEGORICAL_STATIC_COLUMNS']:
        if len(group_df[col].unique()) != 1:
            pd.set_option('display.max_columns', None)
            print(group_df)
            raise ValueError('Static column ' + str(col) + ' has more than one value.')
    for col in cols_dict['NUMERICAL_STATIC_COLUMNS']:
        if len(group_df[col].unique()) != 1:
            pd.set_option('display.max_columns', None)
            print(group_df)
            raise ValueError('Static column ' + str(col) + ' has more than one value.')


def generate_batch(index: int,
                   group_df: pd.DataFrame,
                   one_hot_encoding: bool,
                   prefix_based: bool,
                   ohe_encoders: dict,
                   cols_dict: dict,
                   static_as_sequence: bool):
    """
    Create feature vector for a case id.

    Parameters
    ----------
    index: int
        Case id.
    group_df: pd.DataFrame
        Dataframe with all events for one case id.
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    prefix_based: bool
        True if data is generate for a many-to-one approach.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.
    cols_dict: dict
        Dictionary containing information about static, dynamic, categorical and numerical columns.
    static_as_sequence: bool
        True if event attributes should be duplicated for each timestep.

    Returns
    -------
    list: Dynamic input features
    list: Target labels
    list: Case id'S
    list: Static input features. Only returned if static_as_sequence is False.
    """

    check_static_values(group_df=group_df,
                        cols_dict=cols_dict)

    if static_as_sequence:
        categorical_columns = cols_dict['CATEGORICAL_COLUMNS']
        numerical_columns = cols_dict['NUMERICAL_COLUMNS']
    else:
        categorical_columns = cols_dict['CATEGORICAL_DYNAMIC_COLUMNS']
        numerical_columns = cols_dict['NUMERICAL_DYNAMIC_COLUMNS']

    dynamic_data, labels, cases = generate_dynamic_batch(index=index,
                                                         group_df=group_df,
                                                         one_hot_encoding=one_hot_encoding,
                                                         prefix_based=prefix_based,
                                                         ohe_encoders=ohe_encoders,
                                                         categorical_columns=categorical_columns,
                                                         numerical_columns=numerical_columns)

    static_data = None
    if not static_as_sequence:
        if len(cols_dict['CATEGORICAL_STATIC_COLUMNS']) > 0 or len(cols_dict['NUMERICAL_STATIC_COLUMNS']) > 0:
            static_data = generate_static_batch(group_df=group_df,
                                                one_hot_encoding=one_hot_encoding,
                                                prefix_based=prefix_based,
                                                ohe_encoders=ohe_encoders,
                                                categorical_columns=cols_dict['CATEGORICAL_STATIC_COLUMNS'],
                                                numerical_columns=cols_dict['NUMERICAL_STATIC_COLUMNS'])

    return dynamic_data, labels, cases, static_data


def generate_static_batch(group_df: pd.DataFrame,
                          one_hot_encoding: bool,
                          prefix_based: bool,
                          ohe_encoders: dict,
                          categorical_columns: list,
                          numerical_columns: list):
    """
    Create feature vector for static attributes.

    Parameters
    ----------
    group_df: pd.DataFrame
        Dataframe with all events for one case id.
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    prefix_based: bool
        True if data is generate for a many-to-one approach.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.
    categorical_columns: list
        List of str indicating the columns with categorical features.
    numerical_columns: list
        List of str indicating the columns with numerical features.

    Returns
    -------
    list: Static input features
    """
    data_large = list()
    length = len(group_df)
    if prefix_based:
        start = 1
    else:
        start = length
    for i in range(start, length + 1):
        data_trace = list()

        for col in categorical_columns:
            value = group_df[col].iloc[0]
            idx = ohe_encoders[col].transform(np.asarray(value).reshape(1, -1)).astype(int)[0][0]
            if one_hot_encoding:
                if np.isnan(idx) or idx < 0:
                    data_trace.extend([0] * len(ohe_encoders[col].categories_[0]))
                else:
                    data_trace.extend(buildOHE(idx, len(ohe_encoders[col].categories_[0])))
            else:
                if np.isnan(idx) or idx < 0:
                    data_trace.append(0)  # This is encoding for 'unknown'
                else:
                    data_trace.append(idx + 1)  # Add +1 since 0 is used for masking

        for col in numerical_columns:
            value = group_df[col].iloc[0]
            data_trace.append(value)

        data_large.append(data_trace)

    return data_large


def generate_dynamic_batch(index: int,
                           group_df: pd.DataFrame,
                           one_hot_encoding: bool,
                           prefix_based: bool,
                           ohe_encoders: dict,
                           categorical_columns: list,
                           numerical_columns: list
                           ):
    """
    Create dynamic features.

    Parameters
    ----------
    index: int
        Case id.
    group_df: pd.DataFrame
        Dataframe with all events for one case id.
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    prefix_based: bool
        True if data is generate for a many-to-one approach.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.
    categorical_columns: list
        List of str indicating the columns with categorical features.
    numerical_columns: list
        List of str indicating the columns with numerical features.

    Returns
    -------

    """

    data_large = list()
    labels = list()
    cases = list()

    length = len(group_df)
    if prefix_based:
        start = 1
    else:
        start = length
    labels.append(group_df['remaining_time'].values.tolist())
    cases.extend([index for i in range(len(group_df))])
    for i in range(start, length + 1):
        df_tmp = group_df[:i]
        data_trace = list()
        for index_, row in df_tmp.iterrows():
            data_row = list()
            for col in categorical_columns:
                idx = ohe_encoders[col].transform(np.asarray(row[col]).reshape(1, -1)).astype(int)[0][0]
                if one_hot_encoding:
                    if np.isnan(idx) or idx < 0:
                        data_row.extend([0] * len(ohe_encoders[col].categories_[0]))
                    else:
                        data_row.extend(buildOHE(idx, len(ohe_encoders[col].categories_[0])))
                else:
                    if np.isnan(idx) or idx < 0:
                        data_row.append(0) # This is encoding for 'unknown'
                    else:
                        data_row.append(idx + 1) # Add +1 since 0 is used for masking

            for col in numerical_columns:
                data_row.append(row[col])

            data_trace.append(data_row)

        data_large.append(data_trace)

    return data_large, labels, cases


def get_categorical_column_names(one_hot_encoding: bool,
                                 ohe_encoders: dict,
                                 cols: list):
    """
    Get names of categorical columns.

    Parameters
    ----------
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.
    cols: list
        List of string indicating categorical columns.

    Returns
    -------
    list: List of str with column names that contain categorical values.

    """
    col_names = list()
    if one_hot_encoding:
        for col in cols:
            num_idx = len(ohe_encoders[col].categories_[0])
            col_names.extend([col + '||' + str(i) for i in range(num_idx)])
    else:
        for col in cols:
            col_names.append(col)
    return col_names


def get_column_names(static_as_sequence: bool,
                     cols_dict: dict,
                     one_hot_encoding: bool,
                     ohe_encoders: dict):
    """
    Get column names with dynamic (event) attributes and static (case) attributes, respectively.
    If static_as_sequence is False, case attributes will be duplicated for each timestep and
    therefore regarded as dynamic attributes.

    Parameters
    ----------
    static_as_sequence: bool
        True if event attributes should be duplicated for each timestep.
    cols_dict: dict
        Dictionary containing information about static, dynamic, categorical and numerical columns.
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.

    Returns
    -------
    list: List of str containing names of dynamic columns.
    list: List of str containing names of static columns. If static_as_sequence is False, case attributes will be duplicated for each timestep and
    therefore regarded as dynamic attributes. This list will then be empty.

    """

    if static_as_sequence:
        static_col_names = None
        dynamic_col_names = get_categorical_column_names(one_hot_encoding=one_hot_encoding,
                                                         ohe_encoders=ohe_encoders,
                                                         cols=cols_dict['CATEGORICAL_COLUMNS'])

        for col in cols_dict['NUMERICAL_COLUMNS']:
            dynamic_col_names.append(col)
    else:
        dynamic_col_names = get_categorical_column_names(one_hot_encoding=one_hot_encoding,
                                                         ohe_encoders=ohe_encoders,
                                                         cols=cols_dict['CATEGORICAL_DYNAMIC_COLUMNS'])
        for col in cols_dict['NUMERICAL_DYNAMIC_COLUMNS']:
            dynamic_col_names.append(col)
        static_col_names = get_categorical_column_names(one_hot_encoding=one_hot_encoding,
                                                         ohe_encoders=ohe_encoders,
                                                         cols=cols_dict['CATEGORICAL_STATIC_COLUMNS'])
        for col in cols_dict['NUMERICAL_STATIC_COLUMNS']:
            static_col_names.append(col)

    return dynamic_col_names, static_col_names


def generate_set(df: pd.DataFrame,
                 prefix_based: bool,
                 hdf5_filepath: str,
                 data_name: str,
                 labels_name: str,
                 cases_name: str,
                 scale_label: bool,
                 data_columns_name: str,
                 one_hot_encoding: bool,
                 ohe_encoders: dict,
                 cols_dict: dict,
                 case_id_column: str,
                 label_column: str,
                 maxlen: int,
                 max_label: float,
                 static_data_name: str,
                 static_data_columns_name: str
                 ):
    """
    Creates input features and target labels and stores them to a HDF5 file. Creation is case by case
    in order not to overload main memory.

    Parameters
    ----------
    df: pd.DataFrame
        Event log.
    prefix_based: bool
        Indicates whether data is prepared for prefix-based approaches (many-to-one)
        or trace_based approaches (many-to-many)
    hdf5_filepath: str
        The HDF5 file to be created.
    data_name: str
        The dataset inside the HDF5 file that contains dynamic model input data.
    labels_name: str
        The dataset inside the HDF5 file that contains the target labels.
    cases_name: str
        The dataset inside the HDF5 file that contains the case id's.
    scale_label: bool
        Indicates whether target labels should be scaled.
    data_columns_name: str
        The dataset inside the HDF5 file that stores the columns names corresponding to dynamic input data.
    one_hot_encoding: bool
        True if one hot encoding should be applied. Otherwise, each category is encoded with one integer value.
    ohe_encoders: dict
        Keys are of type string and indicate a column name. The values are of type sklearn.preprocessing.OrdinalEncoder.
    cols_dict: dict
        Dictionary containing information about static, dynamic, categorical and numerical columns.
    case_id_column: str
        Column with case id's.
    label_column: str
        Column that contains the labels.
    maxlen: int
        Maximum trace length.
    max_label: float
        Maximum label value.
    static_data_name: str
        If provided, we save static data not as sequence. If not provided, we save static data as sequence.
    static_data_columns_name: str
        Name of the dataset in the hdf5 file for storing static model inputs.

    Returns
    -------
    int: Maximum trace length.
    float: Maximum label value.

    """

    print('Length of input dataframe: ' + str(len(df)))

    grouped = df.groupby(by=case_id_column)

    if maxlen is None:
        maxlen = grouped.size().max()

    if max_label is None:
        max_label = df[label_column].max()

    data_index = 0
    labels_index = 0
    cases_index = 0
    static_data_index = 0

    if not Path(hdf5_filepath).exists():
        mode = 'w'
    else:
        mode = 'r+'

    if static_data_name is not None:
        static_as_sequence = False
    else:
        static_as_sequence = True

    with h5py.File(hdf5_filepath, mode) as f:
        pbar = tqdm(total=len(grouped))
        for index, group in grouped:

            data, labels, cases, data_static = generate_batch(group_df=group,
                                                              index=index,
                                                              one_hot_encoding=one_hot_encoding,
                                                              prefix_based=prefix_based,
                                                              ohe_encoders=ohe_encoders,
                                                              cols_dict=cols_dict,
                                                              static_as_sequence=static_as_sequence)
            # Needed to get the shape of the data to retrieve models
            data = sequence.pad_sequences(data, maxlen=maxlen, dtype=float)

            if prefix_based:
                labels = np.asarray(list(chain.from_iterable(labels)))
            else:
                labels = sequence.pad_sequences(labels, maxlen=maxlen, dtype=float, value=None)
            if scale_label:
                labels = labels / max_label

            if data_index == 0:
                dynamic_cols, static_cols = get_column_names(static_as_sequence=static_as_sequence,
                                                             cols_dict=cols_dict,
                                                             ohe_encoders=ohe_encoders,
                                                             one_hot_encoding=one_hot_encoding)

                # Store column names
                data_cols = f.create_dataset(data_columns_name, data=dynamic_cols)
                if static_cols is not None:
                    static_data_cols = f.create_dataset(static_data_columns_name, data=static_cols)

                if prefix_based:
                    data_h5 = f.create_dataset(data_name, (len(df), maxlen, data.shape[2]))
                    labels_h5 = f.create_dataset(labels_name, (len(df), ))
                    print('Shape of dynamic dataset: ' + str(data_h5[:].shape))

                    if not static_as_sequence:
                        if data_static is not None:
                            data_static_h5 = f.create_dataset(static_data_name, (len(df), np.asarray(data_static).shape[-1]))
                            print('Shape of static dataset: ' + str(data_static_h5[:].shape))

                else:
                    data_h5 = f.create_dataset(data_name, (len(grouped), maxlen, data.shape[2]))
                    labels_h5 = f.create_dataset(labels_name, (len(grouped), maxlen))
                    print('Shape of dynamic dataset: ' + str(data_h5[:].shape))

                    if not static_as_sequence:
                        if data_static is not None:
                            data_static_h5 = f.create_dataset(static_data_name, (len(grouped), np.asarray(data_static).shape[-1]))
                            print('Shape of static dataset: ' + str(data_static_h5[:].shape))

                cases_h5 = f.create_dataset(cases_name, (len(df), ))
                keys = [key for key in f.keys()]
                if 'max_train_label' not in keys:
                    max_label_h5 = f.create_dataset(name='max_train_label', data=max_label)

            data_h5[data_index: data_index + len(data)] = data
            data_index += len(data)
            labels_h5[labels_index: labels_index + len(labels)] = labels
            labels_index += len(labels)
            cases_h5[cases_index: cases_index + len(cases)] = cases
            cases_index += len(cases)

            if not static_as_sequence:
                if data_static is not None:
                    data_static_h5[static_data_index: static_data_index + len(data_static)] = data_static
                    static_data_index += len(data_static)

            pbar.update(1)
        pbar.close()

    return maxlen, max_label


def _load_dataset_name(filename,
                       config,
                       prefix_based,
                       hdf5_filepath_train,
                       hdf5_filepath_test,
                       scale_label,
                       one_hot_encoding,
                       static_data_as_sequence):
    """
    Initiate creation of training, validation and test set for an event log.

    Parameters
    ----------
    filename: str
        Path to the event log in .csv format.
    config: dict
        Dataset configuration as retrieved from dataset_config.py
    prefix_based: bool
        Indicates whether data is prepared for prefix-based approaches (many-to-one)
        or trace_based approaches (many-to-many)
    hdf5_filepath_train: str
        Filename for storing training and validation data in HDF5 file format.
    hdf5_filepath_test: str
        Filename for storing test data in HDF5 file format.
    scale_label: bool
        Indicates whether the target labels should be scaled.
    one_hot_encoding: bool
        Indicates whether one hot encoding is applied or integer encoding.
    static_data_as_sequence: bool
        Indicates whether case attributes are duplicated for each timestep.

    Returns
    -------
    float: Value of largest label value.

    """
    config = deepcopy(config)
    df = pandas.read_csv(filename, sep=';')

    groups = list()
    current_case_id = 1
    if isinstance(df[config['CASE_ID_COLUMN']].iloc[0], str):
        # Need to create int type for case id to be compatible with hdf5 storage
        for index, group_df in df.groupby(config['CASE_ID_COLUMN']):
            group_df[config['CASE_ID_COLUMN']] = current_case_id
            groups.append(group_df)
            current_case_id += 1

        df = pd.concat(groups)

    cat_dyn_cols = deepcopy(config['CATEGORICAL_DYNAMIC_COLUMNS'])
    cat_stat_cols = deepcopy(config['CATEGORICAL_STATIC_COLUMNS'])
    num_dyn_cols = deepcopy(config['NUMERICAL_DYNAMIC_COLUMNS'])
    num_stat_cols = deepcopy(config['NUMERICAL_STATIC_COLUMNS'])

    # Prepare dataframe for calculation of additional features
    df[config['TIMESTAMP_COLUMN']] = pd.to_datetime(df[config['TIMESTAMP_COLUMN']], format='mixed', infer_datetime_format=True)
    df = df.sort_values(by=[config['CASE_ID_COLUMN'], config['TIMESTAMP_COLUMN']])

    # Calculate target label
    df = feature_calculation.remaining_case_time(df=df,
                                                 timestamp_column=config['TIMESTAMP_COLUMN'],
                                                 case_id_column=config['CASE_ID_COLUMN'],
                                                 new_column_name='remaining_time')

    # Calculate additional features
    df = feature_calculation.day_of_week(df=df,
                                         timestamp_column=config['TIMESTAMP_COLUMN'],
                                         new_column_name='day_of_week')
    cat_dyn_cols.append('day_of_week')

    df = feature_calculation.time_since_midnight(df=df,
                                                 timestamp_column=config['TIMESTAMP_COLUMN'],
                                                 new_column_name='time_since_midnight')
    num_dyn_cols.append('time_since_midnight')

    df = feature_calculation.time_since_sunday(df=df,
                                               timestamp_column=config['TIMESTAMP_COLUMN'],
                                               new_column_name='time_since_sunday')
    num_dyn_cols.append('time_since_sunday')

    df = feature_calculation.time_since_last_event(df=df,
                                                   timestamp_column=config['TIMESTAMP_COLUMN'],
                                                   new_column_name='time_since_last_event',
                                                   case_id_column=config['CASE_ID_COLUMN'])
    num_dyn_cols.append('time_since_last_event')

    df = feature_calculation.time_since_process_start(df=df,
                                                      timestamp_column=config['TIMESTAMP_COLUMN'],
                                                      new_column_name='time_since_start',
                                                      case_id_column=config['CASE_ID_COLUMN'])
    num_dyn_cols.append('time_since_start')

    # Put categorical and numerical cols together in a list for fast access
    categorical_cols = list()
    categorical_cols.extend(cat_dyn_cols)
    categorical_cols.extend(cat_stat_cols)

    numerical_cols = list()
    numerical_cols.extend(num_dyn_cols)
    numerical_cols.extend(num_stat_cols)

    for cat_col in categorical_cols:
        df[cat_col] = df[cat_col].astype(str).astype(pd.CategoricalDtype())

    df[numerical_cols] = df[numerical_cols].fillna(0.)
    df[categorical_cols] = df[categorical_cols].fillna('__NO_VALUE_PROVIDED__')

    cases_sorted = df.groupby(by=config['CASE_ID_COLUMN'])[config['TIMESTAMP_COLUMN']].min().sort_values().index.tolist()
    num_cases = len(cases_sorted)
    num_train_cases = int(num_cases * 0.6)
    num_val_cases = int(num_cases * 0.2)
    training_cases = cases_sorted[:num_train_cases]
    validation_cases = cases_sorted[num_train_cases:num_train_cases + num_val_cases]
    test_cases = cases_sorted[num_train_cases + num_val_cases : ]

    train_df = df[df[config['CASE_ID_COLUMN']].isin(training_cases)].sort_values(by=[config['CASE_ID_COLUMN'], config['TIMESTAMP_COLUMN']])
    validation_df = df[df[config['CASE_ID_COLUMN']].isin(validation_cases)].sort_values(by=[config['CASE_ID_COLUMN'], config['TIMESTAMP_COLUMN']])
    test_df = df[df[config['CASE_ID_COLUMN']].isin(test_cases)].sort_values(by=[config['CASE_ID_COLUMN'], config['TIMESTAMP_COLUMN']])

    ohe_encoders = dict()
    for col in categorical_cols:
        ohe_encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=np.nan).fit(train_df[col].to_numpy().reshape(-1, 1))

    for col in numerical_cols:
        max_value = train_df[col].max()
        if max_value != 0:
            train_df[col] = train_df[col] / max_value
            validation_df[col] = validation_df[col] / max_value
            test_df[col] = test_df[col] / max_value

    # Store information of columns
    dirname = os.path.dirname(hdf5_filepath_train)
    cols_dict = dict()
    cols_dict['CATEGORICAL_DYNAMIC_COLUMNS'] = deepcopy(cat_dyn_cols)
    cols_dict['CATEGORICAL_STATIC_COLUMNS'] = deepcopy(cat_stat_cols)
    cols_dict['NUMERICAL_DYNAMIC_COLUMNS'] = deepcopy(num_dyn_cols)
    cols_dict['NUMERICAL_STATIC_COLUMNS'] = deepcopy(num_stat_cols)
    cols_dict['CATEGORICAL_COLUMNS'] = deepcopy(categorical_cols)
    cols_dict['NUMERICAL_COLUMNS'] = deepcopy(numerical_cols)
    joblib.dump(filename=dirname + '/columns_info.pkl', value=cols_dict)

    # Set variable whether to save static data as sequence or not
    if static_data_as_sequence:
        static_data_names = [None, None, None]
        static_data_columns_names = [None, None, None]
    else:
        static_data_names = ['train_static_X', 'validation_static_X', 'test_static_X']
        static_data_columns_names = ['train_columns_static_X', 'validation_columns_static_X', 'test_columns_static_X']

    maxlen, max_label = generate_set(df=train_df,
                                     prefix_based=prefix_based,
                                     hdf5_filepath=hdf5_filepath_train,
                                     data_name='train_X',
                                     labels_name='train_y',
                                     cases_name='train_cases',
                                     scale_label=scale_label,
                                     data_columns_name='train_columns',
                                     one_hot_encoding=one_hot_encoding,
                                     ohe_encoders=ohe_encoders,
                                     case_id_column=config['CASE_ID_COLUMN'],
                                     label_column='remaining_time',
                                     cols_dict=cols_dict,
                                     max_label=None,
                                     maxlen=None,
                                     static_data_name=static_data_names[0],
                                     static_data_columns_name=static_data_columns_names[0])

    generate_set(df=validation_df,
                 prefix_based=prefix_based,
                 hdf5_filepath=hdf5_filepath_train,
                 maxlen=maxlen,
                 data_name='validation_X',
                 labels_name='validation_y',
                 cases_name='validation_cases',
                 max_label=max_label,
                 scale_label=scale_label,
                 data_columns_name='validation_columns',
                 one_hot_encoding=one_hot_encoding,
                 ohe_encoders=ohe_encoders,
                 case_id_column=config['CASE_ID_COLUMN'],
                 label_column='remaining_time',
                 cols_dict=cols_dict,
                 static_data_name=static_data_names[1],
                 static_data_columns_name=static_data_columns_names[1])

    # Make test set always prefix based
    generate_set(df=test_df,
                 prefix_based=True,
                 hdf5_filepath=hdf5_filepath_test,
                 maxlen=maxlen,
                 data_name='test_X',
                 labels_name='test_y',
                 cases_name='test_cases',
                 max_label=max_label,
                 scale_label=scale_label,
                 data_columns_name='test_columns',
                 one_hot_encoding=one_hot_encoding,
                 ohe_encoders=ohe_encoders,
                 case_id_column=config['CASE_ID_COLUMN'],
                 label_column='remaining_time',
                 cols_dict=cols_dict,
                 static_data_name=static_data_names[2],
                 static_data_columns_name=static_data_columns_names[2])

    return max_label
