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

def buildOHE(index,n):
    L=[0]*n
    L[index]=1
    return L

def load_dataset(name,
                 hdf5_filepath_train,
                 hdf5_filepath_test,
                 scale_label,
                 one_hot_encoding,
                 prefix_based,
                 static_data_as_sequence,
                 data_storage):
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


def generate_batch(index,
                   group_df,
                   one_hot_encoding,  # either True or False. If False, we just return the index number, otherwise full one hot encoding
                   prefix_based,
                   ohe_encoders,
                   cols_dict,
                   static_as_sequence):

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


def generate_static_batch(group_df,
                          one_hot_encoding,
                          prefix_based,
                          ohe_encoders,
                          categorical_columns,
                          numerical_columns):
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


def generate_dynamic_batch(index,
                           group_df,
                           one_hot_encoding,  # either True or False. If False, we just return the index number, otherwise full one hot encoding
                           prefix_based,
                           ohe_encoders,
                           categorical_columns,
                           numerical_columns
                           ):

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


def get_categorical_column_names(one_hot_encoding,
                                 ohe_encoders,
                                 cols):
    col_names = list()
    if one_hot_encoding:
        for col in cols:
            num_idx = len(ohe_encoders[col].categories_[0])
            col_names.extend([col + '||' + str(i) for i in range(num_idx)])
    else:
        for col in cols:
            col_names.append(col)
    return col_names


def get_column_names(static_as_sequence,
                     cols_dict,
                     one_hot_encoding,
                     ohe_encoders):

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


def generate_set(df,
                 prefix_based,
                 hdf5_filepath,
                 data_name,
                 labels_name,
                 cases_name,
                 scale_label,
                 data_columns_name,
                 one_hot_encoding,
                 ohe_encoders,
                 cols_dict,
                 case_id_column,
                 label_column,
                 maxlen,
                 max_label,
                 static_data_name, # If provided, we save static data not as sequence. If not provided, we save static data as sequence.
                 static_data_columns_name
                 ):

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
                labels = list(chain.from_iterable(labels))
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
