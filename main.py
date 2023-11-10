import sys
import os
import random
import numpy as np
from pathlib import Path
import joblib
import h5py
import prepare_dataset
import models
import dataloader
import tensorflow as tf
from tensorflow import keras
import pipeline
from sklearn.metrics import mean_absolute_error

# Please add
data_storage = '' # Location of event logs
STORAGE_LOC = '' # Location to save datasets and results of model training to

# Hyperparameters four our experiments
search_space = {
        'n_neurons': [100, 150, 200], # removed 250 compared to Navarin since they reported that 200 and more overfits
        'n_layers': [1, 2, 3, 4, 5, 6]
    }

# Number of maximum training epochs per model
num_training_epochs = 500

# Parameter required for optina. Does not have to be changed!
num_random_trials = 0


if __name__ == '__main__':
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    seed = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    # Set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # find number of hyperparameter combinations
    num_hp_trials = 1
    for v in search_space.values():
        num_hp_trials = num_hp_trials * len(v)

    # Set filepaths for storing datasets and metadata
    results_location = STORAGE_LOC + '/' + str(dataset) + '/' + model_type + '/'
    hdf5_filepath = results_location + '/data.hdf5'
    hdf5_filepath_test = results_location + '/test_data.hdf5'
    cols_filepath = results_location + '/columns_info.pkl'
    Path(results_location).mkdir(parents=True, exist_ok=True)
    p = str(Path(results_location).resolve())

    # Set parameters for the defined model type that is applied
    scale_label = True
    if model_type == 'lstm_trace_oh_pre':
        prefix_based = False
        one_hot_encoding = True
        static_data_as_sequence = True
    elif model_type == 'lstm_trace_emb_post':
        prefix_based = False
        one_hot_encoding = False
        static_data_as_sequence = False
    elif model_type == 'lstm_trace_oh_post':
        prefix_based = False
        one_hot_encoding = True
        static_data_as_sequence = False
    elif model_type == 'lstm_trace_emb_pre':
        prefix_based = False
        one_hot_encoding = False
        static_data_as_sequence = True
    elif model_type == 'lstm_prefix_oh_pre_navarin':
        prefix_based = True
        one_hot_encoding = True
        scale_label = False
        static_data_as_sequence = True
    elif model_type == 'lstm_prefix_oh_pre':
        prefix_based = True
        one_hot_encoding = True
        static_data_as_sequence = True
    elif model_type == 'lstm_prefix_emb_pre':
        prefix_based = True
        one_hot_encoding = False
        static_data_as_sequence = True
    elif model_type == 'lstm_prefix_oh_post':
        prefix_based = True
        one_hot_encoding = True
        static_data_as_sequence = False
    elif model_type == 'lstm_prefix_emb_post':
        prefix_based = True
        one_hot_encoding = False
        static_data_as_sequence = False
    else:
        raise ValueError('Specified model does not exist.')

    # Create dataset and save to disk as hdf5 file
    # One file is created for training and validation and one is create for test set.
    if not Path(hdf5_filepath).exists():
        print('Creating hdf5 dataset.')
        max_label = prepare_dataset.load_dataset(name=dataset,
                                                   prefix_based=prefix_based,
                                                   hdf5_filepath_train=hdf5_filepath,
                                                   hdf5_filepath_test=hdf5_filepath_test,
                                                   scale_label=scale_label,
                                                   one_hot_encoding=one_hot_encoding,
                                                   static_data_as_sequence=static_data_as_sequence,
                                                   data_storage=data_storage)
    else:
        print('Loading existing hdf5 dataset.')
        with h5py.File(hdf5_filepath, 'r') as f:
            max_label = f['max_train_label'][()]

    columns_info = joblib.load(cols_filepath)

    # Perform hyperparameter tuning
    with h5py.File(hdf5_filepath, 'r') as f:
        keys = [key for key in f.keys()]

        training_generator = dataloader.DataGenerator(batch_size,
                                                      X_dynamic=f['train_X'],
                                                      X_static=f['train_static_X'] if 'train_static_X' in keys else None,
                                                      X_columns_dynamic=f['train_columns'],
                                                      X_columns_static=f['train_columns_static_X'] if 'train_static_X' in keys else None,
                                                      y=f['train_y'],
                                                      shuffle=True,
                                                      columns_config=columns_info,
                                                      split_categorical_inputs= not one_hot_encoding)

        dynamic_input_config, static_input_config, seq_len = training_generator.get_model_input_information()

        validation_generator = dataloader.DataGenerator(batch_size,
                                                        X_dynamic=f['validation_X'],
                                                        X_static=f['validation_static_X'] if 'validation_static_X' in keys else None,
                                                        X_columns_dynamic=f['validation_columns'],
                                                        X_columns_static=f['validation_columns_static_X'] if 'validation_static_X' in keys else None,
                                                        y=f['validation_y'],
                                                        shuffle=False,
                                                        columns_config=columns_info,
                                                        split_categorical_inputs= not one_hot_encoding,
                                                        dynamic_input_config=dynamic_input_config,
                                                        static_input_config=static_input_config)

        trials_df = pipeline.perform_optuna_study(search_space=search_space,
                                                       num_trials=num_hp_trials,
                                                       results_location=results_location,
                                                       training_data_generator=training_generator,
                                                       validation_data_generator=validation_generator,
                                                       dynamic_input_config=dynamic_input_config,
                                                       static_input_config=static_input_config,
                                                       num_training_epochs=num_training_epochs,
                                                       model_type=model_type,
                                                       max_label=max_label,
                                                       scale_label=scale_label)

    # Final Evaluation

    # Get model which performed best on validation set
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']
    trials_df = trials_df[trials_df['value'] == trials_df['value'].max()]
    best_trial = trials_df['number'].iloc[0]

    model_location = results_location + '/trial_' + str(best_trial) + '/'
    with tf.keras.utils.custom_object_scope({'custom_mae_loss': models.custom_mae_loss}):
        training_model = keras.models.load_model(model_location + '/model_weights_best.h5')

    # For inference, we don't want a many-to-many approach, but get a prediction only after the last input event
    # We load a model that predicts in a many-to-one approach and transfer the weights from a many-to-many model to it.
    # If trained model is already many-to-one, weights are also transferred, but this could be ommitted.
    model_inf = models.get_inference_model(model_type=model_type,
                                                 seq_len=seq_len,
                                                 n_neurons = trials_df['params_n_neurons'].iloc[0],
                                                 n_layers = trials_df['params_n_layers'].iloc[0],
                                                 inputs_dynamic = dynamic_input_config,
                                                 inputs_static = static_input_config)
    model_inf.set_weights(training_model.get_weights())

    # Test model
    del training_model
    with h5py.File(hdf5_filepath_test, 'r') as f:
        keys = [key for key in f.keys()]
        test_generator = dataloader.DataGenerator(batch_size=1,
                                                  X_dynamic=f['test_X'],
                                                  X_static=f['test_static_X'] if 'test_static_X' in keys else None,
                                                  X_columns_dynamic=f['test_columns'],
                                                  X_columns_static=f[
                                                      'test_columns_static_X'] if 'test_static_X' in keys else None,
                                                  y=f['test_y'],
                                                  shuffle=False,
                                                  columns_config=columns_info,
                                                  split_categorical_inputs=not one_hot_encoding,
                                                  dynamic_input_config=dynamic_input_config,
                                                  static_input_config=static_input_config)

        res_df = pipeline.predict(model=model_inf,
                                  data_generator=test_generator,
                                  max_label=max_label,
                                  prefix_based=False,
                                  scale_label=scale_label)

        res_df.to_csv(results_location + '/test_preds.csv', index=False)

        print('Result of best model: ', mean_absolute_error(y_true=res_df['labels'], y_pred=res_df['preds']) / 3600. / 24.)
