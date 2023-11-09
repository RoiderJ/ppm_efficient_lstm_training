from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import time
import pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna
import models
import tensorflow.keras as keras
import tensorflow as tf
from pathlib import Path
import shutil
import joblib


def perform_optuna_study(search_space,
                         num_trials,
                         results_location,
                         training_data_generator,
                         validation_data_generator,
                         dynamic_input_config,
                         static_input_config,
                         num_training_epochs,
                         model_type,
                         max_label,
                         scale_label):
    Path(results_location).mkdir(parents=True, exist_ok=True)
    p = str(Path(results_location).resolve()) # Get absolute path

    if Path(p + '/study.db').is_file():
        shutil.copyfile(p + '/study.db', p + '/study_safety_copy.db')

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(storage='sqlite:///' + p + '/study.db',
                                sampler=sampler,  # TPESampler
                                pruner=None,
                                study_name='study',
                                direction='minimize',
                                load_if_exists=True,
                                directions=None)

    df = study.trials_dataframe()

    # Check how many trials were completed correctly
    completed_trials = list()
    for study in study.trials:
        if study.state == optuna.trial.TrialState.COMPLETE:
            completed_trials.append(study)
    num_completed_trials = len(completed_trials)

    # Do not train again if we have already the required number of trials
    if num_completed_trials >= num_trials:
        df = df[df['state'] == 'COMPLETE']
        df.to_csv(results_location + '/trials.csv')
        return df

    # Delete study before recreating to remove studies which failed. Otherwise, GridSearch won't be fully performed.
    optuna.delete_study(study_name='study', storage='sqlite:///' + p + '/study.db')

    sampler = optuna.samplers.GridSampler(search_space)
    # Recreate study and add trials to keep
    p = str(Path(results_location).resolve())
    study = optuna.create_study(storage='sqlite:///' + p + '/study.db',
                                sampler=sampler,  # TPESampler
                                pruner=None,
                                study_name='study',
                                direction='minimize',
                                load_if_exists=True,
                                directions=None)
    study.add_trials(completed_trials)

    study.optimize(Objective(training_data_generator=training_data_generator,
                             validation_data_generator=validation_data_generator,
                             results_location=results_location,
                             dynamic_input_config=dynamic_input_config,
                             static_input_config=static_input_config,
                             num_training_epochs=num_training_epochs,
                             model_type=model_type,
                             max_label=max_label,
                             scale_label=scale_label),
                             n_trials=num_trials - num_completed_trials)

    df = study.trials_dataframe()
    df.to_csv(results_location + '/trials.csv', index=False)
    return df


def case_based_inference(model,
                         data_generator,
                         max_label,
                         scale_label,
                         seq_len):

    preds = model.predict(data_generator,
                          verbose=1,
                          callbacks=[],
                          max_queue_size=64,
                          workers=8,
                          use_multiprocessing=False)
    labels = data_generator.y[:]
    preds = np.reshape(preds, (len(labels), seq_len))[:, -1]

    if scale_label:
        preds = preds * max_label
        labels = labels * max_label

    data = {'preds': preds, 'labels': labels}
    res_df = pd.DataFrame(data=data)

    return res_df


def predict(model,
            data_generator,
            prefix_based,
            max_label,
            scale_label):

    preds = model.predict(data_generator,
                          verbose=1,
                          callbacks=[],
                          max_queue_size=64,
                          workers=8,
                          use_multiprocessing=False)
    labels = data_generator.y[:]

    if not prefix_based:
        labels = np.asarray(labels).flatten()
        cond = ~np.isnan(np.asarray(labels))
        labels = labels[cond.flatten()]
        preds = np.asarray(preds).flatten()
        preds = preds[cond.flatten()]
    else:
        preds = np.asarray(preds).flatten()
        labels = np.asarray(labels).flatten()

    if scale_label:
        preds = preds * max_label
        labels = labels * max_label

    data = {'preds': preds, 'labels': labels}
    res_df = pd.DataFrame(data=data)

    return res_df


def test(model,
         data_generator,
         prefix_based,
         max_label,
         scale_label):
    res_df, durations_df = predict(model=model,
                                   prefix_based=prefix_based,
                                   max_label=max_label,
                                   scale_label=scale_label,
                                   data_generator=data_generator)
    return mean_absolute_error(y_true=res_df['labels'], y_pred=res_df['preds']) / 3600. / 24., durations_df


class Objective:
    def __init__(self,
                 training_data_generator,
                 validation_data_generator,
                 results_location,
                 dynamic_input_config,
                 static_input_config,
                 num_training_epochs,
                 model_type,
                 scale_label,
                 max_label
                 ):
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.results_location = results_location
        self.dynamic_input_config = dynamic_input_config
        self.static_input_config = static_input_config
        self.num_training_epochs = num_training_epochs
        self.model_type = model_type
        self.scale_label = scale_label
        self.max_label = max_label


    def __call__(self, trial):

        print('CURRENT TRIAL: ', str(trial.number))

        results_location_ = self.results_location + '/trial_' + str(trial.number) + '/'
        Path(results_location_).mkdir(parents=True, exist_ok=True)

        n_neurons = trial.suggest_int('n_neurons', 1, 10000, log=True)
        n_layers = trial.suggest_int('n_layers', 1, 1000)

        # Build model
        model = getattr(models, self.model_type)(seq_len=self.training_data_generator.sequence_len,
                                                 n_neurons=n_neurons,
                                                 n_layers=n_layers,
                                                 inputs_dynamic=self.dynamic_input_config,
                                                 inputs_static=self.static_input_config)

        print(model.summary())

        # Fit model
        self.fit(results_filepath=results_location_,
                 model=model,
                 training_generator=self.training_data_generator,
                 validation_generator=self.validation_data_generator,
                 training_epochs=self.num_training_epochs)

        # Load best model
        with tf.keras.utils.custom_object_scope({'custom_mae_loss': models.custom_mae_loss}):
            testmodel = keras.models.load_model(results_location_ + "/model_weights_best.h5")

        # Predict validation set on best model
        mae = test(model=testmodel,
                   prefix_based=False,
                   max_label=self.max_label,
                   scale_label=self.scale_label,
                   data_generator=self.validation_data_generator)

        del model
        del testmodel
        tf.keras.backend.clear_session()
        return mae

    def fit(self,
            results_filepath,
            model,
            training_generator,
            validation_generator,
            training_epochs):
        # Define callbacks
        early_stopping = EarlyStopping(patience=15, monitor='val_loss')
        model_checkpoint = ModelCheckpoint(results_filepath + "/model_weights_best.h5",
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       patience=10,
                                       verbose=0,
                                       mode='auto',
                                       epsilon=0.0001,
                                       cooldown=0,
                                       min_lr=0)

        start_time = time.time()
        history = model.fit(x=training_generator,
                            validation_data=validation_generator,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer],
                            epochs=training_epochs,
                            max_queue_size=64,
                            workers=8,
                            use_multiprocessing=False)
        end_time = time.time()
        peak_epoch = np.argmin(history.history['val_loss']) + 1

        joblib.dump(value=history.history, filename=results_filepath + '/history.pkl')

        pd.DataFrame(data=[[peak_epoch, end_time - start_time, len(history.history['val_loss'])]],
                     columns=['peak_epoch', 'training_duration', 'num_epochs']).to_csv(results_filepath + '/stats.csv',
                                                                                       index=False)
