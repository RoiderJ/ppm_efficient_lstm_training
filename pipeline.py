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


def perform_optuna_study(search_space: dict,
                         num_trials: int,
                         results_location: str,
                         training_data_generator: tf.keras.utils.Sequence,
                         validation_data_generator: tf.keras.utils.Sequence,
                         dynamic_input_config: list,
                         static_input_config: list,
                         num_training_epochs: int,
                         model_type: str,
                         max_label: float,
                         scale_label: bool):
    """
    Perform full hyperparameter tuning with grid search.

    Parameters
    ----------
    search_space: dict
        Keys are of type string and indicate the name of a hyperparameter. Values are of type list and
        give the individual values samples for grid search.
    num_trials: int
        The number of trials to be performed. Sometimes studies might have to be continued, therefore
        this number might be lower than the total number of all hyperparameter combinatins.
    results_location: str
        The folder to which the results will be saved.
    training_data_generator: tf.keras.utils.Sequence
        Data generator for training data.
    validation_data_generator: tf.keras.utils.Sequence
        Data generator for validation data.
    dynamic_input_config: list
        Standardized list to indicate ```[input_name, input_dimension, embedding_dimension, data_type``` for
        each input features, e. g.
        ```[['inp1', inp_dim, embed_dim, 'categorical/numerical']]```.
    static_input_config: list
        Standardized list to indicate ```[input_name, input_dimension, embedding_dimension, data_type``` for
        each input features, e. g.
        ```[['inp1', inp_dim, embed_dim, 'categorical/numerical']]```.
    num_training_epochs: int
        The number of maximum training epochs per model.
    model_type: str
        The model architecture to be used for training.
    max_label: float
        Maximum label value.
    scale_label: bool
        Whether labels are scaled or not.

    Returns
    -------
    pd.DataFrame: Dataframe containing info produced by optuna for hyperparameter tuning results and metadata.

    """
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


def predict(model: tf.keras.Model,
            data_generator: tf.keras.utils.Sequence,
            prefix_based: bool,
            max_label: float,
            scale_label: bool):
    """
    Predict a given dataset.

    Parameters
    ----------
    model: tf.keras.Model
        A trained model that is used for predictions.
    data_generator: tf.keras.utils.Sequence
        The data generator providing the input data.
    prefix_based: bool
        Indicates whether the model architecture is prefix-based
    max_label: float
        The maximum label value.
    scale_label: bool
        Indicates whether the target labels of the data generator are scaled.

    Returns
    -------
    pd.DataFrame: A Dataframe with all the predictions and target labels.

    """

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


def test(model: tf.keras.Model,
         data_generator: tf.keras.utils.Sequence,
         prefix_based: bool,
         max_label: float,
         scale_label: bool):
    """
    Apply a model and return Mean Absolute Error.

    Parameters
    ----------
    model: tf.keras.Model
        A trained model that is used for predictions.
    data_generator: tf.keras.utils.Sequence
        The data generator providing the input data.
    prefix_based: bool
        Indicates whether the model architecture is prefix-based
    max_label: float
        The maximum label value.
    scale_label: bool
        Indicates whether the target labels of the data generator are scaled.

    Returns
    -------
    float: Mean Absolute Error the model achieved.

    """
    res_df = predict(model=model,
                                   prefix_based=prefix_based,
                                   max_label=max_label,
                                   scale_label=scale_label,
                                   data_generator=data_generator)
    return mean_absolute_error(y_true=res_df['labels'], y_pred=res_df['preds']) / 3600. / 24.


class Objective:
    """
    Objective that defines the procedure for one optuna trial.
    """
    def __init__(self,
                 training_data_generator: tf.keras.utils.Sequence,
                 validation_data_generator: tf.keras.utils.Sequence,
                 results_location: str,
                 dynamic_input_config: list,
                 static_input_config: list,
                 num_training_epochs: int,
                 model_type: str,
                 scale_label: bool,
                 max_label: float
                 ):
        """
        Constructor of the class.

        Parameters
        ----------
        training_data_generator: tf.keras.utils.Sequence
            Generator providing training data.
        validation_data_generator: tf.keras.utils.Sequence
            Generator providing validation data.
        results_location: str
            Folder to which the results will be saved.
        dynamic_input_config: list
            Standardized list to indicate ```[input_name, input_dimension, embedding_dimension, data_type``` for
            each input features, e. g.
            ```[['inp1', inp_dim, embed_dim, 'categorical/numerical']]```.
        static_input_config: list
            Standardized list to indicate ```[input_name, input_dimension, embedding_dimension, data_type``` for
            each input features, e. g.
            ```[['inp1', inp_dim, embed_dim, 'categorical/numerical']]```.
        num_training_epochs: int
            Number of maximum training epochs for the model to be trained.
        model_type: str
            The model architecture to be used.
        scale_label: bool
            Indicates whether the target labels of the generators are scaled.
        max_label: float
            The maximum value of all training labels.
        """
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.results_location = results_location
        self.dynamic_input_config = dynamic_input_config
        self.static_input_config = static_input_config
        self.num_training_epochs = num_training_epochs
        self.model_type = model_type
        self.scale_label = scale_label
        self.max_label = max_label


    def __call__(self,
                 trial: optuna.trial.Trial):
        """
        Function to run one hyperparameter trial. Automatically called by optuna.

        Parameters
        ----------
        trial: optuna.trial.Trial
            Trial object passed by optuna.

        Returns
        -------
        float: Mean absolute error which the model achieved on the validation set.

        """

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
            results_filepath: str,
            model: tf.keras.Model,
            training_generator: tf.keras.utils.Sequence,
            validation_generator: tf.keras.utils.Sequence,
            training_epochs: int):
        """
        Train a neural network.

        Parameters
        ----------
        results_filepath: str
            Location to which the training results will be saved.
        model: tf.keras.Model
            Model to be trained.
        training_generator: tf.keras.utils.Sequence
            Generator providing the training data.
        validation_generator: tf.keras.utils.Sequence
            Generator prodiving the validation data.
        training_epochs: int
            Maximum number of training epochs for a model.

        Returns
        -------
        None

        """
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
