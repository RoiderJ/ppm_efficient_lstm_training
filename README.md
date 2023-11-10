# Efficient Training of Recurrent Neural Networks for Remaining Time Prediction in Predictive Process Monitoring

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Installation

### Pip
Install all required packaged using:
```pip install -r requirements.txt```

### Anaconda
First create an Anaconda environment:
```conda create --name <NAME_OF_YOUR_CHOICE> --file requirements.txt```

You can use the environment with:
```conda activate <NAME_OF_YOUR_CHOICE>```


## Running Experiments

1. Open ```main.py``` and set the variables ```data_storage``` and ```STORAGE_LOC```. 
   ```data_storage``` is the folder which contains event logs in csv format. The separator
   used is a semicolon (```;```). ```STORAGE_LOC```is the folder to which 
   the prepprocessed data, results of hyperparameter tuning and final results will be saved to.
   
   The data we used is downloaded from ```https://github.com/verenich/time-prediction-benchmark/blob/master/experiments/logdata/logs.zip```.
   
2. Run ```main.py```:
   ```
   python main.py <dataset> <model_type> <seed> <batch_size>
   ```
   
   The parameters are as follows:

   ```<dataset>``` Name of the dataset to be used without .csv extension
   
   ```<model_type``` The architecture to be used for training

   ```seed``` Seed to reproduce experiments.

   ```<batch_size>``` The batch size used for model training.
   
   Example:
    ```
   python main.py helpdesk prefix_oh_pre 7 64
   ```
   
## Model Architectures

We implemented different models for our experiments. When running 
```main.py```, the exact name of the model to be used needs 
to be provided.

```lstm_prefix_oh_pre_navarin``` Original implementation of DA-LSTM. [1]

```lstm_prefix_oh_pre``` Our Base Model which makes changes to DA-LSTM in order to speed up the training.

```lstm_prefix_emb_pre``` Prefix-based sequence encoding, 
                          Embedding layers for categorical features, 
                          Event attributes are concatenated before LSTM layers

```lstm_prefix_oh_post``` Prefix-based sequence encoding, 
                          One-hot encoding for categorical features, 
                          Event attributes are concatenated after LSTM layers

```lstm_prefix_emb_post``` Prefix-based sequence encoding, 
                           Embedding layers for categorical features, 
                           Event attributes are concatenated after LSTM layers

```lstm_trace_oh_pre``` Trace-based sequence encoding, 
                        One-hot encoding for categorical features, 
                        Event attributes are concatenated before LSTM layers

```lstm_trace_emb_post``` Trace-based sequence encoding, 
                          Embedding layers for categorical features, 
                          Event attributes are concatenated before LSTM layers

```lstm_trace_oh_post``` Trace-based sequence encoding, 
                         One-hot encoding for categorical features, 
                         Event attributes are concatenated after LSTM layers

```lstm_trace_emb_pre``` Trace-based sequence encoding, 
                         Embedding layers for categorical features, 
                         Event attributes are concatenated after LSTM layers

## Required Disk Space

We store preprocessed data and results of model training to the provided
location, indicated in ```main.py``` as parameter ```STORAGE_LOC```.

Specifically we create the following files:

```<STORAGE_LOC>/study.db``` SQLite database to store results of hyperparameter 
                             tuning. Created by optuna.

```<STORAGE_LOC>/data.hdf5``` Training and validation data including labels, 
                              column names, and case id's.

```<STORAGE_LOC>/data.hdf5``` Test data (always prefix-based) including labels, 
                              column names, and case id's.

```<STORAGE_LOC>/columns_info.pkl``` Metadata storing information about 
                                     event and case attributes as well as 
                                     categorical and numerical attributes.

```<STORAGE_LOC>/test_preds.csv``` Final predictions on the test set of the 
                                   best model found during hyperparameter optimization.

Additionally, for each hyperparameter trial a folder is created: ```<STORAGE_LOC>/trial_<TRIAL_NUM>``` 

Each of these folder contains the following files:

```<STORAGE_LOC>/trial_<TRIAL_NUM>/stats.csv``` General statistisc like the number of epoch needed for model training.

```<STORAGE_LOC>/trial_<TRIAL_NUM>/history.pkl``` History object provided by keras.

```<STORAGE_LOC>/trial_<TRIAL_NUM>/model_weights_best.h5``` The best model over all epochs.

## Adding new datasets

New datasets must be added to the configuration file ```dataset_config.py``` in the form of 
a dictionary. The name of the dictionary must be the same as the csv-file (without .csv-extension).
The keys are as follows: 

```CASE_ID_COLUMN``` Case identifier

```ACTIVITY_COLUMN``` Activity identifier

```TIMESTAMP_COLUMN``` Completion timestamp of activities.

```CATEGORICAL_DYNAMIC_COLUMNS``` Categorical features which are event attributes.

```NUMERICAL_DYNAMIC_COLUMNS``` Numerical features which are event attributes.

```CATEGORICAL_STATIC_COLUMNS``` Categorical features which are case attributes. If the log 
                                 has no event attributes, set this to an empty list.

```NUMERICAL_STATIC_COLUMNS``` Numerical features which are case attributes. If the log  
                               has no event attributes, set this to an empty list.

## References

[1] Navarin, Nicolo, et al. "LSTM networks for data-aware remaining
    time prediction of business process instances." 2017 IEEE Symposium Series on Computational Intelligence
    (SSCI). IEEE, 2017.
