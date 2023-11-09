# ppm_efficient_lstm_training

- First provide location of event logs and were to save to data and model training results

- Prameters for main.py
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    seed = int(sys.argv[3])
    batch_size = int(sys.argv[4])
  

- Testing is always done in prefix based approach

- Describe what we save to disk
  - hdf5 file for training plus validaton
  - hdf5 file for test set
  - metadata
  - results on test set
  - a folder for each hyperparameter combination with corresponding model and predictions on validation set
  - sqlite database   
    

- If there are problem with memory, try reducing number of threads and batch sizes

- New datasets can be added to dataset_config