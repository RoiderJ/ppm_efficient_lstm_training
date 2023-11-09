import tensorflow as tf
import numpy as np


def get_cat_col_indizes(hdf5_columns,
                        columns_config_list):
    col_indizes = list()
    for col in columns_config_list:
        found_indizes = list()
        for i in range(len(hdf5_columns)):
            if col == hdf5_columns[i].split('||')[0]:
                found_indizes.append(i)
        col_indizes.append(found_indizes)
    return col_indizes


def get_num_col_indizes(hdf5_columns,
                        columns_config_list):
    found_indizes = list()
    for num in columns_config_list:
        for i in range(len(hdf5_columns)):
            if num == hdf5_columns[i]:
                found_indizes.append(i)
    return [found_indizes]


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size,
                 X_dynamic,
                 X_static,
                 X_columns_dynamic,
                 X_columns_static,
                 y,
                 shuffle,
                 columns_config,
                 split_categorical_inputs, # Expect in this case that embedding layer is applied
                 dynamic_input_config = None,
                 static_input_config = None):

        super().__init__()
        self.batch_size = batch_size
        self.X_dynamic = X_dynamic
        self.X_static = X_static
        self.X_columns_dynamic = None
        self.X_columns_static = None
        self.y = y
        self.num_samples = len(y)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.indizes = np.asarray([i for i in range(self.num_samples)])
        self.shuffle = shuffle
        self.columns_config = columns_config
        self.split_categorical_inputs = split_categorical_inputs
        self.sequence_len = X_dynamic.shape[1]
        if self.shuffle:
            np.random.shuffle(self.indizes)

        # Get column indizes
        dynamic_hdf5_columns = list()
        for col in X_columns_dynamic:
            dynamic_hdf5_columns.append(col.decode('utf-8'))
        self.X_columns_dynamic = dynamic_hdf5_columns
        self.dynamic_cat_col_indizes = get_cat_col_indizes(hdf5_columns=dynamic_hdf5_columns,
                                                           columns_config_list=columns_config['CATEGORICAL_DYNAMIC_COLUMNS'])
        self.dynamic_num_col_indizes = get_num_col_indizes(hdf5_columns=dynamic_hdf5_columns,
                                                           columns_config_list=columns_config['NUMERICAL_DYNAMIC_COLUMNS'])

        self.static_cat_col_indizes = []
        self.static_num_col_inidzes = []
        if X_static is not None:
            static_hdf5_columns = list()
            for col in X_columns_static:
                static_hdf5_columns.append(col.decode('utf-8'))

            self.static_cat_col_indizes = get_cat_col_indizes(hdf5_columns=static_hdf5_columns,
                                                              columns_config_list=columns_config['CATEGORICAL_STATIC_COLUMNS'])

            self.static_num_col_indizes = get_num_col_indizes(hdf5_columns=static_hdf5_columns,
                                                              columns_config_list=columns_config['NUMERICAL_STATIC_COLUMNS'])

            self.X_columns_static = static_hdf5_columns

        self.dynamic_input_config = dynamic_input_config
        self.static_input_config = static_input_config

        if dynamic_input_config is None and static_input_config is None:
            self.get_number_of_categorical_values()

    def prepare_data(self,
                     batch_indizes):
        sorted_indizes = np.sort(batch_indizes)
        # indizes = [i for i in range(len(batch_indizes))]
        #if self.shuffle:
        #    np.random.shuffle(indizes)
        try:
            labels = self.y[sorted_indizes]#[indizes]
            labels = np.expand_dims(labels, axis=-1)
        except:
            print(sorted_indizes)
            print(self.y)
            raise

        set_input_config = False
        if self.dynamic_input_config is None and self.static_input_config is None:
            set_input_config = True

        # Only dynamic data, no embedding
        if self.X_static is None:
            if not self.split_categorical_inputs:
                data = self.X_dynamic[sorted_indizes]#[indizes]
                if set_input_config:
                    self.dynamic_input_config = [['inp1', data.shape[-1], None, 'all_datatypes']]
                    self.static_input_config = []
                return data, labels

            else:
                data = self.X_dynamic[sorted_indizes]#[indizes]

                stacked_data = list()

                dynamic_input_config = list()

                i = 0
                # Add categorical data
                j = 0
                for inds in self.dynamic_cat_col_indizes:
                    if len(inds) > 0:
                        i += 1
                        if set_input_config:
                            dynamic_input_config.append(['inp_' + str(i),
                                                          int(self.num_dynamic_cat_values[j]),
                                                          int(np.around(np.power(self.num_dynamic_cat_values[j],(1/4)))),
                                                          'categorical'])

                        # We will use an embedding layer, therefore we reduce the input dim
                        d = data[:, :, inds]
                        d = np.reshape(d, (d.shape[0], d.shape[1]))
                        stacked_data.append(d)
                        j += 1

                # Add numerical data
                for inds in self.dynamic_num_col_indizes:
                    if len(inds) > 0:
                        i += 1
                        if set_input_config:
                            dynamic_input_config.append(['inp_' + str(i),
                                                         len(inds),
                                                         None,
                                                         'numerical'])
                        stacked_data.append(data[:, :, inds])

                if set_input_config:
                    self.dynamic_input_config = dynamic_input_config
                    self.static_input_config = []

                return stacked_data, labels
        else:
            if not self.split_categorical_inputs:
                dynamic_data = self.X_dynamic[sorted_indizes]#[indizes]
                static_data = self.X_static[sorted_indizes]#[indizes]

                if set_input_config:
                    self.dynamic_input_config = [['inp1', dynamic_data.shape[-1], None, 'all_datatypes']]
                    self.static_input_config = [['inp2', static_data.shape[-1], None, 'all_datatypes']]

                return [dynamic_data, static_data], labels
            else:
                dynamic_data = self.X_dynamic[sorted_indizes]#[indizes]
                static_data = self.X_static[sorted_indizes]#[indizes]

                stacked_data = list()

                dynamic_input_config = list()
                static_input_config = list()

                i = 0
                j = 0
                for inds in self.dynamic_cat_col_indizes:
                    if len(inds) > 0:
                        # We will use an embedding layer, therefore we reduce the input dim
                        d = dynamic_data[:, :, inds]
                        d = np.reshape(d, (d.shape[0], d.shape[1]))
                        stacked_data.append(d)
                        i += 1
                        if set_input_config:
                            dynamic_input_config.append(['inp_' + str(i),
                                                         int(self.num_dynamic_cat_values[j]),
                                                         int(np.around(np.power(self.num_dynamic_cat_values[j],(1/4)))),
                                                         'categorical'])
                        j += 1


                for inds in self.dynamic_num_col_indizes:
                    if len(inds) > 0:
                        stacked_data.append(dynamic_data[:, :, inds])
                        i += 1
                        if set_input_config:
                            dynamic_input_config.append(['inp_' + str(i),
                                                         len(inds),
                                                         None,
                                                         'numerical'])

                j = 0
                for inds in self.static_cat_col_indizes:
                    if len(inds) > 0:
                        stacked_data.append(static_data[:, inds])
                        i += 1
                        if set_input_config:
                            static_input_config.append(['inp_' + str(i),
                                                        int(self.num_static_cat_values[j]),
                                                        int(np.around(np.power(self.num_static_cat_values[j],(1/4)))),
                                                        'categorical'])
                    j += 1

                for inds in self.static_num_col_indizes:
                    if len(inds) > 0:
                        stacked_data.append(static_data[:, inds])
                        i += 1
                        if set_input_config:
                            static_input_config.append(['inp_' + str(i),
                                                        len(inds),
                                                        None,
                                                        'numerical'])

                self.dynamic_input_config = dynamic_input_config
                self.static_input_config = static_input_config

                return stacked_data, labels

    def __getitem__(self,
                    idx: int):
        if idx + 1 == self.num_batches:
            batch_indizes = self.indizes[idx * self.batch_size:]
        else:
            batch_indizes = self.indizes[idx * self.batch_size: idx * self.batch_size + self.batch_size]
        return self.prepare_data(batch_indizes=batch_indizes)

    def __len__(self):
        return self.num_batches

    def get_num_samples(self):
        return self.num_samples

    def on_epoch_end(self):
        # Reshuffle if needed
        if self.shuffle:
            np.random.shuffle(self.indizes)


    def get_max_values_emb(self,
                           cat_col_indizes,
                           X,
                           sequence_data):

        col_maximas = [[] for i in range(len(cat_col_indizes))]

        for batch_num in range(self.num_batches):
            if batch_num - 1 == self.num_batches:
                tmp_data = X[batch_num * self.batch_size:]
            else:
                tmp_data = X[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]

            i = 0
            for idx in cat_col_indizes:
                if sequence_data:
                    col_data = tmp_data[:, :, idx]
                else:
                    col_data = tmp_data[:, idx]
                try:
                    col_maximas[i].append(np.max(col_data))
                except:
                    print(col_data)
                    print(cat_col_indizes)
                    raise
                i += 1

        return np.max(np.asarray(col_maximas), axis=1)


    def get_number_of_categorical_values(self):
        if self.split_categorical_inputs:
            self.num_dynamic_cat_values = self.get_max_values_emb(cat_col_indizes=self.dynamic_cat_col_indizes,
                                                                  X=self.X_dynamic,
                                                                  sequence_data=True)

            if len(self.static_cat_col_indizes) > 0:
                self.num_static_cat_values = self.get_max_values_emb(cat_col_indizes=self.static_cat_col_indizes,
                                                                     X=self.X_static,
                                                                     sequence_data=False)
            else:
                self.num_static_cat_values = None
        else:
            # We can just read out len of one hot encoded vectors
            self.num_dynamic_cat_values = list()
            for col_ in self.columns_config['CATEGORICAL_DYNAMIC_COLUMNS']:
                i = 0
                for col in self.X_columns_dynamic:
                    if col_ == col.split('||')[0]:
                        i += 1
                self.num_dynamic_cat_values.append(i)

            if self.X_columns_static is not None:
                self.num_static_cat_values = list()
                for col_ in self.columns_config['CATEGORICAL_STATIC_COLUMNS']:
                    i = 0
                    for col in self.X_columns_static:
                        if col_ == col.split('||')[0]:
                            i += 1
                    self.num_static_cat_values.append(i)
            else:
                self.num_static_cat_values = None

    def get_model_input_information(self):
        self.__getitem__(idx=0)
        return self.dynamic_input_config, self.static_input_config, self.sequence_len
