import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, TimeDistributed, Input, Dense, BatchNormalization, Embedding, Concatenate, RepeatVector
from tensorflow.keras.optimizers.legacy import Nadam
from tensorflow.keras.models import Sequential


def custom_mae_loss(y_true, y_pred):

    mask = tf.identity(y_true)

    y_pred = tf.where(~tf.math.is_nan(mask), y_pred, 0.)
    y_true = tf.where(~tf.math.is_nan(mask), y_true, 0.)

    y_pred = tf.where(tf.math.is_nan(mask), tf.stop_gradient(y_pred), y_pred)
    y_true = tf.where(tf.math.is_nan(mask), tf.stop_gradient(y_true), y_true)

    absolute_error = tf.math.abs(y_true - y_pred)
    loss = tf.reduce_mean(absolute_error)
    return loss


def get_inference_model(model_type, seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):
    if model_type == 'lstm_case_oh_pre' or model_type == 'lstm_prefix_oh_pre':
        return lstm_prefix_oh_pre(seq_len=seq_len,
                                   n_neurons=n_neurons,
                                   n_layers=n_layers,
                                   inputs_dynamic=inputs_dynamic,
                                   inputs_static=inputs_static)
    elif model_type == 'lstm_case_emb_post' or model_type == 'lstm_prefix_emb_post':
        return lstm_prefix_emb_post(seq_len=seq_len,
                                   n_neurons=n_neurons,
                                   n_layers=n_layers,
                                   inputs_dynamic=inputs_dynamic,
                                   inputs_static=inputs_static)
    elif model_type == 'lstm_case_oh_post' or model_type == 'lstm_prefix_oh_post':
        return lstm_prefix_oh_post(seq_len=seq_len,
                                  n_neurons=n_neurons,
                                  n_layers=n_layers,
                                  inputs_dynamic=inputs_dynamic,
                                  inputs_static=inputs_static)
    elif model_type == 'lstm_case_emb_pre' or model_type == 'lstm_prefix_emb_pre':
        return lstm_prefix_emb_pre(seq_len=seq_len,
                                    n_neurons=n_neurons,
                                    n_layers=n_layers,
                                    inputs_dynamic=inputs_dynamic,
                                    inputs_static=inputs_static)
    elif model_type == 'lstm_prefix_oh_pre_navarin':
        return lstm_prefix_oh_pre_navarin(seq_len=seq_len,
                                           n_neurons=n_neurons,
                                           n_layers=n_layers,
                                           inputs_dynamic=inputs_dynamic,
                                           inputs_static=inputs_static)
    else:
        raise ValueError('Specified Inference model does not exist.')


def lstm_case_oh_pre(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):
    # [['inp1', inp_dim, embed_dim, 'categorical/numerical']]

    num_features = 0
    for i_ in inputs_dynamic:
        num_features += i_[1]
    for i_ in inputs_static:
        num_features += i_[1]

    inp = Input(shape=(seq_len, num_features))
    masked = keras.layers.Masking(mask_value=0.0,
                                  input_shape=(None, seq_len, num_features))(inp)
    x = LSTM(n_neurons,
             use_bias=True,
             input_shape=(seq_len, num_features),
             recurrent_dropout=0.2,
             return_sequences=True)(masked)
    if n_layers > 1:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                     use_bias=True,
                     input_shape=(seq_len, num_features),
                     recurrent_dropout=0.2,
                     return_sequences=True)(x)
    # Access hidden states, take this as the predictions
    x = TimeDistributed((Dense(16, activation='relu')))(x)
    x = TimeDistributed(Dense(1, activation='sigmoid', use_bias=True))(x)
    outp = tf.squeeze(x)
    model = keras.Model(inputs=inp, outputs=outp)
    opt = Nadam()
    model.compile(loss=custom_mae_loss, optimizer=opt)
    return model


def lstm_case_emb_post(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):

    inputs_dyn = list()
    for input in inputs_dynamic:
        if input[3] == 'categorical':
            inputs_dyn.append(tf.keras.Input(shape=(seq_len), name=input[0]))
        elif input[3] == 'numerical':
            inputs_dyn.append(tf.keras.Input(shape=(seq_len, input[1]), name=input[0]))

    concat_inputs_dyn = list()
    for i in range(len(inputs_dyn)):
        if inputs_dynamic[i][3] == 'categorical':
            masked = Embedding(input_dim=inputs_dynamic[i][1] + 1,
                               output_dim=inputs_dynamic[i][2],
                               mask_zero=True,
                               input_shape=inputs_dyn[i].shape)(inputs_dyn[i])
        else:
            masked = keras.layers.Masking(mask_value=0.0,
                                          input_shape=inputs_dyn[i].shape)(inputs_dyn[i])
        concat_inputs_dyn.append(masked)

    conc_dyn = Concatenate(axis=2)(concat_inputs_dyn)

    x = LSTM(n_neurons,
             use_bias=True,
             recurrent_dropout=0.2,
             return_sequences=True)(conc_dyn)
    if n_layers > 1:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                     use_bias=True,
                     recurrent_dropout=0.2,
                     return_sequences=True)(x)

    inputs_stat = list()
    for input in inputs_static:
        if input[3] == 'categorical':
            inp = tf.keras.Input(shape=(1), name=input[0])
            inputs_stat.append(inp)
        elif input[3] == 'numerical':
            inp = tf.keras.Input(shape=(input[1]), name=input[0])
            inputs_stat.append(inp)

    stat_x = list()
    for inp in inputs_stat:
        stat_x.append(RepeatVector(seq_len)(inp))

    embeddings = list()
    for i in range(len(inputs_stat)):
        if inputs_static[i][3] == 'categorical':
            emb = Embedding(input_dim=inputs_static[i][1] + 1,
                            output_dim=inputs_static[i][2],
                            mask_zero=True)(stat_x[i])
            squ = tf.squeeze(emb, axis=2)
            embeddings.append(squ)
        else:
            embeddings.append(stat_x[i])

    embeddings.append(x)

    x = Concatenate(axis=2)(embeddings)
    x = TimeDistributed((Dense(16, activation='relu')))(x)
    x = TimeDistributed(Dense(1, activation='sigmoid', use_bias=True))(x)
    outp = tf.squeeze(x)

    model_inputs = list()
    model_inputs.extend(inputs_dyn)
    model_inputs.extend(inputs_stat)
    model = keras.Model(inputs=model_inputs, outputs=outp)
    opt = Nadam()
    model.compile(loss=custom_mae_loss, optimizer=opt)
    return model


def lstm_case_oh_post(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):

    num_features_dynamic = 0
    for i_ in inputs_dynamic:
        num_features_dynamic += i_[1]

    num_features_static = 0
    for i_ in inputs_static:
        num_features_static += i_[1]

    inp_dyn = Input(shape=(seq_len, num_features_dynamic))
    masked = keras.layers.Masking(mask_value=0.0,
                                  input_shape=(None, seq_len, num_features_dynamic))(inp_dyn)
    x = LSTM(n_neurons,
             use_bias=True,
             input_shape=(seq_len, num_features_dynamic),
             recurrent_dropout=0.2,
             return_sequences=True)(masked)
    if n_layers > 1:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                     use_bias=True,
                     input_shape=(seq_len, num_features_dynamic),
                     recurrent_dropout=0.2,
                     return_sequences=True)(x)

    inp_stat = Input(shape=num_features_static)
    stat_x = RepeatVector(seq_len)(inp_stat)

    x = Concatenate(axis=2)([x, stat_x])
    x = TimeDistributed((Dense(16, activation='relu')))(x)
    # Access hidden states, take this as the predictions
    x = TimeDistributed(Dense(1, activation='sigmoid', use_bias=True))(x)
    outp = tf.squeeze(x)

    model = keras.Model(inputs=[inp_dyn, inp_stat], outputs=outp)
    opt = Nadam()
    model.compile(loss=custom_mae_loss, optimizer=opt)
    print(model.summary())
    return model


def lstm_case_emb_pre(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):

    inputs = list()
    inputs.extend(inputs_dynamic)
    inputs.extend(inputs_static)

    model_inputs = list()
    for input in inputs:
        if input[3] == 'categorical':
            model_inputs.append(tf.keras.Input(shape=(seq_len), name=input[0]))
        elif input[3] == 'numerical':
            model_inputs.append(tf.keras.Input(shape=(seq_len, input[1]), name=input[0]))

    concat_inputs = list()
    for i in range(len(model_inputs)):
        if inputs[i][3] == 'categorical':
            masked = Embedding(input_dim=inputs[i][1] + 1,
                               output_dim=inputs[i][2],
                               mask_zero=True)(model_inputs[i])
        else:
            masked = keras.layers.Masking(mask_value=0.0,
                                          input_shape=(None, seq_len, inputs[i][1]))(model_inputs[i])
        concat_inputs.append(masked)

    conc = Concatenate(axis=2)(concat_inputs)

    x = LSTM(n_neurons,
             use_bias=True,
             recurrent_dropout=0.2,
             return_sequences=True)(conc)
    if n_layers > 1:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                     use_bias=True,
                     recurrent_dropout=0.2,
                     return_sequences=True)(x)

    # Access hidden states, take this as the predictions
    x = TimeDistributed((Dense(16, activation='relu')))(x)
    x = TimeDistributed(Dense(1, activation='sigmoid', use_bias=True))(x)
    outp = tf.squeeze(x)
    model = keras.Model(inputs=model_inputs, outputs=outp)
    opt = Nadam()
    model.compile(loss=custom_mae_loss, optimizer=opt)
    return model


def lstm_prefix_oh_pre_navarin(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):
    num_features = 0
    for i_ in inputs_dynamic:
        num_features += i_[1]
    for i_ in inputs_static:
        num_features += i_[1]

    model = Sequential()

    if n_layers == 1:
        model.add(LSTM(n_neurons,
                       implementation=2,
                       input_shape=(seq_len, num_features),
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_neurons,
                           implementation=2,
                           input_shape=(seq_len, num_features),
                           recurrent_dropout=0.2,
                           return_sequences=True))
            model.add(BatchNormalization())
        model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
        model.add(BatchNormalization())

    # add output layer (regression)
    model.add(Dense(1))

    opt = Nadam()
    # compiling the model, creating the callbacks
    model.compile(loss='mae', optimizer=opt)
    return model


def lstm_prefix_oh_pre(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):
    num_features = 0
    for i_ in inputs_dynamic:
        num_features += i_[1]
    for i_ in inputs_static:
        num_features += i_[1]

    model = Sequential()
    model.add(keras.layers.Masking(mask_value=0.0,
                                   input_shape=(seq_len, num_features)))

    if n_layers == 1:
        model.add(LSTM(n_neurons,
                       implementation=2,
                       input_shape=(seq_len, num_features),
                       recurrent_dropout=0.2))
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_neurons,
                           implementation=2,
                           input_shape=(seq_len, num_features),
                           recurrent_dropout=0.2,
                           return_sequences=True))
        model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))

    model.add(Dense(16, activation='relu'))

    # add output layer (regression)
    model.add(Dense(1, activation='sigmoid'))

    opt = Nadam()
    # compiling the model, creating the callbacks
    model.compile(loss='mae', optimizer=opt)
    return model


def lstm_prefix_emb_pre(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):

    inputs = list()
    inputs.extend(inputs_dynamic)
    inputs.extend(inputs_static)

    model_inputs = list()
    for input in inputs:
        if len(input) > 0:
            if input[3] == 'categorical':
                model_inputs.append(tf.keras.Input(shape=(seq_len), name=input[0]))
            else:
                model_inputs.append(tf.keras.Input(shape=(seq_len, input[1]), name=input[0]))

    concat_inputs = list()
    for i in range(len(model_inputs)):
        if inputs[i][3] == 'categorical':
            masked = Embedding(input_dim=inputs[i][1] + 1,
                                output_dim=inputs[i][2],
                                mask_zero=True)(model_inputs[i])
        else:
            masked = keras.layers.Masking(mask_value=0.0,
                                          input_shape=(None, seq_len, inputs[i][1]))(model_inputs[i])
        concat_inputs.append(masked)

    x = Concatenate(axis=2)(concat_inputs)

    if n_layers == 1:
        x = LSTM(n_neurons,
                 implementation=2,
                 recurrent_dropout=0.2)(x)
    else:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                           implementation=2,
                           recurrent_dropout=0.2,
                           return_sequences=True)(x)
        x = LSTM(n_neurons, implementation=2, recurrent_dropout=0.2)(x)

    x = Dense(16, activation='relu')(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=model_inputs, outputs=outp)

    opt = Nadam()
    model.compile(loss='mae', optimizer=opt)
    return model


def lstm_prefix_oh_post(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):
    num_features_dynamic = 0
    for i_ in inputs_dynamic:
        num_features_dynamic += i_[1]

    num_features_static = 0
    for i_ in inputs_static:
        num_features_static += i_[1]

    inp_dyn = Input(shape=(seq_len, num_features_dynamic))
    x = keras.layers.Masking(mask_value=0.0,
                             input_shape=(None, seq_len, num_features_dynamic))(inp_dyn)

    if n_layers == 1:
        x = LSTM(n_neurons,
                 implementation=2,
                 recurrent_dropout=0.2)(x)
    else:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                           implementation=2,
                           recurrent_dropout=0.2,
                           return_sequences=True)(x)
        x = LSTM(n_neurons, implementation=2, recurrent_dropout=0.2)(x)

    inp_stat = Input(shape=num_features_static)

    x = Concatenate(axis=1)([x, inp_stat])

    x = Dense(16, activation='relu')(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[inp_dyn, inp_stat], outputs=outp)

    opt = Nadam()
    model.compile(loss='mae', optimizer=opt)
    return model


def lstm_prefix_emb_post(seq_len, n_neurons, n_layers, inputs_dynamic, inputs_static):

    dyn_inputs = list()
    for input in inputs_dynamic:
        if input[3] == 'categorical':
            dyn_inputs.append(tf.keras.Input(shape=(seq_len), name=input[0]))
        elif input[3] == 'numerical':
            dyn_inputs.append(tf.keras.Input(shape=(seq_len, input[1]), name=input[0]))

    concat_inputs_dyn = list()
    for i in range(len(dyn_inputs)):
        if inputs_dynamic[i][3] == 'categorical':
            masked = Embedding(input_dim=inputs_dynamic[i][1] + 1,
                               output_dim=inputs_dynamic[i][2],
                               mask_zero=True,
                               input_shape=dyn_inputs[i].shape)(dyn_inputs[i])
        else:
            masked = keras.layers.Masking(mask_value=0.0,
                                          input_shape=dyn_inputs[i].shape)(dyn_inputs[i])
        concat_inputs_dyn.append(masked)

    x = Concatenate(axis=2)(concat_inputs_dyn)


    if n_layers == 1:
        x = LSTM(n_neurons,
                 implementation=2,
                 recurrent_dropout=0.2)(x)
    else:
        for i in range(n_layers - 1):
            x = LSTM(n_neurons,
                     implementation=2,
                     recurrent_dropout=0.2,
                     return_sequences=True)(x)
        x = LSTM(n_neurons, implementation=2, recurrent_dropout=0.2)(x)

    stat_inputs = list()
    for input in inputs_static:
        if input[3] == 'categorical':
            stat_inputs.append(tf.keras.Input(shape=(1), name=input[0]))
        elif input[3] == 'numerical':
            stat_inputs.append(tf.keras.Input(shape=(1, input[1]), name=input[0]))

    embeddings = list()
    for i in range(len(stat_inputs)):
        if inputs_static[i][3] == 'categorical':
            emb = Embedding(input_dim=inputs_static[i][1] + 1,
                            output_dim=inputs_static[i][2],
                            mask_zero=True)(stat_inputs[i])
            squ = tf.squeeze(emb, axis=1)
            embeddings.append(squ)
        else:
            squ = tf.squeeze(stat_inputs[i], axis=1)
            embeddings.append(squ)

    embeddings.append(x)

    x = Concatenate(axis=1)(embeddings)

    x = Dense(16, activation='relu')(x)
    outp = Dense(1, activation='sigmoid')(x)

    model_inputs = list()
    model_inputs.extend(dyn_inputs)
    model_inputs.extend(stat_inputs)
    model = keras.Model(inputs=model_inputs, outputs=outp)

    opt = Nadam()
    model.compile(loss='mae', optimizer=opt)
    return model
