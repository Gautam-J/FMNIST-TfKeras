import kerastuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard


def build_initial_hyper_model(hp):
    model = Sequential(name='initial_hyper_model')

    act_func = hp.Choice('activation', ['relu', 'elu'])

    model.add(Conv2D(hp.Int('conv_units1', 8, 256, 32), 3, padding='same',
                     input_shape=(28, 28, 1), use_bias=False, name='conv1'))
    model.add(BatchNormalization(axis=-1, name='conv1_bn'))
    model.add(Activation(act_func))

    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(hp.Int('conv_units2', 32, 256, 32), 3, padding='same',
                     use_bias=False, name='conv2'))
    model.add(BatchNormalization(axis=-1, name='conv2_bn'))
    model.add(Activation(act_func))

    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Conv2D(hp.Int('conv_units3', 64, 256, 32), 3, padding='same',
                     use_bias=False, name='conv3'))
    model.add(BatchNormalization(axis=-1, name='conv3_bn'))
    model.add(Activation(act_func))

    model.add(MaxPool2D(2, name='maxpool3'))

    dense = hp.Boolean('dense')
    if dense:
        model.add(Flatten(name='flatten'))
        n_layers = hp.Choice('ndense_layers', [1, 2, 3])

        for i in range(n_layers):
            model.add(Dense(hp.Int(f'dense_units{i + 1}', 128, 512, 64),
                            use_bias=False, name=f'dense{i + 1}'))
            model.add(BatchNormalization(axis=-1, name=f'dense{i + 1}_bn'))
            model.add(Activation(act_func))

    else:
        model.add(Conv2D(128, 1, use_bias=False, padding='same',
                         name='conv4'))
        model.add(BatchNormalization(axis=-1, name='conv4_bn'))
        model.add(Activation(act_func))

        model.add(GlobalAveragePooling2D(name='global_avg_pooling'))

    model.add(Dense(10, activation='linear', name='output'))

    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    opt_obj = Adam(learning_rate=hp.Float('lr', min_value=1e-4,
                                          max_value=5e-3, sampling='log'))

    model.compile(loss=loss_obj, metrics=['accuracy'], optimizer=opt_obj)

    return model


def build_fine_hyper_model(hp):
    model = Sequential(name='fine_hyper_model')

    model.add(Conv2D(hp.Int('units1', 6, 10, 2), 3, padding='same',
                     activation='relu', input_shape=(28, 28, 1),
                     name='conv1'))
    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(hp.Int('units2', 50, 80, 5), 3, padding='same',
                     activation='relu', name='conv2'))
    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Conv2D(hp.Int('units3', 80, 100, 5), 3, padding='same',
                     activation='relu', name='conv3'))
    model.add(MaxPool2D(2, name='maxpool3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(hp.Int('units4', 100, 130, 5), activation='relu',
                    name='dense1'))

    model.add(Dense(10, activation='linear', name='output'))

    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    opt_obj = Adam(learning_rate=hp.Float('lr', min_value=1e-4,
                                          max_value=1e-3, sampling='log'))

    model.compile(loss=loss_obj, metrics=['accuracy'], optimizer=opt_obj)

    return model


def build_train_model():
    model = Sequential(name='train_model')

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu',
                     name='conv1', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu',
                     name='conv2'))
    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(32, activation='relu', name='dense1'))
    model.add(Dense(10, activation='linear', name='output'))

    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    opt = Adam(learning_rate=3e-4)

    model.compile(loss=loss_obj, optimizer=opt, metrics=['accuracy'])

    return model


def get_tensorboard(log_dir, fine_tuning=False):
    if fine_tuning:
        return TensorBoard(log_dir=f'./logs/hparam_val/{log_dir}')

    return TensorBoard(log_dir=f'./logs/hparam_train/{log_dir}')


def get_tuner(log_dir, max_epochs, fine_tuning=False):
    if fine_tuning:
        return kt.Hyperband(
            build_fine_hyper_model,
            objective='val_accuracy',
            max_epochs=max_epochs,
            executions_per_trial=1,
            directory=f'logs/hparam_val/run_{log_dir}',
            project_name='fmnist'
        )

    return kt.Hyperband(
        build_initial_hyper_model,
        objective='accuracy',
        max_epochs=max_epochs,
        executions_per_trial=1,
        directory=f'logs/hparam_train/run_{log_dir}',
        project_name='fmnist'
    )
