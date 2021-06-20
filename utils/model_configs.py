import kerastuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard


def build_initial_hyper_model(hp):
    model = Sequential(name='initial_hyper_model')

    model.add(Conv2D(hp.Int('conv_units1', 32, 256, 32), 3, padding='same',
                     input_shape=(28, 28, 1), use_bias=False, name='conv1'))
    model.add(BatchNormalization(axis=-1, name='conv1_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(hp.Int('conv_units2', 32, 256, 32), 3, padding='same',
                     use_bias=False, name='conv2'))
    model.add(BatchNormalization(axis=-1, name='conv2_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Conv2D(hp.Int('conv_units3', 32, 256, 32), 3, padding='same',
                     use_bias=False, name='conv3'))
    model.add(BatchNormalization(axis=-1, name='conv3_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(hp.Int('dense_units1', 128, 1024, 64),
                    use_bias=False, name='dense1'))
    model.add(BatchNormalization(axis=-1, name='dense1_bn'))
    model.add(Activation('relu'))

    model.add(Dense(10, activation='linear', name='output'))

    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    opt_obj = Adam(learning_rate=hp.Float('lr', min_value=1e-4,
                                          max_value=5e-3, sampling='log'))

    model.compile(loss=loss_obj, metrics=['accuracy'], optimizer=opt_obj)

    return model


def build_fine_hyper_model(hp):
    model = Sequential(name='fine_hyper_model')

    model.add(Conv2D(hp.Int('conv_units1', 100, 200, 10, default=160), 3,
                     padding='same', input_shape=(28, 28, 1), use_bias=False,
                     name='conv1'))
    model.add(BatchNormalization(axis=-1, name='conv1_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(hp.Int('conv_units2', 40, 80, 10, default=64), 3,
                     padding='same', use_bias=False, name='conv2'))
    model.add(BatchNormalization(axis=-1, name='conv2_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Conv2D(hp.Int('conv_units3', 64, 128, 16, default=96), 3,
                     padding='same', use_bias=False, name='conv3'))
    model.add(BatchNormalization(axis=-1, name='conv3_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(hp.Int('dense_units1', 350, 500, 64, default=384),
                    use_bias=False, name='dense1'))
    model.add(BatchNormalization(axis=-1, name='dense1_bn'))
    model.add(Activation('relu'))

    model.add(Dense(10, activation='linear', name='output'))

    loss_obj = SparseCategoricalCrossentropy(from_logits=True)
    opt_obj = Adam(learning_rate=3e-4)

    model.compile(loss=loss_obj, metrics=['accuracy'], optimizer=opt_obj)

    return model


def build_train_model():
    model = Sequential(name='train_model')

    model.add(Conv2D(128, 3,
                     padding='same', input_shape=(28, 28, 1), use_bias=False,
                     name='conv1'))
    model.add(BatchNormalization(axis=-1, name='conv1_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool1'))

    model.add(Conv2D(64, 3,
                     padding='same', use_bias=False, name='conv2'))
    model.add(BatchNormalization(axis=-1, name='conv2_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool2'))

    model.add(Conv2D(128, 3,
                     padding='same', use_bias=False, name='conv3'))
    model.add(BatchNormalization(axis=-1, name='conv3_bn'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, name='maxpool3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(256,
                    use_bias=False, name='dense1'))
    model.add(BatchNormalization(axis=-1, name='dense1_bn'))
    model.add(Activation('relu'))

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
        return kt.BayesianOptimization(
            build_fine_hyper_model,
            objective='val_accuracy',
            max_trials=max_epochs,
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
