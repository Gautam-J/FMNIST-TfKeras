from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


def build_initial_hyper_model(hp):

    model = Sequential(name='initial_hyper_model')

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


def build_fine_hyper_model(hp):
    model = Sequential(name='fine_hyper_model')

    return model


def build_train_model():
    model = Sequential(name='train_model')

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu',
                     name='conv1', input_shape=(32, 32, 3)))
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
