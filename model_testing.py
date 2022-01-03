import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

from utils.analysis import get_class_labels
from utils.data import (
    set_random_seed,
    normalize_image
)

set_random_seed()
(_, _), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.reshape(*x_test.shape, 1)
y_test = y_test.reshape(*y_test.shape, 1)

x_test = normalize_image(x_test)

MODEL_PATH = './logs/hparam_train/run_1623504684_5083737/best_model.h5'
model = load_model(MODEL_PATH)

y_pred = np.argmax(model.predict(x_test), axis=1)
correct_indices = np.nonzero(y_pred.reshape(-1,) == y_test.reshape(-1,))[0]

batches = np.split(correct_indices, list(range(9, len(correct_indices), 9)))
cl = get_class_labels()

batch = 0
running = True
while running and batch < len(batches):
    batch += 1
    fig = plt.figure(figsize=(12, 9))

    for idx, i in enumerate(batches[batch]):
        fig.add_subplot(3, 3, idx + 1)
        img = x_test[i] * 255.
        plt.imshow(img, cmap='gray')
        plt.title(f'Index: {i} True: {cl[int(y_test[i][0])]} Predicted: {cl[y_pred[i]]}')

    plt.tight_layout()
    plt.show()
    plt.close()

    ui = input('Press ENTER to view next batch, "q" to quit: ')
    running = not (ui == 'q')
