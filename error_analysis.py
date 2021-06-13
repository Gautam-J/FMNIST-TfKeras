import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

from utils.analysis import (
    setup_logger,
    get_error_pct,
    get_class_labels
)
from utils.data import (
    split_data,
    set_random_seed,
    normalize_image
)

logger = setup_logger('error_analysis_logger', log_file='logs/analysislogs.log')
logger.info('=' * 50)

set_random_seed()
logger.info("RNG seed set")

logger.info('Loading data')
(x, y), (_, _) = fashion_mnist.load_data()

x = x.reshape(*x.shape, 1)
y = y.reshape(*y.shape, 1)

logger.info('Normalizing images')
x = normalize_image(x)

logger.info('Splitting data')
_, x_val, _, y_val = split_data(x, y, split_size=0.05)

logger.debug(f'{x_val.shape = }')
logger.debug(f'{y_val.shape = }')

MODEL_PATH = './logs/hparam_train/run_1623504684_5083737/best_model.h5'
model = load_model(MODEL_PATH)
logger.info(f'Model loaded {MODEL_PATH}')

y_pred = np.argmax(model.predict(x_val), axis=1)
incorrect_indices = np.nonzero(y_pred.reshape(-1,) != y_val.reshape(-1,))[0]

model_val_error_pct = get_error_pct(model, x_val, y_val) * 100

logger.info(f'Number of validation examples: {y_val.shape[0]}')
logger.critical(f'Number of wrongly classified examples: {len(incorrect_indices)}')
logger.critical(f'{model_val_error_pct = :.4f} %')

batches = np.split(incorrect_indices, list(range(9, len(incorrect_indices), 9)))
cl = get_class_labels()

batch = 0
running = True
while running and batch < len(batches):
    batch += 1
    fig = plt.figure(figsize=(12, 9))

    for idx, i in enumerate(batches[batch]):
        fig.add_subplot(3, 3, idx + 1)
        img = x_val[i] * 255.
        plt.imshow(img, cmap='gray')
        plt.title(f'Index: {i} True: {cl[int(y_val[i][0])]} Predicted: {cl[y_pred[i]]}')

    plt.tight_layout()
    plt.show()
    plt.close()

    ui = input('Press ENTER to view next batch, "q" to quit: ')
    running = not (ui == 'q')

logger.info('=' * 50)
