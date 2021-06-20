import time
import argparse
from tensorflow.keras.datasets import fashion_mnist

from utils.model_configs import get_tensorboard, get_tuner
from utils.analysis import (
    setup_logger,
    get_error_pct,
    get_f1_score
)
from utils.data import (
    split_data,
    get_dataset,
    set_random_seed,
    normalize_image
)

parser = argparse.ArgumentParser()
parser.add_argument('-fine', action='store_true',
                    help='Flag to use a fine tuning hyper model.')
args = parser.parse_args()

logger = setup_logger('hparam_logger', log_file='logs/hparamlogs.log')
logger.info('=' * 50)

set_random_seed()
logger.info("RNG seed set")

MAX_EPOCHS = 20
BATCH_SIZE = 64
FINE_TUNING = args.fine
logger.critical(f'{MAX_EPOCHS = }')
logger.critical(f'{BATCH_SIZE = }')
logger.critical(f'{FINE_TUNING = }')

logger.info('Loading data')
(x, y), (x_test, y_test) = fashion_mnist.load_data()

x = x.reshape(*x.shape, 1)
y = y.reshape(*y.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)
y_test = y_test.reshape(*y_test.shape, 1)

logger.info('Normalizing images')
x = normalize_image(x)
x_test = normalize_image(x_test)

logger.info('Splitting data')
x_train, x_val, y_train, y_val = split_data(x, y, split_size=0.05)

logger.info(f'Number of train examples: {y_train.shape[0]}')
logger.info(f'Number of validation examples: {y_val.shape[0]}')
logger.info(f'Number of test examples: {y_test.shape[0]}')

logger.debug(f'{x_train.shape = }')
logger.debug(f'{y_train.shape = }')
logger.debug(f'{x_val.shape = }')
logger.debug(f'{y_val.shape = }')
logger.debug(f'{x_test.shape = }')
logger.debug(f'{y_test.shape = }')

logger.info('Creating tf.data.Dataset')
ds_train = get_dataset(x_train, y_train, batch_size=BATCH_SIZE)
ds_val = get_dataset(x_val, y_val, batch_size=BATCH_SIZE)
ds_test = get_dataset(x_test, y_test, test=True)

LOG_DIR = str(time.time()).replace('.', '_')
board = get_tensorboard(LOG_DIR, FINE_TUNING)
tuner = get_tuner(LOG_DIR, MAX_EPOCHS, FINE_TUNING)

if FINE_TUNING:
    logger.info(f'Starting BayesianOptimization search of {MAX_EPOCHS = }')
else:
    logger.info(f'Starting Hyperband search of {MAX_EPOCHS = }')

if FINE_TUNING:
    tuner.search(ds_train, validation_data=ds_val, batch_size=BATCH_SIZE,
                 callbacks=[board], epochs=MAX_EPOCHS)
else:
    tuner.search(ds_train, batch_size=BATCH_SIZE, callbacks=[board])

if FINE_TUNING:
    logger.info('BayesianOptimization search finished')
else:
    logger.info('Hyperband search finished')

best_hp = tuner.get_best_hyperparameters()[0].values
model = tuner.get_best_models()[0]

for k, v in best_hp.items():
    logger.critical(f'{k}: {v}')

model_train_f1_score = get_f1_score(model, x_train, y_train)
logger.critical(f'{model_train_f1_score = :.4f}')
model_val_f1_score = get_f1_score(model, x_val, y_val)
logger.critical(f'{model_val_f1_score = :.4f}')
model_test_f1_score = get_f1_score(model, x_test, y_test)
logger.critical(f'{model_test_f1_score = :.4f}')

model_train_error_pct = get_error_pct(model, x_train, y_train) * 100
logger.critical(f'{model_train_error_pct = :.4f} %')
model_val_error_pct = get_error_pct(model, x_val, y_val) * 100
logger.critical(f'{model_val_error_pct = :.4f} %')
model_test_error_pct = get_error_pct(model, x_test, y_test) * 100
logger.critical(f'{model_test_error_pct = :.4f} %')

if FINE_TUNING:
    model.save(f'logs/hparam_val/run_{LOG_DIR}/best_model.h5')
else:
    model.save(f'logs/hparam_train/run_{LOG_DIR}/best_model.h5')

logger.critical('Best model saved')
model.summary(print_fn=logger.debug)

logger.info('=' * 50)
