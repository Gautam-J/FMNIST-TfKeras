from sklearn.dummy import DummyClassifier
from tensorflow.keras.datasets import cifar10

from utils.data import set_random_seed, split_data, normalize_image
from utils.analysis import setup_logger, get_error_pct, get_f1_score

logger = setup_logger('baseline_logger', log_file='logs/baselinelogs.txt')
logger.info('=' * 50)

set_random_seed()
logger.info("RNG seed set")

(x, y), (x_test, y_test) = cifar10.load_data()

x = normalize_image(x)
x_test = normalize_image(x_test)

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

model = DummyClassifier()
model.fit(x_train, y_train)

dummy_train_f1_score = get_f1_score(model, x_train, y_train)
logger.critical(f'{dummy_train_f1_score = :.4f}')
dummy_val_f1_score = get_f1_score(model, x_val, y_val)
logger.critical(f'{dummy_val_f1_score = :.4f}')
dummy_test_f1_score = get_f1_score(model, x_test, y_test)
logger.critical(f'{dummy_test_f1_score = :.4f}')

dummy_train_error_pct = get_error_pct(model, x_train, y_train) * 100
logger.critical(f'{dummy_train_error_pct = :.4f} %')
dummy_val_error_pct = get_error_pct(model, x_val, y_val) * 100
logger.critical(f'{dummy_val_error_pct = :.4f} %')
dummy_test_error_pct = get_error_pct(model, x_test, y_test) * 100
logger.critical(f'{dummy_test_error_pct = :.4f} %')

logger.info('=' * 50)
