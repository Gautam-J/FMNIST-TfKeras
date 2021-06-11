import os
import time
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import TensorBoard

from utils.model_configs import build_train_model
from utils.analysis import (
    setup_logger,
    get_error_pct,
    get_f1_score,
    get_class_labels
)
from utils.data import (
    split_data,
    get_dataset,
    set_random_seed,
    normalize_image
)
from utils.visualizations import (
    save_classification_report,
    save_confusion_matrix,
    plot_tf_model
)

logger = setup_logger('stdoutLogger', log_file='logs/trainlogs.txt')
logger.info('=' * 50)

set_random_seed()
logger.info("RNG seed set")

EPOCHS = 2
BATCH_SIZE = 16
logger.critical(f'{EPOCHS = }')
logger.critical(f'{BATCH_SIZE = }')

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
board = TensorBoard(
    log_dir=f'./logs/train/{LOG_DIR}',
    update_freq='batch',
    write_graph=True,
    write_images=True
)

logger.info('Building model')
model = build_train_model()
model.summary(print_fn=logger.debug)

logger.info('Training started')
model.fit(ds_train, validation_data=ds_val,
          epochs=EPOCHS, verbose=1, batch_size=BATCH_SIZE,
          callbacks=[board])
logger.info('Training finished')

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

if not os.path.exists(f'models/{LOG_DIR}'):
    os.makedirs(f'models/{LOG_DIR}')
    logger.info(f'Created ./models/{LOG_DIR}')

y_pred = np.argmax(model.predict(x_test), axis=-1).reshape(-1, 1)
class_labels = get_class_labels()

save_classification_report(y_test, y_pred, class_labels,
                           directory=f'./models/{LOG_DIR}')
logger.info('Classification Report saved')

save_confusion_matrix([class_labels[i[0]] for i in y_test],
                      [class_labels[i[0]] for i in y_pred],
                      class_labels, directory=f'./models/{LOG_DIR}')
logger.info('Confusion Matrix saved')

plot_tf_model(model, directory=f'./models/{LOG_DIR}')
logger.info('Model architecture saved')

model.save(f'models/{LOG_DIR}/model.h5')
# tf.keras.models.load_model('model.h5')
logger.critical('Model saved')

json_config = model.to_json(indent=2)
with open(f'models/{LOG_DIR}/model_config.json', 'w') as f:
    f.write(json_config)

# with open('model_config.json', 'r') as f:
#     json_config = f.read()
# model = tf.keras.models.model_from_json(json_config)
logger.critical('Model configs saved')

model.save_weights(f'models/{LOG_DIR}/model_weights.h5')
# model.load_weights('model_weights.h5')
logger.critical('Model weights saved')

logger.info('=' * 50)
