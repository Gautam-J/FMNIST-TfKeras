import logging
import numpy as np
from sklearn.metrics import hamming_loss, f1_score


def setup_logger(name, formatter=None, log_file=None, level=logging.DEBUG):
    """Set up a python logger to log results.

    Parameters
    ----------
    name : str
        Name of the logger.

    formatter : logging.Formatter, default=None
        A custom formatter for the logger to output. If None, a default
        formatter of format `"%Y-%m-%d %H:%M:%S LEVEL MESSAGE"` is used.

    log_file : str, default=None
        File path to record logs. Must end with a readable extension. If None,
        the logs are not logged in any file, and are logged only to `stdout`.

    level : LEVEL or int, default=logging.DEBUG (10)
        Base level to log. Any level lower than this level will not be logged.

    Returns
    -------
    logger : logging.Logger
        A logger with formatters and handlers attached to it.

    Notes
    -----
    If passing a directory name along with the log file, make sure the
    directory exists.
    """

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger


def get_error_pct(model, x, y_true):
    """Get the percentage of samples wrongly classified.

    Parameters
    ----------
    model : tf.keras.Model or sklearn.model
        The model used to compute error percentange.
    x : ndarray
        The features used to compute error percentage.
    y_true : array
        The corresponding true labels for the given features.

    Returns
    -------
    error_pct : float
        The error percentage of the model, on the given dataset. 0 means there
        was no single incorrect classification, 1 means that all examples were
        misclassified.

    Notes
    -----
    The error percentage is mathematically equal to
    `(1 - accuracy(y_true, y_pred))`, where `accuracy` is the fraction of
    samples correctly classified.

    Sklearn's `hamming_loss` method is used to compute the error.
    """

    y_pred = _get_y_pred(model, x)
    error_pct = hamming_loss(y_true, y_pred)

    return error_pct


def get_f1_score(model, x, y_true):
    """Computes the F1 Score of the predictions of the given model.

    Parameters
    ----------
    model : tf.keras.Model or sklearn.model
        The model to be used to predict classes, which will be used
        to compute the f1 score.
    x : ndarray
        The features used to predict classes.
    y_true : ndarray
        The true labels that will be used to compute f1 score.

    Returns
    -------
    score : float
        The f1 score of the model, for the given features and true labels.

    Notes
    -----
    To compute the f1 score, sklearn's `f1_score` method is used,
    along with `"macro"` average.
    """

    y_pred = _get_y_pred(model, x)
    score = f1_score(y_true, y_pred, average='macro')

    return score


def _get_y_pred(model, x):
    """Private function used to predict labels, given model and features.

    Parameters
    ----------
    model : tf.keras.Model or sklearn.model
        The model to be used for predicting labels.
    x : ndarray
        The features used to predict labels.

    Returns
    -------
    y_pred : array
        The predicted classes of the features,
        taken using `np.argmax` function.

    Notes
    -----
    If a sklearn model is given, the model automatically returns the predicted
    classes. In case of a tf.keras.Model model, since the model returns a
    probability distribution for each sample, `argmax` function is used to
    extract the classes.
    """

    y_pred = model.predict(x)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    return y_pred
