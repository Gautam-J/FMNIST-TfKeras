import random
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.python.framework import random_seed


def get_dataset(x, y, batch_size=32, shuffle_size=1024, test=False):
    """Returns a tf.Data.Dataset for the given data.

    Parameters
    ----------
    x : ndarray
        The features of the dataset.
    y : array
        The corresponding labels for the features.
    batch_size : int, default=32
        Batch size of the data.
    shuffle_size : int, default=1000
        Buffer size for shuffling data.
    test : bool, default=False
        Flag to be passed if the given data is test data.

    Returns
    -------
    dataset : tf.data.Dataset
        The processed dataset.

    Notes
    -----
    If `test=True`, the dataset is returned without any batch split, shuffling,
    caching and prefetching.
    """

    dataset = Dataset.from_tensor_slices((x, y))

    if not test:
        dataset = dataset.batch(batch_size).shuffle(shuffle_size)
        dataset = dataset.cache().prefetch(AUTOTUNE)

    return dataset


def split_data(x, y, split_size=0.02):
    """Shuffles and splits the given data into 2 batches.

    Parameters
    ----------
    x : ndarray
        The features to be split.
    y : ndarray
        The corresponding labels to the features.
    split_size : float or int, default=0.02
        The size of the second batch.

    Returns
    -------
    x_train, ndarray
    x_test, ndarray
    y_train, ndarray
    y_test, ndarray
        The split batch data.

    Notes
    -----
    If a `float` is passed as argument to the `split_size` parameter, the
    second batch will contain that many percentage of samples.
    If an `int` is passed, the second batch will contains that many examples.

    Uses sklearn's `train_test_split` for shuffling and splitting of data.
    A seed of value 42 is always passed for shuffling.
    The split is stratified conditioned to the given labels.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=split_size,
        random_state=42,
        stratify=y
    )

    return (
        x_train, x_test,
        y_train, y_test
    )


def normalize_image(x):
    """Normalized the given data by scaling it by 1 / 255.

    Parameters
    ----------
    x : ndarray
        The data to be normalized.

    Returns
    -------
    x : ndarray
        The normalized data.

    Notes
    -----
    Since multiplication is way faster than division, instead of dividing the
    input by 255, (1. / 255) is multiplied to it. Therefore, the input data
    is casted to a float value.
    """

    x = x.astype(np.float64)
    x *= (1. / 255)

    return x


def set_random_seed(seed=42):
    """Sets the seed for random number generation.
    Includes python's random module, numpy and tensorflow.

    Parameters
    ----------
    seed : int, default=42
        The seed for random number generation.
    """

    random.seed(seed)
    np.random.seed(seed)
    random_seed.set_seed(seed)
