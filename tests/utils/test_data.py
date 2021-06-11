import pytest
import numpy as np

from utils.data import (
    get_dataset,
    split_data,
    normalize_image
)


@pytest.fixture
def get_sample_data():
    x = np.random.randn(100, 4)
    y = np.random.randn(100, 1)

    return x, y


def test_get_dataset_train(get_sample_data):
    x, y = get_sample_data
    ds = get_dataset(x, y)

    assert ds.element_spec[0].shape[0] is None
    assert ds.element_spec[1].shape[0] is None
    assert ds.element_spec[0].shape[1] == x.shape[-1]
    assert ds.element_spec[1].shape[1] == y.shape[-1]


def test_get_dataset_test(get_sample_data):
    x, y = get_sample_data
    ds = get_dataset(x, y, test=True)

    assert ds.element_spec[0].shape[0] == x.shape[-1]
    assert ds.element_spec[1].shape[0] == y.shape[-1]


@pytest.fixture
def get_labelled_data():
    x = np.arange(1, 221).reshape(-1, 2)
    y = np.c_[np.ones((55)), np.zeros((55))].reshape(-1, 1)

    return x, y


@pytest.mark.parametrize('fraction', [0.04, 0.1, 0.5])
def test_split_data_fraction(get_labelled_data, fraction):
    x, y = get_labelled_data
    x_train, x_test, y_train, y_test = split_data(
        x, y, split_size=fraction
    )

    assert x_train.dtype == x.dtype
    assert y_train.dtype == y.dtype
    assert x_test.dtype == x.dtype
    assert y_test.dtype == y.dtype

    assert x_train.shape[-1] == x.shape[-1]
    assert y_train.shape[-1] == y.shape[-1]
    assert x_test.shape[-1] == x.shape[-1]
    assert y_test.shape[-1] == y.shape[-1]

    assert x_train.shape[0] == np.floor(x.shape[0] * (1 - fraction))
    assert y_train.shape[0] == np.floor(y.shape[0] * (1 - fraction))
    assert x_test.shape[0] == np.ceil(x.shape[0] * fraction)
    assert y_test.shape[0] == np.ceil(y.shape[0] * fraction)


@pytest.mark.parametrize('size', [10, 20, 40])
def test_split_data_size(get_labelled_data, size):
    x, y = get_labelled_data
    x_train, x_test, y_train, y_test = split_data(
        x, y, split_size=size
    )

    assert x_train.dtype == x.dtype
    assert y_train.dtype == y.dtype
    assert x_test.dtype == x.dtype
    assert y_test.dtype == y.dtype

    assert x_train.shape[-1] == x.shape[-1]
    assert y_train.shape[-1] == y.shape[-1]
    assert x_test.shape[-1] == x.shape[-1]
    assert y_test.shape[-1] == y.shape[-1]

    assert x_train.shape[0] == (x.shape[0] - size)
    assert y_train.shape[0] == (y.shape[0] - size)
    assert x_test.shape[0] == size
    assert y_test.shape[0] == size


def test_normalize_image():
    x = np.random.randint(low=0, high=256, size=(100, 32, 64, 3))
    result = normalize_image(x)

    assert result.shape == x.shape

    assert np.max(result) <= 1.0
    assert np.min(result) >= 0.0
    assert 0 < np.mean(result) < 1
