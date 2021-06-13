import os
import pytest
import logging
import numpy as np
from unittest.mock import patch

from utils.analysis import (
    setup_logger,
    get_error_pct,
    get_f1_score,
    _get_y_pred
)


@pytest.mark.parametrize("name, level", [('name1', logging.DEBUG),
                                         ('name2', logging.INFO),
                                         ('name3', logging.CRITICAL)])
def test_setup_logger_stream(name, level):
    logger1 = setup_logger(name, level=level)

    assert type(logger1) == logging.Logger
    assert logger1.name == name
    assert logger1.getEffectiveLevel() == level

    assert logger1.hasHandlers()


@pytest.mark.parametrize("name, log_file, level", [
    ('name1', "debug_log.log", logging.DEBUG),
    ('name2', "info_log.log", logging.INFO),
    ('name3', "critical_log.log", logging.CRITICAL)
])
def test_setup_logger_file(name, log_file, level):
    logger2 = setup_logger(
        name,
        level=level,
        log_file=log_file
    )

    assert type(logger2) == logging.Logger
    assert logger2.name == name
    assert logger2.getEffectiveLevel() == level

    assert logger2.hasHandlers()

    assert os.path.exists(f'./{log_file}')
    os.remove(f'./{log_file}')


@pytest.fixture
def get_sample_data():
    x = np.random.randn(3, 2)
    y = np.array([1, 0, 1, 1, 0, 2, 0, 1, 3, 2])

    return x, y


@pytest.mark.parametrize("pred, error", [
    (np.array([1, 0, 1, 1, 0, 2, 0, 1, 3, 2]), 0.0),
    (np.array([0, 1, 2, 0, 1, 3, 1, 0, 2, 3]), 1.0),
    (np.array([1, 1, 3, 2, 0, 2, 0, 2, 3, 2]), 0.4)
])
def test_get_error_pct(get_sample_data, pred, error):
    with patch('sklearn.dummy.DummyClassifier') as model:
        model.predict = lambda x: pred
        result = get_error_pct(model, *get_sample_data)

    assert result == error


@pytest.mark.parametrize("pred, score", [
    (np.array([0, 1, 2, 0, 1, 3, 1, 0, 2, 3]), 0.0),
    (np.array([1, 0, 1, 1, 0, 2, 0, 1, 3, 2]), 1.0),
    (np.array([1, 0, 1, 1, 0, 2, 0, 1, 2, 3]), 0.625)
])
def test_get_f1_score(get_sample_data, pred, score):
    with patch('sklearn.dummy.DummyClassifier') as model:
        model.predict = lambda x: pred
        result = get_f1_score(model, *get_sample_data)

    assert result == score


@pytest.mark.parametrize('pred', [
    np.random.randn(3,),
    np.random.randn(3, 1),
    np.random.randn(3, 10),
])
def test_get_y_pred(get_sample_data, pred):

    with patch('sklearn.dummy.DummyClassifier') as model:
        model.predict = lambda x: pred
        x, _ = get_sample_data
        result = _get_y_pred(model, x)

    assert result.ndim == 1
