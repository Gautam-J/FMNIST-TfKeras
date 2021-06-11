import os
import pytest
import numpy as np

from utils.visualizations import (
    save_classification_report,
    save_confusion_matrix
)


@pytest.fixture
def get_sample_data():
    y_true = np.random.randint(low=0, high=5, size=(100, 1))
    y_pred = np.random.randint(low=0, high=5, size=(100, 1))

    return y_true, y_pred


@pytest.mark.parametrize("labels", [
    ['one', 'two', 'three', 'four', 'five'],
    ['1', '2', '3', '4', '5'],
    ['first', 'second', 'third', 'fourth', 'fifth'],
])
def test_save_classification_report(get_sample_data, labels):
    y_true, y_pred = get_sample_data
    save_classification_report(y_true, y_pred, labels)

    assert os.path.exists('classification_report.png')
    os.remove('classification_report.png')


def test_save_confusion_matrix(get_sample_data):
    y_true, y_pred = get_sample_data
    labels = list(range(5))
    save_confusion_matrix(y_true, y_pred, labels)

    assert os.path.exists('confusion_matrix.png')
    os.remove('confusion_matrix.png')
