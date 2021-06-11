import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('seaborn')


def save_classification_report(y_true, y_pred, class_labels, directory='.'):
    """Computes the classification report and saves it as a heatmap.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    class_labels : list[str] or list[int]
        A list of labels to be used in the heatmap for better readability.
    directory : str, default='.'
        Path to directory where the generated heatmap will be saved.

    Notes
    -----
    The directory passed should already exist. By default, the plot will
    be saved in the current working directory.
    """

    report = classification_report(y_true, y_pred, target_names=class_labels,
                                   output_dict=True)

    df = pd.DataFrame(report).T
    cr = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    cr.yaxis.set_ticklabels(cr.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    cr.xaxis.set_ticklabels(cr.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{directory}/classification_report.png')
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_labels, directory='.'):
    """Computes and saves the normalized confusion matrix.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    class_labels : list[str] or list[int]
        A list of labels to be used in the heatmap for better readability. This
        should be the same as in `y_true` and `y_pred`.
    directory : str, default='.'
        Path to directory where the generated heatmap will be saved.

    Note:
    -----
    The directory passed should already exist. By default, the plot will
    be saved in the current working directory.
    """

    matrix = confusion_matrix(y_true, y_pred, labels=class_labels,
                              normalize='true')

    df = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    hm = sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False)

    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=10)

    hm.set_xlabel('Predicted Label')
    hm.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'{directory}/confusion_matrix.png')
    plt.close()


def plot_tf_model(model, directory='.'):
    """Plots the graph of the given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model for which the graph will be saved.
    directory : str, default='.'
        Path to directory where the generated plot will be saved.

    Note:
    The directory passed should already exist. By default, the plot will
    be saved in the current working directory.
    """

    plot_model(model, to_file=f'{directory}/model.png', show_shapes=True,
               dpi=200, expand_nested=True)
