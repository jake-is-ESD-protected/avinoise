import os
import time
import tensorflow as tf
from avinoise import config
import matplotlib.pyplot as plt
import json

__all__ = ["tensorboard"]

params = config.params()


def _get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(os.curdir, '_logs', run_id)


def tensorboard():
    """
    Description
    -----------
    Create a tensorboard callback for usage on port 6006.

    Notes
    -----
    Session data is stored under `_logs` in the root folder.
    """
    tensorboard_cb = tf.keras.callbacks.TensorBoard(_get_run_logdir())
    return tensorboard_cb


def _genTimestampName():
    return f"AN_{params.modif[:-6].replace(':', '_')}"


def genModelName():
    """
    Description
    -----------
    Generate a name for a saved model using the timestamp
    of the last modification of `params` in the `.h5` format.
    """
    return _genTimestampName() + "h5"


def genWeightsName():
    """
    Description
    -----------
    Generate a name for a saved model weights using the timestamp
    of the last modification of `params` in the `.h5` format.
    """
    return "weights_" + _genTimestampName() + "h5"


def genMetricsImageName():
    """
    Description
    -----------
    Generate a name for the metrics plot using the timestamp
    of the last modification of `params`.
    """
    return _genTimestampName() + "png"


def dict2README(readme_path):
    assert readme_path.endswith(".md")
    header = f"# {_genTimestampName()}\n## Configuration:\n```\n"
    tail = "\n```"
    d_str = json.dumps(params._open())
    with open(readme_path, "w") as f:
        f.write(header + d_str.replace(',', '\n') + tail)


def plotMetrics(history):
    """
    Description
    -----------
    Plot the metrics of a model by its training history. A
    `png` file of the plot is saved to the `Docs` folder.

    Parameters
    ----------
    `history`:
        TF history object. Return value of `model.fit()`.

    Notes
    -----
    You might have to call `plt.show()` after this function.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    name = genMetricsImageName()
    fig.suptitle(name[:-4])

    ax[0].plot(range(params.n_epochs),
               history.history['loss'])
    ax[0].plot(range(params.n_epochs),
               history.history['val_loss'])
    ax[0].set_ylabel('loss'), ax[0].set_title('train_loss vs val_loss')

    ax[1].plot(range(params.n_epochs),
               history.history['categorical_accuracy'])
    ax[1].plot(range(params.n_epochs),
               history.history['val_categorical_accuracy'])
    ax[1].set_ylabel('accuracy'), ax[1].set_title('train_acc vs val_acc')

    for a in ax:
        a.grid(True)
        a.legend(['train', 'val'], loc=4)
        a.set_xlabel('num of Epochs')

    if os.path.exists("Docs"):
        fig.savefig(os.path.join("Docs", name))
    else:
        fig.savefig(name)
