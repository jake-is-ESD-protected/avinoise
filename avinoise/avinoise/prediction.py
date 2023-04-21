from avinoise import config
from keras.models import load_model
from avinoise import extraction
import numpy as np
import matplotlib.pyplot as plt
import os


def predict(filepath, 
            model="./_models/F1M2.h5", 
            weights="./_models/weights_F1M2.h5", 
            sensitivity=1):
    """
    Description
    -----------
    Wrapper for data extraction and `model.predict()`.

    Parameters
    ----------
    `filepath`:
        String. Path to file which should be analyzed.
    `model`:
        String, keras.model. Path to saved model or model
        itself.
    `weights`:
        String. Path to weights.
    `sensitivity`:
        Float. Weighting factor for `contaminated` class.
        Raises "alertness" of system by amplifying the
        predictions for contamination.

    Returns
    -------
    `predicitions`:
        Array. Predictions split into classes and time steps.
        Each second timestep is the overlap of the previous
        and next timestep. By omitting the last "half frame"
        the total number of timesteps is `step_len * 2 - 1`.
    `mel`:
        Array. Raw mel spectrogram for further use.
    """

    params = config.params()
    if isinstance(model, str):
        model = load_model(model)

    # Load weights
    model.load_weights(weights)

    # Extract features
    raw_mel = extraction.mel(filepath, 
                             normalize=params.normalize, 
                             cutoff=params.cutoff,
                             const_file_len=False)

    # truncate data if the last frame is less than 5s
    n_frames = int(params.sr * params.audio_len / params.hop_length + 1)
    truncate = raw_mel[0].shape[0] % n_frames
    mel = raw_mel[:, :-truncate]
    mel = mel[np.newaxis, ...]

    # ready new storage array
    split_mel = np.empty((0, params.n_mels, n_frames))

    # split data into 5s sections
    n_sections = mel.shape[2] // n_frames
    for i in range(n_sections):
        # get base sections
        split_mel = np.append(split_mel,
                                mel[:, :, (i*n_frames):((i+1)*n_frames)],
                                axis=0)
        # ignore last "half-frame"
        if i == n_sections -1:
            continue
        # get ovrlapping sections
        split_mel = np.append(split_mel,
                                mel[:, :, (i*n_frames)+(n_frames//2):((i+1)*n_frames)+(n_frames//2)],
                                axis=0)
    split_mel = split_mel[..., np.newaxis]

    # Predict
    predictions = model.predict(split_mel, verbose=0)
    for pred in predictions:
        pred[0] *= (1/sensitivity)
        pred[1] = 1 - pred[0]
    return predictions, mel


def evaluate(predictions, method="greaterOnce", show_text=True):
    """
    Description
    -----------
    Pretty print predictions obtained from `prediction.predict()`.

    Parameters
    ----------
    `predictions`:
        Array. Predictions from `prediction.predict()`.
    `method`:
        String. Method by which the file is flagged as `CLEAN`
        or `CONTAMINATED`. Available options:  
            - `'greaterOnce'`: One or more frames are more
                likely to be contaminated
            - `'greaterMean'`: The mean of contamination is
                greater than the mean of clean 
        Default is `'greaterOnce'`.
    `show_text`:
        Bool. Show prints on terminal. Default is `True`.
    
    Returns
    -------
    `flag`:
        String. `'CLEAN'` if file has no detectable unwanted noise,
        `'CONTAMINATED'` if unwanted noise was detected.
    """
    if method == "greaterOnce":
        flag = _flagGreaterOnce(predictions)
    elif method == "greaterMean":
        flag = _flagGreaterMean(predictions)
    else:
        raise ValueError(f"Evalation <{method}> method not available")

    params = config.params()
    mean_clean = 0
    mean_contaminated = 0

    if show_text:
        print("s\t| clean\t\t| contaminated")
        print("-------------------------------------------")
        for i, pred in enumerate(predictions):
            print(f"{(i+1)*5}\t| {int(pred[0]*100)}% \t\t| {int(pred[1]*100)}%")
            mean_clean += pred[0]
            mean_contaminated += pred[1]

        mean_clean /= (i + 1)
        mean_contaminated /= (i + 1)
        print(f"mean:   | {int(mean_clean*100)}% \t\t| {int(mean_contaminated*100)}%")
        print(f"Flagged as {flag}")

    return flag


def _flagGreaterOnce(predictions):
    for pred in predictions:
        if pred[0] < pred[1]:
            return "CONTAMINATED"
    return "CLEAN"


def _flagGreaterMean(predictions):
    p = np.transpose(predictions)
    if np.mean(p[0]) < np.mean(p[1]):
        return "CONTAMINATED"
    else:
        return "CLEAN"


def plot(predictions, mel, name="", visual_scale=1):
    """
    Description
    -----------
    Plot spectrogram and classifications of a sound file over time.

    Parameters
    ----------
    `predictions`:
        Array. Predictions from `prediction.predict()`.
    `mel`:
        Array. mel spectrum for plotting, obtainable with
        `prediction.predict()`.
    `name`:
        String. Name of plot. Default is `""`.
    `visual_scale`:
        Float. Contrast booster for spectrogram. Visual only.
        Default is `1`.

    Returns
    -------
    `axs`:
        matplotlib axis object. Contains both plots. `axs[0]` is
        the spectrogram and `axs[1]` is the classification over
        time.
    """
    params = config.params()
    n_frames = int(params.sr * params.audio_len / params.hop_length + 1)
    fig = plt.figure(figsize=(10, 5))
    axs = fig.subplots(2, 1)
    axs[0].set_title(f"{name}")

    im0 = axs[0].imshow(np.power(mel[0], 1/visual_scale),
                  aspect='auto',
                  interpolation='nearest',
                  origin='lower')
    axs[0].set_xticks(np.arange(0, mel.shape[2], n_frames))
    axs[0].set_xticklabels(np.arange(0,
                                     (mel.shape[2] // n_frames) * params.audio_len,
                                     5))
    axs[0].set_xlabel("t / s")
    axs[0].set_ylabel("Mel bands")

    # Plot and label the model output scores for the top-scoring classes.
    im1 = axs[1].imshow(predictions.T,
                  aspect='auto',
                  interpolation='nearest',
                  cmap='gray_r')
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(params.classes, rotation=45)

    axs[1].set_xlabel("Inferences")
    axs[1].set_ylabel("Classification")

    plt.colorbar(im0)
    plt.colorbar(im1)
    plt.tight_layout()

    return axs
