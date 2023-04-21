import librosa as lr
from avinoise import config
import numpy as np
from pathlib import Path
import tensorflow as tf
import os

# global mel filter (performance reasons)
params = config.params()
mel_filter = lr.filters.mel(sr=params.sr,
                            n_fft=params.n_fft,
                            n_mels=params.n_mels,
                            fmin=0.0,
                            fmax=None,
                            htk=False,
                            norm='slaney',
                            dtype=np.float32)


def mel(file, mel_filter=mel_filter, normalize=False, cutoff=0, const_file_len=True):
    """
    Description
    -----------
    Transform and filter an audio file into a mel spectrogram.

    Parameters
    ----------
    `file`:
        String. Path to audio file which should be transformed.
    `mel_filter`:
        `librosa.filters`-object. Supply a filter from the outside to
        increase performance.
    `normalize`:
        Bool. Scale values between 0 and 1. Default is `False`.
    `cutoff`:
        Float. Lower Threshold for scaled data. Anything below `cutoff`
        will be set to `cutoff`. Default is `0`.
    `const_file_len`:
        Bool. Lazy way to indicate if the incoming file is from the
        dataset with constant length or an inference file with arbitrary
        length. Default is `True`. Ignore this.

    Notes
    -----
    Supply a filter from the outside to increase performance if this
    is in a loop.

    Filter example:
    ```
    mel_filter = lr.filters.mel(params.sr,
                                    params.n_fft,
                                    n_mels = params.n_mels,
                                    fmin = 0.0,
                                    fmax = None,
                                    htk = False,
                                    norm = 'slaney',
                                    dtype = np.float32)
    ```
    """
    params = config.params()
    # load audio data
    audio, sr = lr.load(file, sr=params.sr, mono=params.mono,
                        offset=0.0, duration=None,
                        dtype=np.float32, res_type='kaiser_best')
    if const_file_len:
        audio = _ensureEqualLen(audio, params.sr, params.audio_len)

    # transorm data
    stft = lr.stft(audio, n_fft=params.n_fft,
                   hop_length=params.hop_length,
                   win_length=params.win_length,
                   window=params.window, center=params.center,
                   dtype=np.complex64, pad_mode=params.pad_mode)
    stft = np.abs(stft).astype(np.float32)

    # filter stft with mel-filter
    mel_spec = mel_filter.dot(np.power(stft, 2))
    if normalize:
        mel_spec /= np.max(mel_spec)
        mel_spec = np.select([mel_spec <= cutoff, mel_spec > cutoff], 
                             [np.full_like(mel_spec, cutoff), mel_spec])

    return mel_spec


def _ensureEqualLen(x, sr, seconds):
    target_len = int(seconds * sr)
    sig_len = len(x)
    if target_len < sig_len:
        x = x[:target_len]
    elif sig_len < target_len:
        x = np.append(x, np.zeros(target_len - sig_len))
    else:
        pass
    
    if target_len != len(x):
        raise AssertionError(f"Target length <{target_len}> unequal to modified signal length <{len(x)}>!") 
    return x


def getLabelOneHot(file_path):
    """
    Description
    -----------
    Obtain label vector from folder names.

    Parameters
    ----------
    `file_path`:
        String. Path to source file within a class folder.

    Notes
    -----
    The assumed filestructure is
    ```
    >data
        >test
            >class1
                >file1.wav
                ...
            >class2
                ...
            ...
        >train
            >class1
                ...
            >class2
                ...
            ...
    ```
    """
    params = config.params()
    label = Path(file_path).parts[-2]
    label_idx = params.classes.index(label)

    one_hot = tf.one_hot(label_idx, len(params.classes),
                         on_value=None, off_value=None,
                         axis=None, dtype=tf.uint8, name=None)
    return one_hot


def _extract_features(file_path, verbose=False):
    params = config.params()
    # for whatever reason, the supplied path from
    # `dataset.save` gets turned into a tensor
    # containing the string. This gets rid of that.
    if not isinstance(file_path, str):
        file_path = file_path.numpy().decode("utf-8")
    if params.verbose:
        print(f"Extracting {file_path}...")

    labels = getLabelOneHot(file_path)
    mel_vals = np.expand_dims(mel(file_path, 
                                  mel_filter,
                                  normalize=params.normalize,
                                  cutoff=params.cutoff), axis=-1)

    return mel_vals, labels


def extract_features(file_path):
    """
    Description
    -----------
    Wrapper for feature extraction of raw dataset and label transfer.

    Parameters
    ----------
    `file_path`:
        String. Path to audio file which should be extracted.

    Notes
    -----
    This function is called within the `tf.data.Dataset` API via `map`.
    """
    params = config.params()
    import tensorflow as tf
    mel_spec, one_hot = tf.py_function(_extract_features,
                                       [file_path],
                                       [tf.float32, tf.uint8])

    mel_spec.set_shape([params.n_mels, params.n_frames, 1])
    one_hot.set_shape([len(params.classes)])
    return mel_spec, one_hot


def saveFeatures(train_path, test_path):
    """
    Description
    -----------
    Run feature extraction and save features on disk in tf's
    dataset API format.

    Parameters
    ----------
    `train_path`:
        String. Path to train data (contains folders of classes).
    `test_path`:
        String. Path to test data (contains folders of classes).

    """
    params = config.params()
    train_files = os.path.join(train_path, "*", "*.wav")
    test_files = os.path.join(test_path, "*", "*.wav")

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.list_files(train_files)
    train_dataset = train_dataset.map(extract_features,
                                      num_parallel_calls=AUTOTUNE)

    test_dataset = tf.data.Dataset.list_files(test_files)
    test_dataset = test_dataset.map(extract_features,
                                    num_parallel_calls=AUTOTUNE)
    if params.verbose:
        print(f"Data set shape: {train_dataset.element_spec}")
        print("Saving train dataset...")

    tf.data.experimental.save(dataset=train_dataset,
                              path=train_path,
                              compression='GZIP')
    if params.verbose:
        print("Saving test dataset...")

    tf.data.experimental.save(dataset=test_dataset,
                              path=test_path,
                              compression='GZIP')

    return train_dataset.element_spec


def loadFeatures(train_path, test_path):
    """
    Description
    -----------
    Load features from disk previously saved with `saveFeatures()`.

    Parameters
    ----------
    `train_path`:
        String. Path to train data (contains stored features).
    `test_path`:
        String. Path to test data (contains stored features).

    Notes
    -----
    Features are stored in `.pb` and `.metadata` files as well as
    a folder consisting of a string of numbers.
    """
    params = config.params()
    input_specs = tf.TensorSpec(shape=(params.n_mels, params.n_frames, 1),
                                dtype=tf.float32,
                                name=None)
    output_specs = tf.TensorSpec(shape=(len(params.classes),),
                                 dtype=tf.uint8,
                                 name=None)

    loaded_train = tf.data.experimental.load(path=train_path,
                                             element_spec=(input_specs,
                                                           output_specs),
                                             compression='GZIP')
    loaded_test = tf.data.experimental.load(path=test_path,
                                            element_spec=(input_specs,
                                                          output_specs),
                                            compression='GZIP')

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = loaded_train.cache()
    train_dataset = train_dataset.shuffle(buffer_size=4000)
    train_dataset = train_dataset.batch(params.n_batches)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    test_dataset = loaded_test.cache()
    test_dataset = test_dataset.batch(params.n_batches)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    if params.verbose:
        print(f"Data set shape: {train_dataset.element_spec}")

    return train_dataset, test_dataset
