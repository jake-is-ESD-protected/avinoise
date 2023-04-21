from pydub import AudioSegment
import os
from avinoise import config
import random
from augmolino import augmentation
import librosa as lr
import soundfile as sf
import shutil

__all__ = ["convertSampleRate",
           "trainTestSplit",
           "sliceSamples",
           "evenDataset"]

# open backbone params object
params = config.params()


def convertSampleRate(import_path, export_path, sample_rate):
    """
    Description
    -----------
    The function takes the path of an import folder, scans the
    folder for audio files and converts their sample rate to a
    specified value. The new audio files are exported as wav-file
    to an export folder.

    Parameters
    ----------
    `import_path`:
        String. Path to target files.
    `export_path`:
        String. Path to save folder for new files
    `sample_rate`:
        Int. New sample rate.
    """

    # List of all audio files in folder "import_path"
    files = [file for file in os.listdir(
        import_path) if file.lower().endswith(("mp3", "wav", "flac"))]
    for file in files:
        if params.verbose:
            print(f"Resampling {os.path.join(import_path, file)}...")
        # Open audio file
        audio = AudioSegment.from_file(os.path.join(import_path, file))

        # Set sample rate
        new_audio = audio.set_frame_rate(sample_rate)

        file_name = os.path.splitext(file)[0]
        # Save the converted file with .wav extenstion and in format wav
        new_audio.export(os.path.join(
            export_path, f"{file_name}_{sample_rate}.wav"), format="wav")


def trainTestSplit(class_paths, ratio=0.7, overwrite=False):
    """
    Description
    -----------
    Split a folder containing a class of samples into a train
    and test set. This function will create a `data` folder
    according to `params.data_root` path.

    Parameters
    ----------
    `class_paths`:
        List. Paths to the folders containing a class of audio.
        Will be as long as number of classes.
    `ratio`:
        Float. Ratio by which the split is done. `0.7` means
        70% training, 30% testing. Default is `0.7`.
    `overwrite`:
        Bool. Specifies if an already split dataset should be
        overwritten or kept. Default is `False`.
    """

    if not overwrite:
        if os.path.exists(params.data_root):
            return

    class_paths = [params.raw_clean_path, params.raw_contaminated_path]
    raw_data_classes = [os.listdir(class_path) for class_path in class_paths]

    split_folders = [os.path.join(params.data_root, "train"),
                     os.path.join(params.data_root, "test")]

    os.makedirs(params.data_root, exist_ok=True)

    for folder in split_folders:
        os.makedirs(folder, exist_ok=True)
        for c in params.classes:
            os.makedirs(os.path.join(folder, c), exist_ok=True)

    for raw_class, class_path, c in zip(raw_data_classes,
                                        class_paths,
                                        params.classes):

        random.shuffle(raw_class)
        split_files = [raw_class[:int(ratio * len(raw_class))],
                       raw_class[int(ratio * len(raw_class)):]]

        for folder, files in zip(split_folders, split_files):

            for file in files:
                shutil.copy(os.path.join(class_path, file),
                            os.path.join(folder, c, file))


def sliceSamples(root_path, slice_len=5):
    """
    Description
    -----------
    Slice all audio files in a folder into short clips of
    euqal lenght in whole seconds. The last clip which is
    shorter than one second will be omitted.

    Parameters
    ----------
    `root_path`:
        String. Path to whole dataset.
    `slice_len`:
        Int. Length of each slice in whole seconds.
        Default is `5`.

    Notes
    -----
    The original file gets deleted. All files obtained from
    the source file have the equal name with an index added
    at the end.
    """

    for (root, dirs, files) in os.walk(root_path, topdown=True):
        for file in files:
            if file.lower().endswith(("mp3", "wav")):
                cur_path = os.path.join(root, file)
                audio, _ = lr.load(path=cur_path, sr=params.sr)
                # iterate over blocks which are `slice_len` seconds long
                n_blocks = len(audio) // (params.sr * slice_len)
                # ignore the file if it is already `slice_len` long
                if n_blocks < 2:
                    continue
                # trim off end which is < 1 second
                audio = audio[:-(len(audio) % params.sr)]

                if params.verbose:
                    print(f"Slicing {cur_path}...")
                for i in range(n_blocks):
                    sf.write(f"{cur_path[:-4]}_{i + 1}.wav",
                             audio[i * params.sr * slice_len:
                                   (i + 1) * params.sr * slice_len],
                             params.sr)
                os.remove(cur_path)


def evenDataset(base_path):
    """
    Description
    -----------
    The dataset for this project is known to have a strong asymmetry towards
    'clean' audiofiles which consist of aviation noise only. The interesting
    cases however are the overlay of aviation noise and random urban/suburban/
    rural noise, since these trigger the noise monitors for the wrong reason.
    Therefore augmentation shall be used to artificially mix clean aviation
    noise with noise from the \
    [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)
    until the two subsets have the same amount of files. For augmentation the
    module [augmolino](https://github.com/jake-is-ESD-protected/augmolino) \
    shall be used.

    Parameters
    ----------
    `base_path`:
        String. Root folder of dataset.

    Notes
    -----
    The data has been handpicked by class since only noise events of an outside
    urban/suburban/rural source or at least something similar to that are of
    interest in this project.
    """

    # introduce paths to dataset
    class_paths = [os.path.join(base_path, params.classes[0]),
                   os.path.join(base_path, params.classes[1])]
    files = [os.listdir(class_path) for class_path in class_paths]

    # calc number of files which have to be augmented to even out the dataset
    n_bootstraps = len(files[0]) - len(files[1])
    if n_bootstraps == 0:
        return

    # get files that will be augmented to be contaminated
    files_to_bootstrap = random.sample(files[0], n_bootstraps)

    # create mixer augmentation
    mixer = augmentation.mixAudio(ratio=0.5, sample_rate=params.sr)

    # get needed amount of mixing data
    noise_path = params.augmentation_source
    files_noise = [file for file in os.listdir(
        noise_path) if file.lower().endswith(("mp3", "wav", "flac"))]

    if params.verbose:
        print("Evening out dataset...\n\n\n")
    for i, file_clean in enumerate(files_to_bootstrap):
        if params.verbose:
            print(f"Augmenting '{file_clean}', file {i+1}/{n_bootstraps}")

        f_source = os.path.join(class_paths[0], file_clean)
        f_dest = os.path.join(class_paths[1], "mix_" + os.path.basename(
            file_clean))
        f_mix = os.path.join(noise_path, random.sample(files_noise, 1)[0])
        _ = mixer.run(f_source=f_source,
                      f_dest=f_dest,
                      f_mix=f_mix)
