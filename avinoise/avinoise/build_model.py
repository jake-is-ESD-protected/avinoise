import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from avinoise import config

params = config.params()


def build_model(model_type, input_shape, num_classes):
    """
    Description
    -----------
    Create and compile a fixed CNN with `Sequential()`
    and `Conv2D()` based on [SOURCE MISSING!].

    Parameters
    ----------
    `model_type`:
        String. Choose `'1'` or `'2'`. Model 1 is very 
        simple and small, while model 2is more complex.
    `input_shape`:
        Tuple. Input shape of a spectrogram unit, e.g.
        `(mel_coefficients, n_frames, channels)`.
        The `channels` is needed by tensorflow and has
        to be set to `1` in this case.
    `num_classes`:
        Int. Amount of classes of the classification
        problem. Defines the output shape.

    Notes
    -----
    Advanced metrics can be specified inside the `config`
    module.
    """
    model = Sequential()

    if model_type == "1":
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

    elif model_type == "2":
        # https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    
    else:
        raise ValueError(f"Specified model type <{model_type}> not available!")

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=params.metrics)

    return model
