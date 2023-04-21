# use this if you want to train from a headless and/or remote machine

import os
import numpy as np
import tensorflow as tf
from avinoise import config, extraction, build_model, utils
import resampy.core as resampy

print(f"TF version: {tf.__version__}")

params = config.params()
params.summary()

train_path = os.path.join(params.data_root, "train")
test_path = os.path.join(params.data_root, "test")
# io_spec = extraction.saveFeatures(train_path, test_path)
    
loaded_train, loaded_test = extraction.loadFeatures(train_path, test_path)

model = build_model.build_model(params.model_type, loaded_train.element_spec[0].shape[1:], 
                                loaded_train.element_spec[1].shape[1])
model.summary()

hist = model.fit(loaded_train,
                 validation_data=loaded_test,
                 callbacks=[utils.tensorboard()],
                 epochs=params.n_epochs)

from sklearn.metrics import classification_report

y_test_prob = model.predict(loaded_test)
y_test_pred = np.argmax(y_test_prob, axis=1)
y_test_true = np.argmax(np.array([y for x, y in loaded_test.unbatch().as_numpy_iterator()]), 
                        axis=1)

print('Classification report:')
print(classification_report(y_true=y_test_true, y_pred=y_test_pred, target_names=params.classes))

os.makedirs("_models", exist_ok=True)
model.save(os.path.join("_models", utils.genModelName()))
model.save_weights(os.path.join("_models", utils.genWeightsName()))
utils.dict2README(os.path.join("_models", utils._genTimestampName() + "md"))