# use this if you want to train from a headless and/or remote machine

import os
from avinoise import config, prediction
import random
import pandas as pd
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

params = config.params()

model_paths = ["_models/F1M1.h5",
               "_models/F1M2.h5",
               "_models/F2M1.h5",
               "_models/F2M2.h5",]

weights_paths = ["_models/weights_F1M1.h5",
                 "_models/weights_F1M2.h5",
                 "_models/weights_F2M1.h5",
                 "_models/weights_F2M2.h5",]

n_samples = 120
samples = random.sample(os.listdir("raw_data/clean/"), n_samples)
clean_data = ["raw_data/clean/" + s for s in samples]
samples = random.sample(os.listdir("raw_data/contaminated/"), n_samples)
contaminated_data = ["raw_data/contaminated/" + s for s in samples]
data = [clean_data, contaminated_data]
labels = ["CLEAN", "CONTAMINATED"]
postp_methods = ["greaterOnce", "greaterMean"]

fig, axs = plt.subplots(1, 2)

for n, method in enumerate(postp_methods):

    corrects = []

    for model_path, weights_path in zip(model_paths, weights_paths):

        if "F1" in model_path:
            params.add({"normalize": False})
            params.add({"cutoff": 0})
        else:
            params.add({"normalize": True})
            params.add({"cutoff": 0.01})
            
        for clss, label in zip(data, labels):

            correct = 0
            for file in clss:

                preds, mel = prediction.predict(file, model_path, weights_path)
                if prediction.evaluate(preds, method=method, show_text=False) == label:
                    correct += 1
            corrects.append(correct / n_samples)
            print(f"{correct}/{n_samples} correct samples for class <{label}> with model <{model_path}>")

    cleans = [val for i, val in enumerate(corrects) if i % 2 == 0]
    contaminateds = [val for i, val in enumerate(corrects) if i % 2 != 0]
    index = [name.replace("_models/", "").replace(".h5", "") for name in model_paths]
    df = pd.DataFrame({'Clean': cleans,
                    'Contaminated': contaminateds}, index=index)
    df.plot.bar(rot=0, ax=axs[n])
    axs[n].set_title(method)
    axs[n].set_ylabel(r"Accuracy")

plt.tight_layout()
plt.savefig('performance.pgf')
