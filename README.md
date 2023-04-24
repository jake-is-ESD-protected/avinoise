# Classification of aviation noise and false triggers for airport noise monitoring (AVINOISE)

> ### [About](#about) | [Methods](#about) | [Results](#results) | [Paper](#paper) | [Try it yourself](#try-it-yourself) | [Contribute](#contribute)

## About
**AVINOISE** is a project for classifying false triggering of noise monitoring stations [around the **BER airport**](https://travisber.topsonic.aero/), Germany. Such a station may be triggered by a sound other than airplane noise, which unnecessarily alerts the airport's noise control department. There, somebody has to listen through the whole recording to verify its origin to know if an airplane violated noise limits or a completely unrelated sound set of the alarm. Now, the neural network **AVINOISE** will take care of that.
![map](/Docs/map.png)

## Methods
**AVINOISE** was built on **100000** 5 second clips of starting and landing events of airplanes supplied by BER airport and various types of urban or rural noise sourced from the well known [ESC-50](https://github.com/karolpiczak/ESC-50) and [UrbanSound8k](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) which were augmented into the data set. The model employs **convolution layers** for audio recognition and classifies audio slices of 5 seconds with a hop size of 2.5 seconds. The only classes are `clean` and `contaminated`. A "clean" file consists of typical airport noises, while a "contaminated" file features transient irregularities very atypical for the larger perimeter of an airport like a dog barking, car horns or church bells.

|*clean* noise event (just airplanes)|*contaminated* noise event|
|-|-|
|![clean_spec](/Docs/clean.png)|![cont_spec](/Docs/contaminated.png)|

## Results
Two different model architectures were tested with two different pre- and post-processing methods closer described in the [Paper](#paper). The classical ML results using the **KERAS** API are listed below for the different approaches ***FxMx***:

||***F1M1***|***F1M2***|***F2M1***|***F2M2***|
|-|-|-|-|-|
|Accuracy|0.88|**0.91**|0.84|0.85|
|Loss|0.79|**0.37**|0.59|0.55|

Since a flight event is much longer than a single 5 second slice, the obtained classifications over time have to be post-processed and flagged as "contaminated" if enough unwanted noise is present. This is done by two different algorithms:
* *P1*: A single contamination classification is greater than 50%
* *P2*: The mean of contamination classification is greater than 50%

![post_process](/Docs/post_process.png)

The best performing combination ***F1M2P1*** is chosen as the final product of this project. Here is a sample of it running inference:
![inference](/Docs/inference.png)
```
s	    | clean		| contaminated
-------------------------------------------
5	    | 93% 		| 6%
10	    | 0% 		| 100%
15	    | 0% 		| 100%
20	    | 0% 		| 100%
25	    | 0% 		| 100%
30	    | 0% 		| 100%
35	    | 0% 		| 100%
40	    | 0% 		| 99%
45	    | 0% 		| 99%
50	    | 0% 		| 99%
55	    | 0% 		| 99%
60	    | 0% 		| 99%
65	    | 2% 		| 97%
70	    | 99% 		| 0%
75	    | 64% 		| 35%
80	    | 42% 		| 57%
85	    | 0% 		| 99%
90	    | 0% 		| 99%
95	    | 0% 		| 99%
100	    | 97% 		| 2%
105	    | 54% 		| 45%
110	    | 25% 		| 74%
115	    | 18% 		| 81%
120	    | 32% 		| 67%
125	    | 43% 		| 56%
130	    | 92% 		| 7%
135	    | 73% 		| 26%
mean:   | 27% 		| 72%
Flagged as CONTAMINATED
```

### Avinoise vs. [YAMNet](https://hub.tensorflow.google.cn/google/yamnet/1)
To know where this project stands, is is compared against a transfer learning model of **YAMNet**, a well known and reliable universal audio event classifier from Google. Training large datasets through YAMNet models can lead to severe overfitting, and finally after attempts it was found that the highest test accuracy of **90%** could be obtained when the dataset was around 200 to 400 files. This is only **0.2%** of the data set that **F1M2** used, which is why we recommend transfer learning in case of data scarcity. Below is a table for an accuracy and loss comparison for both models:
||***F1M2***|***YAMNet TL***|
|-|-|-|
|Accuracy|0.91|0.90|
|Loss|0.37|0.35|

## Paper
The paper and project were created by Lukas Probst, Jakob Tschavoll and Yijun Zhao, audio technology students of TU-Berlin. The paper can be found [here](Docs/CLASSIFICATION_OF_AVIATION_NOISE_AND_FALSE_TRIGGERS_FOR_AIRPORT_NOISE_MONITORING_lprobst_jtschavoll_zyijun.pdf).

## Try it yourself

Be sure that you have **python 3.9 or higher** and pip, otherwise the project won't run!

Get your local instance by cloning from the [original repo](https://github.com/jake-is-ESD-protected/avinoise)'s `main` branch via 
```
$ git clone https://github.com/jake-is-ESD-protected/avinoise
```

Create a `venv` with any virtual environment tool you like. We recommend the native python venv. You can create it by navigating into the folder you just cloned and typing
```
$ python -m venv .env
```

Don't forget to actually type the `.` symbol. To install all the requirements you first activate your venv with
```
$ .\.env\Scripts\activate           # for Windows
$ source .env/bin/activate          # for Linux
```
and then setup the required libs with
```
$ pip install -r requirements.txt   # install the required libs
$ pip install -e avinoise           # install the local avinoise module
```

**AVINOISE** is now ready to run on your machine.

### Run inference
Within the file [`API.ipynb`](/API.ipynb) you can analyze your flight event files. Simply provide the path to your file and run the notebook. It will produce a printed report and a plot with the spectrogram and the classifications over time.

Alternatively you can create your own `python`-based file in the root folder of this project and create your own API with the internal `avinoise` module:
```python
# my_file.py
from avinoise import prediction
file = "./my_aviation_noise.wav"

preds, mel_spectrogram = prediction.predict(file) # run inference
flag = prediction.evaluate(preds) # evaluate prediction
prediction.plot(preds, mel_spectrogram) # see a plot
```

## Contribute

### Get the repo
Get your local instance by cloning from the [original repo](https://github.com/jake-is-ESD-protected/avinoise)'s `dev` branch via 
```
$ git clone --branch dev https://github.com/jake-is-ESD-protected/avinoise
```
You should then create a new branch which roughly states what you want to add to the codebase. Let's say you want to implement a feature `avinoise.summary()` which prints a summary of all important current parameters for this project as an overview. You would then create a branch like
```
$ git branch parameter-summary          # create branch
$ git checkout parameter-summary        # switch to branch
```
You can verify that the switch worked with ` $ git status`.

### Set up a virtual environment
Create a `venv` with any virtual environment tool you like. We recommend the native python venv. You can create it by navigating into the folder you just cloned and typing
```
$ python -m venv .env
```
Don't forget to actually type the `.` symbol. Now new folders should be created which hold the local instances of the required interpreters and libraries for the project to work.  
To install all the requirements to start adding and testing code, you first activate your venv with
```
$ .\.env\Scripts\activate           # for Windows
$ source .env/bin/activate           # for Linux
```
and then setup the required libs with
```
$ pip install -r requirements.txt   # install the required libs
$ pip install -e avinoise           # install the local avinoise module
```

### Understand project structure
As stated, the module `avinoise` is the backbone of this project. It holds various submodules which are concerned with a specific part of the program flow. They themselves hold various functions which execute simple, stackable jobs. For example, the submodule `config.py` is concerned with storing global parameters for the project so that every function in other submodules sources from the same data. Here you would add your submodules with specific functions.

```
-avinoise
    |-avinoise
        |-__init__.py
        |-config.py
        |-your_module1.py       # your code
        |-your_module2.py       # your code
        |.
        |.
        |.
    |-setup.py
-<Folder1>
-<Folder2>
-<Folder3>
.
.
.
```

Please be careful when editing other modules which are not actually concerned with your branch as it can lead to merge conflicts with other developers. Inform them if you need their advice.

If you want to visually test things, we recommend that you create a **jupyter notebook** in the main folder called `explore.ipynb`. This is just for you as it will be ignored by `.gitignore`.

In the end, all of these highly abstracted submodules will be combined in the jupyter notebook `helper_notebooks/pipeline.ipynb` which demonstrates the creation and verification of the project's model itself.

### Add your code
Within your newly created submodule you can import other submodules and work with them. Be sure to add all of your public classes and functions to the `__all__` variable and prefix your private functions with `_`. Split tasks into easy to test, simple I/O-based functions. This keeps the code clean, modular and testable.

### Push, review and merge
After all tests have passed, push your changes to your branch and submit a pull request into the `dev` branch. Assign all other team members to review your code and wait for their response. If asked, fix any errors and your changes will be merged into `dev`. If your changes work within `dev`, they will be pushed to `main`.
