# Tab-estimator
## File structure

```
Tab-Estimator
├── data
├── model
├── result
├── src
│   ├── config.yaml
│   ├── jams_interpreter.py
│   ├── jams_to_midi.py
│   ├── midi_to_numpy.py
│   ├── network.py
│   ├── predict.py
│   ├── train.py
│   └── visualize.py
├── tensorboard
└── requiments.txt
```

## Details
### data/
This directory contains ground truth midi files converted from the original GuitarSet annotations, and sets of input features and ground truth in npz file format.

### model/
This directory contains trained models and a config file.

### result/
This directory contains the estimation metrics (precision, recall, F1 score, TDR).

### src/
This directory contains the source codes used in this study.

- [config.yaml](https://github.com/KimSehun725/Tab-estimator/blob/master/src/config.yaml)
  contains all the informations about the basic configuration. 
  
- [jams_interpreter.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/jams_interpreter.py)
  modules needed for interpreting jams file.

- [jams_to_midi.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/jams_to_midi.py)
  convert jams files into midi files.

- [midi_to_numpy.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/midi_to_numpy.py)
  convert midi files created from [jams_to_midi.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/jams_to_midi.py) to npz files.
  
- [network.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/network.py)
  model and loss function for training.

- [predict.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/predict.py)
  predict and save the results and metrics using a trained model.

- [train.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/train.py)
  train the model defined in [network.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/network.py).

- [visualize.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/visualize.py)
  plot the results of the estimation.
  
### tensorboard/
This directory contains the training log files. You can see the training log by `tensorboard --logdir [logdir]`

### [requirements.txt](https://github.com/KimSehun725/Tab-estimator/blob/master/requirements.txt)
This file contains the informations on python packages required to run this project.

## Setup
First, make `GuitarSet/` directory and download GuitarSet with following codes:
```
mkdir -p GuitarSet/annotation GuitarSet/audio_mono-mic
wget -qO- https://zenodo.org/record/3371780/files/annotation.zip?download=1 | busybox unzip - -d GuitarSet/annotation
wget -qO- https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1 | busybox unzip - -d GuitarSet/audio_mono-mic
```

Next, create virtual environment and install all the necessary packages using `pip`. Run the following codes to do so.
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the code

In all of the following instructions, we will assume that you are in `Tab-Estimator/` directory.

### Make input and output features
Run [jams_to_midi.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/jams_to_midi.py). 
This will convert the jams files provided by GuitarSet into midi files.
Next, run [midi_to_numpy.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/midi_to_numpy.py).
This will generate npz files in `data/npz`. The npz file contains input acoustic features and one-hot-vector type ground truth.

### Training
Check [config.yaml](https://github.com/KimSehun725/Tab-estimator/blob/master/src/config.yaml) and run [train.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/train.py).
Running [train.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/train.py) will copy the [config.yaml](https://github.com/KimSehun725/Tab-estimator/blob/master/src/config.yaml) to `model/date_and_time/`.
The format of `date_and_time` is YYYYMMDDhhmm.
Full six-fold cross-validation training process with 192 epochs each, will take 6+ hours with following specs:
```
Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz w/ 16GB system memory
Nvidia GeForce RTX 2080ti w/ 11GB VRAM
```
Trained models will be saved in `model/date and time/` every 32 epochs. 

### Predict and evaluate
Run [predict.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/predict.py) with arguments for designating the model. 
The first argument is `date_and_time`, and the second argument is `epochs`.
ex) `python3 src/predict.py 202201012359 192`
This will load the trained model, predict, calculate metrics, and save the results to `result/F0` or `result/tab`.

### Visualize the predictions
Run [visualize.py](https://github.com/KimSehun725/Tab-estimator/blob/master/src/visualize.py) with arguments for designating the model.
ex) `python3 src/visualize.py 202201012359 192`
This will visualize the results and save these to `result/*/visualize/`.
