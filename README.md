# WaveDenoiser

A large deep learning model to reduce background noise from seismic signals in the time domain. The model was developed by Chengping Chai, Derek Rose, Scott Stewart, Nathan Martindale, Mark Adams, Lisa Linville, Christopher Stanley, Anibely Torres Polanco and Philip Bingham.

## Introduction

This model is trained on STEAD (Mousavi et al., 2019) for reducing background noise from seismic signals. The model was trained using earthquake data at local distances (0-350 km). The model may not perform well for earthquake data at larger distances or for non-earthquake sources. The model may have a less-than-optimal performance for earthquake data outside of a source-receiver distance range of 10-110 km and a magnitude range of 0-4.5 due to biases in the training data.

The model uses 57s-long three-component seismograms sampled at 100 Hz as input. The output of the model are denoised three-component seismograms.

## Installation

Creating a virtual environment is highly encouraged. For example, you can run the following command if you use mamba.
```
mamba create -n wavedenoiser python=3.12
mamba activate wavedenoiser
```
To install WaveDenoiser with GPU support, you can use the following command.

```
pip3 install wavedenoiser
```
To install only for CPU (e.g., macOS), you may use the following command.
```
pip3 install torch==2.7 torchvision torchaudio
pip3 install wavedenoiser
```

## Example Usage

```
import h5py
import numpy as np  
from wavedenoiser.wavedenoiser import Denoiser
model = Denoiser()
#
fid = h5py.File("example_waveforms.h5", "r")
data_group = fid["data"]
example_data = []
for akey in data_group.keys():
    dataset = data_group[akey]
    example_data.append(dataset[...])
fid.close()
#
example_data = np.array(example_data)
preds = model.predict(example_data)
```

## Test the Package

First, download the GitHub repostory and go to the top directory of the package. Then, run the following command.
```
cd tests
python run_tests.py
```
![example image](images/example_waveform_2.png)

## Known Limitations

* The model may have a less-than-optimal performance for earthquake data outside of a source-receiver distance
range of 10–110 km and a magnitude range of 0–4.5 because of biases in the training data.
* The model may not perform well for earthquake data at larger distances or for non-earthquake sources.


## Reference

TBD

## License

GNU GENERAL PUBLIC LICENSE version 3

## Credit

The architecture of the deep learning model was adapted from SeisBench (Woollam et al., 2022).
