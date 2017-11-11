# Music Object Detector

This repository is the home of a Faster R-CNN implementation for Music Symbols to implement a fast and reliable Music Symbol detector with Deep Learning.

[![Build Status](https://travis-ci.org/apacha/MusicObjectDetector.svg?branch=master)](https://travis-ci.org/apacha/MusicObjectDetector)

Note my previous projects that [classified entire sheets](https://github.com/apacha/MusicScoreClassifier) or [learnt to classify different music symbols](https://github.com/apacha/MusicSymbolClassifier).

An extensive overview of the results of different parameters is documented in this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1MT4CH9yJD_vM9nT8JgnfmzwAVIuRoQYEyv-5FHMjYVo/edit?usp=sharing).

# Running the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Requirements

- Python 3.6
- Keras 2.0.8
- Tensorflow 1.3.0 (or optionally tensorflow-gpu 1.3.0)
- [Microsoft Visual C++ Build Tools 2015](http://landinghub.visualstudio.com/visual-cpp-build-tools) (for faster data_generator)

Optional: If you want to print the graph of the model being trained, install GraphViz on Windows via http://www.graphviz.org/Download_windows.php and add /bin to the PATH or run `sudo apt-get install graphviz` on Ubuntu (see https://github.com/fchollet/keras/issues/3210)

For installing Tensorflow and Keras we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

## Training the model

The easiest way to start the training is to run `TrainModel.ps` from the PowerShell.

### Manually start the training
For manually starting the training, make sure to first compile the tools 

    cd keras_frcnn/py_faster_rcnn
    python setup.py build_ext --inplace
    
then run TrainModel like this

    MusicObjectDetector> python TrainModel.py --network resnet50 --output_weight_path "resnet50.hdf5"
    

# License

Published under MIT License,

Copyright (c) 2017 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
