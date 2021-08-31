# ProbFire: a probabistic early fire warning system for Indonesia

The source code of the ProbFire seasonal fire event occurrence prediction model
implemented using [scikit-learn](https://scikit-learn.org) Multilayer
Perceptron Classifier.  

### Brief description The model is trained to predict fire occurrence
probabilities in Indonesia at 25km spatial resolution and monthly time step.
Predictions are based on precipitation, temperature, relative humidity and
forest cover and peatland extent datasets.The prediction part is using
Multilayer Perceptron Classifier. ProbFire is trained using ECMWF ERA5
reanalysis climate, and the prediction skill at 1 to 6 month lead times
is evaluated using ECMWF SEAS5 long range climate forecasts.

Model input data (features) is available from zenodo repository
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5206278.svg)](https://doi.org/10.5281/zenodo.5206278)
Detailed description and performance evaluation can be found in [TODO]
publication.

### Usage
[TODO]
