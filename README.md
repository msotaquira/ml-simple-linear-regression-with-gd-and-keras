# Simple linear regression: gradient descent + Keras + TensorFlow

## Overview
Implementation of simple linear regression using the gradient descent algorithm. Keras library (with TensorFlow backend) is used in this implementation.

First a noisy dataset (*y = 9x + 20*) is created; then, the Keras model is built, compiled and trained. After training, plots of both mean square error (cost function) and linear regression are shown. Finally, the model is used to perform a prediction on new data.

## Dependencies
numpy==1.14.0, matplotlib==2.0.0, Keras==2.1.3