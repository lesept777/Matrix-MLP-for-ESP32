# (Matrix) Multi Layer Perceptron library for ESP32
New version of MLP, using matrix maths

# Dependencies
Use my matrix linear algebra library (https://github.com/lesept777/Matrix-maths-for-ESP32-and-AI)

# Description
This library is designed to be used with the Arduino IDE.

Multilayer perceptron is decribed [here](https://en.wikipedia.org/wiki/Multilayer_perceptron). This library implements both training and inference phases on a dataset. Datasets can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address **deep networks**, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron for regression, or several neurons for classification (use SIGMOID or SOFTMAX activation for the last layer).
