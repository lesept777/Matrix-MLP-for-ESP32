# (Matrix) Multi Layer Perceptron library for ESP32
New version of MLP, using matrix maths

# Dependencies
Use my matrix linear algebra library (https://github.com/lesept777/Matrix-maths-for-ESP32-and-AI)

# Description
This library is designed to be used with the Arduino IDE.

Multilayer perceptron is decribed [here](https://en.wikipedia.org/wiki/Multilayer_perceptron). This library implements both training and inference phases on a dataset. Datasets can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

It can address **deep networks**, made of one input layer, multiple hidden layers and one output layer. The output layer can have one neuron for regression, or several neurons for classification (use SIGMOID or SOFTMAX activation for the last layer).

## Main features
* Uses matrix algebra (floating points 32 bits)
* Many activations available
* Can create dataset from csv file or from std::vector or arrays
* The user can split the dataset
* Many training options available:
  * Begin training on small dataset for faster training
  * Learning rate linear or logarithmic evolution
  * Momentum and sigmoid gain can change if the cost does not decrease
  * Weights can be randomly alterred if the cost does not decrease
  * Gradient clipping, gradient scaling for faster convergence
  * Weights clipping to zero for sparsity
  * L1, L2 regularization available
  * Xavier initialization ofr weights
* Verbose level can be set to have more or less convergence information
* Displays network and performance information at the end:
  * Confusion matrix
  * Layers informations
  * Weights statistics
* Can save the network's data on file for future usage (transfer learning)
* etc.
