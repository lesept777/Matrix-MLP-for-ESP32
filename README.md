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
* Single instruction optimization of the training phase, with options
  * The user can also build her/his own training code
* Many training options available:
  * Begin training on small dataset for faster training (don't use this with transfer learning)
  * Linear or logarithmic variation of the learning rate during the training phase
  * Quadratic variation of momentum during the training phase
  * Momentum and sigmoid gain can change if the cost does not decrease
  * Weights can be randomly alterred if the cost does not decrease
  * Gradient clipping, gradient scaling for faster convergence
  * Weights clipping to zero for sparsity
  * Network pruning for lower memory size
  * L1, L2 regularization available
  * Xavier initialization for weights
  * Displays training cost and validation cost, to watch for overfitting
* Verbose level can be set to have more or less convergence information
* Displays network and performance information at the end:
  * Confusion matrix
  * Layers informations
  * Weights statistics
* Can save the network's data on file for future usage (transfer learning)
* etc.

# Multilayer perceptron
Multilayer perceptron is decribed [here](https://en.wikipedia.org/wiki/Multilayer_perceptron). This library implements both training and inference phases on a dataset. Datasets can be created in the sketch or read from a csv file. This library is not intended to work with images, only with arrays of floats.

## Quick start
If you want to test it quickly, try the ["sinus" example](./examples/MMLP_Sinus)

# Guidelines
## Declare a network
To declare a network, just create an array of int with the number of neurons in each layer. The arguments of the constructor are: number of layers, array of neurons, verbose level.
```
// Declare the network
int Neurons[] = {2, 20, 1}; // Number of neurons in each layer (from input to output)
MLP Net(Neurons, 3, 1);     // number of layers, array of neurons, verbose level
```
Declare the activations of each layer and set the hyperparameters.
```
int Activations[] = {RELU, RELU, SIGMOID};
Net.setActivations (Activations);
Net.setHyper(0.25f, 0.5f); // LR & momentum
```
### **Activation functions** currently available: 
* `SIGMOID`: S-shaped curve, between 0 and 1
* `SIGMOID2`: Similar to `SIGMOID`, but between -1 and +1
* `TANH`: Quite similar to `SIGMOID2`
* `RELU`: Rectified Linear Unit
* `LEAKYRELU` and `ELU` variants
* `SELU` : Scaled Exponential Linear Unit (prevents vanishing & exploding gradient problems)
* `IDENTITY`
* `SOFTMAX`

The **sigmoid** and **hyperbolic tangent** activation functions cannot be used in networks with many layers due to the vanishing gradient problem. In the backpropagation process, gradients tend to get smaller and smaller as we move backwards:  neurons in earlier layers learn slower than neurons in the last layers. This leads to longer learning and less accurate prediction. The **rectified linear** activation function overcomes this problem, allowing models to learn faster and perform better.

![RELU SIGMOID](https://miro.medium.com/max/1452/1*29VH_NiSdoLJ1jUMLrURCA.png "Sigmoid and RELU functions")

**Softmax** for classification problems implemented. `SOFTMAX` can only be used for the last layer. If you choose it, the cost function is [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy). Otherwise, it's Squared error.

## Create a dataset
There are several ways to create a dataset:
* read data from a csv file (see [Boston Housing](./examples/MMLP_BostonHousing) example)
* create the data using a function:
  * declare the dataset from vectors of float
  * declare the dataset from arrays of float

The data can be normalized (several possible options) using the `normalizeDataset` method. Then the dataset can be split in 3 parts: training data, validation data and test data. Use `setTrainTest`method:
```
Net.setTrainTest(0.7, 0.2, 0.1); // 70% training, 20% validation, 10% test
Net.setTrainTest(4, 1, 1);       // 4/6 = 66.66% training, 1/6 = 16.66% validation and test
Net.setTrainTest(0.8, 0., 0.2);  // 70% training, 0% validation, 20% test
```
