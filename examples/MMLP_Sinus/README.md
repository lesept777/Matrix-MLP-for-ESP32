# Fitting a sine curve
This example shows how to fit the sine curve, for x from -PI to +PI with automatic training.

The network used here has 4 layers, with neurons {1, 8, 3, 1}. 

## First define the network  and its hyperparameters:
```
  float eta = 0.5f; // learning rate
  float momentum = 0.1f;
  int Neurons[] = {1, 8, 3, 1};   // neurons per layer
  int activation[] = {TANH, TANH, TANH}; // activations of each layer
  int maxEpochs = 300;
  float stopError = 0.02f;
  int batchSize = 20;

  MLP Net (Neurons, 4, 1); // 4 layers, verbose set to 1
  Net.setActivations(activation);
  Net.setHyper(eta, momentum);

```

Learning parameters are:
* Momentum = 0.1
* Learning rate = 0.5
* Sigmoid gain = 1
* Activations are set to TANH
* Training will stop when the error on the test set is lower than 0.02

## Then create the dataset and put data inside. 
2 possibilities are shown:
* Create the dataset from vectors
* Create the dataset from arrays

```
  // Create the dataset
  //////////////    SINUS    //////////////

  int nData = 150;
  //  float x[nData], y[nData];  // <-- dataset from arrays
  std::vector<float> x, y;
  for (int i = 0; i < nData; ++i) {
    float xx = -PI + 2.0f * PI * float(i) / (nData - 1.0f);
    float yy = sin(xx);
    //    x[i] = xx; // <-- dataset from arrays
    //    y[i] = yy; // <-- dataset from arrays
    x.push_back(xx);
    y.push_back(yy);
  }

  ///////////////////////////////////////////
  
  std::vector<std::vector<float> > dataX;
  std::vector<std::vector<float> > dataY;
  //  Net.createDatasetFromArray (dataX, dataY, x, y, nData);  // <-- dataset from arrays
  Net.createDatasetFromVector (dataX, dataY, x, y);
```

## Then split the dataset
```
  Net.setTrainTest(4, 1, 1);
```
4/6 (67%) of the dataset is for training, 1/6 for validation and 1/6 for testing.

## Training and test
Training is done by a single line: optimize the network on the dataset on 300 epochs (maximum) and batch size of 20 data.
```
  Net.run (dataX, dataY, maxEpochs, batchSize, stopError);
```
During the training, as verbose is set to 1, you can see the evolution of the cost. If necessary, when the algorithm senses that the cost doen't decrease, it applies some changes to the parameters. 

This is called 'heuristics' in the source code. It can be user set, or default. In this case, the heuristics is chosen by default. The details can be found in the file MMLP.h, by the end:
* Initialize the weights randomly
* Mutate weights: when the cost doesn't decrease for some time, the weights are changed randomly from up to +/-2.5% or 10%
* Learning rate changes throughout the epochs, logarithmically
* The value of the momentum can change randomly
* Gradient scaling is applied (scaled such as L2 norm equals 1)


The result is shown in the file [Output.txt] (https://github.com/lesept777/Matrix-MLP-for-ESP32/blob/main/examples/MMLP_Sinus/Output.txt)
