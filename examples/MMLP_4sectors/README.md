# Classification of points in 4 square sectors

This example classifies points (x,y) in the domain [0,1] x [0,1] in 4 sectors:

```
/*    sectors:
       _______
      | 1 | 3 |
      | 0 | 2 |
       -------
*/
```
If x>0.5 and y<0.5, the output is 2.

This example uses the SIGMOID and RELU activations and the MSE cost function. The network is made of 4 layers, with :
* input layer: 2 neurons (x and y)
* hidden layer: 30 neurons
* hidden layer: 20 neurons
* output layer: 1 neuron

The dataset is made in the ino file, using the `sector` function
```
int sector (float x, float y) {
  return (x >= 0.5) * 2 + (y >= 0.5);
  /*
     this is equivalent to:
    if (x <  0.5 && y < 0.5)  return 0;
    if (x <  0.5 && y >= 0.5) return 1;
    if (x >= 0.5 && y < 0.5)  return 2;
    if (x >= 0.5 && y >= 0.5) return 3;
  */
}
```
This function returns the sector's number (i.e. the output value) for a given (x,y) point. 

It is a complex task for a MLP, doing classification as if it was regression: an output from 0 to 0.25 will be identified as class number 1, from 0.25 to 0.5 as class number 2, etc.

## Network definition
```
int Neurons[] = {2, 30, 20, 1};
int Activations[] = {SIGMOID, SIGMOID, RELU};
MLP Net(Neurons, 4, 1);

...

Net.setActivations(Activations);
Net.setCost(MSE);
Net.setHyper(0.2f, 0.05f);
Net.size();
```

## Heuristics definition
Here, 'heuristics' stands for the set of rules the algorithm will use during the optimization task. They are defined in a variable (called `heristics` here):
```
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_LRLIN +
                    H_CHAN_SGAIN +
                    H_SELE_WEIGH;
```
All possible rules are described in the file [MMLP.h](../../MMLP.h). Here:
* H_INIT_OPTIM : initialize weights randomly
* H_CHAN_WEIGH : enable brand new random weights when needed
* H_CHAN_LRLIN : learning rate changes linearly during training
* H_CHAN_SGAIN : the gain of the SIGMOID can change (i.e. the slope at origin)
* H_SELE_WEIGH : select best weights over 30 random sets
Then, set the heuristics using `Net.setHeuristics(heuristics);`

## Train
The dataset is split, normalized then the training phase is run:
```
  Net.setTrainTest(4, 1, 1);
  Net.normalizeDataset(dataX, dataY, 1);
  Net.run (dataX, dataY, 100, 10, 0.025f);
  Net.netSave(networkFile);
```

Results are shown in the file [Results.txt](./Results.txt). Not very good, but now you know more about heuristics. See the other [MMLP_4sectors_softmax](../MMLP_4sectors_softmax) example for better results.
