# Classification of points in 4 square sectors

This example classifies points (x,y) in the domain [0,1] x [0,1] in 4 sectors:

```
/*    sectors:
      y
      |_______
      | 1 | 3 |
      | 0 | 2 |
       --------- x
*/
```
For example, if x>0.5 and y<0.5, the output is 2.

The main difference with the other 4sectors example is the use of SOFTMAX activation, and 4 classes (instead of 1) which make it a real classification task.

## Activations
```
int Activations[] = {SIGMOID, SIGMOID, SOFTMAX};
```

## Heuristics
```
  long heuristics = H_INIT_OPTIM +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLOG +
                    H_CHAN_SGAIN +
                    H_GRAD_CLIP  +
                    H_ZERO_WEIGH +
//                    H_DATA_SUBSE +
                    H_SELE_WEIGH;
  Net.setHeuristics(heuristics);
```
The options are:
* H_INIT_OPTIM : initialize with random weights
* H_MUTA_WEIGH : slightly change the weights randomly if the cost does not decrease
* H_CHAN_LRLOG : learning rate decreases logarithmically
* H_CHAN_SGAIN : change SIGMOID gain (slope at origin)
* H_GRAD_CLIP  : gradient clipping
* H_ZERO_WEIGH : force weigths whose absolute value is less than a threshold to zero (sparsify the network)
* H_DATA_SUBSE : begin to train on a subset of the training set (20%)
* H_SELE_WEIGH : select best weights over 30 random sets (does not hold if H_INIT_OPTIM is selected)

The parameters of the heuristics can be customized, using the following methods:
```
    void setHeurShuffleDataset (bool);
    void setHeurZeroWeights (bool, float);
    void setHeurRegulL1 (bool, float = 1.0f);
    void setHeurRegulL2 (bool, float = 1.0f);
    void setHeurChangeMomentum (bool, float = 0.1f, float = 1.5f);
    void setHeurChangeGain (bool, float = 0.5f, float = 2.0f);
    void setHeurInitialize (bool);
    void setHeurGradScale (bool, float);
    void setHeurGradClip (bool, float);
```

## Transfer learning
If the result is not satisfying, it is possible to continue the learning phase, with transfer learning. The command `Net.netSave(networkFile)` saves the network in a file in the SPIFFS memory.

It is possible to run the sketch again, read the file at the beginning and run the training from the saved data.

First, uncomment the line:
```
bool initialize = !Net.netLoad(networkFile);
```
This reads the network from the file and sets `initialize` to `false`. Then uncomment the line:
```
Net.setHeurInitialize(initialize);
```
This line must come after `setHeuristics`. This instruction forces to `false` the flag that initializes random weights (it also prevents the 'select best weights' option). The training will then resume from the saved result.

## Results
The results are much better than the previous case, see files [Results.txt](./Results.txt) and [Results2.txt](./Results2.txt). The library now computes and displays the confusion matrix after the test phase. You can see the effect of `H_ZERO_WEIGH`: more than 50% of the weights are zero.

# Pruning
New options (May 2022): attempts to remove the neurons that do not have an effective impact on the result.

Using heuristics option `H_NEUR_PRUNE`, after the training phase, the algorithm searches the neurons which have the gighest number of zeros in the weight matrices. If the row is full of zeros, this means the neuron is inactive: it does not have any impact on the result of the network and can be removed.

If the number of zeros in the weight matrix associated to a given neuron is higher than a treshold (can be set using `setHeurPruning`) this neuron is also removed.

The file [Pruning.txt](./Pruning.txt) shows the result. With a threshold at 75%, 14 neurons were removed, which decreases the number of synapses (the size of the network) of 40%, with a slight decrease of the test performance.