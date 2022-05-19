# Pruning a network during training

This example shows how to use the pruning options.

Pruning a network is the action of removing some neurons, to reduce the (memory) size of the network. This leads to a lighter metwork, with faster inference, and sometimes a small reduction in performances.

## Heuristics pruning options

Two possibilities exist for now (others will come soon):

- `H_TEST_PRUNE`: prune the network after training, during the test phase
- `H_TRAI_PRUNE`: prune the network during training (and at the end during the test also)

Pruning after training (option `H_TEST_PRUNE`) applies pruning on the trained network. It may lead to lower performances if the pruning is too aggressive, i.e. if too many neurons are removed. Pruning during training will reduce the impact of neurons removal, because the training continues after pruning which alleviates the effect of neurons removal.

## How pruning works

Two kinds of pruning are implemented for now:

- Remove inactive neurons. Inactive neurons are characterized by associated weights that are all zeros. In this case, the neuron does not play any role in the network and can be removed with no impact on performance. The only possible impact can be if later during training the neuron would 'wake up' but this is unlikely, and the training compensates the impact.
- Remove low activity neurons. Low activity neurons are neurons with more than a given number of weights equal to zero. The threshold's value is set using the `setHeurPruning` method.

```
Net.setHeurPruning(true, threshold);
```

The default value is 0.85, meaning that if a neuron has more than 85% of the weights in its row in the matrix that are equal to zero, it is a low activity neuron. `true` is set to enable the pruning (`false` will disable if necessary).

Pruning works better with the `H_ZERO_WEIGH` heuristics option, which forces weights of absolute value under a given threshold to zero. The threshold can be set using the `setHeurZeroWeights`method:

```
Net.setHeurZeroWeights(true, threshold);
```

Default value is 0.15: if during the training a weight has an absolute value lower than the threshold, it is forced to 0.

## Pruning strategy

During training, only inactive weights are removed, this is done two times depending on the training score: when the score passes under 4 times the objective score, and again when it passes under 2 times the objective score:

```
		// Prune inactive neurons
		if (_prune_train) {
			// First prune phase : error < 6 * _stopError
			if (pruneEpochs == 0 && _currError < 4 * _stopError) {
				++pruneEpochs;
				pruneAll();
				size();
			}
			// Second prune phase : error < 3 * _stopError
			if (pruneEpochs == 1 && _currError < 2 * _stopError) {
				++pruneEpochs;
				pruneInactive();
				size();
			}
		}
```

The method `pruneInactive()`is used to select and remove inactive networks.

Pruning during test removes both inactive and low activity neurons. It is done using the `pruneAll()` method which calls both `pruneInactive()` and `pruneLowAct()` methods. Both methods return the number of pruned neurons.

After pruning, `pruneAll()` returns a boolean which is `true` if any neuron was pruned. In this case, a new test phase is performed to provide the new network's statistics.

## Examples

The file [Results.txt](./Results.txt) shows the results obtained for the 4 sectors classification example.

The initial network has 794 total weights and biases (synapses), with 2 hidden layers of 30 and 20 neurons. The objective score is set to 0.04. When the score passes under 0.16, the first pruning is run: 6 neurons are inactive, and the number of synapses is reduced to 656 (-17%).

The second pruning phase is run when the score reaches 0.08 but no action is taken.

The confusion matrix is printed, then the test-phase pruning is run, but no inactive or low activity neurons are found. So the network stays at 656 synapses. The validation on unknown data shows 100% performance. There are still more almost 60% synapses that are equal to 0.

Another example is provided: [Results2.txt](./Results2.txt). We add:

```
  Net.setHeurZeroWeights(true, 0.20);
  Net.setHeurPruning(true, 0.80);
```

before the line

```
  Net.displayHeuristics();
```

to change the thresholds. 11 neurons are pruned in the first phase. The confusion matrix is:

```
TR/PR    0    1    2    3  (Recall)
  0 :   13    1    0    0  ( 92.9%)
  1 :    0   14    0    1  ( 93.3%)
  2 :    0    0   10    1  ( 90.9%)
  3 :    0    0    0   10  (100.0%)
Prec:  100%  93% 100%  83% -> 94.0%
```

No pruning is done at test phase.

In case the pruning degrades the results, running again with transfer learning from the saved network can improve the performance. This is done by uncommenting the following lines in the ino file:

```
    bool initialize = !Net.netLoad(networkFile);
...
    Net.setHeurInitialize(initialize);
```

and setting the threshold of low activity neurons to 100% to cancel its action:

```
    Net.setHeurPruning(true, 1.0);
```
