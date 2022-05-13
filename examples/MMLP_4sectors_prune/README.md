# Pruning a network during training
This example shows how to use the pruning options.

Pruning a network is the action of removing some neurons, to reduce the (memory) size of the network. This leads to a lighter metwork, with faster inference, and sometimes a small reduction in performances.

## Heuristics pruning options
Two possibilities exist for now (others will come soon):
* `H_TEST_PRUNE`: prune the network after training, during the test phase
* `H_TRAI_PRUNE`: prune the network during training (and at the end during the test also)

Pruning after training (option `H_TEST_PRUNE`) applies pruning on the trained network. It may lead to lower performances if the pruning is too aggressive, i.e. if too many neurons are removed. Pruning during training will reduce the impact of neurons removal, because the training continues after pruning which alleviates the effect of neruons removal.

## How pruning works
Two kinds of pruning are implemented for now:
* Remove inactive neurons. Inactive neurons are characterized by associated weights that are all zeros. In this case, the neuron does not play any role in the network and can be removed with no impact on performance. The only possible impact can be if later during training the neuron would 'wake up' but this is unlikely.
* Remove low activity neurons. Low activity neurons are neurons whose associated weights are lower (in absolute value) than a specific threshold. The threshold's value is set using the `setHeurPruning` method.

```net.setHeurPruning(true, threshold);```

