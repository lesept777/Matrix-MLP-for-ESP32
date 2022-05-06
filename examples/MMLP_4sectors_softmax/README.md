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
