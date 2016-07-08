------------------------------------------
# OLD
------------------------------------------
## Model notes:
### NN -------
#### Space:
  - conv-lstm on topomaps

#### Frequency:
  - fully conencted on per sensor lstm on 1d conv of frequencies
  - lstm on fc of 1d conv of frequencies

### Overall:
  - ensembled vs not ensembled

### Classical -------
#### Space:
  - classifier on topomap CSP
  - classifier on topomap CSP through time
### Hybrid -------
#### Space:
  - FFNN on small time window of CSP
  - RNN on CSP
