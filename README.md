# NPBCL : icml20
This is a PyTorch implementation of the Bayesian Structure Adaptation for Continual Learning.

## Requirements
  * Keras 2.2.5
  * PyTorch 1.3.1
  * torchvision 0.4.2
  * scipy 1.3.1
  * sklearn 0.21.3
  * numpy 1.17.2
  * matplotlib 3.1.1
  * gzip, pickle, tarfile, urllib, PIL, math, copy

## Running the experiments
To run all experiments together (might take a while) use : `sh run_all.sh`

Individual experiments can be run using `pyhton3 experiment_name.py`
- Experiment names starts with : `npbcl_xxx.py`
- Copy and save the stored experiment models using : `echo -e "saves\ncache/destination" | python3 save.py`
- Copy and save the generative experiment images : `echo -e "Gens\ncache/destination" | python3 save.py`
