# FP_GNN
This repository is Pytorch ans DGL implementation of the experiments in the following paper:

paper_citation [link]



if you make use of the code/experiment in you work, please cite our paper.



## Installation

Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.

```
pip install -r requirements.txt
```



## Test run

Run test code as follows:

```
python test_code.py
```

The default arguments for test code are 

- aggregator_type: lstm
- checkpoint: ./checkpoint
- dataset: cubicasa_test
- feature_normalize: standard
- gnn_model: sage
- hidden_dim: 128
- num_layers: 6.



