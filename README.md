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

For training and test (for validation):

```python
python train_test.py
```

For test code:

```python
python test_code.py
```



## Scripts and directories

* Scripts
  * main: main script for construction ,training and test for the framework
  * dataset_module: dataset construction
  * models: code implementation of GNN models
  * vectorization: code implementation for image pre-processing, vectorization, and RAG conversion
  * train_test: a script for training and test for GNN models
* Directories
  * checkpoint: pre-trained GNN models
  * output: predicted .shp files
  * dataset: used dataset images and vector files (fps) and pre-processed .bin files (preprocessed) 

