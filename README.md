# FP_GNN
This repository is Pytorch ans DGL implementation of the experiments in the following paper:

Song J, Yu K. Framework for Indoor Elements Classification via Inductive Learning on Floor Plan Graphs. ISPRS International Journal of Geo-Information. 2021; 10(2):97. https://doi.org/10.3390/ijgi10020097

if you make use of the code/experiment in you work, please cite the paper.



## Installation

Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.7.0 and DGL 0.5.2 versions.

Then install the other dependencies.

```
pip install -r requirements.txt
```



## Test run

For training and test:

```python
python train_test.py
```

For test code:

```python
python test_code.py
```



## Scripts and directories

* Scripts
  * main: main script for constructing the dataset and train/test for the framework
  * dataset_module: dataset construction
  * models: code implementation of GNN models
  * vectorization: code implementation for image pre-processing, vectorization, and RAG conversion
  * train_test: a script for training and test for GNN models
* Directories
  * checkpoint: pre-trained GNN models
  * output: predicted .shp files
  * dataset: used dataset images and vector files (fps) and pre-processed .bin files (preprocessed) 



## Note

* The UOS dataset is not available now for security reasons. We will open the dataset to the public as soon as it is approved.
