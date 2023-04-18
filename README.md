# GCLmf
GCLmf: A novel molecular graph contrastive learning framework based on hard negatives and application in toxicity prediction

## Requirements
   * python 3.7 
### install requirements
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html  
pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip install PyYAML
conda install -c conda-forge rdkit=2020.09.1.0
```

## Pre-training
To train the GCLmf, where the configurations and detailed explaination for each variable can be found in ***config_pretrain.yaml***  
`python GCLmf_pre.py` 

## Fine-tuning
### molecular property benchmarks 
`python GCLmf_ft_property.py` 
### toxicity data sets
`python GCLmf_ft_tox.py` 
  






