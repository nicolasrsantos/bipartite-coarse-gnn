This repository is the official PyTorch implementation of "Bipartite Graph Coarsening For Text Classification Using Graph Neural Networks" (CIARP 2023)

## Overview

## Requirements

This code was implemented using Python 3.11.2, CUDA 11.8 and the following packages:

- `pytorch==2.0.1`
- `torch-geometric==2.3.0`
- `numpy==1.24.1`
- `networkx==3.0`
- `scikit-learn=1.2.2`
- `igraph==0.10.4`
- `scipy==1.7.2`
- `pyyaml==6.0`

## How to run the code

Under construction.

## Reference

This section will be updated once the paper is available on springer.

## Acknowledgements

The Multilevel framework for bipartite networks (MFBN) plays an important role in our method. We adapted its original implementation to coarsen a graph using the cosine similarity of the words present in the graphs we used. For more information on MFBN, you can check the following repository: https://github.com/alanvalejo/mfbn.