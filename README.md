This repository is the official PyTorch implementation of "Bipartite Graph Coarsening For Text Classification Using Graph Neural Networks" (CIARP 2023)

## Overview

Under construction.

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

In order to run our method, you must perform the steps described in this section.

### Obtaining GloVe's embeddings

Since our method uses GloVe's embeddings, you **must** create a directory called *embeddings* in the project's root directory. Then, download the embeddings (glove.6B.zip) from this url https://nlp.stanford.edu/projects/glove/ and extract the 300d file (glove.6B.300d.txt) to the directory you created.

### Build graph

To build a graph you can simply run the following command:

    $ python build_graph.py --dataset <dataset_name>

After running this command, a new directory called *graphs* will be created in the *data* directory. Inside the *graphs* folder you will find many files, including the edge index and the masks required by torch geometric, a file containg the documents' classes (.y), the embeddings for each word (.x_word) and document (.x_doc), and a map used by our method.

This repository contain a few datasets available in the *data* directory, including the datasets we used in the experiments we ran.

### Coarsening

### GNN

You can train the GNN to perform text classification by running the following command:

    $ python train.py --dataset <dataset_name> --out_dim <number_of_classes>

The following arguments are implemented to allow the control of the GNN's hyperparameters:

    - `--lr`

    Controls the model's learning rate.

    Default: 1e-3

## Reference

This section will be updated once the paper is available on springer.

## Acknowledgements

The Multilevel framework for bipartite networks (MFBN) plays an important role in our method. We adapted its original implementation to coarsen a graph using the cosine similarity of the words present in the datasets we used. For more information on MFBN, you can check the following repository: https://github.com/alanvalejo/mfbn.