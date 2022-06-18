# Evaluation of natural language processing embeddings in protein function prediction for bacteria
This project was created as part of the **2021-2022 Bachelor Research Project at Delft University of Technology**.
Access to the bachelor thesis is provided on the TU Delft repository: [to-add].

The contents of this repository include the scripts for the implementation and evaluation of the prediction models used in our experiments.

## Requirements
To run all scripts, install the following dependencies:
- numpy v1.22.3
- pandas v1.4.2
- goatools v0.7.11

To generate embeddings from scratch, you can use the [bio-embeddings](https://anaconda.org/conda-forge/bio-embeddings) conda package.

## Usage
### Generating predictions
To generate predictions, run:
``python3 scripts/predict.py``

**Input:** We generate predictions from the embeddings stored in ``data/embeddings``. In this folder, we store the embeddings generated by SeqVec (seqvec), T5XLU50 (t5), ProtBERT (protbert), and ESM1b (esm1b). Each of these 4 folders contains 2 sub-folders, namely ``bacillus`` and ``ecoli``, each containing 3 .pkl files:
- ``test_embeddings.pkl``: the embeddings from the query set;
- ``train_embeddings1.pkl``: half of the embeddings from the database set;
- ``train_embeddings2.pkl``: the other half of the embeddings from the database set.

**Output:** In ``data/predictions``, we store the k-NN predictions for each embedder, as well as the normalized predictions.

### Evaluation
To calculate the F-max and S-min of all models in the evaluation, run: ``python3 scripts/evaluate.py``

**Input:** The k-NN predictions generated from all embedding models, stored in ``data/predictions``. This folder also includes predictions from BLAST, DeepGOPlus, and goPredSim.

**Output:** This script will print the F-max and S-min values for each predictor, per GO class: MF (molecular function), BP (biological process), CC (cellular component). Some values for precision, recall, remaining uncertainty and misinformation will also be printed.

## Authors
**Student:** Bianca-Maria Cosma

**Supervisors:** Aysun Urhan, Abigail Manson, Thomas Abeel

I express my thanks to Aysun, to whom I give credit for providing me with the protein sequence data used in these experiments.
