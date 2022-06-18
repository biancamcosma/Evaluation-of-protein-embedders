import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from evaluation_metrics import calculate_fmax, calculate_smin
from utils import parse_tsv

prefix = "../data/"

# Get actual labels
tsv_file = prefix + "uniprot2go_exp.tsv"
true_labels = parse_tsv(tsv_file)

# Get GO term classes
with open(prefix + "go_classes.pkl", "rb") as f:
    go_classes = pickle.load(f)

# Fetch pre-computed GO term information contents
with open("/tudelft.net/staff-umbrella/abeellab/bcosma/functionpred/data/go_ics.pkl", "rb") as f:
    go_ics = pickle.load(f)

# List of methods to evaluate
methods = ["esm1b", "protbert", "t5", "seqvec", "blast", "goPredSim", "deepgoplus"]

t_values = np.linspace(0.0, 1.0, num=50)

for b in ["bacillus", "ecoli"]:
    print("\nResults for: " + b)
    print("--------")
    # Load test proteins
    test_proteins = set(open(prefix + "testprot_" + b + "_trimmed.txt").read().splitlines())

    for go_class in ["MF", "BP", "CC"]:
        for method in methods:
            print("\nEvaluating " + method + " predictions...")
            print("-----")

            # Load predictions
            predictions_file = prefix + "predictions/" + method + "/final_predictions/predictions_" + b + ".pkl"
            with open(predictions_file, "rb") as f:
                predictions = pickle.load(f)

            # Calculate Fmax
            fmax, precision, recall = calculate_fmax(predictions, true_labels, t_values, test_proteins, go_classes, go_class)
            # Calculate Smin
            smin, remaining_uncertainty, misinformation = calculate_smin((predictions, true_labels, t_values, test_proteins, go_classes, go_class))

            # Print out results
            print("GO class: " + go_class)
            print("F-max: " + str(fmax))
            print("Precision values: " + str(precision[:3]) + " ... " + str(precision[-2:]))
            print("Recall_values: " + str(recall[:3]) + " ... " + str(recall[-2:]))
            print("---")
            print("S-min: " + str(smin))
            print("Misinformation: " + str(misinformation[:3]) + " ... " + str(misinformation[-2:]))
            print("Remaining uncertainty: " + str(remaining_uncertainty[:3]) + " ... " + str(remaining_uncertainty[-2:]))
