import numpy as np
import pandas as pd
import pickle

from goatools.obo_parser import OBOReader
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag

def parse_tsv(file_location):
    """Parses a .tsv file that provides mappings from protein IDs to GO terms.
    
    A line in this file is of the form: proteind_id    GO:0000001;GO:0000003

    Parameters:
    file_location: A string of the full path to the .tsv file.

    Returns:
    A dictionary with keys as protein IDs (strings) and values as an array of corresponding GO annotations (np.ndarray of strings).

    """

    annotations = pd.read_csv(file_location, sep="\t")
    annotations_map = dict()

    for row in annotations.itertuples():
        seqid = row.id
        go_tags = np.array([str(i) for i in row.go_exp.split(';')])

        annotations_map[seqid] = go_tags        
    
    return annotations_map


def get_go_depths(obo_file, annotations):
    """Determines the depths of all GO terms in an .obo file.

    Parameters:
    obo_file: A string of the full path to the .obo file.
    annotations: All dataset annotations. A dictionary with keys as protein IDs and values as arrays of GO terms.

    Returns:
    A dictionary with keys as GO terms (strings) and values as depths (integers).

    """
    all_go_terms = set()
    for go_terms in annotations.values():
        for go_term in go_terms:
            all_go_terms.add(go_term)
        
    # Create GO DAG with optional relationships included
    # SubDAG is needed, as DAG does not have the desired functionality
    godag = get_godag(obo_file, optional_attrs={'relationship'})
    godag = GoSubDag(all_go_terms, godag)

    go_info = godag.go2nt
    
    go_depths = dict()
    for go_id in go_info.keys():
        go_depths[go_id] = go_info[go_id].depth

    # Obsolete terms that are not part of the DAG have depth 0
    obsolete_terms = {"GO:2000775", "GO:0052312", "GO:1902586"}
    for go_term in obsolete_terms:
        go_depths[go_term] = 0

    return go_depths


def normalize_predictions(predictions, prefix):
    """Normalizes a set of predictions and stores them.

    Parameters:
    predictions: A nested dictionary containing the predictions for each test protein. The format is: {protein_id: {GO_id: probability}}.
    prefix: prefix to where we store the results.

    """
    
    # Iterate over test proteins and their predictions
    min_probability = dict()
    max_probability = dict()
    for category in ["MF", "CC", "BP"]:
        min_probability[category] = 2
        max_probability[category] = -2

    # Get GO term classes
    with open(prefix + "go_classes.pkl", "rb") as f:
        go_classes = pickle.load(f)
        
    for test_protein_id, test_protein_predictions in predictions.items():
        # Determine range of prediction probabilities
        for go_term, probability in test_protein_predictions.items():
            category = go_classes[go_term]
            min_probability[category] = min(min_probability[category], probability)
            max_probability[category] = max(max_probability[category], probability)

    # Normalize per class
    for test_protein_id, test_protein_predictions in predictions.items():
        for go_term, probability in test_protein_predictions.items():
            category = go_classes[go_term]
            if (min_probability[category] < 2) and (abs(max_probability[category] - min_probability[category]) > 0.0000000001):
                test_protein_predictions[go_term] = (probability - min_probability[category]) / (max_probability[category] - min_probability[category])

    normalized_predictions_file = prefix + "predictions/" + method + "/final_predictions/normalized/predictions_" + str(bacteria) + ".pkl"
    with open(normalized_predictions_file, "wb") as f:
        pickle.dump(predictions, f)
