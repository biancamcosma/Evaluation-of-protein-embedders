import numpy as np
import pandas as pd
import pickle

from goatools.obo_parser import OBOReader
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.godag.go_tasks import get_go2children

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


def print_stats(protein_ids, go_classes, annotations, go_depths):
    """Prints statistics for a set of protein IDs and their ground-truth labels.

    Parameters:
    protein_ids: A set of protein IDs for which we print the statistics.
    go_classes: A dictionary mapping GO IDs to their class (MF, BP, CC).
    annotations: All dataset annotations. A dictionary with keys as protein IDs and values as arrays of GO terms.
    go_depths: A dictionary mapping GO IDs to their depth in the hierarchy.

    """
    
    # Distinct number of GO terms for each class (not counted twice even if they annotate 2 proteins)
    go_terms_per_class = {"MF": 0, "BP": 0, "CC": 0, "All": 0}

    # Average number of annotations for each protein
    avg_annotations_num = {"MF": 0, "BP": 0, "CC": 0, "All": 0}

    # Weighted average depth of annotations (sum(depth * count) / sum(counts))
    weighted_avg_depth = {"MF": 0, "BP": 0, "CC": 0, "All": 0}
    # Total number of annotations per class (we do count twice if the same GO term annotates 2 proteins)
    go_counts_per_class = {"MF": 0, "BP": 0, "CC": 0, "All": 0}
    # How many times a GO term appears in the datatset annotations
    go_counts = dict()

    all_go_terms = set()

    for protein_id in protein_ids:
        go_terms = annotations[protein_id]

        for go_term in go_terms:
            all_go_terms.add(go_term)

            go_counts_per_class["All"] += 1
            go_counts_per_class[go_classes[go_term]] += 1

            if go_term not in go_counts:
                go_counts[go_term] = 1
            else:
                go_counts[go_term] += 1

    for go_term in all_go_terms:
        go_terms_per_class[go_classes[go_term]] += 1
        go_terms_per_class["All"] +=1

        weighted_avg_depth["All"] += go_depths[go_term] * go_counts[go_term]

        for go_class in ["MF", "BP", "CC"]:
            if go_classes[go_term] == go_class:
                weighted_avg_depth[go_class] += go_depths[go_term] * go_counts[go_term]

    for go_class in ["MF", "BP", "CC", "All"]:
        weighted_avg_depth[go_class] = float(weighted_avg_depth[go_class]) / go_counts_per_class[go_class]

    print("Distinct number of GO annotations for this set:")
    for key in go_terms_per_class.keys():
        print(key + ": " + str(go_terms_per_class[key]) + " terms")

    print("Average number of annotations per protein:")
    for key in go_counts_per_class.keys():
        print(key + ": " + str(float(go_counts_per_class[key])/len(protein_ids)))
    print(go_counts_per_class["CC"] + go_counts_per_class["BP"] + go_counts_per_class["MF"] == go_counts_per_class["All"])

    print("Wighted average GO term depth for this set:")
    for key in weighted_avg_depth.keys():
        print(key + ": " + str(weighted_avg_depth[key]))

        
def convert_deepgo_predictions(predictions_file, terms_file, results_file):
    """Convert DeepGOPlus predictions to a format that can be used in our evaluation.

    Parameters:
    predictions_file: Location of the file storing DeepGOPlus predictions.
    terms_file: Location of the file storing the dataframe with GO terms used by DeepGOPlus.
    results_file: Location of the file where we store the converted predictions.

    """
    
    # Load file with predictions
    with open(predictions_file, "rb") as f:
        deepgo_predictions_df = pickle.load(f)
    # Load file with GO terms (provides mappings from an index to a GO id)
    with open(terms_file, "rb") as f:
        terms = pickle.load(f)

    index_to_term = dict()
    for index, row in terms.iterrows():
        index_to_term[index] = row.terms

    predictions = dict()
    
    for row in deepgo_predictions_df.itertuples():
        protein_id = row.proteins

        predicted_terms = dict()
        i = 0
        for label in row.labels:
            if label == 1:
                predicted_terms[index_to_term[i]] = row.preds[i]

            i = i + 1

        predictions[protein_id] = predicted_terms

    with open(results_file, "wb") as f:
        pickle.dump(predictions, f)


def get_descendants(godag, go_term, descendants_set):
    """Gets the descendants of a given GO term in the GO hierarchy.

    Parameters:
    go_dag: A directed acyclic graph containing GO terms. (as generated by goatools)
    go_term: The ID of the GO term for which we compute the descendants.
    descendants_set: The set of descendants of the go_term.

    """
    
    children = set()
    if go_term in godag:
        children = set(map(lambda x: x.id, godag[go_term].children))
        for child in children:
            descendants_set.add(child)
            get_descendants(godag, child, descendants_set)
    else:
        print("Not found in DAG (obsolete term): " + go_term)


def get_go_ics(go_terms, obo_file, go_counts, go_classes, results_file):
    """Determines and stores the information contents of all GO terms in an .obo file.

    Parameters:
    go_terms: a set of GO terms for which we compute the information content.
    obo_file: A string of the full path to the .obo file.
    go_counts: A dictionary providing the number of annotations for each GO term (the number of proteins it annotates).
    go_classes: A dictionary mapping GO IDs to their class (MF, BP, CC).
    results_file: Location of the file where we store the GO information contents.

    """

    # Get total number of annotations per class
    class_counts= {"MF": 0, "BP": 0, "CC": 0}
    for go_term in go_terms:
        category = go_classes[go_term]
        class_counts[category] = class_counts[category] + go_counts[go_term]

    go_ic = dict()
    for go_term in go_terms:
        # Get set of descendants (including the term itself)
        optional_relationships = {'regulates', 'negatively_regulates', 'positively_regulates'}
        descendants = {go_term}
        get_descendants(godag, go_term, descendants)

        descendants = set(filter(lambda x: x in go_terms, descendants))

        # Determine total number of annotations for descendants
        num_annotations_descendants = 0
        for descendant in descendants:
             num_annotations_descendants = num_annotations_descendants + go_counts[descendant]

        # Calculate information content
        p = float(num_annotations_descendants) / class_counts[go_classes[go_term]]

        # These 3 are obsolete terms. Make IC = 0.
        if go_term in {"GO:0052312", "GO:1902586", "GO:2000775"}:
            go_ic[go_term] = 0
        else:
            go_ic[go_term] = -math.log(p, 2)

    # Store IC values
    with open(results_file, "wb") as f:
        pickle.dump(go_ic, f)
