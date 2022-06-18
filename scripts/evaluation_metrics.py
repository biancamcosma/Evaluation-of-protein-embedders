import numpy as np


def calculate_fmax(predictions, true_labels, t_values, test_proteins, go_classes, evaluated_go_class):
    """Determines the Fmax, precision and recall of a set of predictions.

    Parameters:
    predictions: A nested dictionary containing the predictions for each test protein. The format is: {protein_id: {GO_id: probability}}.
    true_labels: A dictionary providing the ground-truth labels for the test proteins. Each protein ID is mapped to an array of ground-truth GO terms.
    t_values: A numpy array containing the threshold values that are used to calculate precision and recall. These values range from 0 to 1.
    test_proteins: A set of target protein IDs.
    go_classes: A dictionary providing the mapping of each GO term to its class. A GO term is identified by its ID, while classes can be either MF, BP, or CC.
    evaluated_go_class: The GO class for which we are computing the Fmax evaluation.

    Returns:
    The maximum F-measure (Fmax), and precision and recall values for all thresholds in t_values.

    """
    
    n = len(test_proteins)

    # Store precision and recall values
    precision_values = []
    recall_values = []

    f_max = -np.inf

    for t in t_values:
        precision = 0.0
        recall = 0.0

        # Total number of test proteins for which a prediction
        # has been made with probability >= t.
        mt = 0

        for test_protein in test_proteins:
            # This variable stores whether the protein sequence
            # has any predicted GO term with probability >= t.
            counts = 0

            # Fetch predictions for this test protein.
            predicted_go_terms = dict()
            if test_protein in predictions:
                predicted_go_terms = predictions[test_protein]

            # Fetch ground-truth labels and filter them based on desired class.
            actual_go_terms = set(true_labels[test_protein])
            actual_go_terms_from_class = set(filter(lambda x: go_classes[x] == evaluated_go_class, actual_go_terms))

            intersection_size = 0

            # Keep track of the number of terms in the prediction that belong to the desired class.
            class_terms = 0

            for predicted_go_term in predicted_go_terms.items():
                # Check whether the term belongs to the desired class
                if go_classes[predicted_go_term[0]] in evaluated_go_class:
                    class_terms = class_terms + 1

                    # Check threshold condition
                    if predicted_go_term[1] >= t:
                        counts = 1
                        if predicted_go_term[0] in actual_go_terms_from_class:
                            intersection_size = intersection_size + 1

            # If t = 0, override precision -> all terms in the ontology are predicted with probability 0.
            if t == 0:
                precision = precision + (float(len(actual_go_terms_from_class)) / len(go_classes))
            elif (class_terms > 0):
                precision = precision + (float(intersection_size) / class_terms)
                
            if len(actual_go_terms_from_class) > 0:
                recall = recall + (float(intersection_size) / len(actual_go_terms_from_class))
            
            mt = mt + counts

        if mt > 0:
            precision = float(precision) / mt
        recall = float(recall) / n

        # If t is 0, override recall as 1.
        if t == 0:
            recall = 1.0    
        
        precision_values.append(precision)
        recall_values.append(recall)

        if precision + recall > 0:
            f_max = max(f_max, (2 * precision * recall) / (precision + recall))
        else:
            f_max = max(f_max, 0)

    return f_max, precision_values, recall_values


def calculate_smin(predictions, true_labels, t_values, test_proteins, go_classes, evaluated_go_class, go_ics):
    """Determines the Smin, misinformation and remaining uncertainty of a set of predictions.

    Parameters:
    predictions: A nested dictionary containing the predictions for each test protein. The format is: {protein_id: {GO_id: probability}}.
    true_labels: A dictionary providing the ground-truth labels for the test proteins. Each protein ID is mapped to an array of ground-truth GO terms.
    t_values: A numpy array containing the threshold values that are used to calculate precision and recall. These values range from 0 to 1.
    test_proteins: A set of target protein IDs.
    go_classes: A dictionary providing the mapping of each GO term to its class. A GO term is identified by its ID, while classes can be either MF, BP, or CC.
    evaluated_go_class: The GO class for which we are computing the Fmax evaluation.
    go_ics: A dictionary mapping each GO id to the term's information content (ic).

    Returns:
    The minimum semantic distance (Smin), and remaining uncertainty misinformation values for all thresholds in t_values.

    """
    
    n = len(test_proteins)

    # Store remaining uncertainty and missing information values
    ru = []
    mi = []

    smin = np.inf

    for t in t_values:
        remaining_uncertainty = 0.0
        missing_information = 0.0

        for test_protein in test_proteins:
            # Fetch predictions for this test protein.
            predicted_go_terms = dict()
            if test_protein in predictions:
                predicted_go_terms = predictions[test_protein]

            # Fetch all predictions with probability higher than t
            predicted_over_threshold = set()
            for predicted_go_term in predicted_go_terms.keys():
                if predicted_go_terms[predicted_go_term] >= t:
                    predicted_over_threshold.add(predicted_go_term)
            # Filter based on GO class
            predicted_over_threshold = set(filter(lambda x: go_classes[x] == go_class, predicted_over_threshold))

            # Fetch ground-truth labels and filter them based on desired class
            actual_go_terms = set(true_labels[test_protein])
            actual_go_terms_from_class = set(filter(lambda x: go_classes[x] == go_class, actual_go_terms))

            for actual_go_term in actual_go_terms_from_class:
                if actual_go_term not in predicted_over_threshold:
                    remaining_uncertainty = remaining_uncertainty + go_ics[actual_go_term]

            for predicted_go_term in predicted_over_threshold:
                if predicted_go_term not in actual_go_terms_from_class:
                    missing_information = missing_information + go_ics[predicted_go_term]

        remaining_uncertainty = float(remaining_uncertainty) / n
        missing_information = float(missing_information) / n

        if remaining_uncertainty > 0 or missing_information > 0:
            smin = min(smin, math.sqrt((remaining_uncertainty ** 2) + (missing_information ** 2)))

        ru.append(remaining_uncertainty)
        mi.append(missing_information)

    return smin, ru, mi
