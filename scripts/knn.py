import numpy as np
from numpy.linalg import norm
import pandas as pd
import pickle
import time

from utils import parse_tsv, normalize_data

def create_train_matrix(train_embeddings):
    """Creates a matrix where each row is a database embedding, divided by its own norm.

    Parameters:
    train_embeddings: A dictionary mapping a protein ID to its embedding.

    Returns:
    A matrix where each embedding is normalized and represented as a row, and a dictionary mapping each protein ID to its row index in the embedding matrix.

    """
    num_embeddings = len(train_embeddings)
    embedding_size = len(list(train_embeddings.values())[0])

    embedding_matrix = np.zeros((num_embeddings, embedding_size))

    id_to_row = dict()

    i = 0
    for train_embedding_id, train_embedding in train_embeddings.items():
        train_protein_id = train_embedding_id.split("|")[1]

        embedding_matrix[i] = train_embedding
        id_to_row[train_protein_id] = i

        i = i + 1

    return embedding_matrix/np.linalg.norm(embedding_matrix, axis=1, keepdims=True), id_to_row


def knn(percentile, tsv_file, test_file, train_file1, train_file2, bacteria, method, prefix):
    """Generates and stores predictions for a query and databse set based on a k-NN search.

    Parameters:
    percentile: Float value ranging from 0 to 100. This percentile is used to determine the similarity threshold for the k-NN predictions.
    tsv_file: The location of the file containing the mapping from protein IDs to ground-truth GO terms.
    test_file: Location of a .pkl file containing a dictionary that maps the IDs of query proteins to their embeddings.
    train_file1:  Location of a .pkl file containing a dictionary that maps the IDs of database proteins to their embeddings. Half of the database embeddings are here.
    train_file2: Location of a .pkl file containing a dictionary that maps the IDs of database proteins to their embeddings. The remaining half of the database embeddings are here.
    bacteria: The bacterium from which the query set proteins were selected.
    method: The embedding model for which we are generating k-NN predictions.
    prefix: Prefix of the data folder where we store the predictions.

    """
    
    start = time.time()    

    annotations_map = parse_tsv(tsv_file)
    test_embeddings, train_embeddings = get_embeddings(test_file, train_file1, train_file2)

    embedding_matrix, id_to_row = create_train_matrix(train_embeddings)
    
    negative_cos = False

    # Iterate over test set and make predictions
    n = len(test_embeddings)
 
    print("\nPercentile used to compute the similarity threshold: " + str(percentile))
    print("----------")

    predictions = dict()

    i = 1
    for test_embedding_id, test_embedding in test_embeddings.items():
        if i % 100 == 0 or i == n:
            print(str(i) + " out of " + str(n))
   
        # To each test protein, associate a dictionary of GO terms and their associated probabilities
        predicted_go_terms = dict()

        # Determine similarities
        dot_products = np.matmul(embedding_matrix, test_embedding)
        similarities = dot_products / np.linalg.norm(test_embedding)

        t = np.percentile(similarities, percentile)
    
        for train_embedding_id, train_embedding in train_embeddings.items():
            test_protein_id = test_embedding_id.split("|")[1]
            train_protein_id = train_embedding_id.split("|")[1]

            similarity = similarities[id_to_row[train_protein_id]]

            if similarity < 0:
                negative_cos = True

            # Only consider neighbors with similarity over the threshold
            if similarity >= t:
                # Fetch GO terms associated with train protein
                train_protein_go_terms = annotations_map[train_protein_id]

                # Iterate over the train protein go terms
                for go_term in train_protein_go_terms:
                    if go_term in predicted_go_terms:
                        # If this GO term has been predicted before, store the maximum similarity
                        predicted_go_terms[go_term] = max(predicted_go_terms[go_term], similarity)
                    else:
                        predicted_go_terms[go_term] = similarity

        # Make prediction for this test protein
        predictions[test_protein_id] = predicted_go_terms

        i = i + 1

    # Store predictions dictionary
    with open(prefix + "predictions/" + method + "/final_predictions/predictions_" + str(bacteria) + ".pkl" , "wb") as f:
         pickle.dump(predictions, f)
    # Store normalized predictions
    normalize_predictions(predictions, prefix)
        
    end = time.time()
    print("Time taken: " + str(end - start) + " seconds")
