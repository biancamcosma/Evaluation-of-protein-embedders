
embedders = ["protbert", "esm1b", "seqvec", "t5"]
prefix = "../data/"

for b in ["bacillus", "ecoli"]:
    print("Creating predictions for " + b)
    print("---")

    for embedder in embedders:
        print("\n" + "Embedder: " + embedder)
        
        # Load training and test files
        test_embeddings_file = prefix + "embeddings/" + embedder + "/" + str(b) + "/" + "test_embeddings.pkl"
        train_embeddings_file1 = prefix + "embeddings/" + embedder + "/" + str(b) + "/" + "train_embeddings1.pkl"
        train_embeddings_file2 = prefix + "embeddings/" + embedder + "/" + str(b) + "/" + "train_embeddings2.pkl"

        module.knn(99.999,
            prefix + "uniprot2go_exp.tsv",
            test_embeddings_file,
            train_embeddings_file1,
            train_embeddings_file2,
            b,
            embedder,
            prefix)
