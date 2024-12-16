import argparse
import os
import pickle
from pathlib import Path

from src.bm25.bm25 import BM25

if __name__ == "__main__":
    """
    Script for creating and saving a BM25 model.

    Command-line Arguments:
        -dir, --token_dir (Path): Path to the directory containing the tokenized texts for each book (title + description) (default is './data/tokenized_data').
        --output_dir (Path): Path to the directory where the BM25 model will be saved (default is './data/bm25_models').

    Output:
        Saves a pickled BM25 model to the current directory as 'bm25_<language>.pkl'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "data/tokenized_data")
    parser.add_argument("--output_dir", type=Path, default = "data/bm25_models")
    args = parser.parse_args()

    # Load tokenized documents
    with open(f'{args.token_dir}/tokens.pkl', "rb") as f:
        docs = pickle.load(f)

    lang_params = {
        "k1": 1.4, "b": 0.5
    }

    # Create, fit and save BM25 model
    bm25_ind = BM25(k1=lang_params["k1"], b=lang_params["b"])
    bm25_ind.fit(docs)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25_ind, f)

    print("BM25 model created")