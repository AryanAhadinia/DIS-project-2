import argparse
import pickle
from pathlib import Path

from src.bm25.bm25 import BM25

if __name__ == "__main__":
    """
    Script for creating and saving a BM25 model for a specified language.

    Command-line Arguments:
        -dir, --token_dir (Path): Path to the directory containing the tokenized documents (default is './data').
        -lang, --language (str): Language code for which to create the BM25 model. Choices are ['en'].

    Output:
        Saves a pickled BM25 model to the current directory as 'bm25_<language>.pkl'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "../../data/tokenized_data")
    parser.add_argument("-lang", "--language", type=str, default='en', choices=['en'])
    args = parser.parse_args()

    with open(f'{args.token_dir}/tokens_{args.language}.pkl', "rb") as f:
        docs = pickle.load(f)

    lang_params = {
        "en": {"k1": 1.4, "b": 0.5},
    }

    bm25_ind = BM25(k1=lang_params[args.language]["k1"], b=lang_params[args.language]["b"])
    bm25_ind.fit(docs)

    with open(f"bm25_{args.language}.pkl", "wb") as f:
        pickle.dump(bm25_ind, f)

    print("BM25 model created for " + args.language)