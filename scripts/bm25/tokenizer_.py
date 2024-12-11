import argparse
import pickle
from pathlib import Path
import pandas as pd

from typing import List
import spacy
import re
import os
import spacy.cli
from tqdm import tqdm

LANGS = ["en"]

from src.bm25.text_tokenizer import EnglishTokenizer

def main():
    """
    Processes and tokenizes a text corpus in specified languages.

    Command-line Arguments:
        -c, --corpus_df (Path): Path to the input corpus JSON file. Required.
        -o, --output_dir (Path): Path to the directory where output files will be saved. Required.
        -b, --batch_size (int): The batch size for tokenization (default is 64).
        --cores (int): The number of processor cores to use for parallel processing (default is 10).


    Saves:
        A pickled file containing tokenized text data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--corpus_df", type=Path, default="../../data/clean_data/metadata_clean.csv")
    parser.add_argument("-o", "--output_dir", type=Path, default="../../data/tokenized_data")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--cores", type=int, default=10)

    args = parser.parse_args()

    print("corpus_df:", args.corpus_df)
    print("output_dir:", args.output_dir)

    assert args.corpus_df.exists(), "corpus_df does not exist"
    assert args.corpus_df.is_file(), "corpus_df is not a file"

    assert args.output_dir.exists(), "output_dir does not exist"
    assert args.output_dir.is_dir(), "output_dir is not a directory"

    corpus_df = pd.read_csv(args.corpus_df)

    tokenizer = EnglishTokenizer()

    tokenized_texts = tokenizer.tokenize(corpus_df["author"].tolist(),
                                         corpus_df["title"].tolist(),
                                         corpus_df["description"].tolist(),
                                         batch_size=args.batch_size,
                                         n_process=args.cores)

    output_file = (
        args.output_dir
        / f"tokens_{args.language}.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)


if __name__ == "__main__":
    main()