import argparse
import pickle
from pathlib import Path
import pandas as pd

import os

LANGS = ["en"]

from src.bm25.text_tokenizer import EnglishTokenizer

def main():
    """
    Processes and creates tokenized text data for each book.

    Command-line Arguments:
        -c, --corpus_df (Path): Path to the CSV file containing the books metadata (default is 'data/clean_data/metadata_clean.csv').
        -o, --output_dir (Path): Path to the directory where output files will be saved (default is 'data/tokenized_data').
        -b, --batch_size (int): The batch size for tokenization (default is 64).
        --cores (int): The number of processor cores to use for parallel processing (default is 10).


    Saves:
        A pickled file containing the tokenized texts for each book.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--corpus_df", type=Path, default="data/clean_data/metadata_clean.csv")
    parser.add_argument("-o", "--output_dir", type=Path, default="data/tokenized_data")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--cores", type=int, default=10)

    args = parser.parse_args()

    print("corpus_df:", args.corpus_df)
    print("output_dir:", args.output_dir)

    assert args.corpus_df.exists(), "corpus_df does not exist"
    assert args.corpus_df.is_file(), "corpus_df is not a file"

    corpus_df = pd.read_csv(args.corpus_df).dropna()

    # Tokenize the texts using the EnglishTokenizer
    tokenizer = EnglishTokenizer()

    tokenized_texts = tokenizer.tokenize(corpus_df["author"].tolist(),
                                         corpus_df["title"].tolist(),
                                         corpus_df["description"].tolist(),
                                         batch_size=args.batch_size,
                                         n_process=args.cores)
    

    # Save the tokenized texts 
    output_file = (
        args.output_dir
        / f"tokens.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)

    # save the book ids
    book_ids = corpus_df["book_id"].tolist()
    output_file = (
        args.output_dir
        / f"book_ids.pkl"
    )
    
    with open(output_file, "wb") as f:
        pickle.dump(book_ids, f)

    # save the authors
    authors_list = corpus_df["author"].tolist()
    output_file = (
        args.output_dir
        / f"authors.pkl"
    )
    
    with open(output_file, "wb") as f:
        pickle.dump(authors_list, f)


if __name__ == "__main__":
    main()