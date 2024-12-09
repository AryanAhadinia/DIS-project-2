import numpy as np
import pickle

import pandas as pd

import json
import argparse
from pathlib import Path
import gc

from src.bm25_tfidf.tokenizer import (
    FrenchTokenizer,
    EnglishTokenizer,
    GermanTokenizer,
    ItalianTokenizer,
    SpanishTokenizer,
    ArabicTokenizer,
    KoreanTokenizer
)

LANGS = ["en"]

tokenizers = {"en": EnglishTokenizer()}

if __name__ == "__main__":
    """
    Script for evaluating TF-IDF models with a set of queries and calculating accuracy.

    Command-line Arguments:
        --token_dir (Path): Path to the directory containing tokenized query files (default: './data').
        --ids_path (Path): Path to the JSON file containing document IDs (default: './data/ids_dict.json').
        --tfidf_path (Path): Path to the directory containing the TF-IDF models (default: current directory).

    Output:
        Prints the accuracy of the TF-IDF retrieval for each language.
        Optionally, the results can be saved as a CSV file.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--token_dir", type=Path, default="./data")
    parser.add_argument("--ids_path", type=Path, default="./data/ids_dict.json")
    parser.add_argument("--tfidf_path", type=Path, default=".")
    tfidfs = {}
    args = parser.parse_args()
    for lang in LANGS:
        with open(f"{args.tfidf_path}/tfidf_{lang}.pkl", "rb") as f:
            tfidfs[lang] = pickle.load(f)

    # load doc ids dict with json
    with open(args.ids_path, "r") as f:
        ids_dict = json.load(f)

    for lang in LANGS:
        ids_dict[lang] = np.array(ids_dict[lang])

    queries = pd.read_csv(f'{args.token_dir}/train.csv')
    # queries = pd.read_csv(f'{args.token_dir}/test.csv')

    ls = [[] for _ in range(len(queries))]
    queries["docids"] = ls
    for lang in LANGS:
        queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        # queries_lang = queries[queries["lang"] == lang][["query"]].reset_index(drop=True)
        tokens = tokenizers[lang].tokenize([query for query in queries_lang["query"].tolist()])
        ids_ = tfidfs[lang].retrieve_top_k(tokens, k=10)
        doc_ids = [ids_dict[lang][doc_id].tolist() for doc_id in ids_]
        queries.loc[queries["lang"] == lang, "docids"] = pd.Series(doc_ids,
                                                                   index=queries.loc[queries["lang"] == lang].index)

        acc = 0
        for i, row in queries_lang.iterrows():
            if row["positive_docs"] in doc_ids[i]:
                acc += 1
        print(f"Accuracy for {lang} : {acc / len(queries_lang)}")
        gc.collect()

    # queries = queries[["id", "docids"]]
    # queries.to_csv(f"{args.token_dir}/submission.csv", index=False)