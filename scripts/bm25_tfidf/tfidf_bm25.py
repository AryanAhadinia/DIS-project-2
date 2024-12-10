import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


from src.bm25_tfidf.text_tokenizer import (
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
    Script for combining BM25 and TF-IDF models to evaluate document retrieval accuracy.

    Command-line Arguments:
        -dir, --token_dir (Path): Path to the directory containing tokenized query files (default: './data').
        --tfidf_path (Path): Path to the directory containing TF-IDF models (default: current directory).
        --ids_path (Path): Path to the JSON file containing document IDs (default: './data/ids_dict.json').
        --bm25_path (Path): Path to the directory containing BM25 models (default: current directory).

    Output:
        Prints the accuracy of the combined BM25 and TF-IDF retrieval for each language.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    parser.add_argument("--tfidf_path", type=Path, default=".")
    parser.add_argument("--ids_path", type=Path, default="./data/ids_dict.json")
    parser.add_argument("--bm25_path", type=Path, default=".")

    args = parser.parse_args()

    bm25s = {}

    for lang in LANGS:
        with open(f"{args.bm25_path}/bm25_{lang}.pkl", "rb") as f:
            bm25s[lang] = pickle.load(f)

    tfidfs = {}
    for lang in LANGS:
        with open(args.tfidf_path / f"tfidf_{lang}.pkl", "rb") as f:
            tfidfs[lang] = pickle.load(f)

    # queries = pd.read_csv(f'{args.token_dir}/test.csv')
    queries = pd.read_csv(f'{args.token_dir}/train.csv')

    with open(args.ids_path, "r") as f:
        ids_dict = json.load(f)

    for lang in LANGS:
        ids_dict[lang] = np.array(ids_dict[lang])

    ls = [[] for _ in range(len(queries))]
    queries["docids"] = ls

    k_b = 9
    k = 10

    for lang in LANGS:
        queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        # queries_lang = queries[queries["lang"] == lang][["query"]].reset_index(drop=True)
        tokens = tokenizers[lang].tokenize(queries_lang["query"].tolist())
        bm25_ind = bm25s[lang]
        bm25_ind_doc_ids = []
        for tokenized_query in tokens:
            indices, _ = bm25_ind.match(tokenized_query, k=k)
            bm25_ind_doc_ids.append(ids_dict[lang][indices].tolist())

        tfidf_doc_ids = [ids_dict[lang][docid].tolist() for docid in tfidfs[lang].retrieve_top_k(tokens, k=k_b)]

        doc_ids = []
        for i in range(len(queries_lang)):
            docid = bm25_ind_doc_ids[i][:k_b]
            for rec in tfidf_doc_ids[i]:
                if rec not in docid:
                    docid.append(rec)
                if len(docid) == k:
                    break
            l = k_b
            while len(docid) < k:
                if bm25_ind_doc_ids[i][l] not in docid:
                    docid.append(bm25_ind_doc_ids[i][l])
                l += 1
            doc_ids.append(docid)

        # queries.loc[queries["lang"] == lang, "docids"] = pd.Series(doc_ids, index=queries.loc[queries["lang"] == lang].index)

        acc = 0
        for i, row in queries_lang.iterrows():
            if row["positive_docs"] in doc_ids[i]:
                acc += 1
        print(f"Accuracy for {lang} : {acc / len(queries_lang)}")

        gc.collect()

    # queries = queries[["id", "docids"]]
    # queries.to_csv(f"{args.token_dir}/submission.csv", index=False)
