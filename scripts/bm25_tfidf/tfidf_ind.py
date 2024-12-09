from src.bm25_tfidf.tfidf import Tf_Idf_Vectorizer

import pickle 

import argparse
from pathlib import Path



if __name__ == "__main__":
    """
    Script for creating and saving a TF-IDF model for a specified language.

    Command-line Arguments:
        -dir, --token_dir (Path): Path to the directory containing the tokenized documents (default is './data').
        -lang, --language (str): Language code for which to create the TF-IDF model. Choices are ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko'].

    Output:
        Saves a pickled TF-IDF model to the current directory as 'tfidf_<language>.pkl'.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    parser.add_argument("-lang", "--language", type=str, default='en', choices=['en'])
    args = parser.parse_args()

   
    #Load documents tokenization  : 
    with open(f'{args.token_dir}/tokens_{args.language}.pkl', "rb") as f:
        docs = pickle.load(f)

    
    tfidf = Tf_Idf_Vectorizer()
    tfidf.fit_transform(docs)

    #save tfidf model
    with open(f"tfidf_{args.language}.pkl", "wb") as f:
        pickle.dump(tfidf, f)


    print("Tf-Idf model created for " + args.language)
