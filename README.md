# Document Retrieval Project - Team : MAM

This repository contains the implementation of a book recommender system as part of the course CS-423 Distributed Information System at EPFL. The goal is to create a recommender system capable of predicting the ratings of books by users.

## Dataset
You can find the preprocessed data in Kaggle [here](https://www.kaggle.com/datasets/mansarip/dis1-preprocess). The data includes the following:
- **doc_tokens**: Folder of preprocessed documents with tokenized text.
- **models**: Folder of precomputed models for BM25 and TF-IDF indexing.
- **ids_dict.json**: Dictionary mapping document IDs to their respective languages in the order of the corpus.


## Repository Structure
- **scripts/**: Contains scripts for preprocessing the corpus and precomputing the models, running experiments and evaluations.
  - `bm25_tfidf/`: Scripts for BM25 and TF-IDF preprocess, indexing and retrieval.
  - `dpr/`: Scripts for Dense Passage Retrieval (DPR) model data preprocess, training and inference.
- **src/**: Main source code directory.
  - `bm25_tfidf/`: Implementation of BM25 and TF-IDF models and utility functions.
  - `dpr/`: Implementation of DPR models, encoders, and related modules.
- **README.md**: Project overview and documentation (you are reading it!).
- **requirements.txt**: Dependencies required to run the project.
- **setup.py**: Setup script for package installation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/madhueb/DIS_project1.git
   ```
2. Creat a virtual environment and install the dependencies (requirements.txt is for the BM25 and TF-IDF models):
   ```bash
   pip install -r requirements.txt
   ```
3. Install the `camel_tools` module for Arabic text processing:
   ```bash
   camel_data -i disambig-mle-calima-msa-r13
    ```
4. Install the project as a package:
   ```bash
   pip install -e .
   ```
5. Download the preprocessed data from Kaggle and place it in the root directory of the project. You can also build your own preprocessed data using the scripts provided in the `scripts` directory. To tokenize the documents, run the following command:
   ```bash
   python scripts/bm25_tfidf/preprocess/tokenizer_.py <your_args> # for all languages except Arabic
   python scripts/bm25_tfidf/preprocess/tokenizer_.arabic.py <your_args> # for Arabic
   ```
    You might also want to precompute the BM25 and TF-IDF models using the scripts provided in the `scripts` directory. You can also run the following command:
   ```bash
    python scripts/bm25_tfidf/bm25_ind.py <your_args> # for BM25 indexing
    python scripts/bm25_tfidf/tfidf_ind.py <your_args> # for TF-IDF indexing
    ```
6. For the inference you can either use the `bm25_inf.py` and `tfidf_inf.py` scripts in the `scripts/bm25_tfidf` directory. There is also an example notebook in the `scripts/bm25_tfidf` directory with the name `bm25_tfidf_inference.ipynb`.

Regarding the DPR model, you can use the scripts in the `scripts/dpr` directory to train and evaluate the model. The scripts are provided with detailed docstrings and usage instructions.

## Contributors (Alphabetical Order)
- Aryan Ahadinia
- Matin Ansaripour
- Madeleine Hueber



