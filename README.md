# Document Retrieval Project - Team: MAM

This repository hosts the implementation of a book recommender system developed as part of the **CS-423 Distributed Information System** course at EPFL. The objective of this project is to design a system capable of predicting user ratings for books. Multiple approaches have been implemented and are detailed in this repository.

## Dataset  
The dataset provided for this project includes:  
- **book.csv**: Maps book IDs to their ISBNs.  
- **train.csv**: Contains user ratings for books (training set).  
- **test.csv**: Contains user/book pairs for which ratings need to be predicted.  

All data is located in the `/data` folder. Additionally, we enhanced the dataset with metadata (title, author, description) stored in `/data/clean_data/metadata_clean`.

## Repository Structure  
- **data/**: Includes the datasets and variables required for the models.  
- **scripts/**: Contains scripts for preprocessing, training, and evaluation.  
  - `data_preprocess/`: Preprocessing metadata from external APIs.  
  - `bm25/`: BM25 preprocessing, indexing, and retrieval scripts.  
  - `recommender/`: Training, testing, and submission creation scripts for recommender models.  
- **src/**: Main source code directory.  
  - `bm25_tfidf/`: Implementation and utilities for the BM25 model.  
  - `matrix_factorization/`: Implementation and utilities for matrix factorization models.  
- **README.md**: Project overview and documentation (this file).  
- **requirements.txt**: List of dependencies for the project.  
- **setup.py**: Installation script for the package.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/AryanAhadinia/DIS-project-2.git  

2. Creat a virtual environment and install the dependencies (requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
   
3. Install the project as a package:
   ```bash
   pip install -e .
   ```
4. If you wish to execute the code for metadata gathering and tokenization, start by running the `data_preprocessing.py` script to preprocess the data. Please note that the API keys for ISBNdb and Google Books APIs included in the script may be outdated; you can replace them with your own keys if needed. Once the data is preprocessed, run the `tokenizer_.py` script to tokenize it.
However, if you are working with the same books from the training set provided for the project, you do not need to run these scripts. The metadata and tokenized data for these books have already been processed and are available in the `/data` folder.
    
5. Finally to generate a submission run the `content_inf.py` file with `--submit` argument. 

## Contributors (Alphabetical Order)
- Aryan Ahadinia
- Matin Ansaripour
- Madeleine Hueber



