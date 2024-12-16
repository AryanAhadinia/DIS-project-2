from typing import List

import spacy
import re

import spacy.cli
from tqdm import tqdm



class BaseTokenizer:
    """
    Base tokenizer class.
    
    Attributes:
        nlp (spacy.Language): Spacy NLP pipeline.
        stop_words (Set[str]): Set of stop words.
        
    Methods:
        preprocess_text(text: str) -> str: Preprocess text.
        tokenize(texts: List[str], batch_size: int = 32, n_process: int = 4) -> List[List[str]]: Tokenize texts.
        """

    def __init__(self, model_name: str):
        """
        Initialize BaseTokenizer.
        Args:
            model_name (str): Spacy model name.
        """
        spacy.cli.download(model_name)
        self.nlp = spacy.load(model_name, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text : Remove URLs, long sequences of non-alphanumeric characters, and excessive whitespace.
        Args:
            text (str): Input text.
        Returns:
            str: Preprocessed text.
        """
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def tokenize(self, authors: List[str], titles: List[str], texts: List[str], batch_size: int = 32, n_process: int = 4) -> List[List[str]]:
        """
        Tokenize texts.
        Args:
            texts (List[str]): List of texts.
            batch_size (int): Batch size for processing texts.
            n_process (int): Number of processes to use for tokenization.
        Returns:
            List[List[str]]: List of tokenized texts.
        """

        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        docs = self.nlp.pipe(
            preprocessed_texts, batch_size=batch_size, n_process=n_process
        )

        # Toknized texts correspond to the title and the lemmatized tokens of the description of the book.
        tokenized_texts = [
            #author_.split() + title.split() +
            title.split() +
            [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
            ]
            for doc, author_, title in tqdm(zip(docs, authors, titles), total=len(texts), desc="Tokenizing")
        ]

        return tokenized_texts


class EnglishTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("en_core_web_sm")
