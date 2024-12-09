from collections import Counter
import numpy as np
from pathlib import Path
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from tqdm import tqdm
from typing import Dict
from typing import Optional
from typing import List
from typing import Tuple


class BM25:
    """
    BM25 implementation.

    Attributes:
        k1 (float): Controls term frequency saturation.
        b (float): Controls length normalization.
        corpus_size (int): Number of documents in the corpus.
        inverted_index (Dict[str, int]): Mapping of tokens to vocabulary indices.
        vocab_size (int): Number of unique tokens in the corpus.
        bm25 (csc_matrix): BM25 scores for the corpus.

    
    Methods:
        _get_vocab_index(corpus: List[List[str]]) -> Dict[str, int]: Get dictionary of tokens to vocabulary indices.
        _get_term_frequency_matrix(corpus: List[List[str]]) -> coo_matrix: Get term frequency matrix.
        fit(corpus: List[List[str]]) -> None: Fit BM25 model to the documents.
        _scores(query: List[str]) -> np.ndarray: Compute BM25 scores over all documents for a query.
        match(query: List[str], k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]: Match a query to the corpus.
        match_and_eval(query: List[str], target: int) -> Tuple[np.ndarray, float, int]: Match a query to the corpus and evaluate.
        save(path: Path) -> None: Save BM25 model.
    
    """
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 model.
        Args:
            k1 (float): Controls term frequency saturation.
            b (float): Controls length normalization.
        """
        self.k1 = k1
        self.b = b

        self.corpus_size = None
        self.inverted_index = None
        self.vocab_size = None
        self.bm25 = None

    def _get_vocab_index(self, corpus: List[List[str]]) -> Dict[str, int]:
        """
        Get vocabulary index.
        Args:
            corpus (List[List[str]]): List of documents.
            
        Returns:
            Dict[str, int]: Mapping of tokens to vocabulary indices.
        """
        i = 0
        inverted_index = {}
        for doc in tqdm(corpus):
            for token in doc:
                if token not in inverted_index:
                    inverted_index[token] = i
                    i += 1
        return inverted_index

    def _get_term_frequency_matrix(self, corpus: List[List[str]]) -> coo_matrix:
        """
        Get term frequency matrix.
        Args:
            corpus (List[List[str]]): List of documents.
        Returns:
            coo_matrix: Term frequency matrix.
        """
        r, c = [], []
        data = []
        for i, doc in tqdm(enumerate(corpus), total=len(corpus)):
            counter = Counter(doc)
            for token, freq in counter.items():
                r.append(i)
                c.append(self.inverted_index[token])
                data.append(freq)
        return coo_matrix((data, (r, c)), shape=(self.corpus_size, self.vocab_size))

    def fit(self, corpus: List[List[str]]) -> None:
        """
        Fit BM25 model.
        Args:
            corpus (List[List[str]]): List of documents.
        """
        self.corpus_size = len(corpus)
        self.inverted_index = self._get_vocab_index(corpus)
        self.vocab_size = len(self.inverted_index)
        tf = self._get_term_frequency_matrix(corpus)
        doc_occurrences = tf.astype(np.bool_).sum(axis=0)
        idf = np.log(
            (self.corpus_size - doc_occurrences + 0.5) / (doc_occurrences + 0.5) + 1.0
        )
        doc_lens = tf.sum(axis=1)
        normalized_doc_lens = doc_lens / doc_lens.mean()
        k1, b = self.k1, self.b
        rows, cols = tf.nonzero()
        bm25_data = []
        for i, j, tf_ij in tqdm(zip(rows, cols, tf.data), total=len(rows)):
            bm25_data.append(
                idf[0, j]
                * (
                    tf_ij
                    * (k1 + 1.0)
                    / (tf_ij + k1 * (1.0 - b + b * normalized_doc_lens[i, 0]))
                )
            )
        self.bm25 = csc_matrix(
            (bm25_data, (rows, cols)), shape=(self.corpus_size, self.vocab_size)
        )

    def _scores(self, query: List[str]) -> np.ndarray:
        """
        Compute BM25 scores for a query.
        Args:
            query (List[str]): Query tokens.
        Returns:
            np.ndarray: BM25 scores.
        """
        token_ids = np.array(
            [
                self.inverted_index[token]
                for token in query
                if token in self.inverted_index
            ]
        )
        return np.array(self.bm25[:, token_ids].sum(axis=1))[:, 0]

    def match(self, query: List[str], k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a query to the corpus and return the top k results.
        Args:
            query (List[str]): Query tokens.
            k (Optional[int]): Number of results to return.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Matching indices and scores.
        """
        scores = self._scores(query)
        indices = np.argsort(scores)[::-1]
        if k is not None:
            indices = indices[:k]
        return indices, scores[indices]

    def match_and_eval(self, query: List[str], target: int) -> Tuple[np.ndarray, float, int]:
        """
        Match a query to the corpus and evaluate.
        Args:
            query (List[str]): Query tokens.
            target (int): Target document index.
        Returns:
            Tuple[np.ndarray, float, int]: Scores, target score, and target rank.
        """
        scores = self._scores(query)
        score_target = scores[target]
        rank_target = sum(score_target <= score for score in scores)
        return scores, score_target, rank_target

    def save(self, path: Path) -> None:
        """
        Save BM25 model.
        Args:
            path (Path): Path to save the model.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_bm25(path: Path) -> BM25:
    """
    Load BM25 model.
    Args:
        path (Path): Path to the model.
    Returns:
        BM25: BM25 model.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
