# Module 1 Final System - Bigram Language Model
# Purpose: Build a Bigram Language Model

from collections import defaultdict
from nltk import bigrams
from sentence_preprocessing import tokenize
from typing import List


class BigramLanguageModel:
    """Bigram language model built on given corpus and calculates P(w2|w1)
    
    Attributes:
        None
    """

    def __init__(self, corpus: List[str]):
        # Bigram model represented as {'w1': {'list of w2': {'freq', 'proba}}}
        self._bigram_model = defaultdict(
            lambda: defaultdict(lambda: {"freq": 0, "proba": 0})
        )
        
        # Get tokens and bigrams for each document within corpus
        corpus_tokens = [tokenize(doc) for doc in corpus]
        corpus_bigrams = [bigrams(doc_tokens) for doc_tokens in corpus_tokens]

        for doc_bigrams in corpus_bigrams:
            self._calculate_bigram_freq(doc_bigrams)

        self._calculate_bigram_proba()
        return

    def _calculate_bigram_freq(self, doc_bigrams) -> None:
        """Calculates frequency for each bigram and stores in self._bigram_model
        
        Arguments:
            doc {str} -- Single document from corpus
        """
        for (w1, w2) in doc_bigrams:
            self._bigram_model[w1][w2]["freq"] += 1
        return

    def _calculate_bigram_proba(self) -> None:
        """Calculates probability of each bigram P(w2|w1) and stores in self._bigram_model
        """
        for w1 in self._bigram_model.keys():
            words2 = self._bigram_model[w1]
            total_freq = 0
            for w2, val in words2.items():
                total_freq += val["freq"]
            for w2, val in words2.items():
                val["proba"] = val["freq"] / total_freq
        return

    def get_freq(self, w1: str, w2: str) -> int:
        """ Returns frequency of bigram from corpus
        
        Arguments:
            w1 {str} -- First word in bigram
            w2 {str} -- Second word in bigram
        
        Returns:
            int -- Number of counts of (w1, w2)
        """
        return self._bigram_model[w1][w2]["freq"]

    def get_proba(self, w1: str, w2: str) -> float:
        """Returns probability of P(w2|w1) from corpus
        
        Arguments:
            w1 {str} -- First word in bigram
            w2 {str} -- Second word in bigram
        
        Returns:
            float -- Probability value P(w2|w1)
        """
        return self._bigram_model[w1][w2]["proba"]

    def get_w2(self, w1: str) -> str:
        """Returns all words w2 that match (w1,w2) in bigram model
        
        Arguments:
            w1 {str} -- First word in bigram
        
        Returns:
            str -- Matching second words in bigram
        """
        return list(self._bigram_model[w1])
