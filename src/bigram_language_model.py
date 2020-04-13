# Module 1 Final System - Bigram Language Model
# Purpose: Build a Bigram Language Model

from collections import defaultdict
from nltk import bigrams
from sentence_preprocessing import stopword_removal, tokenize
from typing import Dict, List


# TODO: maybe don't use default dict as each access will add a new entry


class BigramLanguageModel:
    """Bigram language model built on given corpus. Preprocessing done is lower case, remove special characters and stopword removal
    
    Attributes:
        None
    """

    def __init__(self, corpus: List[str]):
        # Bigram model represented as {'w1': {'list of w2': {'freq', 'proba}}}
        self._bigram_model = defaultdict(self._create_dd_w2)
        # Priors represented as {'term': {'freq': 'proba'}}
        self._priors = {}

        # Get tokens and bigrams for each document within corpus
        corpus_tokens = [tokenize(doc) for doc in corpus]
        corpus_tokens = [stopword_removal(doc_tokens) for doc_tokens in corpus_tokens]
        corpus_bigrams = [bigrams(doc_tokens) for doc_tokens in corpus_tokens]

        # Calculate bigram freq and probabilities
        for doc_bigrams in corpus_bigrams:
            self._calculate_bigram_freq(doc_bigrams)
        self._calculate_bigram_proba()

        # Calculate priors
        for doc_tokens in corpus_tokens:
            self._calculate_term_freq(doc_tokens)
        self._calculate_term_priors()

        return

    def _calculate_term_freq(self, doc_tokens: List[str]) -> None:
        """Calculates frequencies of each term
        
        Arguments:
            doc_tokens {List[str]} -- List of tokens from document
        """
        for token in doc_tokens:
            if token not in self._priors.keys():
                self._priors[token] = {"freq": 1, "proba": 0}
            else:
                self._priors[token]["freq"] += 1
        return

    def _calculate_term_priors(self) -> None:
        """Calculates prior probability for each individual word
        """
        total = 0
        for term, val in self._priors.items():
            total += val["freq"]
        for term in self._priors.keys():
            self._priors[term]["proba"] = self._priors[term]["freq"] / total
        return

    def _create_dd_w2(self) -> Dict:
        """Helper function to create default dict that can be pickled
        
        Returns:
            Dict -- Default dictionary with default values
        """
        return defaultdict(self._create_dd_val)

    def _create_dd_val(self) -> Dict:
        """Helper function to create default dict that can be pickled
        
        Returns:
            Dict -- Default values
        """
        return {"freq": 0, "proba": 0}

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

    def get_bigram_freq(self, w1: str, w2: str) -> int:
        """ Returns frequency of bigram from corpus
        
        Arguments:
            w1 {str} -- First word in bigram
            w2 {str} -- Second word in bigram
        
        Returns:
            int -- Number of counts of (w1, w2)
        """
        return self._bigram_model[w1][w2]["freq"]

    def get_posterior(self, w1: str, w2: str) -> float:
        """Returns posterior probability of P(w2|w1) from corpus
        
        Arguments:
            w1 {str} -- First word in bigram
            w2 {str} -- Second word in bigram
        
        Returns:
            float -- Probability value P(w2|w1)
        """
        return self._bigram_model[w1][w2]["proba"]

    def get_prior(self, word: str) -> float:
        """Gets prior probability P(word)
        
        Arguments:
            word {str} -- Term
        
        Returns:
            float -- Term prior probability
        """
        return self._priors[word]["proba"] if word in self._priors.keys() else 0

    def get_w2(self, w1: str) -> str:
        """Returns all words w2 that match (w1,w2) in bigram model
        
        Arguments:
            w1 {str} -- First word in bigram
        
        Returns:
            str -- Matching second words in bigram
        """
        return list(self._bigram_model[w1].keys())

    def calculate_seq_proba(
        self, sequence: List[str], smoothing: float = 1e-6
    ) -> float:
        """Calculates probability of sequence by computing product of posteriors
        
        Arguments:
            sequence {List[str]} -- List of words in given order
        
        Keyword Arguments:
            smoothing {float} -- Smoothing parameter in case of 0 probabilities (default: {1e-6})
        
        Returns:
            float -- Probability of sequence P(w_n|w_n-1) * .... * P(w_2|w_1) * P(w1)
        """

        current_word = sequence[0]
        proba = self.get_prior(current_word)

        for next_word in sequence[1:]:
            proba *= self.get_posterior(current_word, next_word) + smoothing
            current_word = next_word

        return proba
