# Module 3 - Dictionary Building
# Purpose: Build a dictionary of terms to be indexed

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# TODO: Optimize preprocessing


def stopword_removal(words: list) -> list:
    """
    Remove stopwords from descriptions
    """
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return words


def stem_tokens(words: list, stemmer) -> str:
    """
    Stem all words in list
    """
    words = [stemmer.stem(word) for word in words]
    return words


def normalize_token(word: str) -> str:
    """
    Normalizes given word based on 
    a) Hyphens (low-cost becomes lowcost)
    b) Periods (U.S.A becomes USA) 
    """

    word = word.replace("-", "")
    word = word.replace(".", "")

    return word


def tokenize(doc: str) -> list:
    """
    Tokenizes a string into several words/tokens, just a wrapper function
    """
    doc = doc.lower()
    doc = doc.replace("'", "")
    words = word_tokenize(doc)

    # Remove tokens that are only special characters or numbers, ex ')'
    words = [word for word in words if not re.match(r"^[^A-Za-z]*$", word)]

    return words


class Dictionary:
    """
    Dictionary representation for the given corpus
    """

    def __init__(
        self,
        corpus: list,
        remove_stopword: bool = True,
        stem: bool = True,
        normalize: bool = True,
    ):
        # Set attributes
        self.remove_stopword = remove_stopword
        self.stem = stem
        self.normalize = normalize

        # Keep raw unprocessed words from corpus
        words_raw_tokens = [tokenize(doc) for doc in corpus]
        self.words_raw = set([word for tokens in words_raw_tokens for word in tokens])

        # Get list of all words from collection
        doc_tokens = [
            self._preprocess_document(doc, remove_stopword, stem, normalize)
            for doc in corpus
        ]

        # Flatten out list
        words = [word for doc in doc_tokens for word in doc]

        # Remove duplicates
        self.words = set(words)

        self.size = len(self.words)
        return

    def contains(self, word: str) -> bool:
        """
        Returns whether given word is included in the dictionary
        """
        return word in self.words

    def preprocess_document(self, doc: str) -> list:
        """
        Preprocess document based on flag attributes
        """
        words = self._preprocess_document(
            doc, self.remove_stopword, self.stem, self.normalize
        )
        return words

    def _preprocess_document(
        self,
        doc: str,
        remove_stopword: bool = True,
        stem: bool = True,
        normalize: bool = True,
    ) -> list:
        """
        Preprocesses a given sentence by tokenizing, removing stopwords, stemming, and normalizing.
        Returns a list of preprocessed tokens
        """
        words = tokenize(doc)

        if remove_stopword:
            words = stopword_removal(words)

        if stem:
            stemmer = PorterStemmer()
            words = stem_tokens(words, stemmer)

        if normalize:
            words = [normalize_token(word) for word in words]

        return words
