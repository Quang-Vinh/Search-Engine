# Module 3 - Dictionary Building
# Purpose: Build a dictionary of terms to be indexed

import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def stopword_removal(words: list) -> list:
    '''
    Remove stopwords from descriptions
    '''
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words


def stem_tokens(words: list, stemmer) -> str:
    '''
    Stem all words in list
    '''
    words = [stemmer.stem(word) for word in words]
    return words


def normalize_token(word: str) -> str:
    '''
    Normalizes given word based on 
    a) Hyphens (low-cost becomes lowcost)
    b) Periods (U.S.A becomes USA) 
    '''

    word = word.replace('-', '')
    word = word.replace('.', '')
        
    return word


def tokenize(sent: str) -> list:
    '''
    Tokenizes a string into several words/tokens, just a wrapper function
    '''
    sent = sent.lower()
    sent = sent.replace("'", '')
    words = word_tokenize(sent)

    # Remove tokens that are only special characters, ex ')'
    words = [word for word in words if not re.match(r'^[^A-Za-z0-9]*$', word)]

    return words


def preprocess_sentence(sent: str, remove_stopword: bool = True, stem: bool = True, normalize: bool = True) -> list:
    '''
    Preprocesses a given sentence by tokenizing, removing stopwords, stemming, and normalizing.
    Returns a list of preprocessed tokens
    '''

    words = tokenize(sent)

    if remove_stopword:
        words = stopword_removal(words)
    
    if stem:
        stemmer = PorterStemmer()
        words = stem_tokens(words, stemmer)
    
    if normalize:
        words = [normalize_token(word) for word in words]

    return words




class Dictionary():
    '''
    Dictionary representation for the given corpus
    '''

    def __init__(self, corpus: list, remove_stopword: bool = True, stem: bool = True, normalize: bool = True):
        # Get list of all words from collection
        sentence_tokens = [preprocess_sentence(sentence, remove_stopword, stem, normalize) for sentence in corpus]

        # Flatten out list
        words = [word for sentence in sentence_tokens for word in sentence]

        # Remove duplicates
        self.words = list(set(words))
        self.words.sort()

        self.size = len(self.words)
        return