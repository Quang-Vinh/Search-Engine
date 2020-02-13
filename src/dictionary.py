# Module 3
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


def stemmatize_tokens(words: list, stemmer) -> str:
    '''
    Stemmatize all words in list
    '''
    words = [stemmer.stem(word) for word in words]
    return words


def normalize_tokens(word: str, remove_hyphen: bool = False, remove_periods: bool = False) -> str:
    '''
    Normalizes given word based on 
    a) Hyphens (low-cost becomes lowcost)
    b) Periods (U.S.A becomes USA) 
    '''
    if remove_hyphen:
        word = word.replace('-', '')

    if remove_periods:
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


def build_dictionary(courses_descriptions: pd.Series, remove_stopword: bool = True, stemmatize: bool = True, normalize: bool = True):
    '''
    Creates dictionary of terms to be indexed
    '''

    # Get list of all words from collection
    descriptions = courses_descriptions.apply(tokenize)
    words = [word for description in descriptions for word in description]

    if remove_stopword:
        words = stopword_removal(words)
    
    if stemmatize:
        stemmer = PorterStemmer()
        words = stemmatize_tokens(words, stemmer) 

    if normalize:
        words = [normalize_tokens(word, True, True) for word in words]

    # Remove duplicates
    words = list(set(words))
    words.sort()

    return words