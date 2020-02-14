# Module 4/8a - Inverted Index Construction with TF-IDF weights
# Purpose: Associate dictionary terms to documents

import numpy as np
import pandas as pd
from collections import defaultdict
from dictionary import Dictionary, preprocess_sentence

class InvertedIndex():
    '''
    Inverted index data structure for given dictionary and corpus. 
    Contains for each term the set of docIDs it is found in and the weight

    Index will be represented as a dictionary in form of {term: {docID: {freq, tf-idf} } }
    '''

    def __init__(self, dictionary: Dictionary, corpus: list, docIDs: list):
        # Set dictionary attribute
        self.dictionary = dictionary
        
        # Initialize index with empty lists for all words in dictionary
        self.index = dict()

        for term in dictionary.words:
            self.index[term] = dict()

        # Populate index with postings docID -> frequency by iterating through each document
        for i in range(0, len(docIDs)):
            tokens = preprocess_sentence(corpus[i])
            frequencies = self._count_frequencies(tokens)

            for term, frequency in frequencies.items():
                if term in dictionary.words:
                    self.index[term][docIDs[i]] = {'freq': frequency}
    
        # Calculate TF-IDF to get postings docID -> frequency, TF-IDF
        for term, postings in self.index.items():
            document_freq = len(postings)
            idf = np.log10(len(corpus) / document_freq)

            # Add TF-IDF weighting for all postings
            for docID, weight in postings.items():
                tf = np.log10(1 + weight['freq'])
                tf_idf = tf * idf
                self.index[term][docID]['tf-idf'] = tf_idf
        
        return


    def _count_frequencies(self, tokens: list) -> dict:
        '''
        Count the frequencies of all tokens in the given list, returns a dictionary mapping term to frequency
        '''
        frequencies = defaultdict(lambda: 0)

        for token in tokens:
            frequencies[token] += 1

        return frequencies


    def get_frequency(self, term: str, docID: str) -> float:
        '''
        Returns Frequency weight for given term and document ID 
        Returns None if either term or docID is not found
        '''
        if term not in self.index.keys():
            return None
        elif docID not in self.index[term].keys():
            return None

        freq = self.index[term][docID]['freq']
        return freq


    def get_tf_idf(self, term: str, docID: str) -> float:
        '''
        Returns TF-IDF weight for given term and document ID
        Returns None if either term or docID is not found
        '''
        if term not in self.index.keys():
            return None
        elif docID not in self.index[term].keys():
            return None
        
        tf_idf = self.index[term][docID]['tf-idf']
        return tf_idf


    def get_terms(self) -> list:
        '''
        Returns all dictionary terms 
        '''
        return self.dictionary.words

    
    def get_postings(self, term: str) -> list:
        '''
        Return postings (docIDs) stored for given term
        Returns None is term is not in dictionary
        '''
        if term not in self.index.keys():
            return None

        postings = list(self.index[term].keys())
        postings.sort()
        return postings
