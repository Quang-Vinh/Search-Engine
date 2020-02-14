# Module 4 - Inverted Index Construction
# Purpose: Associate dictionary terms to documents

import pandas as pd
from collections import defaultdict
from dictionary import Dictionary, preprocess_sentence


class InvertedIndex():
    '''
    Inverted index data structure for given dictionary and corpus. 
    Contains for each term the set of docIDs it is found in and the weight
    '''

    def __init__(self, dictionary: Dictionary, corpus: list, docIDs: list):
        '''
        Index will be represented as a dictionary mapping every term to a list of postings. 
        The postings will be represented as tuples (docID, frequency)
        '''

        # Initialize index with empty lists for all words in dictionary
        self.index = dict()

        for term in dictionary.words:
            self.index[term] = []

        # Populate index with postings by iterating through each document
        for i in range(0, len(docIDs)):
            tokens = preprocess_sentence(corpus[i])
            frequencies = self._count_frequencies(tokens)

            for term, frequency in frequencies.items():
                if term in dictionary.words:
                    self.index[term].append((docIDs[i], frequency))
                else:
                    print(term)
        
        return


    def _count_frequencies(self, tokens: list) -> dict:
        '''
        Count the frequencies of all tokens in the given list, returns a dictionary mapping term to frequency
        '''
        frequencies = defaultdict(lambda: 0)

        for token in tokens:
            frequencies[token] += 1

        return frequencies


    def get_postings(self, term: str) -> list:
        '''
        Return postings associated with given term
        '''
        postings = self.index[term]
        return postings