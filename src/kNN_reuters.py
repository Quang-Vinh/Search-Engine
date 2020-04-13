# Module 6 - Text categorization with kNN
# Purpose: Assign one or more topics to the Reuters documents that are not assigned any topics

from dictionary import Dictionary
from inverted_index import InvertedIndex
from vector_space_model import VectorSpaceModel

from collections import Counter
from typing import List


class kNN_reuters():
    '''KNN implementation for Reuters collections for Topic prediction
    '''

    def __init__(self, k: int = 5):
        self.k = 5
        return 

    def fit(self, X, Y, docIDs):
        '''Preprocesses X text inputs and builds inverted index and vector space model on it to use for faster KNN calculations
        
        Arguments:
            X {[type]} -- Document body
            Y {[type]} -- Document topics
            docIDs {[type]} -- Document IDs corresponding to given X and Y
        '''
        self.X_train = X
        self.Y_train = Y
        self.docIDs_train = docIDs
        self.dictionary = Dictionary(X_train.to_list())
        self.index = InvertedIndex(self.dictionary, X.to_list(), docIDs.to_list())
        self.vsm = VectorSpaceModel(self.index)
        return

    def predict(self, X) -> List[str]:
        '''Predict topic of given document body using VSM. Topic is chosen as most common topics that make up at least 50% of topics in predictions
        
        Arguments:
            X {[type]} -- Document Body to predict topic
        
        Returns:
            [type] -- Predicted list of topics
        '''
        # Get docID of nearest neighbours
        nn = self.vsm.search(X, limit=self.k)

        # Create list of concatenation of all topics, including duplicates
        topics = []
        for docID in nn:
            index = self.docIDs_train[self.docIDs_train == docID].index[0]
            topics += self.Y_train.iloc[index]

        # Assign prediction as most common topics that make up at least 50% of the topic labels
        n = len(topics)
        total_prob = 0
        results = []
        topics = Counter(topics).most_common()
        for (topic, count) in topics:
            results.append(topic)
            total_prob += count / n 
            if total_prob > 0.5:
                break

        return results
