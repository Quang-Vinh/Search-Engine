# Module 5 - Local Query Expansion with Rocchio Algorithm
# Purpose: Perform implicit query expansion using the relevance provided, within the VSM, using the Rocchio Algorithm


# TODO: remove to_full_vector and move it to vector space model


from inverted_index import InvertedIndex
import numpy as np
from typing import List, Tuple


class Rocchio:
    def __init__(
        self,
        index: InvertedIndex,
        alpha: float = 0.8,
        beta: float = 0.3,
        gamma: float = 0.1,
    ):
        self.dictionary = index.dictionary
        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        return

    def update_query_vector(
        self,
        query_vector: List[Tuple[str, float]],
        relevant_doc_ids: set,
        non_relevant_doc_ids: set,
    ) -> List[Tuple[str, float]]:
        """Updates query vector using Rocchio algorithm based on relevant documents
        
        Arguments:
            query_vector {List[Tuple[str, float]]} -- Sparse query vector
            relevant_doc_ids {list} -- Relevant document ids
            non_relevant_doc_ids {list} -- Non relevant document ids
        
        Returns:
            List[Tuple[str, float]] -- Updated sparse query vector
        """
        non_relevant_doc_ids = list(non_relevant_doc_ids)
        relevant_doc_ids = list(relevant_doc_ids)
        nr_len = len(non_relevant_doc_ids)
        r_len = len(relevant_doc_ids)

        # If no relevance information then do nothing
        if r_len == 0 and nr_len == 0:
            return query_vector

        # Get vector form of documents and queries
        query_vector = self._to_full_vector(query_vector)
        relevant_vectors = np.array(
            [self.index.get_docID_vector(doc_id) for doc_id in relevant_doc_ids]
        )
        non_relevant_vectors = np.array(
            [self.index.get_docID_vector(doc_id) for doc_id in non_relevant_doc_ids]
        )

        # Rocchio algorithm for update
        relevant_vector = relevant_vectors.sum(axis=0)
        non_relevant_vector = non_relevant_vectors.sum(axis=0)
        updated_query_vector = query_vector
        if r_len > 0:
            updated_query_vector = (
                query_vector + self.beta * (1 / r_len) * relevant_vector
            )
        if nr_len > 0:
            updated_query_vector = (
                updated_query_vector - self.gamma * (1 / nr_len) * non_relevant_vector
            )

        # Convert back to sparse vector
        terms = list(self.index.get_terms())
        terms.sort()
        updated_query_vector = [
            (term, weight)
            for (term, weight) in zip(terms, updated_query_vector)
            if weight > 0
        ]

        return updated_query_vector

    def _to_full_vector(self, query_vector: List[Tuple[str, float]]) -> np.array:
        """Converts the sparse vector of (term, weights) into a full vector where the dimensions are the sorted terms
        
        Arguments:
            query_vector {List[Tuple[str, float]]} -- Query in form (term, weights)
        
        Returns:
            np.array -- Full query vector
        """
        terms = list(self.index.get_terms())
        terms.sort()
        vector = np.zeros(len(terms))

        for (term, weight) in query_vector:
            index = terms.index(term)
            vector[index] = weight

        return vector
