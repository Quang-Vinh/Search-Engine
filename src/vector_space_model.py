# Module 8 - Vector Space Model
# Purpose: Implementing the Vector Space Model for retrieval

# TODO: use matrix operations to more efficiently calculate similariies between query and document
# TODO: use numpy for vector representations 

import numpy as np
import pandas as pd
from dictionary import Dictionary
from inverted_index import InvertedIndex
from typing import List, Tuple


class VectorSpaceModel:
    """
    Vector space model for retrieval for given index
    Currently does not accept user defined weights for query terms
    """

    def __init__(self, index: InvertedIndex):
        # Set attributes
        self.index = index
        self.dictionary = index.dictionary
        self.docIDs = self.index.docIDs
        return

    def to_vector(self, query: str) -> List[Tuple[str, float]]:
        '''Converts query into vector with all weights 1 for use in search
        
        Arguments:
            query {str} -- Query string
        
        Returns:
            List[Tuple[str, float]] -- Sparse vector 
        '''
        # Preprocess query string
        query_tokens = set(self.dictionary.preprocess_document(query))

        # Convert to vector with weights of 1
        query_vector = [(query_token, 1) for query_token in query_tokens]
        return query_vector

    def search(
        self,
        query: str,
        similarity: str = "inner-product",
        limit: int = 10,
        include_similarities: bool = False,
    ) -> list:
        """
        Given query string searches through documents to find best matches and returns docIDs with best match, but will exclude documents with similarity of 0.
        Query weights for each term are set to 1
        Uses either "inner-product" or "cosine" for similarity 
        Returns a list of tuples (docID, similarity)
        """
        query_vector = self.to_vector(query)
        search_results = self.vector_search(query_vector, similarity=similarity, limit=limit, include_similarities=include_similarities)
        return search_results

    def vector_search(
        self,
        query_vector: List[Tuple[str, float]],
        similarity: str = "inner-product",
        limit: int = 10,
        include_similarities: bool = False,
    ) -> list:
        """
        Given query vector weight weights searches through documents to find best matches and returns docIDs with best match, but will exclude documents with similarity of 0.
        Query weights for each term are set to 1
        Uses either "inner-product" or "cosine" for similarity 
        Returns a list of tuples (docID, similarity)
        """
        if similarity not in ["inner-product", "cosine"]:
            print("Similarity not defined")
            return None

        # Calculate similarity for all documents
        if similarity == "inner-product":
            query_results = self._inner_product(query_vector)
        elif similarity == "cosine":
            query_results = self._cosine_sim(query_vector)

        # Sort docIDs in order of decreasing similarity values
        query_results = list(query_results.items())
        query_results.sort(key=lambda pair: pair[1], reverse=True)

        # Remove documents with 0 similarity
        query_results = [sim for sim in query_results if sim[1] != 0]

        # Limit search results by amount
        if limit < len(query_results):
            query_results = query_results[:limit]

        # If don't include similarity
        if not include_similarities:
            query_results = [docID for docID, _ in query_results]

        return query_results

    def _inner_product(self, query_vector: List[Tuple[str, float]]) -> dict:
        """
        Calculates inner product between query vector and all documents.
        Returns a dictionary {docID: inner_product}
        """
        # Initialize similarity for each docID
        similarities = dict()

        for docID in self.docIDs:
            similarities[docID] = 0

        # Calculate inner product for each docID
        for (word, word_weight) in query_vector:
            docIDs = self.index.get_postings(word)
            if docIDs == None:
                continue

            # Add weight to similarity for every document
            for docID in docIDs:
                doc_weight = self.index.get_tf_idf(word, docID)
                similarities[docID] += word_weight * doc_weight
        return similarities

    def _cosine_sim(self, query_vector: List[Tuple[str, float]]) -> float:
        """
        Calculates cosine similarity between query and document
        """

        cosine_sim = self._inner_product(query_vector)

        # Normalize each inner product by product of lengths of query and doc vectors
        for docID, inner_prod in cosine_sim.items():
            denom = np.sqrt(len(query_vector)) * self._docID_length(docID)
            cosine_sim[docID] = inner_prod / denom

        return cosine_sim

    def _docID_length(self, docID: str) -> float:
        """
        Calculates the length of the tf-idf vector for given docID 
        """
        length = 0
        terms = self.index.get_docID_terms(docID)

        for term in terms:
            length += np.square(self.index.get_tf_idf(term, docID))

        length = np.sqrt(length)

        return length
