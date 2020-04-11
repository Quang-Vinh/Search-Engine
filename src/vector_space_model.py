# Module 8 - Vector Space Model
# Purpose: Implementing the Vector Space Model for retrieval

# TODO: allow user defined weights
# TODO: use matrix operations to more efficiently calculate similariies between query and document

import numpy as np
import pandas as pd
from dictionary import Dictionary
from inverted_index import InvertedIndex


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
        if similarity not in ["inner-product", "cosine"]:
            print("Similarity not defined")
            return None

        # Preprocess query string
        query_tokens = set(self.dictionary.preprocess_document(query))

        # Calculate similarity for all documents
        if similarity == "inner-product":
            query_results = self._inner_product(query_tokens)
        elif similarity == "cosine":
            query_results = self._cosine_sim(query_tokens)

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

    def _inner_product(self, query_tokens: list) -> dict:
        """
        Calculates inner product between query and all documents. Assigns each query token a weight of 1
        Returns a dictionary {docID: inner_product}
        """
        # Initialize similarity for each docID
        similarities = dict()

        for docID in self.docIDs:
            similarities[docID] = 0

        # Calculate inner product for each docID
        for word in query_tokens:

            docIDs = self.index.get_postings(word)
            if docIDs == None:
                continue

            # Add weight to similarity for every document
            for docID in docIDs:
                weight = self.index.get_tf_idf(word, docID)
                similarities[docID] += 1 * weight

        return similarities

    def _cosine_sim(self, query_tokens: list) -> float:
        """
        Calculates cosine similarity between query and document. Assigns each query token a weight of 1
        """

        cosine_sim = self._inner_product(query_tokens)

        # Normalize each inner product by product of lengths of query and doc vectors
        for docID, inner_prod in cosine_sim.items():
            denom = np.sqrt(len(query_tokens)) * self._docID_length(docID)
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
