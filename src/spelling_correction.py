# Module 9 - Spelling correction with weighted edit distance
# Purpose: Provide suggestions of corrected words to the user


import itertools
import numpy as np
import os.path
import pandas as pd
import re


# TODO: maybe add weighted costs for insertion and deletion?


# Matrix for weighted edit distance. Defined as the 1 + (1 / (substitution error + 1))
# Cost of subbing letter1 for letter2 is given by sub_costs[letter2][letter1]
file_path = os.path.abspath(os.path.dirname(__file__))
sub_costs_matrix = pd.read_csv(
    os.path.join(file_path, "../data/sub_errors.csv"), index_col="X"
)
sub_costs_matrix = 1 + (1 / (sub_costs_matrix + 1))


def weighted_edit_distance(word: str, word_target: str) -> float:
    """
    Calculates weighted edit distance between start word and target word. 
    Insertion and deletion have cost 1. Substitution costs are based on given substitution error matrix
    Reference: CSI4107-TolerantRetrieval slides by Caroline Barrière
    """

    # Initialize n * m table
    n, m = len(word), len(word_target)
    opt = np.zeros(shape=[n + 1, m + 1])

    # Set opt(i,0) = i and opt(0,j) = j
    opt[:, 0] = np.arange(0, n + 1)
    opt[0, :] = np.arange(0, m + 1)

    # Fill out table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            insert_cost = opt[i - 1, j] + 1
            del_cost = opt[i, j - 1] + 1
            sub_cost = opt[i - 1, j - 1] + (
                sub_costs_matrix[word_target[j - 1]][word[i - 1]]
                if word[i - 1] != word_target[j - 1]
                else 0
            )

            # Set cost at opt(i, j) as minimum of three costs
            opt[i, j] = min(insert_cost, del_cost, sub_cost)

    # Cost is opt(N,M)
    cost = opt[n, m]

    return cost


class SpellingCorrector:
    """
    Spelling corrector for a given lexicon using weighted edit distance to determine most likely words
    """

    def __init__(self, lexicon: list):
        # Set attributes
        self.lexicon = set([self._preprocess_string(word) for word in lexicon])

        # Remove empty string if it exists
        if "" in self.lexicon:
            self.lexicon.remove("")

        return

    def check_query(
        self, query: str, limit: int = 10, include_costs: bool = False
    ) -> list:
        """
        Given query, use weighted edit distance to return list of alternate suggestions in order of most to least rank in form of (suggestedQuery, totalCost)
        If all words in the query are in the lexicon then return empty list instead
        """

        # If empty return original query
        if query == "":
            return []

        # Remove all special characters except for () and _ * - for boolean queries
        query = re.sub(r"[^a-zA-Z\(\)_\*] ", "", query)

        words = query.split(" ")

        # Preprocess all words
        words = [self._preprocess_string(word) for word in words]

        # Check if words are all in lexicon
        if set([word in self.lexicon for word in words]) == {True}:
            return []

        # Get closest words in lexicon based on edit distance for all words in the query
        word_suggestions = [self.check_word(word) for word in words]

        # Get all combinations of suggestions - https://stackoverflow.com/questions/12935194/combinations-between-two-lists
        suggested_queries = list(itertools.product(*word_suggestions))
        suggested_queries = [
            self._join_words(query, word_costs) for word_costs in suggested_queries
        ]

        # Remove queries which are same as input query - due to current handling of * TODO: fix
        suggested_queries = [
            (word, cost)
            for word, cost in suggested_queries
            if word.lower() != query.lower()
        ]

        # Sort on total cost
        suggested_queries.sort(key=lambda pair: pair[1])

        # Limit results
        if len(suggested_queries) > limit:
            suggested_queries = suggested_queries[:limit]

        # If don't include costs
        if not include_costs:
            suggested_queries = [query for query, _ in suggested_queries]

        return suggested_queries

    def _join_words(self, query: str, word_costs: list) -> tuple:
        """
        Helper function for self.check_query. Given a list of tuples (word, cost) concatenate all words and sum all costs to return (query, totalCost)
        Works also for boolean queries with AND/OR/AND_NOT and () brackets and wildcards *
        """

        # Get new query
        query = query.split(" ")

        for i in range(len(query)):
            # If there's a wildcard, just skip
            if "*" in query[i]:
                continue

            # Keep brackets, replace only the words
            if query[i].lower() in {"and", "or", "and_not"}:
                continue
            else:
                query[i] = re.sub("[a-zA-Z]+", word_costs[i][0], query[i])

        query = " ".join(query)

        # Get cost
        costs = [cost for _, cost in word_costs]
        total_cost = sum(costs)

        return (query, total_cost)

    def check_word(
        self, word: str, same_first_letter: bool = True, limit: int = 10
    ) -> list:
        """
        Given a word, if the word is not in the lexicon then will calculate weighted edit distance between word and all other words
        in the lexicon and return a list of most likely to least likely word replacements in form of (word, weightedEditDistance). 
        Likelihood will be based on the weighted edit distance.
        If the word is in the lexicon then return (word, 0) 
        """

        if word in self.lexicon:
            return [(word, 0)]

        # Preprocess word before comparing with lexicon
        word = self._preprocess_string(word)

        # Compare with only words with same starting letter
        if same_first_letter:
            lexicon = [w for w in self.lexicon if w[0] == word[0]]
        else:
            lexicon = list(self.lexicon)

        # Calculate weighted edit distance between word and all other words
        edit_distances = [(w, weighted_edit_distance(word, w)) for w in lexicon]

        # Sort by most likely to least likely
        edit_distances.sort(key=lambda pair: pair[1])

        # Limit results
        if len(edit_distances) > limit:
            edit_distances = edit_distances[:limit]

        return edit_distances

    def _preprocess_string(self, word: str) -> str:
        """
        Normalize words for spelling correction. 
        Keeps only letters and converts to lower case
        """
        word = re.sub(r"[^a-zA-Z]", "", word)
        word = word.lower()

        return word
