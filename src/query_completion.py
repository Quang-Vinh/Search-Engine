# Module 2 - Query Completion Module
# Purpose: Provide suggestions as a list of possible completions to a query

# TODO: calculate_one_step prob can just be calculated using P(w_n|w_n-1) and doens't require whole sequence

from bigram_language_model import BigramLanguageModel
from sentence_preprocessing import tokenize
from typing import List, Tuple


class QueryCompleter:
    """Query completer that uses a bigram language model to find suggestions for completing given queries
    """

    def __init__(self, bigram_language_model: BigramLanguageModel):
        self.bigram_language_model = bigram_language_model
        return

    def complete_query(
        self, query: str, limit: int = 5, include_score: bool = False
    ) -> List[str]:
        """Finds suggestions for completing the query 
        
        Arguments:
            query {str} -- Unfinished query to complete

        Keyword Arguments:
            limit {int} -- Top amount of queries to return (default: {5})
        
        Returns:
            List[str] -- List of suggested queries in form of (sentence, probability)
        """
        if query == "":
            return []

        query_tokens = tokenize(query)

        suggestions = self.calculate_one_step_prob(query_tokens, limit=limit)
        suggestions = [
            (" ".join(sent_tokens), proba) for (sent_tokens, proba) in suggestions
        ]

        if not include_score:
            suggestions = [query for (query, _) in suggestions]

        return suggestions

    def calculate_one_step_prob(
        self, sent_tokens: List[str], limit: int = 5
    ) -> List[Tuple[List[str], float]]:
        """Calculates the one word predictions for next word in the sentence
        
        Arguments:
            sent_tokens {List[str]} -- List of tokens to find most next probable word
        
        Keyword Arguments:
            limit {int} -- Top amount of queries to return (default: {5})
        
        Returns:
            List[Tuple[List[str], float]] -- List of most likely one step query completions in form of (sentence, probability)
        """
        last_word = sent_tokens[-1]

        # Get possible next words from last word in sentence
        next_words = self.bigram_language_model.get_w2(last_word)

        if len(next_words) == 0:
            return []

        # Get probabilities of adding each next word P(next word | words before) = P(next_word | word at n-1) * proba
        results = [
            (
                sent_tokens + [next_word],
                self.bigram_language_model.calculate_seq_proba(
                    sent_tokens + [next_word]
                ),
            )
            for next_word in next_words
        ]

        # Sort by highest to lowest prob
        results.sort(key=lambda tup: tup[1], reverse=True)

        # Limit results
        results = results[:limit]

        return results
