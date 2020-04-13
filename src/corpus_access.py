# Module 5 - Corpus Access
# Purpose: Access documents from the corpus


import pandas as pd


# Load preprocessed UofO courses dataframe
uo_courses_corpus = pd.read_csv(
    "../collections/processed/UofO_Courses.csv", index_col="docID"
)
reuters_corpus = pd.read_csv(
    "../collections/processed/reuters.csv", index_col="docID"
)
corpora = {"uo_courses": uo_courses_corpus, "reuters": reuters_corpus}


def get_corpus_texts(corpus: str, docIDs: list) -> pd.DataFrame:
    """Returns documents containing matching docIDs for given corpus in same order as input docIDs list
    
    Arguments:
        corpus {str} -- corpus from which to get documents from, options are either 'uo_courses' or 'reuters'
        docIDs {list} -- document docIDs to return
    
    Returns:
        pd.DataFrame -- Dataframe containing documents with given docIDs
    """
    return corpora[corpus].loc[docIDs]
