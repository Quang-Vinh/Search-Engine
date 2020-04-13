# Module 5 - Corpus Access
# Purpose: Access documents from the corpus


import os.path
import pandas as pd


# Load preprocessed corpus dataframes
file_path = os.path.abspath(os.path.dirname(__file__))

uo_courses_corpus = pd.read_csv(
    os.path.join(file_path, "../collections/processed/UofO_Courses.csv"),
    index_col="docID",
)

reuters_corpus = pd.read_csv(
    os.path.join(file_path, "../collections/processed/reuters.csv"), index_col="docID"
)
# Convert topics column from string to list
topics_list = []
for topics in reuters_corpus.topics:
    topics = topics[1:-1].replace("'", "").replace(",", "").split(" ")
    topics = [] if topics == [""] else topics
    topics_list.append(topics)
reuters_corpus.topics = topics_list


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
