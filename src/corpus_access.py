# Module 5 - Corpus Access
# Purpose: Access documents from the corpus


import os.path
import pandas as pd
from typing import List


def _convert_topic_to_array(topics_list: pd.Series) -> List[List[str]]:
    """Parse string column and returns a column of type list
    
    Arguments:
        topics_list {pd.Series} -- Topic column with string representation of array
    
    Returns:
        List[List[str]] -- List of topics per corpus
    """
    topics_list = []
    for topics in reuters_corpus.topics:
        topics = topics[1:-1].replace("'", "").replace(",", "").split(" ")
        topics = [] if topics == [""] else topics
        topics_list.append(topics)
    return topics_list


def get_corpus_texts(corpus: str, docIDs: list, topic: str = None) -> pd.DataFrame:
    """Returns documents containing matching docIDs for given corpus in same order as input docIDs list filtered by topic
    
    Arguments:
        corpus {str} -- corpus from which to get documents from, options are either 'uo_courses' or 'reuters'
        docIDs {list} -- document docIDs to return
        topic {str} -- Topic to filter results
    
    Returns:
        pd.DataFrame -- Dataframe containing documents with given docIDs and topic
    """
    results = corpora[corpus].loc[docIDs]

    # Get subset of texts of given topic
    if topic and corpus == "reuters":
        topic_filter = _contains_topic(results["topics"], topic)
        results = results.loc[topic_filter]

    return results


def _contains_topic(topics_list: List[str], topic: str) -> List[bool]:
    """Checks which indices in topics list contains the topic
    
    Arguments:
        topics_list {List[str]} -- Topics list to check from
        topic {str} -- Topic to check
    
    Returns:
        List[bool] -- Boolean index
    """
    contains = [topic in topics for topics in topics_list]
    return contains


# Load preprocessed corpus dataframes
file_path = os.path.abspath(os.path.dirname(__file__))

uo_courses_corpus = pd.read_csv(
    os.path.join(file_path, "../collections/processed/UofO_Courses.csv"),
    index_col="docID",
)

reuters_corpus = pd.read_csv(
    os.path.join(file_path, "../collections/processed/reuters_with_topics.csv"),
    index_col="docID",
)
reuters_corpus["topics"] = _convert_topic_to_array(reuters_corpus["topics"])

corpora = {"uo_courses": uo_courses_corpus, "reuters": reuters_corpus}

# Topics for reuters collection
reuters_topics = pd.read_table(
    os.path.join(
        file_path, "../collections/raw/reuters21578/all-topics-strings.lc.txt"
    ),
    header=None,
)
reuters_topics = reuters_topics[0].values
