# Module 5 - Corpus Access
# Purpose: Access documents from the corpus


import pandas as pd 


# Load preprocessed UofO courses dataframe
uo_courses_corpus = pd.read_csv('../collections/UofO_Courses_preprocessed.csv')


def get_uo_courses(docIDs: list) -> pd.DataFrame:
    '''
    Returns dataframe containing matching docIDs for UofO courses
    '''
    return uo_courses_corpus.query('docID in @docIDs')