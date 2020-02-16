# Module 5 - Corpus Access
# Purpose: Access documents from the corpus


import pandas as pd 


# Load preprocessed UofO courses dataframe
uo_courses_corpus = pd.read_csv('../collections/UofO_Courses_preprocessed.csv', index_col = 'docID')


def get_uo_courses(docIDs: list) -> pd.DataFrame:
    '''
    Returns dataframe containing matching docIDs for UofO courses in same order as input docIDs list
    '''
    return uo_courses_corpus.loc[docIDs]