# Module 2 
# Purpose: Convert a collection of documents into a formatted corpus

import pandas as pd
import re
from bs4 import BeautifulSoup


def parse_html(file_path: str) -> pd.DataFrame:
    '''
    Reads the html file and returns a dataframe containing structured information
    Note: There are courses without any description, these will be ignored
    '''


    with open(file_path, 'r', encoding='utf-8') as infile:
        contents = infile.read()
        soup = BeautifulSoup(contents, 'html.parser')


    # Get course descriptions
    course_descriptions = soup.find_all('p', {'class': 'courseblockdesc'})

    # Get titles
    courses= [tag.find_previous('p', {'class': 'courseblocktitle'}).get_text() for tag in course_descriptions]

    # Get course id and name from titles
    course_faculties = [re.search('[A-z]{3}', course)[0] for course in courses]
    course_codes = [re.search('[0-9]{4}', course)[0] for course in courses]
    course_titles = [re.sub(
                        '[A-z]{3} [0-9]{4} ',
                        '',
                        re.sub(' \\(3.+\\)', '', course)) for course in courses]

    # Get course descriptions text
    course_descriptions = [course_description.get_text().replace('\n', '') for course_description in course_descriptions]

    
    courses_pd = pd.DataFrame({'faculty': course_faculties,
                              'code': course_codes, 
                              'title': course_titles,
                              'description': course_descriptions})

    return courses_pd



def filter_english(courses_df: str) -> pd.DataFrame:
    return


def preprocess(data_path: str, output_path: str) -> None:
    return













