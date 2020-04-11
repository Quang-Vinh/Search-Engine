# Module 2 - Corpus Preprocessing
# Purpose: Convert a collection of documents into a formatted corpus

import pandas as pd
import re
from bs4 import BeautifulSoup


# TODO: French parts of title still showing up


def parse_html(file_path: str) -> pd.DataFrame:
    """
    Reads the html file and returns a dataframe containing structured information
    Note: There are courses without any description, these will be ignored
    """

    with open(file_path, "r", encoding="utf-8") as infile:
        contents = infile.read()
        soup = BeautifulSoup(contents, "html.parser")

    # Get course descriptions
    course_descriptions = soup.find_all("p", {"class": "courseblockdesc"})

    # Get titles
    courses = [
        tag.find_previous("p", {"class": "courseblocktitle"}).get_text()
        for tag in course_descriptions
    ]

    # Get course id and name from titles
    course_faculties = [re.search("[A-z]{3}", course)[0] for course in courses]
    course_codes = [re.search("[0-9]{4,5}", course)[0] for course in courses]
    course_titles = [re.sub("[A-z]{3} [0-9]{4} ", "", course) for course in courses]

    # Get course descriptions text
    course_descriptions = [
        course_description.get_text().replace("\n", "")
        for course_description in course_descriptions
    ]

    courses_df = pd.DataFrame(
        {
            "faculty": course_faculties,
            "code": course_codes,
            "title": course_titles,
            "description": course_descriptions,
        }
    )

    # Add docIDs
    courses_df["docID"] = courses_df["faculty"] + courses_df["code"]

    # Remove duplicates
    courses_df.drop_duplicates(subset="docID", inplace=True)

    return courses_df


def filter_english(courses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of courses, keeps only the english courses based on the second digit of the course code
    """
    english_courses = courses_df["code"].apply(lambda code: int(code[1]) < 5)
    return courses_df.loc[english_courses]


def preprocess(data_path: str, output_path: str) -> None:
    """
    Pipeline for preprocessing steps for html of uOttawa courses
    """
    df = parse_html(data_path)
    df = filter_english(df)

    df.to_csv(output_path, index=False)
    return df
