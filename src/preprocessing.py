# Module 2 - Corpus Preprocessing
# Purpose: Convert a collection of documents into a formatted corpus

import pandas as pd
import re
from bs4 import BeautifulSoup


# TODO: French parts of title still showing up


def parse_uo_courses(file_path: str) -> pd.DataFrame:
    """
    Reads the uocourses html file and returns a dataframe containing structured information
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
            "body": course_descriptions,
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


def preprocess_uo_courses(data_path: str) -> None:
    """
    Pipeline for preprocessing steps for html of uOttawa courses
    """
    df = parse_uo_courses(data_path)
    df = filter_english(df)
    return df


def _is_norm(text_tag) -> bool:
    """ Checks whether given text tag is of type 'NORM' or not, small helper function
    
    Arguments:
        text_tag {bs4.element.tag} -- <TEXT> tag from reuters collection, nested within <REUTERS> tag
    
    Returns:
        bool -- If the type of text tag is 'NORM' or not
    """
    try:
        return text_tag["type"] == "NORM"
    except:
        return True


def preprocess_reuters_file(file_path: str) -> pd.DataFrame:
    """ Parse a reuters .sgm file and returns a dataframe where each row is a text
    
    Arguments:
        file_path {str} -- path to reuters .sgm file
    
    Returns:
        [pd.DataFrame] -- Pandas dataframe containing parsed reuters texts
    """
    with open(file_path, "r", encoding="iso-8859-1") as infile:
        contents = infile.read()
        contents = contents.replace("<BODY>", "\n<TEXT_BODY>\n")
        contents = contents.replace("</BODY>", "\n</TEXT_BODY>\n")
        soup = BeautifulSoup(contents, "lxml")

    cols = ["title", "type", "author", "body"]
    reuters = soup.find_all("reuters")
    texts = [reuter.find("text") for reuter in reuters]

    # Keep only the texts that are of type "NORM", these ones are mainly only ones with <BODY> and <TITLE> tags
    index_keep = [_is_norm(text) for text in texts]
    n_texts = len(reuters)
    reuters = [reuters[i] for i in range(n_texts) if index_keep[i]]
    texts = [texts[i] for i in range(n_texts) if index_keep[i]]

    # Create text data
    text_data = [
        (
            text.title.text,
            text.type.text if text.type != None else None,
            text.author.text if text.author != None else None,
            text.text_body.text,
        )
        for text in texts
    ]
    text_df = pd.DataFrame(text_data, columns=cols)

    # Add reuters meta data
    IDs = [int(reuter["newid"]) for reuter in reuters]
    text_df["docID"] = IDs

    return text_df


def preprocess_reuters_all(folder_path: str) -> pd.DataFrame:
    """ Preprocesses all reuters .sgm files and returns a dataframe containing all texts
    
    Arguments:
        folder_path {str} -- Path to reuters21578 folder
    
    Returns:
        pd.DataFrame -- Dataframe containing all texts from reuters collection
    """
    reuters_size = 22
    text_dfs = []

    for i in range(reuters_size):
        print(f"Processing file {i} / {reuters_size}")
        text_dfs.append(
            preprocess_reuters_file(folder_path + "/reut2-{0:0=3d}.sgm".format(i))
        )
    text_dfs = pd.concat(text_dfs)

    return text_dfs
