# Script to setup dictionary/indexes/models for UofO courses


from src.preprocessing import preprocess_uo_courses, preprocess_reuters_all
from src.dictionary import Dictionary
from src.inverted_index import InvertedIndex, save_index

import pickle
from pytictoc import TicToc


if __name__ == "__main__":
    uo_courses_file_path = "../collections/raw/UofO_Courses.html"
    uo_courses_out_path = "../collections/processed/UofO_Courses_preprocessed.csv"
    uo_courses_index_path = "../indexes/UofO_courses_index.pkl"
    reuters_folder_path = "../collections/raw/reuters21578"
    reuters_out_path = "../collections/processed/reuters_preprocessed.csv"
    reuters_index_path = "../indexes/reuters_index.pkl"

    t = TicToc()
    t.tic()

    # Preprocess courses html
    print("Preprocessing UO courses html")
    courses = preprocess_uo_courses(uo_courses_file_path)
    courses.to_csv(uo_courses_out_path, index=False)

    # Preprocess reuters colelction
    print("Preprocessing Reuters collection")
    reuters_texts = preprocess_reuters_all(reuters_folder_path)
    reuters_texts.to_csv(reuters_out_path, index=False)

    # Create dictionary
    print("Create dictionary UO courses")
    courses_dictionary = Dictionary(courses["description"].to_list())
    print("Create dictionary Reuters collection")
    reuters_dictionary = Dictionary(reuters_texts["body"].to_list())

    # Create inverted index
    print("Create inverted UO courses")
    courses_index = InvertedIndex(
        courses_dictionary, courses["description"].to_list(), courses["docID"].to_list()
    )
    save_index(courses_index, uo_courses_index_path)

    print("Create inverted index Reuters collection")
    reuters_index = InvertedIndex(
        reuters_dictionary,
        reuters_texts["body"].to_list(),
        reuters_texts["docID"].to_list(),
    )
    save_index(reuters_index, reuters_index_path)

    t.toc()
