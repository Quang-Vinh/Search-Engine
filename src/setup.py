# Script to setup dictionary/indexes/models for UofO courses


from preprocessing import preprocess
from dictionary import Dictionary
from inverted_index import InvertedIndex, save_index

import pickle


if __name__ == '__main__':
    uo_courses_file_path = '../collections/UofO_Courses.html'
    uo_courses_out_path = '../collections/UofO_Courses_preprocessed.csv'
    uo_courses_index_path = '../indexes/UofO_courses_index.pkl'

    # Preprocess courses html
    courses = preprocess(uo_courses_file_path, uo_courses_out_path)

    # Create dictionary
    courses_dictionary = Dictionary(courses['description'].to_list())

    # Create inverted index
    courses_index = InvertedIndex(courses_dictionary, courses['description'].to_list(), courses['docID'].to_list())

    # Save index
    save_index(courses_index, uo_courses_index_path)