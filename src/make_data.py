# Script to setup dictionary/indexes/models for UofO courses and reuters collection

from bigram_language_model import BigramLanguageModel
from dictionary import Dictionary
from inverted_index import InvertedIndex
from kNN_reuters import kNN_reuters
from preprocessing import preprocess_uo_courses, preprocess_reuters_all

import argparse
import os.path
from pathlib import Path
import pickle
from pytictoc import TicToc



if __name__ == "__main__":

    # Parse cmd arguments
    parser = argparse.ArgumentParser(description='Preprocess the data and create models')
    parser.add_argument('--knn', action='store_true')
    args = parser.parse_args()

    # Path to store preprocessed models and data
    file_path = os.path.abspath(os.path.dirname(__file__))
    uo_courses_file_path = os.path.join(
        file_path, "../collections/raw/UofO_Courses.html"
    )
    uo_courses_out_path = os.path.join(
        file_path, "../collections/processed/UofO_Courses.csv"
    )
    uo_courses_index_path = os.path.join(
        file_path, "../models/indexes/UofO_courses_index.pkl"
    )
    uo_bigram_model_path = os.path.join(
        file_path, "../models/bigram_language_models/UofO_bigram_model.pkl"
    )
    reuters_folder_path = os.path.join(file_path, "../collections/raw/reuters21578")
    reuters_out_path = os.path.join(file_path, "../collections/processed/reuters.csv")
    reuters_index_path = os.path.join(file_path, "../models/indexes/reuters_index.pkl")
    reuters_bigram_model_path = os.path.join(
        file_path, "../models/bigram_language_models/reuters_bigram_model.pkl"
    )
    reuters_topics_out_path = os.path.join(
        file_path, "../collections/processed/reuters_with_topics.csv"
    )

    # Create directories for models
    Path(os.path.join(file_path, "../models/bigram_language_models")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(file_path, "../models/indexes/")).mkdir(
        parents=True, exist_ok=True
    )

    tictoc = TicToc()
    t = TicToc()

    tictoc.tic()

    # Preprocess collections
    t.tic()
    print("Preprocessing UO courses html")
    courses = preprocess_uo_courses(uo_courses_file_path)
    courses.to_csv(uo_courses_out_path, index=False)
    print("Preprocessing Reuters collection")
    reuters_texts = preprocess_reuters_all(reuters_folder_path)
    reuters_texts.to_csv(reuters_out_path, index=False)
    t.toc()

    # Create dictionary
    t.tic()
    print("\nCreate dictionary UO courses")
    courses_dictionary = Dictionary(courses["body"].to_list())
    print("Create dictionary Reuters collection")
    reuters_dictionary = Dictionary(reuters_texts["body"].to_list())
    t.toc()

    # Create inverted index
    t.tic()
    print("\nCreate inverted UO courses")
    courses_index = InvertedIndex(
        courses_dictionary, courses["body"].to_list(), courses["docID"].to_list()
    )
    pickle.dump(courses_index, open(uo_courses_index_path, "wb"))

    print("Create inverted index Reuters collection")
    reuters_index = InvertedIndex(
        reuters_dictionary,
        reuters_texts["body"].to_list(),
        reuters_texts["docID"].to_list(),
    )
    pickle.dump(reuters_index, open(reuters_index_path, "wb"))
    t.toc()

    # Create bigram language models
    t.tic()
    print("\nCreate bigram language model for UO courses")
    uo_bigram_model = BigramLanguageModel(courses["body"].to_list())
    pickle.dump(uo_bigram_model, open(uo_bigram_model_path, "wb"))
    print("Create bigram language model Reuters collections")
    reuter_bigram_model = BigramLanguageModel(reuters_texts["body"].to_list())
    pickle.dump(reuter_bigram_model, open(reuters_bigram_model_path, "wb"))
    t.toc()

    # kNN predict topics

    if  args.knn:
        t.tic()
        print("\nPredict unknown topics for Reuters collections")

        # Split train and testing based on unknown topics
        train_index = reuters_texts["topics"].apply(len) > 0
        X = reuters_texts["body"]
        X_train = X.loc[train_index]
        X_test = X.loc[-train_index].reset_index(drop=True)
        Y = reuters_texts["topics"]
        Y_train = Y.loc[train_index]
        Y_test = Y.loc[-train_index].reset_index(drop=True)
        docIDs = reuters_texts["docID"]
        docIDs_train = docIDs.loc[train_index].reset_index(drop=True)

        # Use k=5, common
        knn = kNN_reuters(5)
        knn.fit(X_train, Y_train, docIDs_train)

        # Get predictions
        predicted_topics = [knn.predict(X) for X in X_test]
        reuters_texts.loc[~train_index, 'topics'] = predicted_topics
        reuters_texts.to_csv(reuters_topics_out_path, index=False)
        t.toc()

    print("\nCompleted preprocessing total time:")
    tictoc.toc()
    print()
