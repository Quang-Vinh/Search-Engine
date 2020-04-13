import dictionary
import pickle


def get_inv_index(index_file):
    return pickle.load(open("../models/indexes/{}".format(index_file), "rb"))


def get_bigram_index(index):
    # returns the bigram : words index
    words = index.dictionary.words_raw
    bigram_index = {}
    for word in words:
        if "$" + word[0] not in bigram_index:
            bigram_index["$" + word[0]] = set()
        bigram_index["$" + word[0]].add(word)
        if word[-1] + "$" not in bigram_index:
            bigram_index[word[-1] + "$"] = set()
        bigram_index[word[-1] + "$"].add(word)
        for i in range(len(word) - 1):
            if word[i : i + 2] not in bigram_index:
                bigram_index[word[i : i + 2]] = set()
            bigram_index[word[i : i + 2]].add(word)
    return bigram_index


def get_wildcarded_bigrams(term):
    term_split = term.split("*")
    term_bigrams = set()
    for i in range(len(term_split)):
        endl = False
        endr = False
        if term_split[i]:
            if i == 0 or (i - 1 == 0 and not term_split[0]):
                endl = True
            if i == len(term_split) - 1 or (
                i - 1 == len(term_split) - 1 and not term_split[-1]
            ):
                endr = True
            term_bigrams = term_bigrams.union(get_bigrams(term_split[i], endl, endr))
    return term_bigrams


def get_bigrams(term, endl=True, endr=True):
    term_bigrams = set()
    for i in range(len(term) - 1):
        term_bigrams.add(term[i : i + 2])
    if endl:
        term_bigrams.add("$" + term[0])
    if endr:
        term_bigrams.add(term[-1] + "$")
    return term_bigrams


def get_indexed_words(term, index_file):
    bigram_index = get_bigram_index(index_file)
    term_bigrams = get_wildcarded_bigrams(term)
    matched_words = bigram_index[next(iter(term_bigrams))]
    for i in term_bigrams:
        matched_words = matched_words.intersection(bigram_index[i])
    return matched_words
