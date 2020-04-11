import dictionary
import pickle


def get_inv_index():
    return pickle.load(open("../indexes/UofO_Courses_index.pkl", "rb"))


def get_bigram_index():
    # returns the bigram : words index
    words = get_inv_index().dictionary.words_raw
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
    for section in term_split:
        if section:
            term_bigrams = term_bigrams.union(get_bigrams(section, False))
    return term_bigrams


def get_bigrams(term, ends=True):
    term_bigrams = set()
    for i in range(len(term) - 1):
        term_bigrams.add(term[i : i + 2])
    if ends:
        term_bigrams.add("$" + term[0])
        term_bigrams.add(term[-1] + "$")
    return term_bigrams


def get_indexed_words(term):
    bigram_index = get_bigram_index()
    term_bigrams = get_wildcarded_bigrams(term)
    matched_words = bigram_index[next(iter(term_bigrams))]
    for i in term_bigrams:
        matched_words = matched_words.intersection(bigram_index[i])
    return matched_words
