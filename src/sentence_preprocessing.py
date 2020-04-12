import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def stopword_removal(words: list) -> list:
    """
    Remove stopwords from descriptions
    """
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return words


def stem_tokens(words: list, stemmer) -> str:
    """
    Stem all words in list
    """
    words = [stemmer.stem(word) for word in words]
    return words


def normalize_token(word: str) -> str:
    """
    Normalizes given word based on 
    a) Hyphens (low-cost becomes lowcost)
    b) Periods (U.S.A becomes USA) 
    """

    word = word.replace("-", "")
    word = word.replace(".", "")

    return word


def tokenize(doc: str) -> list:
    """
    Tokenizes a string into several words/tokens, just a wrapper function
    """
    doc = doc.lower()
    doc = doc.replace("'", "")
    words = word_tokenize(doc)

    # Remove tokens that are only special characters or numbers, ex ')'
    words = [word for word in words if not re.match(r"^[^A-Za-z]*$", word)]

    return words