from numpy import dot
from numpy.linalg import norm
from typing import Callable
from string import punctuation            # Set of all punctuation
from nltk.corpus import stopwords           # Define the set of stopwords
from nltk import word_tokenize
stopwords = set(stopwords.words('english'))


def cosine(a, b):
    return 1 - (dot(a[0], b[0]) / (norm(a[0]) * norm(b[0])))


def simple_lemma(word: str):
    return word


def clean_sentence(sent: str, lemma_func: Callable[[str], str]=None,
                   keep_stop_words: bool=True) -> str:
    """
    Clean a sentence

    Tokenize the word and then lemmatize each individual word before
    rejoining it all together. Optionally removing stop words along the way

    Args:
        sent(str): Sentence to clean
        lemma_func(Callable[[str], str]): A function that takes in a word
                                          and outputs a word, Normally
                                          lemmatizer
        keep_stop_words(bool): Flag to keep the stop words in the sentence
    Rets:
        (str): Cleaned sentence
    """
    if lemma_func is None:
        lemma_func = simple_lemma
    # Lemmatize each word in the sentence and remove the stop words if
    # the flag is set
    return " ".join([lemma_func(token) for token in
                     word_tokenize(sent.lower()) if
                     (token not in stopwords or keep_stop_words) and
                     (token not in punctuation)])
