from abc import ABC, abstractmethod
from typing import Callable, Tuple, List
from string import punctuation            # Set of all punctuation
from nltk.corpus import stopwords           # Define the set of stopwords
from nltk import word_tokenize
from functools import lru_cache           # Annotation for storing func results
from utils import cosine
import sent2vec
import spacy


stopwords = set(stopwords.words('english'))


class AbstractEmojiTranslator(ABC):
    def __init__(self, emoji_data_file: str, s2v_model_file: str,
                 lemma_func: Callable[[str], str]):
        self.emoji_file = emoji_data_file

        self.s2v = sent2vec.Sent2vecModel()
        print(s2v_model_file)
        self.s2v.load_model(s2v_model_file)

        self.lemma_func = lemma_func

        self.emoji_embeddings = self.generate_emoji_embeddings()
        self.nlp = spacy.load("en")

    def clean_n_gram(n_grams: List[str]) -> bool:
        """
        Validate that a given n_gram is good. Good is defined as the series
        of n-grams contains no n-grams containing only stop words
        """
        stopwords = "the and but".split()
        return list(filter(lambda x: x not in stopwords, n_grams))

    def clean_sentence(self, sent: str, lemma_func: Callable[[str], str]=None,
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
            lemma_func = self.lemma_func
        # Lemmatize each word in the sentence and remove the stop words if
        # the flag is set
        return " ".join([lemma_func(token) for token in
                         word_tokenize(sent.lower()) if
                         (token not in stopwords or keep_stop_words) and
                         (token not in punctuation)])

    @lru_cache(maxsize=1000)
    def closest_emoji(self, sent: str) -> Tuple[str, int, str]:
        """
        Get the closest emoji to the given sentence

        Loop through the list of emoji embeddings and keep track of which one
        has the lowest cosine distance from the input sentence's embedding.
        This is the "closest" emoji. The lru_cache designation means that
        python will store the last [maxsize] calls to this function with their
        return value to reduce computation. This is cleared after every call to
        the summary function.

        Args:
            sent(List[str]): Sentence to check
        Ret:
            (Tuple[str, int]) Closest emoji, cosine similarity of emoji

        """
        # Embed the sentence using sent2vec
        emb = self.s2v.embed_sentence(sent)

        # Start the lowest cosine at higher than it could ever be
        lowest_cos = 1_000_000

        # The best emoji starts as an empty string placeholder
        best_emoji = ""
        best_desc = ""

        # Loop through the dictionary
        for emoji in self.emoji_embeddings:
            # Get the current emoji's embedding
            emoji_emb = emoji[1]

            # Check the cosine difference between the emoji's embedding and
            # the sentence's embedding
            curr_cos = cosine(emoji_emb, emb)

            # If it lower than the lowest then it is the new best
            if curr_cos < lowest_cos:
                lowest_cos = curr_cos
                best_emoji = emoji[0]
                best_desc = emoji[2]

        # Return a 2-tuple containing the best emoji and its cosine differnece
        return best_emoji, lowest_cos, best_desc

    def generate_emoji_embeddings(self, lemma_func: Callable[[str], str]=None,
                                  keep_stop_words:
                                  bool=True) -> List[Tuple[str,
                                                           List[float], str]]:
        """
        Generate the sent2vec emoji embeddings from the input file

        Run each emoji within the data file through the sent2vec sentence
        embedder. This is a very naive way of doing it because one emoji may
        have multiple entries in the data file so it has multiple vectors in
        the emoji_embeddings array

        Args:
            lemma_func(Callable[[str], str]): Lemmatization function for
                                              cleaning. A function that takes
                                              in a word and outputs a word,
                                              normally lemmatization.
            keep_stop_words(bool): Keep the stop words in the cleaned sentence
        Rets:
            (List[Tuple[str, List[float]]]): A list of 2-tuples containing the
                                             emoji and its' embedding
        """
        if lemma_func is None:
            lemma_func = self.lemma_func

        # Initialize the list that will hold all of the embedings
        emoji_embeddings = []

        # Open the file that stores the emoji, description 2-tuple list
        with open(self.emoji_file) as emojis:
            for defn in emojis:
                # The file is tab-delim
                split = defn.split("\t")

                # Get the emoji and the description from the current line
                emoji = split[-1].replace("\n", "")
                desc = self.clean_sentence(split[0], lemma_func,
                                           keep_stop_words)

                # Add each emoji and embedded description to the list
                emoji_embeddings.append((emoji,
                                         self.s2v.embed_sentence(desc),
                                         desc))

        # Return the embeddings
        return emoji_embeddings

    def validate_n_gram(self, n_grams: List[str]) -> bool:
        """
        Validate that a given n_gram is good. Good is defined as the
        series of n-grams contains no n-grams containing only stop words
        """

        return not any([all(map(lambda x: x in stopwords,
                                [word for word in word_tokenize(n_gram)]))
                        for n_gram in n_grams])

    @abstractmethod
    def summarize(self, sent: str) -> str:
        pass
