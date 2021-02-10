# Standard library
from typing import List, Tuple, Callable  # Datatypes for the function typing
from abc import ABC, abstractmethod       # Abstract class helpers
from string import punctuation            # Set of all punctuation
from functools import lru_cache           # Annotation for storing func results
from time import time                     # For timing

# Import numpy for some helper functions
import numpy as np

# NLTK
from nltk import word_tokenize     # Tokenizing a sentence into words
from nltk import Tree              # Tree like data structure for grammar trees
from nltk.corpus import stopwords  # Define the set of stopwords

# Sent2vec sentence embedding class
import sent2vec

# Spacy
import spacy  # For creating the grammar tree for the POS tagging translator

# Import the result class to hold the emoji results
from .EmojiSummarizationResult import EmojiSummarizationResult

# Get the cosine function from the utilities class. Would use the one
# from scipy but it won't install on my machine
from .utils import cosine

# Ignore simple warnings
import warnings
warnings.simplefilter('ignore')
stopwords = set(stopwords.words('english'))


class AbstractEmojiTranslator(ABC):

    """Abstract class for all Emoji Translation classes to inherit
    from. This includes an abstract summarize method that MUST
    be overridden

    Attributes:
        emoji_embeddings (List[str, List[float], str]): List of embedded
                                                        emoji descriptions
        emoji_file (str): Location of the emoji data file
        lemma_func (Callable[[str], str]): Lemmatization function
        nlp (Spacy.lang): Spacy NLP object
        s2v (Sent2vecModel): Sent2Vec model
        s2v_file (str): Sent2vec model file location
    """

    # Sent2Vec model singleton class level datamembers
    s2v = None
    s2v_file = ""

    @staticmethod
    def get_s2v(model_file: str) -> sent2vec.Sent2vecModel:
        """Instantiate a s2v model if one doesn't already exist.
        If one does exist then return the pre-existing model

        Args:
            model_file (str): File for the model data

        Returns:
            sent2vec.Sent2vecModel: Sent2vec model
        """
        # Check if a model is already instantiated
        if AbstractEmojiTranslator.s2v is None:
            # If not then create one
            AbstractEmojiTranslator.s2v = sent2vec.Sent2vecModel()
            AbstractEmojiTranslator.s2v.load_model(model_file)
            AbstractEmojiTranslator.s2v_file = model_file

        # Return the current static model
        return AbstractEmojiTranslator.s2v

    def __init__(self, emoji_data_file: str, s2v_model_file: str,
                 lemma_func: Callable[[str], str]):
        self.emoji_file = emoji_data_file

        self.s2v = AbstractEmojiTranslator.get_s2v(s2v_model_file)

        self.lemma_func = lemma_func

        self.emoji_embeddings = self.generate_emoji_embeddings()
        self.nlp = spacy.load("en_core_web_sm")

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


class ExhaustiveChunkingTranslation(AbstractEmojiTranslator):

    def combinations_of_sent(self, sent: str) -> List[List[str]]:
        """
        Return all possible n-gram combinations of a sentence

        Args:
            sent(str): Sentence to n-gram-ify
        Rets:
            (List[List[str]]): List of all possible n-gram combinations
        """

        def combinations_of_sum(sum_to: int,
                                combo: List[int]=None) -> List[List[int]]:
            """
            Return all possible combinations of ints that sum to some int

            Args:
                sum_to(int): The number that all sub-arrays should sum to
                combo(List[int]): The current combination of number that the
                                  recursivealgo should subdivide, not needed
                                  for first run but used in every consequent
                                  recursive run of the function
            """
            # Initialize the list for combinations
            combos = []

            # If the current combo list is none (first run through)
            # then generate it with all 1s and length = sum_to
            if combo is None:
                combo = [1 for x in range(sum_to)]
                combos.append(combo)

            # Base case: If the length  of the combination is 0 then
            # end the recursion because we are at the top of the "tree"
            if len(combo) == 0:
                return None

            # For each
            for i in range(1, len(combo)):
                combo_to_query = combo[:i - 1] + \
                    [sum(combo[i - 1:i + 1])] + combo[i + 1:]
                combos.append(combo_to_query)
                [combos.append(combo) for combo in combinations_of_sum(
                    sum_to, combo_to_query) if combo is not None]

            return combos

        def combinations_of_sent_helper(sent):
            sent = word_tokenize(sent)
            combos = np.unique(combinations_of_sum(len(sent)))
            sent_combos = []
            for combo in combos:
                sent_combo = []
                curr_i = 0
                for combo_len in combo:
                    space_joined = " ".join(sent[curr_i:combo_len + curr_i])
                    if space_joined not in sent_combo:
                        sent_combo.append(space_joined)
                    curr_i += combo_len

                if (sent_combo not in sent_combos and
                        self.validate_n_gram(sent_combo)):
                    sent_combos.append(sent_combo)
            return sent_combos

        return combinations_of_sent_helper(sent)

    def score_summarization_result_average(self, summarization:
                                           EmojiSummarizationResult) -> float:
        """
        Score a EmojiSummarizationResult

        Get the average of all uncertainty scores and return that as the score

        Args:
            summarization(EmojiSummarizationResult): Summarization to score

        Rets:
            (float): Numerical summarization score
        """
        return (sum(summarization.uncertainty_scores) /
                len(summarization.uncertainty_scores))

    def summarize(self, sent: str, lemma_func: Callable[[str], str]=None,
                  keep_stop_words: bool=True, scoring_func:
                  Callable[[EmojiSummarizationResult],
                           float]=None) -> EmojiSummarizationResult:
        """
        Summarize the given sentence into emojis

        Split the sentence into every possible combination of n-grams and
        see which returns the highest score when each n-gram is translated to
        an emoji using the closest emoji in the dataset

        Args:
            sent(str): Sentence to summarize
            lemma_func(Callable[[str], str]): Lemmatization function for c
                                              leaning. A function that takes in
                                              a word and outputs a word,
                                              normally lemmatization
            keep_stop_words(bool): Keep the stop words in the cleaned sentence
        Rets:
            (Tuple[List[str], List[float], List[str]]): (Emoji Sentence,
            List of Uncertainty values for the corresponding emoji,
            list of n-grams used to generate the corresponding emoji)
        """
        if lemma_func is None:
            lemma_func = self.lemma_func

        if scoring_func is None:
            scoring_func = self.score_summarization_result_average

        # Clean the sentence
        sent = self.clean_sentence(sent, lemma_func=lemma_func,
                                   keep_stop_words=keep_stop_words)

        # Generate all combinations of sentences
        sent_combos = self.combinations_of_sent(sent)
        # Init "best" datamembers as empty or exceedingly high
        best_summarization = EmojiSummarizationResult()
        best_summarization_score = 100_000_000
        # Iterate through every combination of sentence combos
        for sent_combo in sent_combos:
            # Start the local data members as empty
            local_summarization = EmojiSummarizationResult()
            # Iterate through each n_gram adding the uncertainty and
            # emoji to the lists
            for n_gram in sent_combo:
                close_emoji, cos_diff, close_desc = self.closest_emoji(n_gram)
                local_summarization.emojis += close_emoji
                local_summarization.emojis_n_grams.append(close_desc)
                local_summarization.uncertainty_scores.append(cos_diff)

            local_summarization.n_grams = sent_combo

            # Check if the average uncertainty is less than the best
            # TODO: Maybe a median check would be helpful as well?
            if scoring_func(local_summarization) < best_summarization_score:
                # Update the best emojis
                best_summarization = local_summarization
                best_summarization_score = scoring_func(best_summarization)

        # Clear the function cache on closest_emoji because it is
        # unlikely the next run will make use of them
        self.closest_emoji.cache_clear()

        # Return the emoji "sentence", list of all the cosine similarities,
        # and all of the n-grams
        return best_summarization


class PartOfSpeechEmojiTranslator(AbstractEmojiTranslator):

    def pos_n_gram(self, sentence: str, keep_stop_words: bool=True) -> List[str]:
        """
        Generate an n-gram based on the POS tagged dependency tree of the sentence that is "simplified" down according
        to a few assumptions that dictate a good sentence split. These assumptions are as follows:
            1. If two words are leafs and on the same level with the same parent they can be grouped as an n-gram
            2. If there is a sequence of parent-child relationships with only 1 child they can be grouped as one
               n-gram


        """
        stopword = "the in has be".split()
        pos_tagged_n_grams = []

        def to_nltk_tree(node):
            current_node = node
            backlog = []
            while current_node.n_lefts + current_node.n_rights == 1:
                backlog.append((current_node.orth_, current_node.i))
                current_node = list(current_node.children)[0]

            backlog.append((current_node.orth_, current_node.i))
            if current_node.n_lefts + current_node.n_rights > 1:
                good_children = [child for child in current_node.children if len(list(child.children)) > 0]
                bad_children = [(child.orth_, child.i) for child in current_node.children if child not in good_children]
                pos_tagged_n_grams.append(backlog)
                pos_tagged_n_grams.append(bad_children)
                return Tree(backlog, [Tree(bad_children, [])] + [to_nltk_tree(child) for child in good_children])
            else:
                pos_tagged_n_grams.append(backlog)
                return Tree(backlog, [])

        def strip_nothing_unigrams(n_grams):
            return [n_gram for n_gram in n_grams if not (len(n_gram.split(" ")) == 1 and n_gram.split(" ")[0] in stopword)]

        query = " ".join([word for word in sentence.split() if word not in stopword or keep_stop_words])
        doc = self.nlp(query)
        to_nltk_tree(list(doc.sents)[0].root);
        # print(nltk_tree)

        sort_inner = [sorted(nltk_child, key=lambda x: x[1]) for nltk_child in pos_tagged_n_grams]

        nltk_averages = []
        for nltk_child in sort_inner:
            if nltk_child == []:
                continue
            nltk_averages.append((nltk_child, max(x[1] for x in nltk_child)))

        sorted_outer = list(sorted(nltk_averages, key=lambda x: x[1]))

        n_grams = []
        for nltk_average in sorted_outer:
            n_grams.append(" ".join(word[0] for word in nltk_average[0]))


        if not keep_stop_words:
            new_n_grams = []
            for n_gram in n_grams:
                new_n_gram = " ".join([word for word in word_tokenize(n_gram) if word not in stopword])
    #             print(new_n_gram)
                new_n_grams.append(new_n_gram)
            return new_n_grams
        else:
            return n_grams



    def summarize(self, sent:str, keep_stop_words:bool=True, lemma_func: Callable[[str], str]=lambda x: x) -> EmojiSummarizationResult:
        """
        Summarize a sentence using POS n-gram chunking

        Args:
            sent(str): Sentence to summarize
            keep_stop_words(bool, Optional): Flag to keep the stop words when cleaning the sentence and n-grams
            lemma_func(Callable[[str], str], Optional): Function to use to lemmatize the sentence

        Rets:
            EmojiSummarizationResult: Result of the emoji summarization
        """
        time_now = time()

        # Clean the sentence
        sent = self.clean_sentence(sent, keep_stop_words=True, lemma_func=lemma_func)

        # Get the n-grams using the part of speech tagging
        pos_n_grams = self.pos_n_gram(sent, keep_stop_words=keep_stop_words)

        # Clean the n_grams
        n_grams = AbstractEmojiTranslator.clean_n_gram(pos_n_grams)

        # Create an Emoji Summarization Result
        esr = EmojiSummarizationResult()

        # Translate every n_gram in that n-gram sequence
        for n_gram in n_grams:
            print(n_gram)
            # Get the closest emoji to the current n-gram
            emoji, similarity, desc = self.closest_emoji(n_gram)

            # Add the closest emoji to the sumary
            esr.emojis += emoji
            esr.emojis_n_grams.append(desc)
            esr.n_grams.append(n_gram)
            esr.uncertainty_scores.append(similarity)

        # Stop the timer
        esr.elapsed_time = time() - time_now

        # Return the summary
        return esr
