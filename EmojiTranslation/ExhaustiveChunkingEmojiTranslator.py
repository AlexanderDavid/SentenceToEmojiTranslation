# Standard library
from typing import List, Callable  # Datatypes for the function typing

# Scipy suite
import numpy as np         # For function annotation

# NLTK
from nltk import word_tokenize              # Tokenizing a sentence into words
# from nltk.stem import WordNetLemmatizer     # Different stemming algorithms

# Import sentence vectorizer
import sent2vec

# Import the result class to hold the emoji results
from EmojiSummarizationResult import EmojiSummarizationResult
from AbstractEmojiTranslator import AbstractEmojiTranslator

# Ignore simple warnings
import warnings
warnings.simplefilter('ignore')

# Parse all the stop words


class ExhaustiveChunkingTranslation(AbstractEmojiTranslator):
    def __init__(self, emoji_data_file: str, s2v_model_file: str,
                 lemma_func: Callable[[str], str]):
        self.emoji_file = emoji_data_file

        self.s2v = sent2vec.Sent2vecModel()
        print(s2v_model_file)
        self.s2v.load_model(s2v_model_file)

        self.lemma_func = lemma_func

        self.emoji_embeddings = self.generate_emoji_embeddings()

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
        print(sent)

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


if __name__ == '__main__':
    e = ExhaustiveChunkingTranslation("./data/emoji_joined.txt",
                                      "./data/wiki_unigrams.bin", lambda x: x)
    while True:
        print(e.summarize(input(">")))
