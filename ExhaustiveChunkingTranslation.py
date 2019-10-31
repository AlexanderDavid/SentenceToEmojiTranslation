# Standard library
from typing import List, Tuple, Callable  # Datatypes for the function typing
from functools import lru_cache           # Annotation for storing func results
from dataclasses import dataclass, field  # Struct functions and annotations
from string import punctuation            # Set of all punctuation

# Scipy suite
import numpy as np                          # For function annotation
from scipy.spatial.distance import cosine   # Distance between vectors
import warnings                             # cosine distance gives warnings
                                            # when div by 0 so ignore

# NLTK
from nltk import word_tokenize              # Tokenizing a sentence into words
# from nltk.stem import WordNetLemmatizer     # Different stemming algorithms
from nltk.corpus import stopwords           # Define the set of stopwords

# Import sentence vectorizer
import sent2vec

# Ignore simple warnings
warnings.simplefilter('ignore')

# Parse all the stop words
stopwords = set(stopwords.words('english'))


@dataclass
class EmojiSummarizationResult:
    """
    "Struct" for keeping track of an Emoji Summarization result

    Data Members:
        emojis(str): String of emojis that represent the summarization
        n_grams(List[str]): List of variable length n-grams that each emoji represents
        uncertainty_scores(List[float]): List of the cosine distance between each n_gram and emoji
        time_elapsed(float): How long it took to complete the summary
    """
    emojis: str = ""
    n_grams: List[str] = field(default_factory=list)
    uncertainty_scores: List[float] = field(default_factory=list)
    elapsed_time: float = 0
    
class ExhaustiveChunkingTranslation:
    def __init__(self, emoji_data_file: str, s2v_model_file: str, lemma_func: Callable[[str], str]):
        self.emoji_file = emoji_data_file

        sent2vec = sent2vec.Sent2vecModel()
        self.s2v = sent2vec.load_model(s2v_model_file)

        self.lemma_func = lemma_func
        
    def clean_sentence(self, sent: str, lemma_func: Callable[[str], str]=None, keep_stop_words: bool=True) -> str:
        """
        Clean a sentence

        Tokenize the word and then lemmatize each individual word before rejoining it all together.
        Optionally removing stop words along the way

        Args:
            sent(str): Sentence to clean
            lemma_func(Callable[[str], str]): A function that takes in a word and outputs a word,
                                              normally used to pass in the lemmatization function to be mapped
                                              on every word the sentence
            keep_stop_words(bool): Keep the stop words in the sentence
        Rets:
            (str): Cleaned sentence
        """
        if lemma_func is None:
            lemma_func = self.lemma_func
        # Lemmatize each word in the sentence and remove the stop words if the flag is set
        return " ".join([lemma_func(token) for token in word_tokenize(sent.lower()) if (token not in stopwords or keep_stop_words) and (token not in punctuation)])
    
    def generate_emoji_embeddings(self, lemma_func: Callable[[str], str]=None, keep_stop_words: bool=True) -> List[Tuple[str, List[float]]]:
        """
        Generate the sent2vec emoji embeddings from the input file

        Run each emoji within the emoji_joined data file from the emoji2vec paper through
        the sent2vec sentence embedder. This is a very naive way of doing it because one
        emoji may have multiple entries in the data file so it has multiple vectors in the
        emoji_embeddings array

        Args:
            lemma_func(Callable[[str], str]): Lemmatization function for cleaning. A function that takes in a word and outputs a word,
                                              normally used to pass in the lemmatization function to be mapped
                                              on every word the sentence
            keep_stop_words(bool): Keep the stop words in the cleaned sentence
        Rets:
            (List[Tuple[str, List[float]]]): A list of 2-tuples containing the emoji and 
                                             one vector representation of it
        """
        if lemma_func is None:
            lemma_func = self.lemma_func
            
        # Initialize the list that will hold all of the embedings
        emoji_embeddings = []

        # Open the file that stores the emoji, description 2-tuple list
        with open(emoji_file) as emojis:
            for defn in emojis:
                # The file is tab-delim
                split = defn.split("\t")

                # Get the emoji and the description from the current line
                emoji = split[-1].replace("\n", "")
                desc = clean_sentence(split[0], lemma_func, keep_stop_words)

                # Add each emoji and embedded description to the list
                emoji_embeddings.append((emoji, s2v.embed_sentence(desc)))

        # Return the embeddings
        return emoji_embeddings
    
    @lru_cache(maxsize=1000)
    def closest_emoji(self, sent: str) -> Tuple[str, int]:
        """
        Get the closest emoji to the given sentence

        Loop through the list of emoji embeddings and keep track of which one has the
        lowest cosine distance from the input sentence's embedding. This is the "closest"
        emoji. The lru_cache designation means that python will store the last [maxsize]
        calls to this function with their return value to reduce computation. This is
        cleared after every call to the summary function.

        Args:
            sent(List[str]): Sentence to check
        Ret:
            (Tuple[str, int]) Closest emoji, cosine similarity of emoji

        """    
        # Embed the sentence using sent2vec 
        emb = s2v.embed_sentence(sent)

        # Start the lowest cosine at higher than it could ever be
        lowest_cos = 1_000_000

        # The best emoji starts as an empty string placeholder
        best_emoji = ""

        # Loop through the dictionary
        for emoji in emoji_embeddings:
            # Get the current emoji's embedding
            emoji_emb = emoji[1]

            # Check the cosine difference between the emoji's embedding and
            # the sentence's embedding
            curr_cos = cosine(emoji_emb, emb)

            # If it lower than the lowest then it is the new best
            if curr_cos < lowest_cos:
                lowest_cos = curr_cos
                best_emoji = emoji[0]

        # Return a 2-tuple containing the best emoji and its cosine differnece
        return best_emoji, lowest_cos
    
    def validate_n_gram(self, n_grams:List[str]) -> bool:
        """
        Validate that a given n_gram is good. Good is defined as the series of n-grams contains no n-grams containing only stop words
        """

        return not any([all(map(lambda x: x in stopwords, [word for word in word_tokenize(n_gram)])) for n_gram in n_grams])

    def combinations_of_sent(self, sent: str) -> List[List[str]]:
        """
        Return all possible n-gram combinations of a sentence

        Args:
            sent(str): Sentence to n-gram-ify
        Rets:
            (List[List[str]]): List of all possible n-gram combinations
        """

        def combinations_of_sum(sum_to: int, combo: List[int]=None) -> List[List[int]]:
            """
            Return all possible combinations of ints that sum to some int

            Args:
                sum_to(int): The number that all sub-arrays should sum to
                combo(List[int]): The current combination of number that the recursive
                                  algo should subdivide, not needed for first run but used
                                  in every consequent recursive run of the function
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
                combo_to_query = combo[:i-1] + [sum(combo[i - 1:i + 1])] + combo[i+1:]
                combos.append(combo_to_query)
                [combos.append(combo) for combo in combinations_of_sum(sum_to, combo_to_query) if combo is not None]

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

                if sent_combo not in sent_combos and validate_n_gram(sent_combo):
                    sent_combos.append(sent_combo)
            return sent_combos

        return combinations_of_sent_helper(sent)
    
    
    def score_summarization_result_average(self, summarization: EmojiSummarizationResult) -> float:
        """
        Score a EmojiSummarizationResult

        Get the average of all uncertainty scores and return that as the score

        Args:
            summarization(EmojiSummarizationResult): Summarization to score

        Rets:
            (float): Numerical summarization score
        """
        return sum(summarization.uncertainty_scores) / len(summarization.uncertainty_scores)
    
    def summarize(self, sent:str, lemma_func: Callable[[str], str]=None, 
                  keep_stop_words: bool=True, scoring_func: Callable[[EmojiSummarizationResult], float]=None) -> EmojiSummarizationResult: 
        """
        Summarize the given sentence into emojis

        Split the sentence into every possible combination of n-grams and see which returns the highest score
        when each n-gram is translated to an emoji using the closest emoji in the dataset

        Args:
            sent(str): Sentence to summarize
            lemma_func(Callable[[str], str]): Lemmatization function for cleaning. A function that takes in a word and outputs a word,
                                              normally used to pass in the lemmatization function to be mapped
                                              on every word the sentence
            keep_stop_words(bool): Keep the stop words in the cleaned sentence
        Rets:
            (Tuple[List[str], List[float], List[str]]): (Emoji Sentence, 
            List of Uncertainty values for the corresponding emoji,
            list of n-grams used to generate the corresponding emoji)
        """
        if lemma_func is None:
            lemma_func = self.lemma_func
        
        if scoring_func is None:
            scoring_func = score_summarization_result_average
        
        # Clean the sentence
        sent = clean_sentence(sent, lemma_func=lemma_func, keep_stop_words=keep_stop_words)
        print(sent)

        # Generate all combinations of sentences
        sent_combos = combinations_of_sent(sent)
        # Init "best" datamembers as empty or exceedingly high
        best_summarization = EmojiSummarizationResult()
        best_summarization_score = 100_000_000
        # Iterate through every combination of sentence combos
        for sent_combo in sent_combos:
            # Start the local data members as empty
            local_summarization = EmojiSummarizationResult()
            # Iterate through each n_gram adding the uncertainty and emoji to the lists
            for n_gram in sent_combo:
                close_emoji, cos_diff = closest_emoji(n_gram)
                local_summarization.emojis += close_emoji
                local_summarization.uncertainty_scores.append(cos_diff)

            local_summarization.n_grams = sent_combo

            # Check if the average uncertainty is less than the best
            # TODO: Maybe a median check would be helpful as well?
            if scoring_func(local_summarization) < best_summarization_score:
                # Update the best emojis
                best_summarization = local_summarization
                best_summarization_score = scoring_func(best_summarization)

        # Clear the function cache on closest_emoji because it is unlikely the next run will make use of them
        closest_emoji.cache_clear()

        # Return the emoji "sentence", list of all the cosine similarities, and all of the n-grams
        return best_summarization