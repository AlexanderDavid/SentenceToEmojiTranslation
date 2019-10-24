#!/usr/bin/env python
# coding: utf-8

# # Naive Sentence to Emoji Translation 
# ## Purpose
# To workshop a naive version of an sentence to emoji translation algorithm. The general idea is that sentences can be "chuncked" out into n-grams that are more related to a single emoji. The related-ness of an n-gram to an emoji is directly related to the cosine similarity of the sent2vec representation of the sentence and the sent2vec representation of one of the emoji's definitions. The emoji definitons are gathered from the [emoji2vec](https://github.com/uclmr/emoji2vec) github repo and the sent2vec model is from the [sent2vec](https://github.com/epfml/sent2vec) github repo. 
# 
# ## Issues
# - The generation of the summary is so incredibly slow
# - There are some issues with lemmatization (e.g. poop != pooped when lemmatized)
# - /opt/conda/lib/python3.7/site-packages/scipy/spatial/distance.py:720: RuntimeWarning: invalid value encountered in float_scalars
#   dist = 1.0 - uv / np.sqrt(uu * vv)
# 
# ## Ideas
# - Add bias for fewer emojis. Some of the generated sentences are just the sentence translated into 1-grams and  it is really easy to find an emoji that represents a one word. If some how the sentence was scored both based on sum similarity and the length of the sentence that might produce better summarizations
# - Use a larger sent2vec model

# In[3]:


# Installs TODO: Add these to docker
# !pip install spacy
# !pip install tabulate
# !pip install ../sent2vec/.

# Standard Library
from typing import List, Tuple, Callable # Datatypes for the function typing
from functools import lru_cache          # Function annotation for storing results 
from dataclasses import dataclass, field # C-like struct functions and class annotation
from string import punctuation

# Scipy suite
import numpy as np                        # For function annotation
from scipy.spatial.distance import cosine # Distance between sentence and emoji in sent2vec vector space

# NLTK 
from nltk import word_tokenize, pos_tag                                 # Tokenizing a sentence into words and tagging POS
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer # Different stemming algorithms
from nltk.corpus import stopwords                                       # Define the set of stopwords in english
stopwords = set(stopwords.words('english'))

# Import spacy (NLP)
import spacy

# Import sentence vectorizer
import sent2vec

# IPython output formatting
from tabulate import tabulate                           # Tabulation from 2-d array into html table
from IPython.display import display, HTML, clear_output # Nice displaying in the output cell
import warnings; warnings.simplefilter('ignore')        # cosine distance gives warnings when div by 0 so
                                                        # ignore all of these
    
# Timing functions
from time import time, localtime, strftime


# In[4]:


# Paramatize the file locations
emoji_file = "./data/emoji_joined.txt" # https://github.com/uclnlp/emoji2vec/blob/master/data/raw_training_data/emoji_joined.txt
wikipedia_file = "./data/wikipedia_utf8_filtered_20pageviews.csv" # https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/


# In[5]:


# Initialize the sent2vec model
s2v = sent2vec.Sent2vecModel()
s2v.load_model('../models/wiki_unigrams.bin') # https://drive.google.com/open?id=0B6VhzidiLvjSa19uYWlLUEkzX3c


# In[6]:


# Intitialize the lemmatizers
# !python -m spacy download en
lemmatizerSpacy = spacy.load('en', disable=['parser', 'ner'])
ps = PorterStemmer()
sb = SnowballStemmer("english")
lemmatizerNLTK = WordNetLemmatizer()


# ## Sentence Cleaning
# The general idea with sentence cleaning is that the sentences need to be put into the same "format" for better analysis. There are two main aspects of cleaning: 1) removal, and 2) modification. Removal is primarily for tokens that do not contribute to the sentence at all. These include ".", "and", "but". Normally this is a standard step in sentence cleaning but it has actually has zero effect on the output that I can see. However, token modification changes the spelling of tokens to uniform all tokens that use the same root. For example "rocked", "rock", "rocking" should all be reduced to their lemma of "rock". There are two different ways to do this: [stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html). 

# In[7]:


def clean_sentence(sent: str, lemma_func: Callable[[str], str]=lemmatizerNLTK.lemmatize, keep_stop_words: bool=True) -> str:
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
    # Lemmatize each word in the sentence and remove the stop words if the flag is set
    return " ".join([lemma_func(token) for token in word_tokenize(sent.lower()) if (token not in stopwords or keep_stop_words) and (token not in punctuation)])


# #### Emoji Vectorization and Related

# In[8]:


# Define the array to store the (emoji, repr) 2-tuple
def generate_emoji_embeddings(lemma_func: Callable[[str], str]=lemmatizerNLTK.lemmatize, keep_stop_words: bool=True) -> List[Tuple[str, List[float]]]:
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

emoji_embeddings = generate_emoji_embeddings()


# In[9]:


@lru_cache(maxsize=1000)
def closest_emoji(sent: str) -> Tuple[str, int]:
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


# #### N-Gram Generation and Related

# In[10]:


def pos_n_gram_split(sent: str) -> List[str]:
    """
    Split a sentence into the best n-gram based on part of speech tagging
    
    Note: Do not clean the sentence before passing into this function
    
    Args:
        sent(str): Sentence to split into n-grams
    Rets:
        (List[str]): n-grams
    """
    
    sent_split_pos = pos_tag(word_tokenize(sent))
    print(sent_split_pos)
    

# In[11]:


def validate_n_gram(n_grams:List[str]) -> bool:
    """
    Validate that a given n_gram is good. Good is defined as the series of n-grams contains no n-grams containing only stop words
    """
            
    return not any([all(map(lambda x: x in stopwords, [word for word in word_tokenize(n_gram)])) for n_gram in n_grams])


# In[12]:


def combinations_of_sent(sent: str) -> List[List[str]]:
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


# ### Summarization Algorithm and Related

# In[13]:


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


# In[62]:


#weighted on real estate an n-gram occupies
def score_summarization_result_weighted_average(summarization: EmojiSummarizationResult) -> float:
    weighted_sum = 0
    sentence_length = 0
    for i in range(len(summarization.uncertainty_scores)):
        sentence_length += len(summarization.n_grams[i].split(" "))
        weighted_sum += summarization.uncertainty_scores[i] * len(summarization.n_grams[i].split(" "))
  
    return weighted_sum/sentence_length

def score_summarization_result_geometric_average(summarization: EmojiSummarizationResult) -> float:
    return np.prod(summarization.uncertainty_scores)**(1/len(summarization.uncertainty_scores))

# Can do with logs - better?
def score_summarization_result_weighted_geometric_average(summarization: EmojiSummarizationResult) -> float:
    weighted_prod = 1
    sentence_length = 0
    for i in range(len(summarization.uncertainty_scores)):
        sentence_length += len(summarization.n_grams[i].split(" "))
        weighted_prod += summarization.uncertainty_scores[i] ** len(summarization.n_grams[i].split(" "))
        
    return weighted_prod ** (1/sentence_length)

def score_summarization_result_harmonic_average(summarization: EmojiSummarizationResult) -> float:
    return len(summarization.n_grams) / sum([1/uncertainty_score for uncertainty_score in summarization.uncertainty_scores])

def score_summarization_result_weighted_harmonic_average(summarization: EmojiSummarizationResult) -> float:
    total = 0
    for i in range(len(summarization.uncertainty_scores)):
        total += 1/(len(summarization.n_grams[i].split(" ")) * summarization.uncertainty_scores[i])
        
    return total


# In[15]:


def score_summarization_result_average(summarization: EmojiSummarizationResult) -> float:
    """
    Score a EmojiSummarizationResult
    
    Get the average of all uncertainty scores and return that as the score
    
    Args:
        summarization(EmojiSummarizationResult): Summarization to score
        
    Rets:
        (float): Numerical summarization score
    """
    return sum(summarization.uncertainty_scores) / len(summarization.uncertainty_scores)


# In[27]:


def summarize(sent:str, lemma_func: Callable[[str], str]=lemmatizerNLTK.lemmatize, 
              keep_stop_words: bool=True, scoring_func: Callable[[EmojiSummarizationResult], float]=score_summarization_result_average) -> EmojiSummarizationResult: 
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
    # Clean the sentence
    sent = clean_sentence(sent, lemma_func, keep_stop_words)
    
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


# ### Verification and Related

# In[54]:


def format_summary(sents: List[str], lemma_func: Callable[[str], str]=lemmatizerNLTK.lemmatize, keep_stop_words: bool=True, generate_embeddings: bool=True,
                  scoring_func: Callable[[EmojiSummarizationResult], float]=score_summarization_result_average) -> HTML:
    """
    Summarize a collection of sentences and display it nicely with IPython
    
    Args:
        sents(List[str]): List of sentences to translate
        lemma_func(Callable[[str], str]), optional: Lemmatization function for cleaning. A function that takes in a word and outputs a word,
                                          normally used to pass in the lemmatization function to be mapped
                                          on every word the sentence
        keep_stop_words(bool), optional: Keep the stop words in the cleaned sentence
        generate_embeddings(bool), optional: Regenerate the emoji embeddings for the case that the lemmatazation/stop_word params have changed
        
    Rets:
        IPython.HTML: HTML List to be displayed with IPython
    
    """

    # Generate emoji embeddings in case the cleaning parameters have changed
    if generate_embeddings:
        time_now = time()
        global emoji_embeddings
        emoji_embeddings = generate_emoji_embeddings(lemma_func, keep_stop_words)
        print("Completed emoji embeddings, time elapsed: {}\n".format(time() - time_now))
    
    # Create the 2d array for the talbe
    table = []
    
    # Iterate through each sentence to be summarized
    for sent in sents:
        # Start timer
        time_now = time()
        
        # Summarize it
        summarization_res = summarize(sent, lemma_func, keep_stop_words, scoring_func)
        
        # Get elapsed time
        elapsed_time = time() - time_now
        
        # Update elapsed time
        summarization_res.elapsed_time = elapsed_time
        
        # Print status update
        # print("Completed sentence: {}, time elapsed: {}".format(sents.index(sent), elapsed_time))

        # Append pertinent data to the table
        table.append([sent, round(scoring_func(summarization_res), 3), 
                           [round(x, 3) for x in summarization_res.uncertainty_scores],
                           summarization_res.n_grams, 
                           summarization_res.elapsed_time,
                           summarization_res.emojis])
        
        # Print out an update
    
    # Return the table with the headers
    return tabulate(table, tablefmt='pipe', 
                                headers=["Input Sentence", "Summary Score", "Individual N-Gram Scores", 
                                         "N-Grams", "Elapsed Time", "Emoji Results"])


# In[55]:
