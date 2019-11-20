# Sentence to Emoji Translation
A naive chunking approach to summarizing a sentence to emoji translation using sent2vec and some of emoji2vec's dataset. This is a final senior research project for my undergraduate degree at Clarion University.

## Algorithm Explanation
The flow of this algorithm is as follows:
  1. Chunk a sentence into every possible n-gram combination
  2. Translate each n-gram sequence into an emoji sequence
  3. Return the emoji sequence with the highest score
  
The algorithm is separated into 4 main parts:
  1. Emoji Vectorization
  2. Sentence Chunking
  3. N-gram to Emoji comparison
  4. Summary scoring

### Emoji Vectorization
The basis of our algorithm is the assumption that by vectorizing an emoji description you can get a good representation of that emoji in a vector space. A snippet from the dataset we are using is seen below:
```
smiley moon	🌝
spooky	🌝
flag for belarus	🇧🇾
regional indicator symbol letters by	🇧🇾
belarusian flag	🇧🇾
vain	🐩
miniature poodle	🐩
dog	🐩
sophisticated	🐩
```
The format is a keyword (or sentence when we use Emojipedia), a tab, then the emoji it represents. Each of these entries in the dataset are vectorized and put into a list. The vectorization we are using is sent2vec so it produces a 700 dimension vector. The emoji vectorization is not unique, as seen above one emoji can have multiple descriptions that do not mean the same thing so each of those is vectorized and appended to the list separately.

### Sentence Chunking
The current method for chunking the sentences is rather naive. We just brute force every n-gram combination by first calculating every way to sum the length of the sentence with the integers and then represent each of those sums with n-grams with the lengths of the integers.

### N-gram to Emoji Comparison
A direct comparison is drawn between the n-grams and the emojis by vectorizing the n-gram and calculating the cosine difference between the two vectors.

### Summary Scoring
We score a summary by averaging the cosine difference of all n-gram-emoji combinations. The lower the average difference the higher the score. 

### Ideas for Improvement
[ ] Use Emoji2Vec-Sent2Vec Vectoriztion instead of naive list
[x] Remove n-gram sequences that contain n-grams with only stop words
[x] Modify scoring algorithm to weight based on n-gram length 

## Some Results
| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                        |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.687 | [0.374, 1.0]               | ['we need to rent a room for our', 'party']    |       10.8195  | 🚪🎈            |
| She folded her handkerchief neatly.     |           0.471 | [0.471]                    | ['she folded her handkerchief neatly']         |        3.02839 | 🙏              |
| I'd rather be a bird than a fish.       |           0.753 | [0.507, 1.0]               | ["i 'd rather be a bird than a", 'fish']       |       10.5912  | 🐥♓            |
| Tom got a small piece of pie.           |           0.758 | [0.332, 1.0, 0.741, 0.958] | ['tom got a', 'small', 'piece', 'of pie']      |        8.73315 | 💯🚼🎩🍏        |
| The lake is a long way from here.       |           0.51  | [0.424, 0.591, 0.515]      | ['the lake is a', 'long', 'way from here']     |        4.05432 | 💯🤥🈁          |
| Rock music approaches at high velocity. |           0.822 | [1.0, 1.0, 0.465]          | ['rock', 'music', 'approach at high velocity'] |        6.74701 | 🎸🎻🚄  |

As you can see this is not yet producing optimal summaries. Some work needs to be done to unify performance across different sentence types.


## Prerequisites
- [NLTK](https://www.nltk.org/)
- [sent2vec](https://github.com/epfml/sent2vec)
- [spacy]
- [Tabulate](https://pypi.org/project/tabulate/)
- [Jupyter Notebook](https://jupyter.org/)

## Getting off the Ground
All of the Jupyter Notebooks we did our development in are in the obvious folder. Read through those to get an idea of how the algorithm works internally if you want. The python module that we created is in the EmojiTranslation folder.
