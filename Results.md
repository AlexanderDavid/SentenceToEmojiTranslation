# Varying Lemmatizer and Stop Words
## Emoji2Vec Dataset
### No lemmatization, with stop words
Completed emoji embeddings, time elapsed: 0.9186346530914307

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                          |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:-------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.687 | [0.373, 1.0]                    | ['we need to rent a room for our', 'party']      |       12.933   | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.464 | [0.464]                         | ['she folded her handkerchief neatly']           |        4.1507  | 🙏              |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']   |       12.4961  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.777 | [0.186, 1.0, 1.0, 0.741, 0.958] | ['tom got', 'a', 'small', 'piece', 'of pie']     |        7.70545 | 💸💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.646 | [0.181, 1.0, 0.404, 1.0]        | ['the lake is', 'a', 'long way from', 'here']    |        9.97463 | 🎏💯🤥🈁        |
| Rock music approaches at high velocity. |           0.815 | [1.0, 1.0, 0.445]               | ['rock', 'music', 'approaches at high velocity'] |        5.83142 | 🎸🎻🎩          |

### No lemmatization, no stop words
Completed emoji embeddings, time elapsed: 0.9026648998260498

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                   |        2.76511 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.527 | [0.527]                    | ['folded handkerchief neatly']                |        1.66054 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']                    |        2.77706 | 🐥♓            |
| Tom got a small piece of pie.           |           0.732 | [0.186, 1.0, 0.741, 1.0]   | ['tom got', 'small', 'piece', 'pie']          |        4.1679  | 💸🚼🎩🍏        |
| The lake is a long way from here.       |           0.364 | [0.223, 0.591, 0.278]      | ['lake', 'long', 'way']                       |        1.66299 | 🛁🤥🌌          |
| Rock music approaches at high velocity. |           0.822 | [1.0, 1.0, 0.466]          | ['rock', 'music', 'approaches high velocity'] |        4.16493 | 🤘🎻🎩          |

### Wordnet Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 0.9625589847564697

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                        |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:-----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.687 | [0.373, 1.0]                    | ['we need to rent a room for our', 'party']    |       12.9235  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.471 | [0.471]                         | ['she folded her handkerchief neatly']         |        4.18612 | 🙏              |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish'] |       12.6363  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.777 | [0.186, 1.0, 1.0, 0.741, 0.958] | ['tom got', 'a', 'small', 'piece', 'of pie']   |        7.76357 | 💸💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.646 | [0.181, 1.0, 0.404, 1.0]        | ['the lake is', 'a', 'long way from', 'here']  |       10.0005  | 🎏💯🤥🈁        |
| Rock music approaches at high velocity. |           0.814 | [1.0, 1.0, 0.442]               | ['rock', 'music', 'approach at high velocity'] |        5.7918  | 🎸🎻🎩          |

### Wordnet Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 0.9974956512451172

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                     |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:--------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                 |        2.76752 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.532 | [0.532]                    | ['folded handkerchief neatly']              |        1.65341 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']                  |        2.78737 | 🐥♓            |
| Tom got a small piece of pie.           |           0.732 | [0.186, 1.0, 0.741, 1.0]   | ['tom got', 'small', 'piece', 'pie']        |        4.16009 | 💸🚼🎩🍏        |
| The lake is a long way from here.       |           0.364 | [0.223, 0.591, 0.278]      | ['lake', 'long', 'way']                     |        1.6893  | 🛁🤥🌌          |
| Rock music approaches at high velocity. |           0.82  | [1.0, 1.0, 0.459]          | ['rock', 'music', 'approach high velocity'] |        4.15355 | 🤘🎻🎩          |

### Spacy Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 24.179542064666748

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                            |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:---------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.72  | [0.44, 1.0]                     | ['-PRON- need to rent a room for', '-PRON- party'] |       12.7295  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.515 | [0.515]                         | ['-PRON- fold -PRON- handkerchief neatly']         |        3.93098 | 🙏              |
| I'd rather be a bird than a fish.       |           0.887 | [1.0, 0.549, 1.0, 1.0]          | ['i', 'have rather be a bird than', 'a', 'fish']   |       12.6723  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.802 | [0.312, 1.0, 1.0, 0.741, 0.958] | ['tom get', 'a', 'small', 'piece', 'of pie']       |        7.78937 | 🚥💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.701 | [0.4, 1.0, 0.404, 1.0]          | ['the lake be', 'a', 'long way from', 'here']      |       10.12    | 👁‍🗨💯🤥🈁        |
| Rock music approaches at high velocity. |           0.815 | [1.0, 1.0, 0.445]               | ['rock', 'music', 'approaches at high velocity']   |        5.86197 | 🎸🎻🎩          |

### Spacy Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 22.51989436149597

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                   |        2.79105 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.539 | [0.539]                    | ['fold handkerchief neatly']                  |        1.71628 | 🙏              |
| I'd rather be a bird than a fish.       |           0.803 | [0.607, 1.0]               | ['have rather bird', 'fish']                  |        2.7738  | 🐥♓            |
| Tom got a small piece of pie.           |           0.763 | [0.312, 1.0, 0.741, 1.0]   | ['tom get', 'small', 'piece', 'pie']          |        4.13023 | 🚥🚼🎩🍏        |
| The lake is a long way from here.       |           0.364 | [0.223, 0.591, 0.278]      | ['lake', 'long', 'way']                       |        1.66909 | 🛁🤥🌌          |
| Rock music approaches at high velocity. |           0.822 | [1.0, 1.0, 0.466]          | ['rock', 'music', 'approaches high velocity'] |        4.19319 | 🤘🎻🎩          |

### Porter Stemmer, with stop words
Completed emoji embeddings, time elapsed: 1.172374963760376

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.753 | [0.237, 1.0, 0.988, 0.788]      | ['we need to rent', 'a', 'room', 'for our parti'] |       13.1578  | ℹ️💯🏥🎈         |
| She folded her handkerchief neatly.     |           0.496 | [0.496]                         | ['she fold her handkerchief neatli']              |        4.20029 | 🙏              |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']    |       12.7307  | ℹ️🐥💯♓         |
| Tom got a small piece of pie.           |           0.784 | [0.186, 1.0, 1.0, 0.951]        | ['tom got', 'a', 'small', 'piec of pie']          |        7.82785 | 💸💯🚼🍏        |
| The lake is a long way from here.       |           0.702 | [0.181, 1.0, 0.591, 0.739, 1.0] | ['the lake is', 'a', 'long', 'way from', 'here']  |       10.0646  | 🎏💯🤥🌌🈁      |
| Rock music approaches at high velocity. |           0.882 | [1.0, 1.0, 0.645]               | ['rock', 'music', 'approach at high veloc']       |        5.88243 | 🎸🎻🎩          |

### Porter Stemmer, no stop words
Completed emoji embeddings, time elapsed: 1.2754266262054443

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.799 | [0.598, 1.0]               | ['need rent room', 'parti']              |        2.7833  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.606 | [0.606]                    | ['fold handkerchief neatli']             |        1.69999 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']               |        2.76588 | 🐥♓            |
| Tom got a small piece of pie.           |           0.724 | [0.186, 1.0, 0.985]        | ['tom got', 'small', 'piec pie']         |        4.15958 | 💸🚼🍏          |
| The lake is a long way from here.       |           0.668 | [0.39, 0.946]              | ['lake long', 'way']                     |        1.65972 | 🤥🌌            |
| Rock music approaches at high velocity. |           0.905 | [1.0, 1.0, 0.714]          | ['rock', 'music', 'approach high veloc'] |        4.1626  | 🤘🎻🎩          |

### Snowball Stemmer, with stop words
Completed emoji embeddings, time elapsed: 1.067199468612671

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.753 | [0.237, 1.0, 0.988, 0.788]      | ['we need to rent', 'a', 'room', 'for our parti'] |       12.8767  | ℹ️💯🏥🎈         |
| She folded her handkerchief neatly.     |           0.465 | [0.575, 0.355]                  | ['she fold', 'her handkerchief neat']             |        4.1999  | 🙏👚            |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']    |       12.727   | ℹ️🐥💯♓         |
| Tom got a small piece of pie.           |           0.784 | [0.186, 1.0, 1.0, 0.951]        | ['tom got', 'a', 'small', 'piec of pie']          |        7.7823  | 💸💯🚼🍏        |
| The lake is a long way from here.       |           0.702 | [0.181, 1.0, 0.591, 0.739, 1.0] | ['the lake is', 'a', 'long', 'way from', 'here']  |        9.99415 | 🎏💯🤥🌌🈁      |
| Rock music approaches at high velocity. |           0.882 | [1.0, 1.0, 0.645]               | ['rock', 'music', 'approach at high veloc']       |        7.47921 | 🎸🎻🎩          |

### Snowball Stemmer, no stop words
Completed emoji embeddings, time elapsed: 1.124800205230713

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.799 | [0.598, 1.0]               | ['need rent room', 'parti']              |        3.55909 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.522 | [0.522]                    | ['fold handkerchief neat']               |        2.13142 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']               |        3.28313 | 🐥♓            |
| Tom got a small piece of pie.           |           0.724 | [0.186, 1.0, 0.985]        | ['tom got', 'small', 'piec pie']         |        4.16571 | 💸🚼🍏          |
| The lake is a long way from here.       |           0.668 | [0.39, 0.946]              | ['lake long', 'way']                     |        1.706   | 🤥🌌            |
| Rock music approaches at high velocity. |           0.905 | [1.0, 1.0, 0.714]          | ['rock', 'music', 'approach high veloc'] |        4.1878  | 🤘🎻🎩          |

## Emojipedia Dataset (First sentence of description)

### No lemmatization, with stop words
Completed emoji embeddings, time elapsed: 0.26052331924438477

| Input Sentence                          |   Summary Score | Individual N-Gram Scores     | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:-----------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.44  | [0.199, 0.703, 0.518, 0.34]  | ['we need to rent', 'a', 'room', 'for our party'] |       2.88409  | 🈹💾🚪🥳         |
| She folded her handkerchief neatly.     |           0.389 | [0.399, 0.481, 0.287]        | ['she folded', 'her', 'handkerchief neatly']      |       0.829407 | 🥟💆🎀           |
| I'd rather be a bird than a fish.       |           0.58  | [0.488, 0.703, 0.55]         | ["i 'd rather be a bird than", 'a', 'fish']       |       2.71226  | 🐦💾🐟          |
| Tom got a small piece of pie.           |           0.448 | [0.184, 0.703, 0.616, 0.287] | ['tom got', 'a', 'small', 'piece of pie']         |       1.53654  | 💆💾🚣🥧         |
| The lake is a long way from here.       |           0.401 | [0.247, 0.703, 0.252]        | ['the lake is', 'a', 'long way from here']        |       2.49323  | 🚤💾🈁          |
| Rock music approaches at high velocity. |           0.539 | [0.655, 0.524, 0.439]        | ['rock', 'music', 'approaches at high velocity']  |       1.14726  | 🧗🎵🚄           |

### No lemmatization, no stop words
Completed emoji embeddings, time elapsed: 0.24212169647216797

| Input Sentence                          |   Summary Score | Individual N-Gram Scores    | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:----------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.461 | [0.422, 0.501]              | ['need rent room', 'party']                   |       0.543367 | 🚪🥳             |
| She folded her handkerchief neatly.     |           0.399 | [0.486, 0.311]              | ['folded', 'handkerchief neatly']             |       0.325372 | 🥟👝             |
| I'd rather be a bird than a fish.       |           0.536 | [0.486, 0.585]              | ["'d rather bird", 'fish']                    |       0.53641  | 🐦🐟            |
| Tom got a small piece of pie.           |           0.425 | [0.279, 0.644, 0.45, 0.325] | ['tom got', 'small', 'piece', 'pie']          |       0.809613 | 💆🚣📄🍰        |
| The lake is a long way from here.       |           0.343 | [0.346, 0.339]              | ['lake', 'long way']                          |       0.322598 | 🚤🥒            |
| Rock music approaches at high velocity. |           0.564 | [0.685, 0.557, 0.451]       | ['rock', 'music', 'approaches high velocity'] |       0.808164 | 🧗🎵🚄           |

### Wordnet Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 0.30639004707336426

| Input Sentence                          |   Summary Score | Individual N-Gram Scores     | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:-----------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.428 | [0.17, 0.703, 0.381, 0.458]  | ['we need to rent', 'a', 'room for our', 'party'] |       2.7611   | 🈹💾🚪🥳         |
| She folded her handkerchief neatly.     |           0.395 | [0.404, 0.481, 0.3]          | ['she folded', 'her', 'handkerchief neatly']      |       0.821405 | 🥟💆👘           |
| I'd rather be a bird than a fish.       |           0.582 | [0.488, 0.703, 0.556]        | ["i 'd rather be a bird than", 'a', 'fish']       |       2.83004  | 🐦💾🍣          |
| Tom got a small piece of pie.           |           0.449 | [0.184, 0.703, 0.616, 0.292] | ['tom got', 'a', 'small', 'piece of pie']         |       1.53374  | 💆💾🚣🥧         |
| The lake is a long way from here.       |           0.434 | [0.364, 0.703, 0.237]        | ['the lake is', 'a', 'long way from here']        |       2.01294  | 🚤💾🥾           |
| Rock music approaches at high velocity. |           0.565 | [0.655, 0.575, 0.465]        | ['rock', 'music', 'approach at high velocity']    |       1.14433  | 🧗🎵🚄           |

### Wordnet Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 0.2833833694458008

| Input Sentence                          |   Summary Score | Individual N-Gram Scores    | N-Grams                                     |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:----------------------------|:--------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.471 | [0.422, 0.521]              | ['need rent room', 'party']                 |       0.544961 | 🚪🥳             |
| She folded her handkerchief neatly.     |           0.399 | [0.486, 0.311]              | ['folded', 'handkerchief neatly']           |       0.329441 | 🥟👝             |
| I'd rather be a bird than a fish.       |           0.531 | [0.486, 0.577]              | ["'d rather bird", 'fish']                  |       0.541451 | 🐦🍣            |
| Tom got a small piece of pie.           |           0.425 | [0.279, 0.644, 0.45, 0.325] | ['tom got', 'small', 'piece', 'pie']        |       0.809405 | 💆🚣📄🍰        |
| The lake is a long way from here.       |           0.429 | [0.519, 0.339]              | ['lake', 'long way']                        |       0.340311 | 🚤🥒            |
| Rock music approaches at high velocity. |           0.594 | [0.685, 0.612, 0.484]       | ['rock', 'music', 'approach high velocity'] |       0.804777 | 🧗🎵🚄           |

### Spacy Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 30.82019877433777

| Input Sentence                          |   Summary Score | Individual N-Gram Scores     | N-Grams                                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:-----------------------------|:---------------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.443 | [0.163, 0.703, 0.458, 0.449] | ['-PRON- need to rent', 'a', 'room for', '-PRON- party'] |       2.85166  | 🚎💾🚪🥳         |
| She folded her handkerchief neatly.     |           0.353 | [0.353]                      | ['-PRON- fold -PRON- handkerchief neatly']               |       0.788352 | 🥡               |
| I'd rather be a bird than a fish.       |           0.569 | [0.461, 0.703, 0.542]        | ['i have rather be a bird than', 'a', 'fish']            |       2.89151  | 🥀💾🎣          |
| Tom got a small piece of pie.           |           0.492 | [0.331, 0.703, 0.616, 0.32]  | ['tom get', 'a', 'small', 'piece of pie']                |       1.55225  | 💆💾🚣🍞        |
| The lake is a long way from here.       |           0.516 | [0.437, 0.651, 0.703, 0.275] | ['the lake', 'be', 'a', 'long way from here']            |       2.01716  | 🚤🧞💾🥕         |
| Rock music approaches at high velocity. |           0.572 | [0.67, 0.572, 0.475]         | ['rock', 'music', 'approaches at high velocity']         |       1.17187  | 🧗🎵🚄           |

### Spacy Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 18.736726999282837

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.454 | [0.422, 0.486]             | ['need rent room', 'party']                   |       0.555363 | 🚪🥳             |
| She folded her handkerchief neatly.     |           0.372 | [0.372]                    | ['fold handkerchief neatly']                  |       0.33495  | 🥡               |
| I'd rather be a bird than a fish.       |           0.527 | [0.479, 0.575]             | ['have rather bird', 'fish']                  |       0.55914  | 🐦🐟            |
| Tom got a small piece of pie.           |           0.453 | [0.367, 0.644, 0.35]       | ['tom get', 'small', 'piece pie']             |       0.826017 | 💆🚣🍞          |
| The lake is a long way from here.       |           0.439 | [0.538, 0.339]             | ['lake', 'long way']                          |       0.333523 | 🚤🥒            |
| Rock music approaches at high velocity. |           0.587 | [0.691, 0.59, 0.478]       | ['rock', 'music', 'approaches high velocity'] |       0.821203 | 🧗🎵🚄           |

### Porter Stemmer, with stop words
Completed emoji embeddings, time elapsed: 0.46215343475341797

| Input Sentence                          |   Summary Score | Individual N-Gram Scores            | N-Grams                                     |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:------------------------------------|:--------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.46  | [0.367, 0.552]                      | ['we need to rent a room for our', 'parti'] |       2.80985  | 🚪🥳             |
| She folded her handkerchief neatly.     |           0.373 | [0.379, 0.367]                      | ['she fold', 'her handkerchief neatli']     |       0.822813 | 🥡🎀             |
| I'd rather be a bird than a fish.       |           0.645 | [0.577, 0.703, 0.656]               | ["i 'd rather be a bird than", 'a', 'fish'] |       2.84725  | 🐦💾🎣          |
| Tom got a small piece of pie.           |           0.472 | [0.197, 0.703, 0.656, 0.525, 0.281] | ['tom got', 'a', 'small piec', 'of', 'pie'] |       1.54378  | 💆💾🚣💊🍞      |
| The lake is a long way from here.       |           0.465 | [0.404, 0.703, 0.288]               | ['the lake is', 'a', 'long way from here']  |       2.00287  | 🚤💾🥕          |
| Rock music approaches at high velocity. |           0.558 | [0.651, 0.643, 0.379]               | ['rock', 'music', 'approach at high veloc'] |       1.15525  | 🧗🎵🚄           |

### Porter Stemmer, no stop words
Completed emoji embeddings, time elapsed: 0.42127180099487305

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.503 | [0.424, 0.582]             | ['need rent room', 'parti']              |       0.551874 | 🚪👗            |
| She folded her handkerchief neatly.     |           0.424 | [0.424]                    | ['fold handkerchief neatli']             |       0.348933 | 🥡               |
| I'd rather be a bird than a fish.       |           0.627 | [0.561, 0.693]             | ["'d rather bird", 'fish']               |       0.539492 | 🐦🎣            |
| Tom got a small piece of pie.           |           0.436 | [0.268, 0.737, 0.304]      | ['tom got', 'small', 'piec pie']         |       0.825557 | 💆🚣🥪           |
| The lake is a long way from here.       |           0.491 | [0.572, 0.41]              | ['lake', 'long way']                     |       0.33101  | 🚤🥒            |
| Rock music approaches at high velocity. |           0.592 | [0.675, 0.703, 0.397]      | ['rock', 'music', 'approach high veloc'] |       0.831188 | 🧗🎵🚄           |

### Snowball Stemmer, with stop words
Completed emoji embeddings, time elapsed: 0.39820289611816406

| Input Sentence                          |   Summary Score | Individual N-Gram Scores            | N-Grams                                     |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:------------------------------------|:--------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.46  | [0.367, 0.552]                      | ['we need to rent a room for our', 'parti'] |       2.94195  | 🚪🥳             |
| She folded her handkerchief neatly.     |           0.399 | [0.379, 0.492, 0.327]               | ['she fold', 'her', 'handkerchief neat']    |       0.821262 | 🥡💆🧶            |
| I'd rather be a bird than a fish.       |           0.616 | [0.488, 0.703, 0.656]               | ["i 'd rather be a bird than", 'a', 'fish'] |       2.88236  | 🐦💾🎣          |
| Tom got a small piece of pie.           |           0.468 | [0.175, 0.703, 0.656, 0.525, 0.281] | ['tom got', 'a', 'small piec', 'of', 'pie'] |       1.55203  | 💆💾🚣💊🍞      |
| The lake is a long way from here.       |           0.465 | [0.404, 0.703, 0.288]               | ['the lake is', 'a', 'long way from here']  |       2.05877  | 🚤💾🥕          |
| Rock music approaches at high velocity. |           0.558 | [0.651, 0.643, 0.379]               | ['rock', 'music', 'approach at high veloc'] |       1.16155  | 🧗🎵🚄           |

### Snowball Stemmer, no stop words
Completed emoji embeddings, time elapsed: 0.3679392337799072

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.503 | [0.424, 0.582]             | ['need rent room', 'parti']              |       0.55168  | 🚪👗            |
| She folded her handkerchief neatly.     |           0.431 | [0.431]                    | ['fold handkerchief neat']               |       0.327883 | 🥡               |
| I'd rather be a bird than a fish.       |           0.59  | [0.486, 0.693]             | ["'d rather bird", 'fish']               |       0.544478 | 🐦🎣            |
| Tom got a small piece of pie.           |           0.436 | [0.268, 0.737, 0.304]      | ['tom got', 'small', 'piec pie']         |       0.824745 | 💆🚣🥪           |
| The lake is a long way from here.       |           0.491 | [0.572, 0.41]              | ['lake', 'long way']                     |       0.330163 | 🚤🥒            |
| Rock music approaches at high velocity. |           0.592 | [0.675, 0.703, 0.397]      | ['rock', 'music', 'approach high veloc'] |       0.821996 | 🧗🎵🚄           |


## Emojipedia and Emoji2Vec Datasets Combined

### No lemmatization, with stop words
Completed emoji embeddings, time elapsed: 1.219926118850708

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                          |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:-------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.687 | [0.373, 1.0]                    | ['we need to rent a room for our', 'party']      |       15.6985  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.465 | [0.575, 0.481, 0.339]           | ['she folded', 'her', 'handkerchief neatly']     |        5.01055 | 🙏💆👳          |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']   |       15.0588  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.777 | [0.186, 1.0, 1.0, 0.741, 0.958] | ['tom got', 'a', 'small', 'piece', 'of pie']     |        9.45181 | 💸💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.663 | [0.247, 1.0, 0.404, 1.0]        | ['the lake is', 'a', 'long way from', 'here']    |       12.0614  | 🚤💯🤥🈁        |
| Rock music approaches at high velocity. |           0.815 | [1.0, 1.0, 0.445]               | ['rock', 'music', 'approaches at high velocity'] |        7.01168 | 🎸🎻🎩          |

### No lemmatization, no stop words
Completed emoji embeddings, time elapsed: 1.2764132022857666

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                   |        3.30334 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.527 | [0.527]                    | ['folded handkerchief neatly']                |        2.00463 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']                    |        3.29656 | 🐥♓            |
| Tom got a small piece of pie.           |           0.755 | [0.279, 1.0, 0.741, 1.0]   | ['tom got', 'small', 'piece', 'pie']          |        4.96605 | 💆🚼🎩🍏        |
| The lake is a long way from here.       |           0.405 | [0.346, 0.591, 0.278]      | ['lake', 'long', 'way']                       |        1.95806 | 🚤🤥🌌          |
| Rock music approaches at high velocity. |           0.822 | [1.0, 1.0, 0.466]          | ['rock', 'music', 'approaches high velocity'] |        4.94741 | 🤘🎻🎩          |

### Wordnet Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 1.2770252227783203

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                        |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:-----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.687 | [0.374, 1.0]                    | ['we need to rent a room for our', 'party']    |       15.4177  | 🚪🎈            |
| She folded her handkerchief neatly.     |           0.473 | [0.574, 0.481, 0.364]           | ['she folded', 'her', 'handkerchief neatly']   |        5.01544 | 🙏💆🎽          |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish'] |       15.0352  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.777 | [0.186, 1.0, 1.0, 0.741, 0.958] | ['tom got', 'a', 'small', 'piece', 'of pie']   |        9.29472 | 💸💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.692 | [0.364, 1.0, 0.404, 1.0]        | ['the lake is', 'a', 'long way from', 'here']  |       12.0837  | 🚤💯🤥🈁        |
| Rock music approaches at high velocity. |           0.822 | [1.0, 1.0, 0.465]               | ['rock', 'music', 'approach at high velocity'] |        7.00032 | 🎸🎻🚄          |

### Wordnet Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 1.2538228034973145

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                     |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:--------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                 |        3.29873 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.532 | [0.532]                    | ['folded handkerchief neatly']              |        1.98755 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']                  |        3.29582 | 🐥♓            |
| Tom got a small piece of pie.           |           0.755 | [0.279, 1.0, 0.741, 1.0]   | ['tom got', 'small', 'piece', 'pie']        |        4.93947 | 💆🚼🎩🍏        |
| The lake is a long way from here.       |           0.491 | [0.519, 0.463]             | ['lake', 'long way']                        |        1.97809 | 🚤🤥            |
| Rock music approaches at high velocity. |           0.828 | [1.0, 1.0, 0.484]          | ['rock', 'music', 'approach high velocity'] |        4.95554 | 🤘🎻🚄          |

### Spacy Lemmatizer, with stop words
Completed emoji embeddings, time elapsed: 55.59264016151428

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                            |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:---------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.72  | [0.44, 1.0]                     | ['-PRON- need to rent a room for', '-PRON- party'] |       15.0302  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.515 | [0.515]                         | ['-PRON- fold -PRON- handkerchief neatly']         |        4.7247  | 🙏              |
| I'd rather be a bird than a fish.       |           0.887 | [1.0, 0.549, 1.0, 1.0]          | ['i', 'have rather be a bird than', 'a', 'fish']   |       15.2554  | 👤🐥💯♓        |
| Tom got a small piece of pie.           |           0.806 | [0.331, 1.0, 1.0, 0.741, 0.958] | ['tom get', 'a', 'small', 'piece', 'of pie']       |        9.40935 | 💆💯🚼🎩🍏      |
| The lake is a long way from here.       |           0.709 | [0.432, 1.0, 0.404, 1.0]        | ['the lake be', 'a', 'long way from', 'here']      |       12.1125  | 🧞💯🤥🈁         |
| Rock music approaches at high velocity. |           0.825 | [1.0, 1.0, 0.475]               | ['rock', 'music', 'approaches at high velocity']   |        7.88819 | 🎸🎻🚄          |

### Spacy Lemmatizer, no stop words
Completed emoji embeddings, time elapsed: 41.01856708526611

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                       |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:----------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.736 | [0.472, 1.0]               | ['need rent room', 'party']                   |        3.32467 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.539 | [0.539]                    | ['fold handkerchief neatly']                  |        1.98845 | 🙏              |
| I'd rather be a bird than a fish.       |           0.803 | [0.607, 1.0]               | ['have rather bird', 'fish']                  |        3.3347  | 🐥♓            |
| Tom got a small piece of pie.           |           0.777 | [0.367, 1.0, 0.741, 1.0]   | ['tom get', 'small', 'piece', 'pie']          |        4.98315 | 💆🚼🎩🍏        |
| The lake is a long way from here.       |           0.501 | [0.538, 0.463]             | ['lake', 'long way']                          |        1.97258 | 🚤🤥            |
| Rock music approaches at high velocity. |           0.826 | [1.0, 1.0, 0.478]          | ['rock', 'music', 'approaches high velocity'] |        4.99455 | 🤘🎻🚄          |

### Porter Stemmer, with stop words
Completed emoji embeddings, time elapsed: 1.6035850048065186

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.753 | [0.237, 1.0, 0.988, 0.788]      | ['we need to rent', 'a', 'room', 'for our parti'] |       15.4555  | ℹ️💯🏥🎈         |
| She folded her handkerchief neatly.     |           0.496 | [0.496]                         | ['she fold her handkerchief neatli']              |        5.02829 | 🙏              |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']    |       15.0708  | ℹ️🐥💯♓         |
| Tom got a small piece of pie.           |           0.787 | [0.197, 1.0, 1.0, 0.951]        | ['tom got', 'a', 'small', 'piec of pie']          |        9.3489  | 💆💯🚼🍏        |
| The lake is a long way from here.       |           0.747 | [0.404, 1.0, 0.591, 0.739, 1.0] | ['the lake is', 'a', 'long', 'way from', 'here']  |       12.0638  | 🚤💯🤥🌌🈁      |
| Rock music approaches at high velocity. |           0.882 | [1.0, 1.0, 0.645]               | ['rock', 'music', 'approach at high veloc']       |        7.05595 | 🎸🎻🎩          |

### Porter Stemmer, no stop words
Completed emoji embeddings, time elapsed: 1.5541434288024902

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.799 | [0.598, 1.0]               | ['need rent room', 'parti']              |        3.3243  | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.606 | [0.606]                    | ['fold handkerchief neatli']             |        2.0212  | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']               |        3.33549 | 🐥♓            |
| Tom got a small piece of pie.           |           0.751 | [0.268, 1.0, 0.985]        | ['tom got', 'small', 'piec pie']         |        4.9837  | 💆🚼🍏          |
| The lake is a long way from here.       |           0.714 | [0.482, 0.946]             | ['lake long', 'way']                     |        2.0052  | 🚤🌌            |
| Rock music approaches at high velocity. |           0.905 | [1.0, 1.0, 0.714]          | ['rock', 'music', 'approach high veloc'] |        5.02363 | 🤘🎻🎩          |

### Snowball Stemmer, with stop words
Completed emoji embeddings, time elapsed: 1.4531970024108887

| Input Sentence                          |   Summary Score | Individual N-Gram Scores        | N-Grams                                           |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:--------------------------------|:--------------------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.753 | [0.237, 1.0, 0.988, 0.788]      | ['we need to rent', 'a', 'room', 'for our parti'] |       15.4345  | ℹ️💯🏥🎈         |
| She folded her handkerchief neatly.     |           0.467 | [0.575, 0.492, 0.333]           | ['she fold', 'her', 'handkerchief neat']          |        4.95312 | 🙏💆👚          |
| I'd rather be a bird than a fish.       |           0.9   | [1.0, 0.601, 1.0, 1.0]          | ['i', "'d rather be a bird than", 'a', 'fish']    |       15.0655  | ℹ️🐥💯♓         |
| Tom got a small piece of pie.           |           0.784 | [0.186, 1.0, 1.0, 0.951]        | ['tom got', 'a', 'small', 'piec of pie']          |        9.3079  | 💸💯🚼🍏        |
| The lake is a long way from here.       |           0.747 | [0.404, 1.0, 0.591, 0.739, 1.0] | ['the lake is', 'a', 'long', 'way from', 'here']  |       12.0054  | 🚤💯🤥🌌🈁      |
| Rock music approaches at high velocity. |           0.882 | [1.0, 1.0, 0.645]               | ['rock', 'music', 'approach at high veloc']       |        6.96598 | 🎸🎻🎩          |

### Snowball Stemmer, no stop words
Completed emoji embeddings, time elapsed: 1.4212706089019775

| Input Sentence                          |   Summary Score | Individual N-Gram Scores   | N-Grams                                  |   Elapsed Time | Emoji Results   |
|:----------------------------------------|----------------:|:---------------------------|:-----------------------------------------|---------------:|:----------------|
| We need to rent a room for our party.   |           0.799 | [0.598, 1.0]               | ['need rent room', 'parti']              |        3.32762 | 🏥🎈            |
| She folded her handkerchief neatly.     |           0.522 | [0.522]                    | ['fold handkerchief neat']               |        1.98618 | 🙏              |
| I'd rather be a bird than a fish.       |           0.807 | [0.615, 1.0]               | ["'d rather bird", 'fish']               |        3.31288 | 🐥♓            |
| Tom got a small piece of pie.           |           0.751 | [0.268, 1.0, 0.985]        | ['tom got', 'small', 'piec pie']         |        4.96969 | 💆🚼🍏          |
| The lake is a long way from here.       |           0.714 | [0.482, 0.946]             | ['lake long', 'way']                     |        1.99059 | 🚤🌌            |
| Rock music approaches at high velocity. |           0.905 | [1.0, 1.0, 0.714]          | ['rock', 'music', 'approach high veloc'] |        5.00762 | 🤘🎻🎩          |