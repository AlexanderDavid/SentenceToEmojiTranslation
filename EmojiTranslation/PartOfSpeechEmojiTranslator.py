from AbstractEmojiTranslator import AbstractEmojiTranslator
from EmojiSummarizationResult import EmojiSummarizationResult
from typing import List, Callable
import spacy


class PartOfSpeechEmojiTranslator(AbstractEmojiTranslator):

    def pos_n_gram(self, sentence: str, keep_stop_words: bool=False) -> List[str]:
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
                good_children = [child for child in current_node.children if len(
                    list(child.children)) > 0]
                bad_children = [(child.orth_, child.i)
                                for child in current_node.children if child not in good_children]
                pos_tagged_n_grams.append(backlog)
                pos_tagged_n_grams.append(bad_children)
                return Tree(backlog, [Tree(bad_children, [])] + [to_nltk_tree(child) for child in good_children])
            else:
                pos_tagged_n_grams.append(backlog)
                return Tree(backlog, [])

        def strip_nothing_unigrams(n_grams):
            return [n_gram for n_gram in n_grams if not (len(n_gram.split(" ")) == 1 and n_gram.split(" ")[0] in stopword)]

    #     query = " ".join([word for word in sentence.split() if word not in stopword or keep_stop_words])
        doc = self.nlp(sentence)
        to_nltk_tree(list(doc.sents)[0].root)
        # print(nltk_tree)

        sort_inner = [sorted(nltk_child, key=lambda x: x[1])
                      for nltk_child in pos_tagged_n_grams]

        nltk_averages = []
        for nltk_child in sort_inner:
            if nltk_child == []:
                continue
            nltk_averages.append((nltk_child, max(x[1] for x in nltk_child)))

        sorted_outer = list(sorted(nltk_averages, key=lambda x: x[1]))

        n_grams = []
        for nltk_average in sorted_outer:
            n_grams.append(" ".join(word[0] for word in nltk_average[0]))

        return n_grams

    def summarize(self, sent: str, keep_stop_words: bool=False,
                  lemma_func: Callable[[str], str]=lambda x: x) -> EmojiSummarizationResult:
        """
        Summarize a sentence using POS n-gram chunking

        Args:
            sent(str): Sentence to summarize
            keep_stop_words(bool, Optional): Flag to keep the stop words when
                                             cleaning the sentence and n-grams
            lemma_func(Callable[[str], str], Optional): Function to use to
                                                        lemmatize the sentence

        Rets:
            EmojiSummarizationResult: Result of the emoji summarization
        """

        # Clean the sentence
        sent = self.clean_sentence(
            sent, keep_stop_words=True, lemma_func=lemma_func)

        # Get the n-grams using the part of speech tagging
        pos_n_grams = self.pos_n_gram(sent, keep_stop_words=keep_stop_words)

        # Clean the n_grams
        n_grams = self.clean_n_gram(pos_n_grams)

        # Create an Emoji Summarization Result
        esr = EmojiSummarizationResult()

        # Translate every n_gram in that n-gram sequence
        for n_gram in n_grams:
            # Get the closest emoji to the current n-gram
            emoji, similarity, desc = self.closest_emoji(n_gram)

            # Add the closest emoji to the sumary
            esr.emojis += emoji
            esr.emojis_n_grams.append(desc)
            esr.n_grams.append(n_gram)
            esr.uncertainty_scores.append(similarity)

        # Return the summary
        return esr


if __name__ == '__main__':
    e = PartOfSpeechEmojiTranslator("./data/emoji_joined.txt",
                                    "./data/wiki_unigrams.bin", lambda x: x)
    while True:
        print(e.summarize(input(">")))
