import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from EmojiTranslation.Translators import *

poset = PartOfSpeechEmojiTranslator("./data/emoji_joined.txt",
                                    "/home/alex/Projects/Exhaustive-Chunking-Emoji-Translation/data/wiki_unigrams.bin", lambda x: x)
