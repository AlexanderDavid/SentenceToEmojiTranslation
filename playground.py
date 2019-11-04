from EmojiTranslation.Translators import *

if __name__ == '__main__':
    p = PartOfSpeechEmojiTranslator("./data/emoji_joined.txt",
                                    "./data/wiki_unigrams.bin", lambda x: x)
    e = ExhaustiveChunkingTranslation("./data/emoji_joined.txt",
                                      "./data/wiki_unigrams.bin", lambda x: x)
    while True:
        inp = input(">")
        print(e.summarize(input(">")))
        print(p.summarize(input(">")))
