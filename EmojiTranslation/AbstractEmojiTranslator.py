from abc import ABC, abstractmethod


class AbstractEmojiTranslator(ABC):
    @abstractmethod
    def summarize(self, sent: str) -> str:
        pass


class B(AbstractEmojiTranslator):
    def __init__(self):
        self.test = True


b = B()
b.summarize()
