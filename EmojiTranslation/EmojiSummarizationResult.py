from dataclasses import dataclass
from dataclasses import field
from typing import List


@dataclass
class EmojiSummarizationResult:
    """
    "Struct" for keeping track of an Emoji Summarization result

    Data Members:
        emojis(str): String of emojis that represent the summarization
        n_grams(List[str]): List of variable length n-grams that each
                            emoji represents
        uncertainty_scores(List[float]): List of the cosine distance
                                         between each n_gram and emoji
        time_elapsed(float): How long it took to complete the summary
    """
    emojis: str = ""
    emojis_n_grams: str = field(default_factory=list)
    n_grams: List[str] = field(default_factory=list)
    uncertainty_scores: List[float] = field(default_factory=list)
    elapsed_time: float = 0
