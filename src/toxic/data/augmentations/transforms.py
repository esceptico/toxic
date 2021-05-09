import random
import string
from typing import Any, List, Callable


class BaseTransform:
    def __init__(self, p):
        self.p = p

    def apply(self, data, **params) -> Any:
        raise NotImplementedError

    def __call__(self, data, **params):
        if random.random() < self.p:
            return self.apply(data, **params)
        return data


class CutOutLettersTransform(BaseTransform):
    """Remove random letters from data"""
    def __init__(self, p=0.5, cutout_probability=0.1):
        super().__init__(p=p)
        self.cutout_probability = cutout_probability

    def apply(self, data: str, **params):
        letters = []
        for letter in data:
            if random.random() >= self.cutout_probability:
                letters.append(letter)
        return ''.join(letters)


class AppendRandomLettersTransform(BaseTransform):
    """Appends random letters into random place of string"""
    def __init__(self, p=0.5, append_probability=0.1, char_set: List[str] = None):
        super().__init__(p=p)
        self.append_probability = append_probability
        self.char_set = char_set

    def apply(self, data: str, **params):
        char_set = self.char_set if self.char_set is not None else list(data + string.punctuation)

        letters = []
        for letter in data:
            letters.append(letter)
            if random.random() < self.append_probability:
                random_letter = random.choice(char_set)
                letters.append(random_letter)
        return ''.join(letters)


class SwapWordsTransform(BaseTransform):
    """Swap random words"""
    def __init__(self, p=0.5, swap_probability=0.1, tokenize_fn: Callable[[str], List[str]] = None):
        super().__init__(p=p)
        self.swap_probability = swap_probability
        self.tokenize_fn = tokenize_fn if tokenize_fn is not None else self._default_tokenize_fn

    def _default_tokenize_fn(self, text):
        return text.split()

    def apply(self, data: str, **params):
        words = self.tokenize_fn(data)
        words_count = len(words)

        for idx, word in enumerate(words):
            if random.random() < self.swap_probability:
                rand_idx = random.randrange(words_count)
                words[idx], words[rand_idx] = words[rand_idx], words[idx]
        return ' '.join(words)


class RandomUnknownTransform(BaseTransform):
    def __init__(self, p=0.5, unk_token='<unk>'):
        super().__init__(p=p)
        self.unk_token = unk_token

    def apply(self, data: str, **params):
        length = int(0.2 * len(data))
        start = random.randint(0, len(data) - length)
        end = start + length
        part_to_replace = data[start:end]
        return data.replace(part_to_replace, self.unk_token)


class RemoveWhitespaceTransform(BaseTransform):
    def __init__(self, p=0.5, remove_probability=0.5):
        super().__init__(p=p)
        self.remove_probability = remove_probability

    def apply(self, data: str, **params):
        letters = []
        for char in data:
            if char == ' ':
                if random.random() < self.remove_probability:
                    continue
                letters.append(char)
        return ''.join(letters)
