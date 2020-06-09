import numpy as np
from nltk import word_tokenize
from typing import Dict


class Encoder:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = np.array(['' for _ in range(len(vocab))], dtype=object)
        for k, v in self.vocab.items():
            self.inv_vocab[v] = k

    def encode_words(self, sentence: str) -> np.ndarray:
        encoded_sent = list(map(lambda word: self.vocab[word], word_tokenize(sentence)))
        encoded_sent = np.array(encoded_sent, dtype=np.int)
        return encoded_sent

    @staticmethod
    def encode_chars(sentence: str) -> np.ndarray:
        encoded_sent = list(map(ord, sentence))
        encoded_sent = np.array(encoded_sent, dtype=np.int)
        return encoded_sent
