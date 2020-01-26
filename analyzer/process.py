import json
import numpy as np
from pathlib import Path
from typing import Union
from nltk import word_tokenize
from tqdm import tqdm

from .dataset import Dataset
from .encoder import Encoder
from .metrics import levenshtein


class Metrics:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, key: str, value: Union[int, float, str]):
        self.metrics[key] = value

    def save_metrics(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open('w') as fp:
            json.dump(self.metrics, fp, indent=2)


class Cache(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class DatasetProcessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset.load()
        self.encoder = Encoder(dataset.get_vocab())
        self.metrics = Metrics()
        self.cache = Cache()

    def _precompute_word_levenshtein(self):
        print('Precomputing levenshtein distance for words...')
        self.cache.word_ld = np.array(list(map(
            lambda sents: levenshtein(self.encoder.encode_words(sents[0]), self.encoder.encode_words(sents[1])),
            tqdm(self.dataset)
        )), dtype=np.int)

    def _precompute_char_levenshtein(self):
        print('Precomputing levenshtein distance for chars...')
        self.cache.char_ld = np.array(list(map(
            lambda sents: levenshtein(self.encoder.encode_chars(sents[0]), self.encoder.encode_chars(sents[1])),
            tqdm(self.dataset)
        )), dtype=np.int)

    def _compute_mean_word_levenshtein(self):
        print('Computing mean word levenshtein distance per sentence...')
        sum_distance = sum(tqdm(self.cache.word_ld))
        mean_distance = sum_distance / len(self.dataset)
        self.metrics.add_metric('mean word LD', mean_distance)

    def _compute_mean_char_levenshtein(self):
        print('Computing mean char levenshtein distance per sentence...')
        sum_distance = sum(tqdm(self.cache.char_ld))
        mean_distance = sum_distance / len(self.dataset)
        self.metrics.add_metric('mean char LD', mean_distance)

    def _compute_mean_word_length(self):
        print('Computing mean words per sentence...')
        total_words = sum(map(lambda sent: len(word_tokenize(sent)), tqdm(self.dataset.get_original_sent())))
        mean_words = total_words / len(self.dataset)
        self.metrics.add_metric('mean words per sent.', mean_words)

    def _compute_mean_char_length(self):
        print('Computing mean chars per sentence...')
        total_chars = sum(map(len, tqdm(self.dataset.get_original_sent())))
        mean_chars = total_chars / len(self.dataset)
        self.metrics.add_metric('mean chars per sent.', mean_chars)

    def _compute_sent_amount(self):
        print('Computing total amount of sentences...')
        self.metrics.add_metric('# sents.', len(self.dataset))

    def _compute_changed_sent_ratio(self):
        print('Computing changed sentences ratio...')
        total_changed = sum(tqdm(self.cache.word_ld > 0))
        changed_ratio = total_changed / len(self.dataset)
        self.metrics.add_metric('changed sents. ratio', changed_ratio)

    def _compute_sents_with_one_change_ratio(self):
        print('Computing ratio of sentences with one word changed...')
        changed = sum(tqdm(self.cache.word_ld == 1))
        changed_ratio = changed / len(self.dataset)
        self.metrics.add_metric('one word changed sents. ratio', changed_ratio)

    def _compute_sents_with_two_changes_ratio(self):
        print('Computing ratio of sentences with two words changed...')
        changed = sum(tqdm(self.cache.word_ld == 2))
        changed_ratio = changed / len(self.dataset)
        self.metrics.add_metric('two words changed sents. ratio', changed_ratio)

    def _save_metrics(self):
        path = Path('resources', 'metrics', f'{self.dataset.get_dataset_name()}.json')
        print(f'Saving metrics to {path}')
        self.metrics.save_metrics(path)

    def compute_metrics(self):
        self._precompute_word_levenshtein()
        self._precompute_char_levenshtein()
        print()

        self._compute_sent_amount()
        self._compute_mean_word_levenshtein()
        self._compute_mean_char_levenshtein()
        self._compute_mean_word_length()
        self._compute_mean_char_length()
        self._compute_changed_sent_ratio()
        self._compute_sents_with_one_change_ratio()
        self._compute_sents_with_two_changes_ratio()
        print()

        self._save_metrics()
