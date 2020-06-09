import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from nltk import word_tokenize
from tqdm import tqdm

import plotly.offline as py
import plotly.graph_objects as go

from .dataset import Dataset
from .encoder import Encoder
from .metrics import levenshtein, levenshtein_sid, GPT2PerplexityScorer


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


class Histogram:
    def __init__(self):
        self.values = []

    def add_value(self, value: float):
        self.values.append(value)

    def save_histogram(self, save_path: Path):
        values = np.array(self.values) - 12
        fig = go.Figure(data=[go.Histogram(x=values, histnorm='probability', nbinsx=100)])
        py.plot(fig, filename=str(save_path))


class ErrorCorrectionDetector:
    def __init__(self, encoder: Encoder,
                 save_error_corrections: bool = False,
                 save_non_error_corrections: bool = False):
        self._encoder = encoder
        self._save_error_corrections = save_error_corrections
        self._save_non_error_corrections = save_non_error_corrections
        self._error_corrections = []
        self._non_error_corrections = []

    def get_error_corrections_count(self, sent1: str, sent2: str, mask1: np.ndarray, mask2: np.ndarray) -> int:
        sent1 = word_tokenize(sent1)
        sent2 = word_tokenize(sent2)

        count = 0
        i, j = 0, 0

        while i < len(mask1) and j < len(mask2):
            while i < len(mask1) and mask1[i] == 0:
                i += 1
            while j < len(mask2) and mask2[j] == 0:
                j += 1
            if i < len(mask1) and j < len(mask2):
                if sent1[i].isalpha() and sent2[j].isalpha():
                    word1 = self._encoder.encode_chars(sent1[i])
                    word2 = self._encoder.encode_chars(sent2[j])
                    ld = levenshtein(word1, word2)
                    threshold = np.log2(max(len(word1), len(word2)))
                    if ld <= threshold + 0.1:
                        if self._save_error_corrections:
                            self._error_corrections.append((sent1[i], sent2[j]))
                        count += 1
                    else:
                        if self._save_non_error_corrections:
                            self._non_error_corrections.append((sent1[i], sent2[j]))
                i += 1
                j += 1

        return count

    def save_error_corrections(self, path: Path):
        data = np.array(self._error_corrections, dtype=np.str)
        df = pd.DataFrame(data=data, index=None, columns=['source', 'target'])
        df.to_csv(path, sep='\t', index=False)

    def save_non_error_corrections(self, path: Path):
        data = np.array(self._non_error_corrections, dtype=np.str)
        df = pd.DataFrame(data=data, index=None, columns=['source', 'target'])
        df.to_csv(path, sep='\t', index=False)


class Cache(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class DatasetProcessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset.load()
        self.encoder = Encoder(dataset.get_vocab())
        self.error_correction_detector = ErrorCorrectionDetector(self.encoder, True, True)

        self.metrics = Metrics()
        self.metrics_dir = Path('resources', 'metrics', self.dataset.get_dataset_name())

        self.cache = Cache()

        self.perplexity_scorer = GPT2PerplexityScorer()
        self.perplexity_hist = Histogram()

    def _precompute_word_sid(self):
        print('Precomputing substitutions, insertions, deletions...')
        self.cache.word_sid = list(map(
            lambda sents: levenshtein_sid(self.encoder.encode_words(sents[0]), self.encoder.encode_words(sents[1])),
            tqdm(self.dataset)
        ))

    def _precompute_char_ld(self):
        print('Precomputing levenshtein distance for chars...')
        self.cache.char_ld = np.array(list(map(
            lambda sents: levenshtein(self.encoder.encode_chars(sents[0]), self.encoder.encode_chars(sents[1])),
            tqdm(self.dataset)
        )), dtype=np.int)

    def _compute_mean_word_levenshtein(self):
        print('Computing mean word levenshtein distance per sentence...')
        sum_distance = sum(sum(sid[:3]) for sid in self.cache.word_sid)
        mean_distance = sum_distance / len(self.dataset)
        self.metrics.add_metric('mean word LD', mean_distance)

    def _compute_mean_char_levenshtein(self):
        print('Computing mean char levenshtein distance per sentence...')
        sum_distance = sum(self.cache.char_ld)
        mean_distance = sum_distance / len(self.dataset)
        self.metrics.add_metric('mean char LD', mean_distance)

    def _compute_mean_word_length(self):
        print('Computing mean words per sentence...')
        total_words = sum(map(lambda sent: len(word_tokenize(sent)), self.dataset.get_original_sent()))
        mean_words = total_words / len(self.dataset)
        self.metrics.add_metric('mean words per sent.', mean_words)

    def _compute_mean_char_length(self):
        print('Computing mean chars per sentence...')
        total_chars = sum(map(len, self.dataset.get_original_sent()))
        mean_chars = total_chars / len(self.dataset)
        self.metrics.add_metric('mean chars per sent.', mean_chars)

    def _compute_sent_amount(self):
        print('Computing total amount of sentences...')
        self.metrics.add_metric('# sents.', len(self.dataset))

    def _compute_changed_sent_ratio(self):
        print('Computing changed sentences ratio...')
        total_changed = sum(sum(sid[:3]) != 0 for sid in self.cache.word_sid)
        changed_ratio = total_changed / len(self.dataset)
        self.metrics.add_metric('changed sents. ratio', changed_ratio)

    def _compute_sents_with_one_change_ratio(self):
        print('Computing ratio of sentences with one word changed...')
        changed = sum(sum(sid[:3]) == 1 for sid in self.cache.word_sid)
        changed_ratio = changed / len(self.dataset)
        self.metrics.add_metric('one word changed sents. ratio', changed_ratio)

    def _compute_sents_with_two_changes_ratio(self):
        print('Computing ratio of sentences with two words changed...')
        changed = sum(sum(sid[:3]) == 2 for sid in self.cache.word_sid)
        changed_ratio = changed / len(self.dataset)
        self.metrics.add_metric('two words changed sents. ratio', changed_ratio)

    def _compute_mean_sid(self):
        print('Computing mean substitutions, insertions and deletions...')
        substitutions = sum(sid[0] for sid in self.cache.word_sid)
        insertions = sum(sid[1] for sid in self.cache.word_sid)
        deletions = sum(sid[2] for sid in self.cache.word_sid)

        mean_substitutions = substitutions / len(self.dataset)
        mean_insertions = insertions / len(self.dataset)
        mean_deletions = deletions / len(self.dataset)

        self.metrics.add_metric('mean word substitutions', mean_substitutions)
        self.metrics.add_metric('mean word insertions', mean_insertions)
        self.metrics.add_metric('mean word deletions', mean_deletions)

    def _compute_sid_sentence_ratio(self):
        print('Computing ratio of sentences with substitutions, insertions and deletions...')
        only_substitutions = sum(sid[0] != 0 and sid[1] == 0 and sid[2] == 0 for sid in self.cache.word_sid)
        substitutions = sum(sid[0] != 0 for sid in self.cache.word_sid)
        insertions = sum(sid[1] != 0 for sid in self.cache.word_sid)
        deletions = sum(sid[2] != 0 for sid in self.cache.word_sid)

        ratio_only_substitutions = only_substitutions / len(self.dataset)
        ratio_substitutions = substitutions / len(self.dataset)
        ratio_insertions = insertions / len(self.dataset)
        ratio_deletions = deletions / len(self.dataset)

        self.metrics.add_metric('ratio of sentences only with substitutions', ratio_only_substitutions)
        self.metrics.add_metric('ratio of sentences with substitutions', ratio_substitutions)
        self.metrics.add_metric('ratio of sentences with insertions', ratio_insertions)
        self.metrics.add_metric('ratio of sentences with deletions', ratio_deletions)

    def _compute_error_corrections(self):
        print('Computing ratio of substitutions which are error corrections...')

        error_corrections, substitutions = 0, 0
        for (sent1, sent2), sid in tqdm(zip(self.dataset, self.cache.word_sid)):
            flag = sid[0] != 0 and sid[1] == 0 and sid[2] == 0
            if not flag:
                continue
            substitutions += sid[0]
            mask1, mask2 = sid[3], sid[4]
            error_corrections += self.error_correction_detector.get_error_corrections_count(sent1, sent2, mask1, mask2)

        ratio_error_corrections = error_corrections / substitutions
        self.metrics.add_metric('ratio of error corrections among substitutions', ratio_error_corrections)

    def _compute_perplexity_change(self):
        print('Computing perplexity change under GPT-2 model')

        for sent1, sent2 in tqdm(self.dataset):
            try:
                ppl1 = self.perplexity_scorer.evaluate_ppl(self.perplexity_scorer.tokenize_sent(sent1))
                ppl2 = self.perplexity_scorer.evaluate_ppl(self.perplexity_scorer.tokenize_sent(sent2))
                diff = ppl2 - ppl1
                if abs(diff) > 1000 or np.isnan(diff):
                    continue
                self.perplexity_hist.add_value(diff)
            except:
                pass

        ppl = np.array(self.perplexity_hist.values)
        mean_ppl_change = float(ppl.mean())
        median_ppl_change = float(np.median(ppl))

        self.metrics.add_metric('mean perplexity change', mean_ppl_change)
        self.metrics.add_metric('median perplexity change', median_ppl_change)

    def _save_metrics(self):
        path = self.metrics_dir / f'{self.dataset.get_dataset_name()}.json'
        print(f'Saving metrics to {path}')
        self.metrics.save_metrics(path)

    def _save_error_corrections(self):
        path = self.metrics_dir / 'corrections.tsv'
        self.error_correction_detector.save_error_corrections(path)

    def _save_non_error_corrections(self):
        path = self.metrics_dir / 'non-corrections.tsv'
        self.error_correction_detector.save_non_error_corrections(path)

    def _save_ppl_histogram(self):
        path = self.metrics_dir / 'ppl_histogram.html'
        self.perplexity_hist.save_histogram(path)

    def compute_metrics(self):
        self._precompute_word_sid()
        self._precompute_char_ld()
        print()

        self._compute_sent_amount()
        self._compute_mean_word_levenshtein()
        self._compute_mean_char_levenshtein()
        self._compute_mean_word_length()
        self._compute_mean_char_length()
        self._compute_changed_sent_ratio()
        self._compute_sents_with_one_change_ratio()
        self._compute_sents_with_two_changes_ratio()
        self._compute_mean_sid()
        self._compute_sid_sentence_ratio()
        self._compute_error_corrections()
        # self._compute_perplexity_change()
        print()

        self._save_metrics()
        self._save_error_corrections()
        self._save_non_error_corrections()
        # self._save_ppl_histogram()

    def extract_edits(self):
        self._precompute_word_sid()
        self._extract_substitutions()
        self._extract_insertions()
        self._extract_deletions()

    def _extract_substitutions(self):
        sentences = []
        print('Extracting substitutions')

        for i in tqdm(range(len(self.dataset))):
            sid = self.cache.word_sid[i]
            sent1, sent2 = self.dataset[i]

            if sid[0] != 0 and sid[1] == 0 and sid[2] == 0:
                words1, words2 = [], []
                for j, word in enumerate(word_tokenize(sent1)):
                    if sid[3][j] == 1:
                        words1.append(f'<sub>{word}</sub>')
                    else:
                        words1.append(word)
                for j, word in enumerate(word_tokenize(sent2)):
                    if sid[4][j] == 1:
                        words2.append(f'<sub>{word}</sub>')
                    else:
                        words2.append(word)

                sent1 = ' '.join(words1)
                sent2 = ' '.join(words2)
                sentences.append((sent1, sent2))

        path = self.metrics_dir / 'substitutions.txt'
        with path.open('w') as output:
            for sent1, sent2 in sentences:
                output.write(sent1 + '\n' + sent2 + '\n\n')

    def _extract_insertions(self):
        sentences = []
        print('Extracting insertions')

        for i in tqdm(range(len(self.dataset))):
            sid = self.cache.word_sid[i]
            sent1, sent2 = self.dataset[i]

            if sid[1] != 0:
                words1, words2 = [], []
                for j, word in enumerate(word_tokenize(sent1)):
                    if sid[3][j] == 1:
                        words1.append(f'<ins>{word}</ins>')
                    else:
                        words1.append(word)
                for j, word in enumerate(word_tokenize(sent2)):
                    if sid[4][j] == 1:
                        words2.append(f'<ins>{word}</ins>')
                    else:
                        words2.append(word)

                sent1 = ' '.join(words1)
                sent2 = ' '.join(words2)
                sentences.append((sent1, sent2))

        path = self.metrics_dir / 'insertions.txt'
        with path.open('w') as output:
            for sent1, sent2 in sentences:
                output.write(sent1 + '\n' + sent2 + '\n\n')

    def _extract_deletions(self):
        sentences = []
        print('Extracting deletions')

        for i in tqdm(range(len(self.dataset))):
            sid = self.cache.word_sid[i]
            sent1, sent2 = self.dataset[i]

            if sid[1] == 0 and sid[2] != 0:
                words1, words2 = [], []
                for j, word in enumerate(word_tokenize(sent1)):
                    if sid[3][j] == 1:
                        words1.append(f'<del>{word}</del>')
                    else:
                        words1.append(word)
                for j, word in enumerate(word_tokenize(sent2)):
                    if sid[4][j] == 1:
                        words2.append(f'<del>{word}</del>')
                    else:
                        words2.append(word)

                sent1 = ' '.join(words1)
                sent2 = ' '.join(words2)
                sentences.append((sent1, sent2))

        path = self.metrics_dir / 'deletions.txt'
        with path.open('w') as output:
            for sent1, sent2 in sentences:
                output.write(sent1 + '\n' + sent2 + '\n\n')
