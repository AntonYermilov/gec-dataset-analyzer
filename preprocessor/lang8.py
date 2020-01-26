from pathlib import Path
from typing import List, Tuple
from .base import DatasetPreprocessor


class Lang8Preprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path: Path):
        """
        Initializes preprocessor for Lang-8 dataset (https://sites.google.com/site/naistlang8corpora/)
        :param dataset_path: a path to the directory with lang-8 entries (i.e. entries.train or entries.test)
        """
        super().__init__(dataset_path, 'lang8')

    def extract_sentences(self) -> Tuple[List[int], List[str], List[str]]:
        original_sent, edited_sent = list(), list()

        for dataset_file in self.dataset_path.rglob('*'):
            with dataset_file.open('r') as inp:
                for line in inp.readlines():
                    columns = line.split('\t')
                    if len(columns) < 5:
                        continue

                    original_sent_column = 4
                    edited_sent_column = 4 if len(columns) == 5 else 5

                    original_sent.append(columns[original_sent_column].rstrip())
                    edited_sent.append(columns[edited_sent_column].rstrip())

        return list(range(len(original_sent))), original_sent, edited_sent
