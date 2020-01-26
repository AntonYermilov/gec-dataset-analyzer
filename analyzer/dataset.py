import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterable, Dict, Optional
from nltk import word_tokenize
from tqdm import tqdm


class DatasetNotFoundError(ValueError):
    def __init__(self, dataset_name: str):
        super().__init__()
        self.message = f'Dataset `{dataset_name}` was not found, try to generate it with dataset preprocessor'


class Dataset:
    def __init__(self, dataset_name: str):
        self._dataset_name = dataset_name
        self._dataset_path = Path('resources', 'preprocessed_datasets', dataset_name, 'dataset.tsv')
        if not self._dataset_path.exists():
            raise DatasetNotFoundError(dataset_name)

        self._dataset: Optional[pd.DataFrame] = None
        self._vocab: Optional[Dict[str, int]] = None

    def load(self):
        print(f'Loading `{self._dataset_name}` dataset')
        self._dataset = pd.read_csv(self._dataset_path, index_col=False, sep='\t').replace(np.nan, '', regex=True)
        self._vocab = set()
        for ind, row in tqdm(self._dataset.iterrows(), total=len(self._dataset)):
            for word in word_tokenize(row['original_sent']):
                self._vocab.add(word)
            for word in word_tokenize(row['edited_sent']):
                self._vocab.add(word)
        self._vocab = dict((word, ind) for ind, word in enumerate(self._vocab))
        return self

    def get_dataset_name(self) -> str:
        return self._dataset_name

    def get_original_sent(self) -> np.ndarray:
        return self._dataset['original_sent'].to_numpy()

    def get_edited_sent(self) -> np.ndarray:
        return self._dataset['edited_sent'].to_numpy()

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, ind) -> Tuple[str, str]:
        row = self._dataset.iloc[ind]
        return row['original_sent'], row['edited_sent']

    def __iter__(self) -> Iterable[Tuple[str, str]]:
        for ind, row in self._dataset.iterrows():
            yield row['original_sent'], row['edited_sent']
