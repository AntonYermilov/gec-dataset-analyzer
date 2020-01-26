from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import pandas as pd


class DatasetPreprocessor(ABC):
    def __init__(self, dataset_path: Path, preprocessor_name: str):
        """
        Initializes dataset preprocessor with the dataset path
        :param dataset_path: a path to the specific dataset
        :param preprocessor_name: the name of a preprocessor, which specifies the directory dataset would be saved to
        """
        self.dataset_path: Path = dataset_path
        self.save_path: Path = Path('resources', 'preprocessed_datasets') / preprocessor_name / 'dataset.tsv'

    def preprocess(self):
        """
        Preprocess dataset and save it to the save_file in tsv format.
        All datasets are saved as tables with the following columns: sent_id, original_sent, edited_sent
        """
        indices, original_sent, edited_sent = self.extract_sentences()
        assert len(indices) == len(original_sent) == len(edited_sent)

        df = pd.DataFrame(list(zip(indices, original_sent, edited_sent)),
                          columns=['sent_id', 'original_sent', 'edited_sent'])

        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        df.to_csv(self.save_path, sep='\t', index=False)

    @abstractmethod
    def extract_sentences(self) -> Tuple[List[int], List[str], List[str]]:
        """
        Reads dataset and extracts original and edited sentences
        :return: a list of original sentences indices and parallel lists of original and edited sentences
        """
        pass
