from pathlib import Path
from typing import List, Tuple
from .base import DatasetPreprocessor


class JFLEGPreprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path: Path):
        """
        Initializes preprocessor for JFLEG dataset (https://github.com/keisks/jfleg)
        :param dataset_path: a path to the directory with JFLEG sentences (i.e. {dev,test}.{src,ref[0-3]})
        """
        super().__init__(dataset_path, 'jfleg')

    def extract_sentences(self) -> Tuple[List[int], List[str], List[str]]:
        files = {}
        for dataset_file in self.dataset_path.rglob('*'):
            files[dataset_file.name] = dataset_file.absolute()

        indices, original_sent, edited_sent = list(), list(), list()
        ind_start = 0
        for name in ['dev', 'test']:
            for ind in range(4):
                with files[f'{name}.src'].open('r') as orig_fp, files[f'{name}.ref{ind}'].open('r') as edit_fp:
                    for line_num, (orig_line, edit_line) in enumerate(zip(orig_fp.readlines(), edit_fp.readlines())):
                        indices.append(ind_start + line_num)
                        original_sent.append(orig_line.strip())
                        edited_sent.append(edit_line.strip())
            with files[f'{name}.src'].open('r') as orig_fp:
                ind_start += len(orig_fp.readlines())

        return indices, original_sent, edited_sent
