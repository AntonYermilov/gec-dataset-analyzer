from pathlib import Path
from typing import List, Tuple
from .base import DatasetPreprocessor


class AESWPreprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path: Path):
        """
        Initializes preprocessor for AESW dataset (http://textmining.lt/aesw/)
        :param dataset_path: a path to the directory with tokenized AESW sentences (i.e. datasets in .tok format)
        """
        super().__init__(dataset_path, 'aesw')

    def extract_sentences(self) -> Tuple[List[int], List[str], List[str]]:
        sent_ids, original_sent, edited_sent = set(), dict(), dict()

        for dataset_file in self.dataset_path.rglob('*'):
            with dataset_file.open('r') as inp:
                for line in inp.readlines():
                    tp, sent_id, text = line.split('\t')

                    tp = int(tp)
                    sent_ids.add(sent_id)

                    if sent_id not in original_sent:
                        original_sent[sent_id] = ''
                    if sent_id not in edited_sent:
                        edited_sent[sent_id] = ''
                    if tp <= 0:
                        original_sent[sent_id] += text.rstrip()
                    if tp >= 0:
                        edited_sent[sent_id] += text.rstrip()

        sent_ids = sorted(list(sent_ids))
        original_sent = [original_sent[sent_id] for sent_id in sent_ids]
        edited_sent = [edited_sent[sent_id] for sent_id in sent_ids]

        return list(range(len(original_sent))), original_sent, edited_sent
