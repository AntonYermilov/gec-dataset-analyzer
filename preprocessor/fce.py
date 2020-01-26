import nltk
from pathlib import Path
from typing import List, Tuple
from lxml import etree
from .base import DatasetPreprocessor


class FCEPreprocessor(DatasetPreprocessor):
    def __init__(self, dataset_path: Path):
        """
        Initializes preprocessor for FCE dataset (https://ilexir.co.uk/datasets/index.html)
        :param dataset_path: path to the directory with FCE XMLs
        """
        super().__init__(dataset_path, 'fce')

    def extract_sentences(self) -> Tuple[List[int], List[str], List[str]]:
        original_sent, edited_sent = list(), list()

        for dataset_file in self.dataset_path.rglob('*'):
            if not dataset_file.is_file() or not dataset_file.name.endswith('.xml'):
                continue

            original_paragraphs, edited_paragraphs = [], []

            with dataset_file.open('r') as inp:
                tree = etree.parse(inp)
                for elem in tree.xpath('//c'):
                    parent = elem.getparent()
                    parent.remove(elem)
                for elem in tree.xpath('//p'):
                    original_paragraphs.append(''.join(elem.xpath('.//text()')).strip())

            with dataset_file.open('r') as inp:
                tree = etree.parse(inp)
                for elem in tree.xpath('//i'):
                    parent = elem.getparent()
                    parent.remove(elem)
                for elem in tree.xpath('//p'):
                    edited_paragraphs.append(''.join(elem.xpath('.//text()')).strip())

            for original_paragraph, edited_paragraph in zip(original_paragraphs, edited_paragraphs):
                original_paragraph = nltk.sent_tokenize(original_paragraph)
                edited_paragraph = nltk.sent_tokenize(edited_paragraph)
                if len(original_paragraph) == 0 or len(edited_paragraph) == 0:
                    continue

                if len(original_paragraph) != len(edited_paragraph):
                    i, j = 1, 1
                    original_sent.append(original_paragraph[0])
                    edited_sent.append(edited_paragraph[0])
                    while i < len(original_paragraph) and j < len(edited_paragraph):
                        dist1 = abs(len(original_sent[-1]) + len(original_paragraph[i]) - len(edited_sent[-1]))
                        dist2 = abs(len(original_sent[-1]) - len(edited_sent[-1]) - len(edited_paragraph[j]))
                        dist3 = abs(len(original_paragraph[i]) - len(edited_paragraph[j]))
                        if dist3 <= min(dist1, dist2):
                            original_sent.append(original_paragraph[i])
                            edited_sent.append(edited_paragraph[j])
                            i += 1
                            j += 1
                        else:
                            if dist1 <= dist2:
                                original_sent[-1] += original_paragraph[i]
                                i += 1
                            else:
                                edited_sent[-1] += edited_paragraph[j]
                                j += 1
                    if i < len(original_paragraph):
                        original_sent[-1] += ' '.join(original_paragraph[i:])
                    if j < len(edited_paragraph):
                        edited_sent[-1] += ' '.join(edited_paragraph[j:])
                else:
                    original_sent.extend(original_paragraph)
                    edited_sent.extend(edited_paragraph)

                assert len(original_sent) == len(edited_sent)

        return list(range(len(original_sent))), original_sent, edited_sent
