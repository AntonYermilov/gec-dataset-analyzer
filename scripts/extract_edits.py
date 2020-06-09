from metric_analyzer.dataset import Dataset
from metric_analyzer.encoder import Encoder
from metric_analyzer.metrics import levenshtein_sid

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_edit(w1, w2, m1, m2, e, encoder):
    s1, s2 = [], []

    for i in w1[m1 == e]:
        s1.append(encoder.inv_vocab[i])
    for i in w2[m2 == e]:
        s2.append(encoder.inv_vocab[i])

    return ' '.join(s1), ' '.join(s2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    args = parser.parse_args()

    dataset = Dataset(args.dataset, only_edited=True, sample_rate=1.0)
    dataset.load(load_vocab=True)

    encoder = Encoder(dataset.get_vocab())

    edits = []

    for j, (sent1, sent2) in enumerate(dataset):
        words1 = encoder.encode_words(sent1)
        words2 = encoder.encode_words(sent2)
        s, i, d, m1, m2 = levenshtein_sid(words1, words2)

        max_edits = max(m1.max(), m2.max())
        for e in range(1, max_edits + 1):
            edits.append(extract_edit(words1, words2, m1, m2, e, encoder))

    print(f'edits size: {len(edits)}')

    df = pd.DataFrame(data=edits,
                      columns=['source_phrase', 'target_phrase'])
    df.to_csv(Path('edits.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
