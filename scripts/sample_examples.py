from metric_analyzer.dataset import Dataset
from metric_analyzer.encoder import Encoder
from metric_analyzer.metrics import levenshtein_sid

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def tag_sentences(w1, w2, m1, m2, encoder):
    w3, w4 = [], []

    u1 = set(m1)
    u2 = set(m2)

    for i in range(len(w1)):
        if m1[i] != 0 and (i == 0 or m1[i - 1] != m1[i]):
            if m1[i] in u2:
                w3.append('<strong>')
            else:
                w3.append('<strong>')
        w3.append(encoder.inv_vocab[w1[i]])
        if m1[i] != 0 and (i + 1 == len(w1) or m1[i + 1] != m1[i]):
            if m1[i] in u2:
                w3.append('</strong>')
            else:
                w3.append('</strong>')

    for i in range(len(w2)):
        if m2[i] != 0 and (i == 0 or m2[i - 1] != m2[i]):
            if m2[i] in u1:
                w4.append('<strong>')
            else:
                w4.append('<strong>')
        w4.append(encoder.inv_vocab[w2[i]])
        if m2[i] != 0 and (i + 1 == len(w2) or m2[i + 1] != m2[i]):
            if m2[i] in u1:
                w4.append('</strong>')
            else:
                w4.append('</strong>')

    return ' '.join(w3), ' '.join(w4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='seed for random sampling')
    parser.add_argument('--count', type=int, default=100, help='number of samples to produce')
    parser.add_argument('--max-edits', type=int, default=1, help='number of allowed edits in texts')
    parser.add_argument('--max-length', type=int, default=150, help='max length of texts to sample')
    args = parser.parse_args()

    random = np.random.RandomState(seed=args.seed)
    samples_count = args.count
    max_edits = args.max_edits
    max_length = args.max_length

    dataset = Dataset('papeeria', only_edited=True, sample_rate=1.0)
    dataset.load(load_vocab=True)

    encoder = Encoder(dataset.get_vocab())

    tagged_sents = []

    for j, (sent1, sent2) in enumerate(dataset):
        if max(len(sent1), len(sent2)) > max_length:
            continue

        words1 = encoder.encode_words(sent1)
        words2 = encoder.encode_words(sent2)
        s, i, d, m1, m2 = levenshtein_sid(words1, words2)

        if m1.max() > max_edits or m2.max() > max_edits:
            continue

        tagged_sents.append(tag_sentences(words1, words2, m1, m2, encoder))

    tagged_sents = np.array(tagged_sents)
    print(f'dataset size: {len(tagged_sents)}')

    indices = np.arange(len(tagged_sents))
    random.shuffle(indices)

    df = pd.DataFrame(data=tagged_sents[indices[:samples_count]],
                      columns=['original_sent', 'edited_sent'])
    df.to_csv(Path('samples.csv'), sep=',', index=False)


if __name__ == '__main__':
    main()
