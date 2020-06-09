from metric_analyzer.dataset import Dataset
from metric_analyzer.encoder import Encoder
from metric_analyzer.metrics import levenshtein_sid

import re
import numpy as np
from nltk import word_tokenize


def main():
    dataset = Dataset('papeeria', only_edited=True, sample_rate=1.0)
    dataset.load(load_vocab=True)

    encoder = Encoder(dataset.get_vocab())

    mask = np.ones(len(dataset), dtype=np.bool)
    # mask[5000:] = 0

    for j, (sent1, sent2) in enumerate(dataset):
        if mask[j] == 0:
            continue

        if min(len(sent1), len(sent2)) < 55:
            mask[j] = 0
            continue

        words1 = encoder.encode_words(sent1)
        words2 = encoder.encode_words(sent2)
        s, i, d, m1, m2 = levenshtein_sid(words1, words2)

        found = False

        for k, v in zip(words1, m1):
            if v == 0:
                continue

            word = encoder.inv_vocab[k]
            if word in {'MATH', 'CITE', 'FIGURE', 'TABLE', 'REF'} or \
                    word.isnumeric() or \
                    not re.fullmatch(r"^[a-zA-Z.,?!;:'\-]+$", word):
                found = True
                break

        if found:
            mask[j] = 0
            continue

        for k, v in zip(words2, m2):
            if v == 0:
                continue

            word = encoder.inv_vocab[k]
            if word in {'MATH', 'CITE', 'FIGURE', 'TABLE', 'REF'} or \
                    word.isnumeric() or \
                    not re.fullmatch(r"^[a-zA-Z.,?!;:'\-]+$", word):
                found = True
                break

        if found:
            mask[j] = 0
            continue

        if m1[0] != 0 or m2[0] != 0:
            mask[j] = 0
            continue

        if m1[-1] != 0 and not encoder.inv_vocab[words1[-1]].isalnum():
            mask[j] = 0
            continue
        if m2[-1] != 0 and not encoder.inv_vocab[words2[-1]].isalnum():
            mask[j] = 0
            continue

        pass

    dataset._dataset = dataset._dataset[mask]

    print('sent_id\toriginal_sent\tedited_sent')
    for i, (sent1, sent2) in enumerate(dataset):
        sent1 = ' '.join(filter(len, word_tokenize(sent1)))
        sent2 = ' '.join(filter(len, word_tokenize(sent2)))
        print(str(i) + '\t' + sent1.strip() + '\t' + sent2.strip())


if __name__ == '__main__':
    main()
