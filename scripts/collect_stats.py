from metric_analyzer.dataset import Dataset

import numpy as np
from nltk import sent_tokenize


def main():
    dataset = Dataset('aesw', only_edited=True, sample_rate=1.0)
    dataset.load(load_vocab=False)

    lens = []
    for text in dataset.get_original_sent():
        for sent in sent_tokenize(text):
            lens.append(len(sent))
    for text in dataset.get_edited_sent():
        for sent in sent_tokenize(text):
            lens.append(len(sent))

    lens = np.array(lens)
    lens.sort()

    for p in [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        i = int(len(lens) * (1 - p))
        print(p, lens[i])


if __name__ == '__main__':
    main()
