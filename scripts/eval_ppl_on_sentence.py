from metric_analyzer.metrics import GPT2PerplexityScorer
from metric_analyzer.metrics import levenshtein_sid
from metric_analyzer import Dataset

import json
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True, help='sentence to evaluate')
    args = parser.parse_args()

    sentence = args.sentence
    scorer = GPT2PerplexityScorer()

    tokens = scorer.tokenize_sent(sentence)

    print(f'tokens: {scorer.tokenizer.tokenize(sentence)}')
    print(f'token_ids: {tokens}')

    prev_loss = 0
    for i in range(len(tokens)):
        loss = scorer.evaluate_loss(tokens[:i + 1])
        if not np.isnan(loss):
            print(f'ppl for {i+1} tokens: {loss}, ppl diff: {loss - prev_loss}')
            prev_loss = loss


if __name__ == '__main__':
    main()
