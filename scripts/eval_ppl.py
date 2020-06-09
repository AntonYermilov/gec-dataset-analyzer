from metric_analyzer.metrics import GPT2PerplexityScorer
from metric_analyzer.metrics import levenshtein_sid
from metric_analyzer import Dataset

import json
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='seed for random sampling')
    parser.add_argument('--count', type=int, default=-1, help='number of samples to produce')
    parser.add_argument('--output-summary', type=str, required=True, help='json path to save summary')
    parser.add_argument('--output-full', type=str, required=True, help='json path to save full stats')
    args = parser.parse_args()

    random = np.random.RandomState(seed=args.seed)
    samples_count = args.count

    scorer = GPT2PerplexityScorer()

    dataset = Dataset(args.dataset, only_edited=True, sample_rate=1.0)
    dataset.load(load_vocab=False)

    if samples_count == -1:
        samples_count = len(dataset)

    if args.dataset == 'papeeria':
        indices0 = np.arange(5000)
        indices1 = 8000 + np.arange(12000)
        random.shuffle(indices0)
        random.shuffle(indices1)
        indices = np.hstack([indices0[:args.count * 3 // 4], indices1])
    else:
        indices = np.arange(len(dataset))
        random.shuffle(indices)

    stats = []

    for i in tqdm(indices):
        sent1, sent2 = dataset[i]

        tok1 = scorer.tokenize_sent(sent1)
        tok2 = scorer.tokenize_sent(sent2)

        if tok1 == tok2:
            continue

        if min(len(tok1), len(tok2)) < 5:
            continue

        loss1 = scorer.evaluate_loss(tok1)
        loss2 = scorer.evaluate_loss(tok2)

        ppl1 = np.exp(loss1 / len(tok1))
        ppl2 = np.exp(loss2 / len(tok2))

        if np.isnan(ppl1) or np.isnan(ppl2):
            continue

        S, I, D, m1, m2 = levenshtein_sid(np.array(tok1), np.array(tok2))
        LD = S + I + D

        stats.append({
            'sent1': sent1,
            'sent2': sent2,
            'ppl1': ppl1,
            'ppl2': ppl2,
            'ppl_diff': ppl2 - ppl1,
            'len1': len(tok1),
            'len2': len(tok2),
            'ld': LD,
            'norm_ppl_diff': 1.0 * (ppl2 - ppl1) / LD,
            'loss1': loss1,
            'loss2': loss2,
            'norm_loss_diff': (loss2 - loss1) / LD,
            'norm2_loss_diff': (loss2 / len(tok2) - loss1 / len(tok1)) / LD
        })

        samples_count -= 1
        if samples_count == 0:
            break

    ppl1_values = np.array([stat['ppl1'] for stat in stats])
    ppl2_values = np.array([stat['ppl2'] for stat in stats])
    ppl_diffs = np.array([stat['ppl_diff'] for stat in stats])
    norm_ppl_diffs = np.array([stat['norm_ppl_diff'] for stat in stats])
    lds = np.array([stat['ld'] for stat in stats])
    loss1_values = np.array([stat['loss1'] for stat in stats])
    loss2_values = np.array([stat['loss2'] for stat in stats])
    norm_loss_diffs = np.array([stat['norm_loss_diff'] for stat in stats])
    norm2_loss_diffs = np.array([stat['norm2_loss_diff'] for stat in stats])

    summary = {
        'count': len(stats),
        'mean_ppl1': ppl1_values.mean(),
        'median_ppl1': np.median(ppl1_values),
        'mean_ppl2': ppl2_values.mean(),
        'median_ppl2': np.median(ppl2_values),
        'mean_ppl_diff': ppl_diffs.mean(),
        'median_ppl_diff': np.median(ppl_diffs),
        'mean_norm_ppl_diff': norm_ppl_diffs.mean(),
        'median_norm_ppl_diff': np.median(norm_ppl_diffs),
        'mean_ld': lds.mean(),
        'median_ld': np.median(lds),
        'mean_loss1': loss1_values.mean(),
        'median_loss1': np.median(loss1_values),
        'mean_loss2': loss2_values.mean(),
        'median_loss2': np.median(loss2_values),
        'mean_norm_loss_diff': norm_loss_diffs.mean(),
        'median_norm_loss_diff': np.median(norm_loss_diffs),
        'mean_norm2_loss_diff': norm2_loss_diffs.mean(),
        'median_norm2_loss_diff': np.median(norm2_loss_diffs),
    }

    with open(args.output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(args.output_full, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
