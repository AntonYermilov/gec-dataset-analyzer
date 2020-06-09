import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.offline as py
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markup-path', type=str, help='path to the markup file')
    args = parser.parse_args()

    markup_path = Path(args.markup_path)
    df = pd.read_csv(markup_path)

    count = 0

    stat_names = np.array(['spelling', 'punctuation', 'grammar', 'semantics', 'better', 'worse', 'different', 'natural'])

    all_votes = np.zeros_like(stat_names, dtype=np.int32)
    major_agreement = np.zeros_like(stat_names, dtype=np.int32)
    total_agreement = np.zeros_like(stat_names, dtype=np.int32)

    mapping = {
        'spelling': 'Answer.edit-type.spelling',
        'punctuation': 'Answer.edit-type.punctuation',
        'grammar': 'Answer.edit-type.grammar',
        'semantics': 'Answer.edit-type.semantics',
        'better': 'Answer.qual_pos.pos',
        'worse': 'Answer.qual_neg.neg',
        'different': 'Answer.qual_diff.diff',
        'natural': 'Answer.natural.natural'
    }

    for group_name, group in df.groupby('HITId'):
        mask_maj = np.zeros_like(stat_names, dtype=np.int32)
        mask_all = np.zeros_like(stat_names, dtype=np.int32)

        for i, prop in enumerate(stat_names):
            mask_all[i] = sum(group[mapping[prop]])
            if sum(group[mapping[prop]]) >= 2:
                mask_maj[i] = 1

        mask_all[4] = 3 - mask_all[5] - mask_all[6]
        if np.all(mask_maj[4:7] == 0):
            mask_maj[4] = 1
        mask_total = mask_all == 3

        all_votes += mask_all
        major_agreement += mask_maj
        total_agreement += mask_total

        # if mask_maj[6]:
        #     print(group['Input.original_sent'].to_numpy()[0])
        #     print(group['Input.edited_sent'].to_numpy()[0])
        #     print()

    L, R = 0, 4

    layout = go.Layout(yaxis={'range': [0, 1], 'dtick': 0.2})
    figure = go.Figure(
        data=[
            # go.Bar(x=stat_names, y=total_agreement, name='total agreement'),
            go.Bar(x=stat_names[L:R], y=major_agreement[L:R] / 200, name='major agreement'),
            # go.Bar(x=stat_names, y=all_votes, name='all votes'),
        ],
        layout=layout
    )
    py.plot(figure, filename='visualize_votes.html')


if __name__ == '__main__':
    main()
