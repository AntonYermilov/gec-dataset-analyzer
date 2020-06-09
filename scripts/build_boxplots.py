import json
from pathlib import Path

import pandas as pd
import plotly.offline as py
import plotly.express as px


def main():
    dataset_name, ppl_ratio, ld, norm_loss_delta = [], [], [], []

    for dataset in ['fce', 'lang8', 'jfleg', 'aesw', 'papeeria']:
        dataset_path = Path('resources', 'metrics', dataset, 'stats_full.json')
        stats = json.loads(dataset_path.read_text())
        for stat in stats:
            if stat['ppl2'] / stat['ppl1'] > 10:
                continue
            dataset_name.append(dataset)
            ppl_ratio.append(stat['ppl2'] / stat['ppl1'])
            ld.append(stat['ld'])
            norm_loss_delta.append(stat['norm2_loss_diff'])

    df1 = pd.DataFrame(data=list(zip(dataset_name, ppl_ratio)), columns=['dataset', 'delta(s_1, s_2)'])
    fig1 = px.box(df1, x='dataset', y='delta(s_1, s_2)')
    fig1.update_layout(yaxis={'range': [0, 7], 'dtick': 1})
    fig1.show()

    # df2 = pd.DataFrame(data=list(zip(dataset_name, ld)), columns=['dataset', 'ED(s_1, s_2)'])
    # fig2 = px.box(df2, x='dataset', y='ED(s_1, s_2)')
    # # fig2.update_layout(yaxis={'range': [0, 7.5], 'dtick': 0.25})
    # fig2.show()
    #
    # df3 = pd.DataFrame(data=list(zip(dataset_name, norm_loss_delta)), columns=['dataset', 'Delta L_N(s_1, s_2)'])
    # fig3 = px.box(df3, x='dataset', y='Delta L_N(s_1, s_2)')
    # # fig2.update_layout(yaxis={'range': [0, 7.5], 'dtick': 0.25})
    # fig3.show()


if __name__ == '__main__':
    main()
