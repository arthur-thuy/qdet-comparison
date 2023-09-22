import numpy as np
import os
import pandas as pd

from qdet_utils.constants import RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from qdet_utils.constants import TF_Q_ALL, TF_Q_ONLY
from qdet_utils.evaluation import METRICS

LIST_TF_ENCODINGS = [TF_Q_ALL]
TF_MODELS = ["transformer"]
LIST_DATASET_NAMES = [RACE_PP_4K]
RANDOM_SEEDS = [123]


def main():
    # work separately on each dataset
    for dataset in LIST_DATASET_NAMES:

        output_df_train = pd.DataFrame()
        output_df_test = pd.DataFrame()

        # transformers
        for model in TF_MODELS:
            for encoding in LIST_TF_ENCODINGS:
                if encoding != TF_Q_ONLY and dataset == AM:
                    continue
                new_row_dict_train, new_row_dict_test = get_dict_results_for_model(dataset, f'{model}_{encoding}')
                output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
                output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        output_df_train.to_csv(os.path.join(OUTPUT_DIR, dataset, 'general_evaluation_train.csv'), index=False)
        output_df_test.to_csv(os.path.join(OUTPUT_DIR, dataset, 'general_evaluation_test.csv'), index=False)


def get_dict_results_for_model(dataset, config):
    new_row_dict_train = dict()
    new_row_dict_test = dict()
    for random_seed in RANDOM_SEEDS:
        local_df = pd.read_csv(os.path.join(OUTPUT_DIR, dataset, 'seed_' + str(random_seed), f'eval_metrics_{config}.csv'))
        for metric in METRICS:
            new_row_dict_train[f'train_seed_{random_seed}_{metric}'] = local_df[f'train_{metric}'].values[0]
            new_row_dict_test[f'test_seed_{random_seed}_{metric}'] = local_df[f'test_{metric}'].values[0]
    for metric in METRICS:
        list_train_results = [new_row_dict_train[f'train_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict_train[f'train_{metric}_mean'] = np.mean(list_train_results)
        new_row_dict_train[f'train_{metric}_std'] = np.std(list_train_results)

        list_test_results = [new_row_dict_test[f'test_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict_test[f'test_{metric}_mean'] = np.mean(list_test_results)
        new_row_dict_test[f'test_{metric}_std'] = np.std(list_test_results)
    new_row_dict_train['model'] = config
    new_row_dict_test['model'] = config
    return new_row_dict_train, new_row_dict_test


main()
