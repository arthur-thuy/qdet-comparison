import logging
import os
import pandas as pd

from qdet_utils.constants import DATA_DIR, DEV, TEST, TRAIN
from qdet_utils.data_utils.race import prepare_racepp_dataset
from qdet_utils.data_utils.arc import prepare_arc_dataset
from qdet_utils.data_utils.am import prepare_assistments_dataset
from qdet_utils.data_utils.mapping_text2props import convert_to_text2props_format_and_store_data
from qdet_utils.data_utils.mapping_r2de import convert_to_r2de_format_and_store_data


# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # RACE++
    logger.info("Starting preparation RACE++")
    race_data_dir = 'data/raw/RACE'
    race_c_data_dir = 'data/raw/race-c-master/data'
    dict_out_dfs = prepare_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
    # Conversion to R2DE format
    convert_to_r2de_format_and_store_data(dict_out_dfs[-1][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp')
    convert_to_r2de_format_and_store_data(dict_out_dfs[4][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_4k')
    convert_to_r2de_format_and_store_data(dict_out_dfs[8][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_8k')
    convert_to_r2de_format_and_store_data(dict_out_dfs[12][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_12k')
    # conversion to text2props format
    convert_to_text2props_format_and_store_data(dict_out_dfs[-1][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp')
    convert_to_text2props_format_and_store_data(dict_out_dfs[4][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_4k')
    convert_to_text2props_format_and_store_data(dict_out_dfs[8][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_8k')
    convert_to_text2props_format_and_store_data(dict_out_dfs[12][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_12k')

    # ARC
    logger.info("Starting preparation ARC")
    arc_data_dir = 'data/raw/ARC-V1-Feb2018'
    df_train, df_dev, df_test = prepare_arc_dataset(arc_data_dir, 'data/processed')
    convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')
    convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')

    logger.info("Starting preparation ARC Balanced")
    balanced_df_train = pd.DataFrame(columns=df_train.columns)
    for diff in range(3, 10):
        if diff in {5, 8}:
            balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff].sample(n=500)], ignore_index=True)
        else:
            balanced_df_train = pd.concat([balanced_df_train, df_train[df_train['difficulty'] == diff]], ignore_index=True)
    balanced_df_train = balanced_df_train.sample(frac=1.0)
    balanced_df_train.to_csv(os.path.join('data/processed', f'arc_balanced_train.csv'), index=False)
    df_dev.to_csv(os.path.join('data/processed', f'arc_balanced_dev.csv'), index=False)
    df_test.to_csv(os.path.join('data/processed', f'arc_balanced_test.csv'), index=False)
    convert_to_r2de_format_and_store_data(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')
    convert_to_text2props_format_and_store_data(balanced_df_train, df_dev, df_test, DATA_DIR, 'arc_balanced')

    # ASSISTments
    logger.info("Starting preparation AM")
    am_data_dir = 'data/interim/assistments'
    df_train, df_dev, df_test = prepare_assistments_dataset(am_data_dir, 'data/processed')
    convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')
    convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')


if __name__ == "__main__":
    main()
