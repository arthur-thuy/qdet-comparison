import pandas as pd
import json
import os
from os import listdir

from qdet_utils.constants import (
    DF_COLS,
    CORRECT_ANSWER,
    OPTIONS,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    SPLIT,
    DIFFICULTY,
    DEV,
    TEST,
    TRAIN,

)

HIGH = 'high'
MIDDLE = 'middle'
COLLEGE = 'college'

LEVEL_TO_DIFFICULTY_MAP = {MIDDLE: 0, HIGH: 1, COLLEGE: 2}


def update_df_with_new_reading_passage(df, reading_passage_data, split, level):
    answers = reading_passage_data['answers']
    options = reading_passage_data['options']
    questions = reading_passage_data['questions']
    article = reading_passage_data['article']
    context_id = reading_passage_data['id']

    for idx in range(len(questions)):
        # this is just to check that there are no anomalies
        assert ord('A') <= ord(answers[idx]) <= ord('Z')
        df = pd.concat([df, pd.DataFrame([{CORRECT_ANSWER: ord(answers[idx])-ord('A'),
                                           OPTIONS: options[idx],
                                           OPTION_0: options[idx][0],
                                           OPTION_1: options[idx][1],
                                           OPTION_2: options[idx][2],
                                           OPTION_3: options[idx][3],
                                           QUESTION: questions[idx],
                                           CONTEXT: article,
                                           CONTEXT_ID: context_id[:-4],
                                           Q_ID: '%s_q%d' % (context_id[:-4], idx),
                                           SPLIT: split,
                                           DIFFICULTY: LEVEL_TO_DIFFICULTY_MAP[level]}])])
    return df


def prepare_and_return_race_df(data_dir: str, split: str):
    df = pd.DataFrame(columns=DF_COLS)
    for level in [HIGH, MIDDLE]:
        for filename in listdir(os.path.join(data_dir, split, level)):
            with open(os.path.join(data_dir, split, level, filename), 'r') as f:
                reading_passage_data = json.load(f)
            df = update_df_with_new_reading_passage(df, reading_passage_data, split, level)
    return df


def prepare_and_return_race_c_df(data_dir: str, split: str):
    df = pd.DataFrame(columns=DF_COLS)
    for filename in listdir(os.path.join(data_dir, split)):
        with open(os.path.join(data_dir, split, filename), 'r') as f:
            reading_passage_data = json.load(f)
        df = update_df_with_new_reading_passage(df, reading_passage_data, split, level=COLLEGE)
    return df


def prepare_racepp_dataset(race_data_dir, race_c_data_dir, output_data_dir):  # , training_size=None, random_state=None):
    out_dfs = []
    for split in [TRAIN, DEV, TEST]:
        df_race = prepare_and_return_race_df(data_dir=race_data_dir, split=split)
        df_race_c = prepare_and_return_race_c_df(data_dir=race_c_data_dir, split=split)
        df = pd.concat([df_race, df_race_c])
        assert set(df.columns) == set(DF_COLS)
        df.to_csv(os.path.join(output_data_dir, f'race_pp_{split}.csv'), index=False)
        out_dfs.append(df.copy())
    return out_dfs


def prepare_subsampled_racepp_dataset(df_train, df_dev, df_test, training_size, output_data_dir, random_state=None):
    # TODO:
    #  this is the sampling per class (in the original implementation, it is balanced!!), I should also
    #  implement the sampling "unbalanced"
    df_train = pd.concat([df_train[df_train[DIFFICULTY] == 0].sample(training_size, random_state=random_state),
                          df_train[df_train[DIFFICULTY] == 1].sample(training_size, random_state=random_state),
                          df_train[df_train[DIFFICULTY] == 2].sample(training_size, random_state=random_state)])
    df_train.to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TRAIN}.csv'), index=False)
    df_dev.to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{DEV}.csv'), index=False)
    df_test.to_csv(os.path.join(output_data_dir, f'race_pp_{training_size}_{TEST}.csv'), index=False)
    return [df_train, df_dev, df_test]


def prepare_race_dataset(race_data_dir, output_data_dir):
    for split in [TRAIN, DEV, TEST]:
        df_race = prepare_and_return_race_df(data_dir=race_data_dir, split=split)
        assert set(df_race.columns) == set(DF_COLS)
        df_race.to_csv(os.path.join(output_data_dir, f'race_{split}.csv'), index=False)


def prepare_race_c_dataset(race_c_data_dir, output_data_dir):
    for split in [TRAIN, DEV, TEST]:
        df_race_c = prepare_and_return_race_c_df(data_dir=race_c_data_dir, split=split)
        assert set(df_race_c.columns) == set(DF_COLS)
        df_race_c.to_csv(os.path.join(output_data_dir, f'race_c_{split}.csv'), index=False)