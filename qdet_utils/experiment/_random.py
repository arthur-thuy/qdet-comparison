from typing import Optional
import os
import pandas as pd
import pickle
import random

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import Q_ID, OUTPUT_DIR, DATA_DIR, TF_DIFFICULTY


class RandomExperiment(BaseExperiment):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = DATA_DIR,
        output_root_dir: str = OUTPUT_DIR,
        random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.true_difficulties = None
        self.min_diff = None
        self.max_diff = None

    def get_dataset(self, *args, **kwargs):
        self.df_train = pd.read_csv(
            os.path.join(
                self.data_dir, f"tf_{self.dataset_name}_text_difficulty_train.csv"
            ),
            dtype={Q_ID: str},
        )
        self.df_test = pd.read_csv(
            os.path.join(
                self.data_dir, f"tf_{self.dataset_name}_text_difficulty_test.csv"
            ),
            dtype={Q_ID: str},
        )
        self.y_true_train = self.df_train[TF_DIFFICULTY].values
        self.y_true_test = self.df_test[TF_DIFFICULTY].values

    def init_model(
        self,
        pretrained_model: Optional[str] = None,
        model_name: str = "model",
        *args,
        **kwargs,
    ):
        self.model_name = model_name

    def train(self, *args, **kwargs):
        if self.discrete_regression:
            self.true_difficulties = list(set(self.y_true_train))
        else:
            self.min_diff = min(self.y_true_train)
            self.max_diff = max(self.y_true_train)

    def predict(self, save_predictions: bool = True):
        if self.discrete_regression:
            self.y_pred_train = [
                random.choice(self.true_difficulties) for _ in range(len(self.df_train))
            ]
            self.y_pred_test = [
                random.choice(self.true_difficulties) for _ in range(len(self.df_test))
            ]
        else:
            self.y_pred_train = [
                random.random() * (self.max_diff - self.min_diff) + self.min_diff
                for _ in range(len(self.df_train))
            ]
            self.y_pred_test = [
                random.random() * (self.max_diff - self.min_diff) + self.min_diff
                for _ in range(len(self.df_test))
            ]
        if save_predictions:
            os.makedirs(self.output_dir, exist_ok=True)
            pickle.dump(
                self.y_pred_test,
                open(
                    os.path.join(
                        self.output_dir, "predictions_test_" + self.model_name + ".p"
                    ),
                    "wb",
                ),
            )
            pickle.dump(
                self.y_pred_train,
                open(
                    os.path.join(
                        self.output_dir, "predictions_train_" + self.model_name + ".p"
                    ),
                    "wb",
                ),
            )
