from typing import Optional
import numpy as np
import os
import pandas as pd
import pickle

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import Q_ID, OUTPUT_DIR, DATA_DIR, TF_DIFFICULTY


class MajorityExperiment(BaseExperiment):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = DATA_DIR,
        output_root_dir: str = OUTPUT_DIR,
        random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.value = None

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

    # train with randomized CV and save model
    def train(self, *args, **kwargs):
        if self.discrete_regression:
            self.value = (
                pd.DataFrame({"d": self.y_true_train})
                .groupby("d")
                .size()
                .reset_index()
                .sort_values(0, ascending=False)["d"]
                .values[0]
            )
        else:
            self.value = np.mean(self.y_true_train)

    def predict(self, save_predictions: bool = True):
        self.y_pred_train = [self.value] * len(self.df_train)
        self.y_pred_test = [self.value] * len(self.df_test)
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
