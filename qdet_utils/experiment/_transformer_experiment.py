import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
)
from transformers import EarlyStoppingCallback
from sklearn.metrics import mean_squared_error

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import (
    OUTPUT_DIR,
    DATA_DIR,
    AM,
    TF_Q_ONLY,
    TF_Q_CORRECT,
    TF_CORRECT,
    TF_TEXT,
    TF_LABEL,
    TF_DESCRIPTION,
    TF_QUESTION_ID,
    TF_ANSWERS,
    DISTILBERT,
    TEST,
    TRAIN,
    VALIDATION,
    TF_DIFFICULTY,
)


class TransformerExperiment(BaseExperiment):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = DATA_DIR,
        output_root_dir: str = OUTPUT_DIR,
        random_seed: Optional[int] = None,
        regression: bool = True,
    ):
        super().__init__(
            dataset_name, data_dir, output_root_dir, random_seed, regression
        )
        self.dataset = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.data_collator = None
        self.input_mode = None
        self.trainer = None
        self.max_length = None

    def get_dataset(
        self, input_mode, num_labels: Optional[int] = None, *args, **kwargs
    ):
        df_train_original = pd.read_csv(
            os.path.join(
                self.data_dir, f"tf_{self.dataset_name}_text_difficulty_train.csv"
            )
        )
        df_test_original = pd.read_csv(
            os.path.join(
                self.data_dir, f"tf_{self.dataset_name}_text_difficulty_test.csv"
            )
        )
        df_dev_original = pd.read_csv(
            os.path.join(
                self.data_dir, f"tf_{self.dataset_name}_text_difficulty_dev.csv"
            )
        )
        self.input_mode = input_mode
        # Update the texts depending on the INPUT_MODE
        if input_mode != TF_Q_ONLY:
            if self.dataset_name == AM:
                raise ValueError()  # AM cannot get in here
            # load answers to integrate the stem
            df_answers = pd.read_csv(
                os.path.join(self.data_dir, f"tf_{self.dataset_name}_answers_texts.csv")
            )
            if input_mode == TF_Q_CORRECT:
                df_answers = df_answers[df_answers[TF_CORRECT] == True]
            answers_dict = dict()
            for q_id, text in df_answers[[TF_QUESTION_ID, TF_DESCRIPTION]].values:
                if q_id not in answers_dict.keys():
                    answers_dict[q_id] = ""
                answers_dict[q_id] = f"{answers_dict[q_id]} [SEP] {text}"
            df_answers = pd.DataFrame(
                answers_dict.items(), columns=[TF_QUESTION_ID, TF_ANSWERS]
            )
            df_train_original = pd.merge(
                df_answers,
                df_train_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_train_original[TF_DESCRIPTION] = (
                df_train_original[TF_DESCRIPTION] + df_train_original[TF_ANSWERS]
            )
            df_test_original = pd.merge(
                df_answers,
                df_test_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_test_original[TF_DESCRIPTION] = (
                df_test_original[TF_DESCRIPTION] + df_test_original[TF_ANSWERS]
            )
            df_dev_original = pd.merge(
                df_answers,
                df_dev_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_dev_original[TF_DESCRIPTION] = (
                df_dev_original[TF_DESCRIPTION] + df_dev_original[TF_ANSWERS]
            )

        df_train_original = df_train_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_test_original = df_test_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_dev_original = df_dev_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )

        if self.regression:
            df_train_original = df_train_original.astype({TF_LABEL: float})
            df_test_original = df_test_original.astype({TF_LABEL: float})
            df_dev_original = df_dev_original.astype({TF_LABEL: float})

        dataset = DatasetDict(
            {
                TRAIN: Dataset.from_pandas(
                    df_train_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                TEST: Dataset.from_pandas(
                    df_test_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                VALIDATION: Dataset.from_pandas(
                    df_dev_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
            }
        )
        if not self.regression:
            dataset = dataset.cast_column(
                TF_LABEL,
                ClassLabel(
                    num_classes=num_labels,
                    names=self.get_difficulty_labels(self.dataset_name),
                ),
            )
        self.dataset = dataset.remove_columns(["__index_level_0__"])

        print(self.dataset[TRAIN].features)

        # added by Arthur: create self.y_true_test, self.y_true_train
        self.y_true_train = df_train_original[TF_LABEL].values
        self.y_true_test = df_test_original[TF_LABEL].values

    def init_model(
        self,
        pretrained_model: Optional[str] = DISTILBERT,
        model_name: str = "model",
        max_length: int = 256,
        pretrained_tokenizer: Optional[PreTrainedTokenizerFast] = None,
        num_labels: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_length = max_length
        if pretrained_tokenizer is None:
            pretrained_tokenizer = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

        if self.regression and num_labels > 1:
            raise ValueError(
                f"Regression is set to True, but num_labels is {num_labels}."
            )
        if not self.regression and num_labels == 1:
            raise ValueError(
                f"Regression is set to False, but num_labels is {num_labels}."
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=num_labels
        )  # default loss for regression (num_labels=1) is MSELoss()
        # TODO possibly move the two lines below somewhere else
        self.tokenized_dataset = self.dataset.map(
            self._preprocess_function, batched=True
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(
        self,
        epochs: int = 10,
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        early_stopping_patience: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        *args,
        **kwargs,
    ):
        training_args = TrainingArguments(
            output_dir=os.path.join(
                self.output_dir, f"{self.model_name}_{self.input_mode}"
            ),
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            metric_for_best_model="r_squared" if self.regression else "accuracy",
            # # evaluate per epoch:
            # evaluation_strategy="epoch",
            # save_strategy="epoch",
            # load_best_model_at_end=True,
            # logging_strategy="epoch",
            # evaluate per step:
            evaluation_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset[TRAIN],
            eval_dataset=self.tokenized_dataset[VALIDATION],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=(
                compute_metrics_regression
                if self.regression
                else compute_metrics_classification
            ),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
        )
        self.trainer.train()
        # TODO possibly change the names below so that they use the training args as well (e.g. learning rate, etc.).
        self.tokenizer.save_pretrained(
            os.path.join(
                self.output_dir, f"{self.model_name}_{self.input_mode}_tokenizer"
            )
        )
        self.trainer.save_model(
            os.path.join(self.output_dir, f"{self.model_name}_{self.input_mode}_model")
        )
        with open(
            os.path.join(training_args.output_dir, "train_logs.pickle"), "wb"
        ) as handle:
            pickle.dump(
                self.trainer.state.log_history, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def predict(self, batch_size: int = 16, save_predictions: bool = True):
        if self.trainer is None:
            test_args = TrainingArguments(
                output_dir=os.path.join(
                    self.output_dir, f"{self.model_name}_{self.input_mode}"
                ),
                do_train=False,
                do_predict=True,
                per_device_eval_batch_size=batch_size,
            )
            self.trainer = Trainer(
                model=self.model,
                args=test_args,
                compute_metrics=(
                    compute_metrics_regression
                    if self.regression
                    else compute_metrics_classification
                ),
            )
        self.tokenized_dataset = self.dataset.map(
            self._preprocess_function, batched=True
        )
        train_results = self.trainer.predict(self.tokenized_dataset[TRAIN])
        self.y_pred_train = train_results.predictions
        test_results = self.trainer.predict(self.tokenized_dataset[TEST])
        self.y_pred_test = test_results.predictions
        if save_predictions:
            pickle.dump(
                self.y_pred_test,
                open(
                    os.path.join(
                        self.output_dir,
                        f"predictions_test_{self.model_name}_{self.input_mode}.p",
                    ),
                    "wb",
                ),
            )
            pickle.dump(
                self.y_pred_train,
                open(
                    os.path.join(
                        self.output_dir,
                        f"predictions_train_{self.model_name}_{self.input_mode}.p",
                    ),
                    "wb",
                ),
            )

    def _preprocess_function(self, examples):
        return self.tokenizer(
            examples[TF_TEXT], truncation=True, max_length=self.max_length, padding=True
        )


def compute_metrics_regression(eval_pred):
    """Determines which metrics to use for regression evaluation."""
    r_squared = evaluate.load("r_squared")
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    # pearsonr = evaluate.load("pearsonr")

    return {
        "r_squared": r_squared.compute(predictions=predictions, references=labels),
        "rmse": rmse,
        # "pearsonr": pearsonr.compute(predictions=predictions, references=labels),
    }


def compute_metrics_classification(eval_pred):
    """Determines which metrics to use for classification evaluation."""
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
    return accuracy.compute(predictions=predictions, references=labels)
