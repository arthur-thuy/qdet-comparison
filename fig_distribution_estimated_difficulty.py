import pandas as pd
import pickle

from qdet_utils.plot_utils import plot_violinplot_race, plot_violinplot_arc, plot_hexbin_am
from qdet_utils.constants import RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from qdet_utils.constants import TF_Q_ALL, TF_Q_ONLY

LIST_TF_ENCODINGS = [TF_Q_ALL]
TF_MODEL = "transformer"
DATASET_NAME = RACE_PP_4K
RANDOM_SEED = 123
INPUT_MODE = "question_all"


BERT_QA = 'BERT $Q_A$'
BERT_QO = 'BERT $Q_O$'
DISTILBERT_QA = 'DistilBERT $Q_A$'
DISTILBERT_QO = 'DistilBERT $Q_O$'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# RACE

# # BERT_QA
# df = pd.read_csv('data/transformers_predictions/race_pp/predictions_test_BERT_3_question_all_0.csv')
# plot_violinplot_race(df, title=BERT_QA, output_filename='bert')

# DISTILBERT_QA
df = pd.read_csv(f'data/processed/{DATASET_NAME}_test.csv')
df['predicted_difficulty'] = pickle.load(open(f'output/{DATASET_NAME}/seed_{RANDOM_SEED}/predictions_test_{TF_MODEL}_{INPUT_MODE}.p', 'rb'))
plot_violinplot_race(df, title=DISTILBERT_QA, output_filename='distilbert')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # ARC

# # BERT_QA
# df = pd.read_csv('data/transformers_predictions/arc_balanced/predictions_test_BERT_question_all_0.csv')
# plot_violinplot_arc(df, BERT_QA, output_filename='balanced_bert_qa')

# # DISTILBERT_QA
# df = pd.read_csv('data/transformers_predictions/arc_balanced/predictions_test_DistilBERT_question_all_0.csv')
# plot_violinplot_arc(df, DISTILBERT_QA, output_filename='balanced_distilbert_qa')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # AM

# # BERT_QO
# df = pd.read_csv('data/transformers_predictions/am/predictions_test_BERT_question_only_0.csv')
# plot_hexbin_am(df, title=BERT_QO, output_filename='bert_qo')

# # DISTILBERT_QO
# df = pd.read_csv('data/transformers_predictions/am/predictions_test_DistilBERT_question_only_0.csv')
# plot_hexbin_am(df, title=DISTILBERT_QO, output_filename='distilbert_qo')
