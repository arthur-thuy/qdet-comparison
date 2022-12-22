from src.constants import DATA_DIR, DEV, TEST, TRAIN
from src.data_utils.race import prepare_racepp_dataset
from src.data_utils.arc import prepare_arc_dataset
from src.data_utils.am import prepare_assistments_dataset
from src.data_utils.text2props_mapping import convert_to_text2props_format_and_store_data
from src.data_utils.r2de_mapping import convert_to_r2de_format_and_store_data

# RACE++
print("DOING RACE++")
race_data_dir = 'data/raw/RACE'
race_c_data_dir = 'data/raw/race-c-master/data'
dict_out_dfs = prepare_racepp_dataset(race_data_dir, race_c_data_dir, DATA_DIR)
# Conversion to R2DE format
convert_to_r2de_format_and_store_data(dict_out_dfs[-1][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp')
convert_to_r2de_format_and_store_data(dict_out_dfs[4][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_4k')
convert_to_r2de_format_and_store_data(dict_out_dfs[8][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_8k')
convert_to_r2de_format_and_store_data(dict_out_dfs[12][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_12k')
# converstion to text2props format
convert_to_text2props_format_and_store_data(dict_out_dfs[-1][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp')
convert_to_text2props_format_and_store_data(dict_out_dfs[4][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_4k')
convert_to_text2props_format_and_store_data(dict_out_dfs[8][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_8k')
convert_to_text2props_format_and_store_data(dict_out_dfs[12][TRAIN], dict_out_dfs[-1][DEV], dict_out_dfs[-1][TEST], DATA_DIR, 'race_pp_12k')

# ARC
print("DOING ARC")
arc_data_dir = 'data/raw/ARC-V1-Feb2018'
df_train, df_dev, df_test = prepare_arc_dataset(arc_data_dir, 'data/processed')
convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')
convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'arc')

# ASSISTments
print("[INFO] DOING AM...")
am_data_dir = 'data/interim/assistments'
df_train, df_dev, df_test = prepare_assistments_dataset(am_data_dir, 'data/processed')
convert_to_r2de_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')
convert_to_text2props_format_and_store_data(df_train, df_dev, df_test, DATA_DIR, 'am')