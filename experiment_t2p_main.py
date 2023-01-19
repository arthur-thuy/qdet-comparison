from sklearn.preprocessing import normalize
from text2props.modules.estimators_from_text import FeatureEngAndRegressionPipeline
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.regression import RegressionModule

from qdet_utils.experiment import Text2propsExperiment
from qdet_utils.text2props_configs import (
    get_config,
    get_text2props_regression_components_from_config,
    get_text2props_feat_eng_components_from_config,
    get_dict_params_by_config,
)

from experiment_t2p_params import (
    dataset_name,
    random_seed,
    feature_eng_config,
    regression_config,
    n_iter,
    n_jobs,
)

experiment = Text2propsExperiment(dataset_name=dataset_name, random_seed=random_seed)
experiment.get_dataset()

difficulty_range = experiment.get_difficulty_range(dataset_name)
config = get_config(feature_engineering_config=feature_eng_config, regression_config=regression_config)
feat_eng_and_regr_pipeline = FeatureEngAndRegressionPipeline(
    FeatureEngineeringModule(get_text2props_feat_eng_components_from_config(config, random_seed), normalize_method=normalize),
    RegressionModule(get_text2props_regression_components_from_config(config, difficulty_range, random_seed))
)
experiment.init_model(None, config, feat_eng_and_regr_pipeline)

dict_params = get_dict_params_by_config(config)
experiment.train(dict_params, n_iter, n_jobs)
experiment.predict()
experiment.evaluate()
