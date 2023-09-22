import yaml

from qdet_utils.experiment import (
    MajorityExperiment,
    RandomExperiment,
)


def main(experiment_config):
    model_type = experiment_config['model_type']
    dataset_name = experiment_config['dataset_name']

    if model_type == 'majority':
        model_name = experiment_config['model_name'] if experiment_config['model_name'] is not None else 'majority'
        experiment = MajorityExperiment(dataset_name=dataset_name)
        experiment.get_dataset()
        experiment.init_model(None, model_name, None)
        experiment.train(None, None, None, None)
        experiment.predict()
        experiment.evaluate(compute_correlation=False)
        return

    random_seed = experiment_config['random_seed'] if experiment_config['random_seed'] is not None else None

    if model_type == 'random':
        model_name = experiment_config['model_name'] if experiment_config['model_name'] is not None else 'random'
        experiment = RandomExperiment(dataset_name=dataset_name, random_seed=random_seed)
        experiment.get_dataset()
        experiment.init_model(None, model_name, None)
        experiment.train(None, None, None, None)
        experiment.predict()
        experiment.evaluate()
        return

    print("Unknown model_type!")


if __name__ == "__main__":
    config = yaml.safe_load(open('experiment_config.yaml', 'r'))
    main(config)
