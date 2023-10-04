import yaml
import logging

from qdet_utils.experiment import TransformerExperiment

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(experiment_config):
    dataset_name = experiment_config["dataset_name"]

    random_seed = (
        experiment_config["random_seed"]
        if experiment_config["random_seed"] is not None
        else None
    )

    model_name = (
        experiment_config["model_name"]
        if experiment_config["model_name"] is not None
        else "transformer"
    )
    experiment = TransformerExperiment(
        dataset_name=dataset_name, random_seed=random_seed
    )
    logger.info("Starting dataset loading")
    experiment.get_dataset(experiment_config["input_mode"])
    logger.info("Starting model initialization")
    experiment.init_model(
        experiment_config["pretrained_model"],
        model_name,
        experiment_config["max_length"],
        experiment_config["pretrained_tokenizer"],
    )
    ### evaluate before training ###
    # logger.info("Starting training (just init, zero epochs)")
    # experiment.train(
    #     0,
    #     experiment_config["batch_size"],
    #     experiment_config["eval_batch_size"],
    #     experiment_config["early_stopping_patience"],
    #     experiment_config["learning_rate"],
    #     experiment_config["weight_decay"],
    # )
    logger.info("Starting prediction before training")
    experiment.predict(experiment_config["eval_batch_size"])
    logger.info("Starting evaluation before training")
    experiment.evaluate(
        save_name=f"{experiment_config['model_name']}_{experiment_config['input_mode']}_init"
    )
    ###

    logger.info("Starting training")
    experiment.train(
        experiment_config["epochs"],
        experiment_config["batch_size"],
        experiment_config["eval_batch_size"],
        experiment_config["early_stopping_patience"],
        experiment_config["learning_rate"],
        experiment_config["weight_decay"],
    )
    logger.info("Starting prediction")
    experiment.predict(experiment_config["eval_batch_size"])
    logger.info("Starting evaluation")
    experiment.evaluate(
        save_name=f"{experiment_config['model_name']}_{experiment_config['input_mode']}"
    )
    return


if __name__ == "__main__":
    config = yaml.safe_load(open("experiment_config.yaml", mode="r", encoding="utf-8"))
    main(config)
