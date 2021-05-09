import hydra
from omegaconf import DictConfig

from src.toxic.data.data_module import DataModule
from src.toxic.modelling.model import Model
from src.toxic.training import Trainer
from src.toxic.utils.random import set_seed


@hydra.main(config_path='conf', config_name='config')
def train(config: DictConfig):
    set_seed(config.trainer.seed)
    data = DataModule(config.data)
    model = Model(config)
    trainer = Trainer(config, data, model)
    trainer.train()


if __name__ == '__main__':
    train()
