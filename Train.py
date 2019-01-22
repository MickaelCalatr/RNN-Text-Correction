#!/usr/bin/env python3.5

from Sources.Model.Training.train import Train
from Sources.Model.Configuration import config
from Sources.Dataset.dataset import Dataset

def train_model():
    dataset = Dataset(config.cross_validation)
    # dataset.open_dataset(config.input_file)
    dataset.load_structure(config.input_file)

    training = Train(config)
    training.run(dataset)

if __name__ == "__main__":
    train_model()
