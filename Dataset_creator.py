#!/usr/bin/env python3.5
from Sources.Dataset.DatasetCreator import *

def dataset_creator():
    dataset = DatasetCreator()
    dataset.run_test()

if __name__ == "__main__":
    dataset_creator()
