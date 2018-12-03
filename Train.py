#!/usr/bin/env python3.5

from Sources.Training import Train

def train_model():
    train = Train()
    train.build_model()
    train.start()

if __name__ == "__main__":
    train_model()
