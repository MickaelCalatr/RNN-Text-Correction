#!/usr/bin/env python3.5

from Sources.Model.Testing.Test import Test
from Sources.Model.Configuration import config

def test_model():
    testing = Test(config)
    testing.test_model(config.test)

if __name__ == "__main__":
    test_model()
