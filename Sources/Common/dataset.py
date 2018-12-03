import os
import json

from os.path import isfile, join
import numpy as np

# ## Loading the Data
def load_file(path, file_to_read):
    """Load a book from its file"""
    input_file = os.path.join(path + file_to_read)
    with open(input_file) as f:
        data = json.load(f)
    return data


# ## Loading of the raw dataset
def load_dataset(data):
    size = data["total"]
    raw_data = data["elements"]
    return size, raw_data
#
# def convert_to_int(line):
#     int_line = []
#     for c in line:
#         int_line.append(ord(c))
#     return int_line
#
# def dataset_convert_to_int(data, labels):
#     int_dataset = []
#     int_labels = []
#
#     for line in data:
#         int_dataset.append(convert_to_int(line))
#     for label in labels:
#         int_labels.append(convert_to_int(label))
#     return np.asarray(int_dataset), np.asarray(int_labels)

def randomise_dataset(data, labels):
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
    return data, labels
