import os
import json

import pandas as pd
import numpy as np

from os.path import isfile, join
from sklearn.model_selection import train_test_split

from Sources.Dataset.dataset_augmentation import dataset_augmentation
from Sources.Common.Log import log
from Sources.Common.Json import Dic
from Sources.Common.directory_handler import create_folder

class Dataset:
    def __init__(self, cross_validation):
        self.cross_validation = cross_validation
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.train_data = []
        self.test_data = []
        self.train_label = []
        self.test_label = []

    def create_dic(self, data, labels):
        count = 0
        for line in data:
            for c in line:
                if c not in self.vocab_to_int:
                    self.vocab_to_int[c] = count
                    count += 1
        for line in labels:
            for c in line:
                if c not in self.vocab_to_int:
                    self.vocab_to_int[c] = count
                    count += 1
        codes = ['<PAD>', '<EOS>', '<GO>']
        for code in codes:
            self.vocab_to_int[code] = count
            count += 1
        for c, value in self.vocab_to_int.items():
            self.int_to_vocab[value] = c
        self.save_dic()

    def save_dic(self):
        json = Dic()
        json.int_to_vocab = self.int_to_vocab
        json.vocab_to_int = self.vocab_to_int
        create_folder('./Model/')
        json.save_json('./Model/', 'saved_dic.json')

    def formating_dataset(self, data, labels):
        int_sentences = []
        int_labels = []
        for s in data:
            int_sentence = []
            for c in s:
                int_sentence.append(self.vocab_to_int[c])
            int_sentences.append(int_sentence)
        for s in labels:
            int_sentence = []
            for c in s:
                int_sentence.append(self.vocab_to_int[c])
            int_labels.append(int_sentence)
        return int_sentences, int_labels

    def create_dataset(self, file_to_read):
        log("[1] Loading data...")
        data = self.load_file(file_to_read)
        log("[1] Loading: Done !")

        log("[2] Dataset loading...")
        size, raw_dataset = self.load_dataset(data)
        log("[2] Dataset loading: Done !")
        log("     Dataset size: " + str(size) + " elements.")

        log("[3] Dataset augmentation...")
        augmentation_size, data, labels = dataset_augmentation(raw_dataset)
        self.create_dic(data, labels)
        log("Total Vocab: {}".format(len(self.vocab_to_int)))
        log(sorted(self.vocab_to_int))
        log("[3] Dataset augmentation: Done !")
        log("     Dataset augmentated of : " + str(int(((augmentation_size - size) / size) * 100)) + "%.")

        log("[4] Dataset formating...")
        int_sentences, int_labels = self.formating_dataset(data, labels)
        log("[4] Dataset formating: Done !")

        lengths = []
        for sentence in int_sentences:
            lengths.append(len(sentence))
        lengths = pd.DataFrame(lengths, columns=["counts"])
        lengths.describe()

        log("[5] Dataset splitting...")
        # Split the data into training and testing sentences
        int_sentences = np.array(int_sentences)
        int_labels = np.array(int_labels)
        int_sentences, int_labels = self.unison_shuffle(int_sentences, int_labels)
        self.train_data, self.test_data = train_test_split(int_sentences, test_size=self.cross_validation, shuffle=False)
        self.train_label, self.test_label = train_test_split(int_labels, test_size=self.cross_validation, shuffle=False)
        log("[5] Dataset splitting: Done !")
        log("Dataset Loaded !")
        log("====================")

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # ## Loading the Data
    def load_file(self, file_to_read):
        """Load a book from its file"""
        input_file = os.path.join(file_to_read)
        with open(input_file) as f:
            data = json.load(f)
        return data


    # ## Loading of the raw dataset
    def load_dataset(self, data):
        size = data["total"]
        raw_data = data["elements"]
        return size, raw_data


    def randomise_dataset(self, data, labels):
        randomize = np.arange(len(labels))
        np.random.shuffle(randomize)
        data = data[randomize]
        labels = labels[randomize]
        return data, labels
