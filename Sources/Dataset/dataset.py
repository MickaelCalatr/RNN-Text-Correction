import os
import json
import linereader
import random
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='INFO: %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

from random import shuffle
from random import randint
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
        self.fds = []
        self.current_fd = 0
        self.i = 0
        self.lines_to_read = []
        self.total_elements = 0
        self.counter = 0
        self.count = 0

    def create_void_dic(self):#, data, labels):
        self.count = 0
        alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z', '+',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','&','°',
                   '<PAD>', '<EOS>', '<GO>', ' ', ',', '.', '_', '-','%','*',
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','(',')',
                   'n','o','p','q','r','s','t','u','v','w','x','y','z']
        for c in alphabet:
            if c not in self.vocab_to_int:
                self.vocab_to_int[c] = self.count
                self.count += 1
        for c, value in self.vocab_to_int.items():
            self.int_to_vocab[value] = c
        self.save_dic()

    def create_dic(self, data, labels):
        for line in data:
            for c in line:
                if c not in self.vocab_to_int:
                    self.vocab_to_int[c] = self.count
                    self.count += 1
        for line in labels:
            for c in line:
                if c not in self.vocab_to_int:
                    self.vocab_to_int[c] = self.count
                    self.count += 1
        for c, value in self.vocab_to_int.items():
            self.int_to_vocab[value] = c
        self.save_dic()

    def save_dic(self):
        json = Dic()
        print(self.vocab_to_int)
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
        data = None
        for s in labels:
            int_sentence = []
            for c in s:
                int_sentence.append(self.vocab_to_int[c])
            int_labels.append(int_sentence)
        label = None
        return int_sentences, int_labels

    def open_dataset(self, filenames):
        log("[1] Loading data...")
        self.fd = open(filename)
        i = 1
        for line in self.fd:
            self.lines_to_read.append(i)
            i += 1
        self.total_elements = i - 1
        self.create_void_dic()
        log("[1] Loading: Done !")
    #
    # def get_batch(self, batch_size):
    #
    #     label = []
    #     line = []
    #     for i in range(batch_size):
    #         j = np.random.choice(self.lines_to_read)
    #         raw_data = self.fd.getline(j)
    #         raw_data = raw_data.rstrip('\n')
    #         if len(raw_data) > 5:
    #             label.append(raw_data.split(';;')[0])
    #             line.append(raw_data.split(';;')[1])
    #         del self.lines_to_read[j]
    #
    #     int_sentences, int_labels = self.formating_dataset(line, label)
    #     int_sentences = np.array(int_sentences)
    #     int_labels = np.array(int_labels)
    #     int_sentences, int_labels = self.unison_shuffle(int_sentences, int_labels)
    #     return int_sentences.tolist(), int_labels.tolist()

    def get_batch(self, batch_size):
        label = []
        line = []
        for i in range(batch_size):
            label.append(self.train_label[self.i])
            line.append(self.train_data[self.i])
            self.i += 1
            if self.i >= self.total_elements:
                self.current_fd += 1
                if self.current_fd == len(self.fds):
                    self.finish()
                self.create_dataset(self.fds[self.current_fd])
        if len(label) < batch_size:
            self.finish()
        return line, label

    def finish(self):
        import sys
        logging.info("Finished !")
        sys.exit(0)

    def load_structure(self, path):
        log("[1] Loading data...")
        self.fds = [path + f for f in os.listdir(path)]
        #shuffle(self.fds)
        self.create_void_dic()
        self.create_dataset(self.fds[self.current_fd])
        self.current_fd = 0
        log("[1] Loading: Done !")

    def create_dataset(self, file_to_read):
        self.train_data = []
        self.test_data = []
        self.train_label = []
        self.test_label = []
        log("[2] Dataset loading of {}...".format(file_to_read))
        # size, raw_dataset = self.load_dataset(self.fds[self.current_fd])
        size, raw_dataset = self.load_file(file_to_read)
        log("[2] Dataset loading: Done !")
        log("     Dataset size: " + str(size) + " elements.")

        log("[3] Dataset augmentation...")
        augmentation_size, data, labels = dataset_augmentation(raw_dataset)
        raw_dataset = None
        self.create_dic(data, labels)
        log("Total Vocab: {}".format(len(self.vocab_to_int)))
        log(sorted(self.vocab_to_int))
        log("[3] Dataset augmentation: Done !")
        log("     Dataset augmentated of : " + str(int(((augmentation_size - size) / size) * 100)) + "%.")

        log("[4] Dataset formating...")
        # int_sentences, int_labels = self.formating_dataset(data, labels)
        data, labels = self.formating_dataset(data, labels)
        log("[4] Dataset formating: Done !")

        lengths = []
        for sentence in data:
            lengths.append(len(sentence))
        lengths = pd.DataFrame(lengths, columns=["counts"])
        lengths.describe()

        log("[5] Dataset splitting...")
        # Split the data into training and testing sentences
        data = np.array(data)
        labels = np.array(labels)
        data, labels = self.unison_shuffle(data, labels)
        self.train_data, self.test_data = train_test_split(data, test_size=self.cross_validation, shuffle=False)
        self.train_label, self.test_label = train_test_split(labels, test_size=self.cross_validation, shuffle=False)
        self.total_elements = len(self.train_label)
        self.i = 0#randint(0, self.total_elements - 1)
        data = None
        labels = None
        log("[5] Dataset splitting: Done !")
        log("New Dataset Loaded !")

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    #
    # # ## Loading the Data
    # def load_file(self, file_to_read):
    #     """Load a book from its file"""
    #     input_file = os.path.join(file_to_read)
    #     with open(input_file) as f:
    #         data = json.load(f)
    #     return data
    # ## Loading the Data
    def load_file(self, input_file):
        """Load a book from its file"""
        data = []
        with open(input_file) as f:
            for i, raw_data in enumerate(f):
                data.append(raw_data)
        return i, data

    # # ## Loading of the raw dataset
    # def load_dataset(self, data):
    #     size = data["total"]
    #     raw_data = data["elements"]
    #     return size, raw_data


    def randomise_dataset(self, data, labels):
        randomize = np.arange(len(labels))
        np.random.shuffle(randomize)
        data = data[randomize]
        labels = labels[randomize]
        return data, labels
