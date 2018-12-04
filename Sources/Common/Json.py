import json

class Dic:
    def jdefault(o):
        return o.__dict__

    def save_json(self, path, filename):
        with open(path + filename, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__, indent=4)

    def __init__(self):
        self.vocab_to_int = {}
        self.int_to_vocab = {}

class Element:
    def __init__(self, line, label):
        self.label = label
        self.line = line

class Dataset:
    def jdefault(o):
        return o.__dict__

    def add_element(self, line, label):
        element = Element(line, label)
        self.elements.append(element)

    def save_json(self, path, filename):
        self.total = len(self.elements)
        with open(path + filename, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.__dict__, indent=4)

    def __init__(self):
        self.elements = []
        self.total = 0
