from jsonmerge import Merger
import os
import re
from os import listdir
from os.path import isfile, join
from pprint import pprint

import json


def load_file(file_to_read):
    """Load a book from its file"""
    input_file = os.path.join(file_to_read)
    with open(input_file) as f:
        data = json.load(f)
    return data

schema = {
    "properties": {
        "elements": {
            "mergeStrategy": "append"
        }
    }
}

def low_mermory_merge():
    path = "./json/"
    total_key = 'total'
    element_key = 'elements'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    total_files = len(onlyfiles)
    total_elements = 0
    i = 0
    with open('dataset.json', 'w') as outfile:
        for i, json_file in enumerate(onlyfiles):
            with open((path + json_file)) as f:
                ok_write = False
                for line in f:
                    if total_key in line:
                        ok_write = False
                    if ok_write:
                        outfile.writelines(line)
                    else:
                        if total_key in line:
                            total_elements += int(re.findall(r'\d+', line)[0])
                        elif element_key in line:
                            ok_write = True
            print("File {} on {}.".format(i, total_files), end='\r')


def merger():
    path = "../Dataset/json/"
    total_key = 'total'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    merger = Merger(schema)
    i = 1
    a = load_file(path + onlyfiles[0])
    while i <= len(onlyfiles) - 1:
        b = load_file(path + onlyfiles[i])
        tmp_total = a[total_key]
        a = merger.merge(a, b)
        a[total_key] = tmp_total + b[total_key]
        i = i + 1
    with open('dataset.json', 'w') as outfile:
        json.dump(a, outfile, default=lambda o: o.__dict__, indent=4)

low_mermory_merge()
