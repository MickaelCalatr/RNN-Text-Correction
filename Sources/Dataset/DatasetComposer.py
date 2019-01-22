import csv
import os.path
import configparser
import argparse
import re, sys
from units import unit
import units.predefined

from collections import defaultdict


def read_csv(columns, input_file):
    with open(input_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value)
                columns[k].append(v)
    return columns

def get_columns(brands):
    unics_brand = []
    for brand in brands:
        if brand not in unics_brand:
            unics_brand.append(brand)
    return unics_brand

def save_file(directory, to_save, filename):
    with open(directory + filename, 'w') as f:
        for i, brand in enumerate(to_save):
            if i != len(to_save) - 1:
                f.write(brand + "\n")
            else:
                f.write(brand)

def multiply_vol(vols):
    l = ['L', 'LITRES']
    cl = ['CL']
    DIV = '_'
    volume = []
    for s in vols:
        line = s
        line.replace(',', '.')
        for i, c in enumerate(line):
            if c.isnumeric() or c == '.':
                continue
            else:
                res = ""
                if line[i:] == 'L':
                    liter = float(unit('L')(float(line[:i])))
                    for item in l:
                        res += line[:i] + item + DIV
                    for i, item in enumerate(cl):
                        res += str(float(unit('cL')(liter)) * 100) + item + DIV
                        res += str(int(float(unit('cL')(liter)) * 100)) + item + DIV
                else:
                    for item in cl:
                        res += line[:i] + item + DIV
                volume.append(res[:-1])
                break
    return volume

def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def my_split(s):
    return re.split(r'(\d+)', s)


def askUser(word):
    print('What is this: [{}].'.format(word))
    result = input()
    if len(result) == 0:
        return word
    if result == '0':
        return ''
    return result

def extracData(data):
    dic = {}
    labels = set()
    for i, raw in enumerate(data):
        line = ''
        if '°' in raw:
            line = raw.split('°')[0] + '°'
        else:
            line = my_split(raw)[0].strip()
        print('LINE {}: [{}] ===> [{}]'.format(i, raw, line))
        tab = line.split(' ')
        result = ''
        for w in tab:
            w = w.strip()
            if len(w) <= 3 and '°' not in w and w not in dic:
                res = askUser(w)
                dic[w] = res
                w = res
            elif w in dic:
                w = dic[w]
            result += w + ' '
        labels.add(result.strip())
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    return labels


def parse_csv(directory, inputs):
    save_dir = directory + 'Files/'
    create_folder(save_dir)

    columns = defaultdict(list) # each value in each column is appended to a list
    for input_file in inputs:
        columns = read_csv(columns, input_file)
    # print(columns)
    data = get_columns(columns['libelle'])
    a = set()
    for i in data:
        if i.find('°') != -1:
            a.add(str(i[i.find('°') - 3 : i.find('°') + 1]).strip())
    print(a)
    # labels = extracData(data)
    # save_file('./', labels, 'labels')

parse_csv('./', ['/home/mickael/RNN-Text-Correction/Parser_Dataset/csv_files/Liste_produits_RICARD_virgule.csv'])
