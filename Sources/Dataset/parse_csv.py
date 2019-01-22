import csv
import os.path
import configparser
import argparse
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

def split_nb_vol(lines):
    nb_vol = []
    vol = []
    for line in lines:
        for i, c in enumerate(line):
            if c.isnumeric() or c == '.' or c == ',':
                continue
            else:
                if line[:i] not in nb_vol:
                    nb_vol.append(line[:i])
                if line[i:] not in vol:
                    vol.append(line[i:])
                break
    return nb_vol, vol

def parse_semantic(lines, key):
    flavours = []
    for line in lines:
        tab = line.split(' ')
        for s in tab:
            if key in s and s.split(':')[1].upper() not in flavours:
                flavours.append(s.split(':')[1].upper())
    return flavours

def multiply_vol(vols):
    l = ['L']
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

def parse_csv(directory, inputs):
    save_dir = directory + 'Files/'
    create_folder(save_dir)

    columns = defaultdict(list) # each value in each column is appended to a list
    for input_file in inputs:
        columns = read_csv(columns, input_file)
    # print(columns)
    to_save = get_columns(columns['sd_marque'])
    save_file(save_dir, to_save, '0_Brands')

    to_save = get_columns(columns['sd_conditionnement'])
    save_file(save_dir, to_save, '1_Cond')

    to_save = get_columns(columns['sd_volume'])
    to_save = multiply_vol(to_save)
    save_file(save_dir, sorted(to_save), '2_Vol')
    # to_save_nb, to_save_vol = split_nb_vol(to_save)
    # save_file(to_save_nb, 'Vol_nb')
    to_parse = get_columns(columns['semantic'])
    to_save = parse_semantic(to_parse, 'parfum')
    save_file(save_dir, sorted(to_save), '3_Flavours')
    to_save = parse_semantic(to_parse, 'extra')
    save_file(save_dir, sorted(to_save), '4_Extra')
    to_save = parse_semantic(to_parse, 'col')
    save_file(save_dir, sorted(to_save), '5_Col')

parse_csv('./', ['/home/mickael/RNN-Text-Correction/Parser_Dataset/csv_files/Liste_produits_RICARD.csv'])
