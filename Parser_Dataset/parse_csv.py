#!/usr/bin/env python3.5

import csv
import configparser
import argparse
from units import unit
import units.predefined

from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputs", help = "This is the csv files.", nargs='+', type=str, required=True)
args = vars(ap.parse_args())


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

def save_file(to_save, filename):
    with open(filename, 'w') as f:
        for brand in to_save:
            f.write(brand + "\n")

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
    l = ['L', 'LITRE', 'LS', 'LITRES']
    cl = ['CL', 'C', 'CLS', 'CENTILITRES']
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

if __name__ == "__main__":
    columns = defaultdict(list) # each value in each column is appended to a list
    for input_file in args['inputs']:
        columns = read_csv(columns, input_file)

    to_save = get_columns(columns['sd_marque'])
    save_file(to_save, 'Brands')

    to_save = get_columns(columns['sd_conditionnement'])
    save_file(to_save, 'Cond')

    to_save = get_columns(columns['sd_volume'])
    to_save = multiply_vol(to_save)
    save_file(sorted(to_save), 'Vol')
    # to_save_nb, to_save_vol = split_nb_vol(to_save)
    # save_file(to_save_nb, 'Vol_nb')
    to_parse = get_columns(columns['semantic'])
    to_save = parse_semantic(to_parse, 'parfum')
    save_file(sorted(to_save), 'Flavours')
    to_save = parse_semantic(to_parse, 'extra')
    save_file(sorted(to_save), 'Extra')
    to_save = parse_semantic(to_parse, 'col')
    save_file(sorted(to_save), 'Col')
