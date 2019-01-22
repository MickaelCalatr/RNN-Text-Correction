import csv
from collections import defaultdict
from Sources.Model.Configuration import config

def read_csv(columns, input_file):
    with open(input_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value)
                columns[k].append(v)
    return columns

def test_file(filenames):
    testing = Test(config)
    testing.load_model()

    for filename in filenames:
        data = {}
        columns = defaultdict(list) # each value in each column is appended to a list
        columns = read_csv(columns, filename)
        libelle = column['libelle']
        code = column['sd_code_sogge']
        key = '1sogge-'
        id = 0
        for i in range(46, len(code)):
            result = testing.test_line(libelle[i])
            if result not in data:
                id += 1
                data[result] = key + str(id)
            print(libelle[i], data[result])

test_file(config['input'])
