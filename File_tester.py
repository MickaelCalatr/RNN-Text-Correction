import csv
from collections import defaultdict
from Sources.Model.Testing.Test import Test
from Sources.Model.Configuration import config

def read_csv(columns, input_file):
    with open(input_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value)
                columns[k].append(v)
    return columns


def test_file(filename):
    testing = Test(config)
    testing.load_model()

    # for filename in filenames:
    data = {}
    columns = defaultdict(list) # each value in each column is appended to a list
    columns = read_csv(columns, filename)
    libelle = columns['libelle']
    sku = columns['sku']
    code = columns['sd_code_sogge']
    result = columns['sd_code_check']
    key = '1sogge-'
    id = 0
    with open(filename,'r') as csvinput:
        with open('./output.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            # row.append('sku')
            # row.append('libelle')
            row.insert(3, 'result')
            # row.append('sd_code_sogge')
            row.insert(4, 'sd_code_check')
            all.append(row)
            for i in range(0, len(libelle)):
                text = ' '.join(x.strip() for x in libelle[i].split(' ') if len(x.strip()) > 0)
                result = testing.test_line(text)

                if result not in data:
                    id += 1
                    data[result] = key + str(id)

                row = next(reader)
                row.insert(3, result)
                row.insert(4, data[result])
                # row.append(sku[i])
                # row.append(libelle[i])
                # row.append(result)
                # row.append(code[i])
                # row.append(data[result])
                all.append(row)
            writer.writerows(all)
            #print(text, '\t       ',  result, data[result])

test_file(config.input_file)
