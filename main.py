#!/usr/bin/env python3.5

import sys, os
from Common.Json import Dataset

def main():
    files = ["brand", "Col", "Flavours", "metric", "Qty", "type"]
    DIV = "_"

    brands = [line.rstrip('\n') for line in open(files[0])]
    cols = [line.rstrip('\n') for line in open(files[1])]
    flavours = [line.rstrip('\n') for line in open(files[2])]
    metrics = [line.rstrip('\n') for line in open(files[3])]
    qties = [line.rstrip('\n') for line in open(files[4])]
    types = [line.rstrip('\n') for line in open(files[5])]
    i = 0
    data = Dataset()
    for brand in brands:
        for tmpCol in cols:
            unitCol = tmpCol.split(DIV)
            for col in unitCol:
                for flavour in flavours:
                    for tmpQty in qties:
                        unitQty = tmpQty.split(DIV)
                        for qty in unitQty:
                            for tmpType in types:
                                unitType = tmpType.split(DIV)
                                for typ in unitType:
                                    if typ == "/":
                                        typ = " "
                                    for metric in metrics:
                                        content = metric.split(DIV)
                                        for cont in content:
                                            label = (brand + " " + unitCol[0] + " " + flavour + " " + unitQty[0] + content[0] + " " + unitType[0])
                                            line = (brand + " " + col + " " + flavour + " " + qty + " " + cont + " " + typ)
                                            data.add_element(line, label)
                                            i += 1
    data.save_json("./", "dataset.json")
    print(i)

main()
