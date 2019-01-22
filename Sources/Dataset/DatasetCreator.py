import sys
import os
import re
import configparser
import argparse
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='INFO: %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

from os import listdir
from os.path import isfile, join
from random import *
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor


from Sources.Dataset.parse_csv import parse_csv
from Sources.Common.directory_handler import create_folder

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help = "This is the csv files.", nargs='+', type=str, required=True)
ap.add_argument("-p", "--path", help = "This is the path to save the dataset.", type=str, default='./Dataset/')
ap.add_argument("-d", "--divider", help="This is the divider used to parse the same values", type=str, default='_')
ap.add_argument("-f", "--filename", help="This is the name of the final file.", type=str, default='dataset.ds')
ap.add_argument("-fy", "--force_yes", help="Use it if you don't want to modify the input file.", type=bool, default=False)
ap.add_argument("-ip", "--input_precalculated", help="This is the Files used to create the dataset.", type=str, default=None)
ap.add_argument("-ic", "--input_computed", help="This is the Files used to create the dataset.", type=str, default=None)

args = vars(ap.parse_args())

# class WriteThread (threading.Thread):
#     def __init__(self, ):
#         threading.Thread.__init__(self)
#         self.fd = fd
#         self.fd_shuffle = fd_shuffle
#         self.fd_25 = fd_25
#         self.fd_75 = fd_75
#
class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.fds = []

    def add(self, fd):
        self.lock.acquire()
        try:
            self.fds.append(fd)
        finally:
            self.lock.release()

class MergerSelect:
    def __init__(self, base_name, path, max=200000):
        self.path = path
        self.base_name = base_name
        self.counter = 0
        self.number_files = 1
        self.max = max
        self.fd = open(self.concat(), 'w')

    def concat(self):
        return self.path + str(self.number_files) + '_' + self.base_name

    def write(self, line):
        if self.counter >= self.max:
            self.fd.close()
            self.counter = 0
            self.number_files += 1
            self.fd = open(self.concat(), 'w')
        self.fd.writelines(line)
        self.counter += 1

class DatasetCreator:
    def __init__(self):
        self.div = args['divider']
        self.path = args['path']
        self.filename = args['filename']
        self.tmp_path = self.path + 'tmp/'
        self.brands = []
        self.conds = []
        self.vols = []
        self.flavours = []
        self.extras = []
        self.cols = []
        self.fds = []
        self.to_divide = 0
        self.total_elements = 0
        self.wrote = 0
        self.elements = 0

    def count_elements(self, tab):
        tmp = 0
        for i in [l for l in tab]:
            tmp += len(i.split(self.div))
        return tmp

    def get_final_number_result(self):
        self.to_divide = self.count_elements(self.brands)
        total = 1
        total *= self.count_elements(self.brands)
        total *= self.count_elements(self.conds)
        total *= self.count_elements(self.vols)
        total *= self.count_elements(self.flavours)
        total *= self.count_elements(self.extras)
        total *= self.count_elements(self.cols)
        return total + int(total / 5) + 1


    def write_in_files(self, line):
        if self.wrote >= self.to_divide:
            self.fds[-1].close()
            self.fds.append(open(self.tmp_path + str(self.elements), 'w'))
            self.wrote = 0
            print('INFO: {} on {} elements.'.format(self.elements, self.total_elements), end='\r')
        self.fds[-1].writelines(line)
        self.wrote += 1
        self.elements += 1

    def load_file(self, path):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(onlyfiles)
        self.brands = self.open_file(path + files[0])
        self.conds = self.open_file(path + files[1])
        self.vols = self.open_file(path + files[2])
        self.flavours = self.open_file(path + files[3])
        self.extras = self.open_file(path + files[4])
        self.cols = self.open_file(path + files[5])

    def run (self):
        # if args['input_computed'] == None:
        create_folder(self.path)
        create_folder(self.tmp_path)
        if args['input_precalculated'] == None:
            parse_csv(self.tmp_path, args['input'])
            self.load_file(self.tmp_path + 'Files/')
            ok = args['force_yes']
            while ok == False:
                print("Check the files in {}".format(self.tmp_path + 'Files/'))
                responce = input("Continue ?\n")
                if responce.lower() == 'yes':
                    ok = True
                elif responce.lower() == 'no':
                    sys.exit()
        else:
            self.load_file(args['input_precalculated'])
        self.total_elements = self.get_final_number_result()
        logging.info('{} elements will be created.'.format(self.total_elements))

        self.dataset_maker_test()
        logging.info('{} elements created.'.format(self.total_elements))
        # else:
        #     self.tmp_path = args['input_computed']
        #     self.fds = [self.tmp_path + fd for fd in os.listdir(self.tmp_path) if os.path.isfile(fd)]
        # self.merge_all_files()
        logging.info('Dataset created.\n\nFinished !')

    def run_test(self):
        create_folder(self.path)
        create_folder(self.tmp_path)
        path = args['input_precalculated']
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        files = sorted(onlyfiles)
        self.brands = self.open_file(path + files[0])
        self.conds = self.open_file(path + files[1])
        self.vols = self.open_file(path + files[2])
        self.cols = self.open_file(path + files[3])
        # self.total_elements = self.get_final_number_result()
        # logging.info('{} elements will be created.'.format(self.total_elements))

        self.dataset_maker_test()
        # logging.info('{} elements created.'.format(self.total_elements))
        # else:
        #     self.tmp_path = args['input_computed']
        #     self.fds = [self.tmp_path + fd for fd in os.listdir(self.tmp_path) if os.path.isfile(fd)]
        # self.merge_all_files()
        logging.info('Dataset created.\n\nFinished !')
    @staticmethod
    def open_file(filename):
        return [line.rstrip('\n') for line in open(filename)]

    # ## Shuffle a line to mix the characteristics
    @staticmethod
    def shuffle_line(line):
        data = line.split(" ")
        shuffle(data)
        tmp_line = ' '.join(data)
        return tmp_line

    def getAlcohol(self):
        alc = ['', '7.5°', '12.5°', '14.8°', '16°', '17°', '18°', '19.5°', '20°', '21°', '38°', '35°', '37°', '40°', '41.7°', '42.3°', '43°', '45°', '46°', '48°', '49°', '50°', '53°', '55.5°', '55°', '59.7°']
        years = ['', '10ANS', '12ANS', '16ANS', '18ANS', '21ANS', '38ANS']

        if randint(0, 3) == 0:
            alcohol = alc[randint(0, len(alc) - 1)]
            if len(alcohol) == 0:
                res_alc = ' '
            else:
                res_alc = ' ' + alcohol
        else:
            res_alc = ''
        if randint(0, 4) == 0:
            year = years[randint(0, len(years) - 1)]
            if len(year) != 0:
                year = ' ' + year
        else:
            year = ''
        false_year = ''
        if len(year) > 0:
            false_year = year[:-2]
        return res_alc + year + ' ', res_alc + false_year + ' '

    def dataset_maker_test(self):
        total = 1
        total *= self.count_elements(self.conds)
        total *= self.count_elements(self.vols)
        total *= self.count_elements(self.cols)
        total_brand = self.count_elements(self.brands)
        print("Total element : {}\n{} elements will be created.".format(total, int(total / total_brand)))
        i = 0
        b = 0
        fd = open('dataset_test.ds', 'w')
        for j in range(100):
            for tmp_col in self.cols:
                tab_col = tmp_col.split(self.div)
                for col in tab_col:
                    if len(col) == 0 or col == ' ':
                        col = ''
                    else:
                        col = col + ' '
                    for tmp_vol in self.vols:
                        tab_vol = tmp_vol.split(self.div)
                        for vol in tab_vol:
                            for tmp_cond in self.conds:
                                tab_cond = tmp_cond.split(self.div)
                                for cond in tab_cond:
                                    brand = self.brands[b]
                                    alcohol, false_alc = self.getAlcohol()
                                    if tab_col[0] == ' ':
                                        label = (brand + alcohol + '1X' + tab_vol[0])
                                    else:
                                        label = (brand + alcohol + tab_col[0] + 'X' + tab_vol[0])

                                    line = (brand + false_alc + col + 'X' + vol +  ' ' + cond)
                                    fd.writelines(label + ';;' + line + '\n')
                                    # if i % 5:
                                    #     line = (brand + alcohol + col + " " + cond + vol)
                                    #     fd.writelines(label + ';;' + line + '\n')
                                    i += 1
                                    b += 1
                                    if b == len(self.brands):
                                        b = randint(0, len(self.brands) - 1)
        fd.close()
        # i = 0
        # b = 0
        # fd = open('dataset_test.ds', 'w')
        # for brand in self.brands:
        #     for tmp_col in self.cols:
        #         tab_col = tmp_col.split(self.div)
        #         for col in tab_col:
        #             if len(col) == 0 or col == ' ':
        #                 col = ''
        #             else:
        #                 col = col + ' '
        #             for tmp_vol in self.vols:
        #                 tab_vol = tmp_vol.split(self.div)
        #                 for vol in tab_vol:
        #
        #                     for tmp_cond in self.conds:
        #                         tab_cond = tmp_cond.split(self.div)
        #                         for cond in tab_cond:
        #                             alcohol = self.getAlcohol()
        #                             if tab_col[0] == ' ':
        #                                 label = (brand + alcohol + '1X' + tab_vol[0])
        #                             else:
        #                                 label = (brand + alcohol + tab_col[0] + 'X' + tab_vol[0])
        #
        #                             line = (brand + alcohol + col + 'X' + vol + cond)
        #                             fd.writelines(label + ';;' + line + '\n')
        #                             # if i % 5:
        #                             #     line = (brand + alcohol + col + " " + cond + vol)
        #                             #     fd.writelines(label + ';;' + line + '\n')
        #                             i += 1
        # fd.close()

    def dataset_maker(self):
        print_thread = PrintThread(self.tmp_path, self.total_elements, self.to_divide)
        print_thread.setDaemon(True)
        print_thread.start()
        i = 0
        for tmp_brand in self.brands:
            tab_brand = tmp_brand.split(self.div)
            for brand in tab_brand:

                for tmp_extra in self.extras:
                    tab_extra = tmp_extra.split(self.div)
                    extra_trim = tab_extra[0].rstrip()
                    if len(extra_trim) == 0:
                        extra_trim = ' '
                    else:
                        extra_trim = ' ' + extra_trim + ' '
                    for extra in tab_extra:
                        if len(extra) == 0 or extra == ' ':
                            extra = ' '
                        else:
                            extra = " " + extra + " "
                        for tmp_flavour in self.flavours:
                            tab_flavour = tmp_flavour.split(self.div)
                            for flavour in tab_flavour:
                                for tmp_col in self.cols:
                                    tab_col = tmp_col.split(self.div)
                                    for col in tab_col:

                                        for tmp_vol in self.vols:
                                            tab_vol = tmp_vol.split(self.div)
                                            for vol in tab_vol:

                                                for tmp_cond in self.conds:
                                                    tab_cond = tmp_cond.split(self.div)
                                                    cond_trim = tmp_cond[0].rstrip()
                                                    if len(cond_trim) != 0:
                                                        cond_trim = ' ' + cond_trim
                                                    for cond in tab_cond:
                                                        label = (tab_brand[0] + extra_trim + flavour + ' ' + tab_col[0] + 'X' + tab_vol[0] + cond_trim)
                                                        line = (brand + extra + flavour + ' ' + col + " " + vol + " " + cond)
                                                        print_thread.print_files(label + ';;' + line + '\n')
                                                        # self.write_in_files((label + ';;' + line + '\n'))

                                                        # label = (tab_brand[0] + " " + tab_flavour[0] + " " + tab_col[0] + 'X' + tab_vol[0] + " " + tab_cond[0])
                                                        if i % 5:
                                                            line = (brand + extra + flavour+ ' ' + col + " " + cond + " X" + vol)
                                                            print_thread.print_files(label + ';;' + line + '\n')
                                                        # self.write_in_files((label + ';;' + line + '\n'))
                                                        #
                                                        # if i % 5 == 0:
                                                        #     rand_f = randint(0, len(self.flavours) - 1)
                                                        #     # label = (tab_brand[0] + " " + tab_flavour[0] + " " + tab_col[0] + 'X' + tab_vol[0] + " " + tab_cond[0])
                                                        #     if len(self.flavours[rand_f].split(self.div)) > 0:
                                                        #         rand_i = randint(0, len(self.flavours[rand_f].split(self.div)) - 1)
                                                        #         line = (brand + " " + extra + flavour + self.flavours[rand_f].split(self.div)[rand_i] + " " + col + " " + vol + " " + cond)
                                                        #     else:
                                                        #         line = (brand + " " + extra + flavour + self.flavours[rand_f][:-1] + " " + col + " " + vol + " " + cond)
                                                        #     print_thread.print_files(label + ';;' + line + '\n')
                                                            # print_thread.queue.put(label + ';;' + line + '\n')
                                                        i += 1
                                                        # self.write_in_files((label + ';;' + line + '\n'))
        print_thread.queue.join()
        print_thread.close()


class PrintThread(threading.Thread):
    def __init__(self, tmp_path, total, diferent_files, max=200000):
        threading.Thread.__init__(self)
        self.queue = queue.Queue()
        self.diferent_files = diferent_files * 2
        self.max = total // (diferent_files)
        self.fds = []
        self.current_fd = []
        self.wrote = 0
        self.number_files = 0
        self.current_file = 0

        self.tmp_path = tmp_path
        self.total_elements = total
        for fd in range(self.diferent_files):
            self.current_fd.append(open(self.concat_name(self.number_files), 'a'))
            self.number_files += 1

    def print_files(self, line):
        # if self.wrote // self.diferent_files >= self.max:
        #     self.fds += self.current_fd
        #     for fd in self.current_fd:
        #         fd.close()
        #     self.current_fd = []
        #     for fd in range(self.diferent_files):
        #         self.current_fd.append(open(self.concat_name(self.number_files), 'w'))
        #         self.number_files += 1
        #     self.wrote = 0
        #     self.current_file = 0
        self.current_fd[self.current_file].write(line)
        self.wrote += 1
        self.current_file += 1
        if self.current_file >= self.diferent_files:
            self.current_file = 0

    def concat_name(self, name):
        return self.tmp_path + str(name) + '_' + str(self.wrote) + '.ds'

    def close(self):
        self.fds += self.current_fd
        for fd in self.fds:
            fd.close()

    def run(self):
        while True:
            result = self.queue.get()
            self.print_files(result)
            self.queue.task_done()
