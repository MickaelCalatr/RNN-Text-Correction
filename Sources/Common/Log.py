import logging
from Sources.Model.Configuration import config

if config.verbose:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
else:
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def log(string):
    logging.warning(string)
