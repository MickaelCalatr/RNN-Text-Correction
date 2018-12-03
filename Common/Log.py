from configuration import conf

def log(string):
    if (conf.verbose):
        print(string)
