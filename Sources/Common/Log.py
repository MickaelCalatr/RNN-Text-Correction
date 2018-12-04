from Sources.Model.Configuration import config

def log(string):
    if (config.verbose):
        print(string)
