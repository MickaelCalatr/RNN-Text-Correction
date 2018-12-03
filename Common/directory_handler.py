import os.path

def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def check_path(path):
    if path[-1] != '/':
        path += "/"
    return path
