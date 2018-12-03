
def open_file(path):
    """Load Dataset from file"""
    input_file = os.path.join(path)
    with open(input_file) as f:
        dataset = f.read()
    return dataset
