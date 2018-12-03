import configparser
import json

from Common.directory_handler import check_path

VERSION = "1.8.3"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help = "This is the dataset.", type=str, default="./dataset.json")
ap.add_argument("-c", "--config_file", help = "Config file of the Network training.", type=str, default="./config.ini")

ap.add_argument("-s", '--save_directory', help = "Directory to save all CNN files.", type=str, default="./Model/")
ap.add_argument("-v", '--version', action='version', version='%(prog)s V' + str(VERSION))
ap.add_argument("-l", "--log_verbose", help="Verbose mode to print the log.", type=bool, default=False)
ap.add_argument("-lr", "--learning_rate", help="This field change the learning rate.", type=float, default=0.0)
FLAGS = vars(ap.parse_args())

class Configuration:
    def __init__(self, args):
        super(Config, self).__init__()
        self.input_file = args['input']
        self.verbose = args['log_verbose']
        self.directory = check_path(args['save_directory'])
        self.train_directory = self.directory + "train/"
        self.summary_directory = self.directory + "log/"
        self.
        # Network
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(args['config_file'])

        if args['learning_rate'] != 0.0:
            self.learning_rate = args['learning_rate']
        else:
            self.learning_rate = get(conf, 'Network', 'Learning_rate')
        self.display_step = get(conf, 'Network', 'Display_step')
        self.test_step = get(conf, 'Network', 'Test_step')
        self.batch_size = get(conf, 'Network', 'Batch_size')
        self.cross_validation = get(conf, 'Network', 'Cross_validation')


    def get(self, conf, section, key=None):
        result = {}
        options = conf.options(section)
        for option in options:
            try:
                result[option] = self.smartcast(conf.get(section, option))
                if result[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                result[option] = None
        if key == None:
            return result
        return result[key]


    def smartcast(self, value):
        tests = [int, float]
        for test in tests:
            try:
                return test(value)
            except ValueError:
                continue
        return value

    def save(self, path_to_save):
        CNN_Section = "CNN"
        Image_Section = "Image"
        cfgfile = open(path_to_save, 'w')
        Config = configparser.ConfigParser()
        Config.add_section(CNN_Section)
        Config.add_section(Image_Section)
        Config.set(CNN_Section, 'Model', 'model.meta')
        Config.set(CNN_Section, 'Train_directory', self.directory)
        Config.set(Image_Section, 'Size', self.CNN.get('Image', 'size'))
        Config.set(Image_Section, 'Channels', self.CNN.get('Image', 'channels'))
        Config.write(cfgfile)
        cfgfile.close()
