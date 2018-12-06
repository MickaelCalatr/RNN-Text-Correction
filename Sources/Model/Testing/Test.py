import time
import os
import json

import numpy as np
import tensorflow as tf

from Sources.Model.Network import Network

class Test(object):
    def __init__(self, conf):
        super(Test, self).__init__()
        self.conf = conf
        self.num_layers = conf.num_layers
        self.batch_size = conf.batch_size
        self.keep_probab = 0.75
        self.display_step = conf.display_step
        self.checkpoint = self.conf.directory + 'kp={},nl={}.ckpt'.format(self.keep_probab, self.num_layers)
        self.epochs = 100
        self.stop = conf.stop
        self.vocab_to_int = {}
        self.int_to_vocab = {}

    def load_dic(self):
        input_file = os.path.join('./Model/saved_dic.json')
        with open(input_file) as f:
            data = json.load(f)
            self.vocab_to_int = data['vocab_to_int']
            self.int_to_vocab = data['int_to_vocab']
        print("Total Vocab: {}".format(len(self.vocab_to_int)))
        print(sorted(self.vocab_to_int))

    def test_model(self, text):
        print("[1] Loading dic...")
        self.load_dic()
        print("[1] Loading: Done !")

        final = []
        text = self.text_to_ints(text)

        network = Network(self.conf)
        model = network.build_graph(self.batch_size, self.vocab_to_int)
        with tf.Session() as sess:
            # Load saved model
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint)

            #Multiply by batch_size to match the model's input parameters
            answer_logits = sess.run(model.predictions, {model.inputs: [text] * self.batch_size,
                                                         model.inputs_length: [len(text)] * self.batch_size,
                                                         model.targets_length: [len(text) + 1],
                                                         model.keep_prob: [1.0]})[0]

        # Remove the padding from the generated sentence
        pad = self.vocab_to_int["<PAD>"]

        print('\nText')
        print('  Input Words:    {}'.format("".join([self.int_to_vocab[str(i)] for i in text if i != pad])))
        print('  Response Words: {}'.format("".join([self.int_to_vocab[str(i)] for i in answer_logits if i != pad])))


    def text_to_ints(self, text):
        '''Prepare the text for the model'''
        ret = [self.vocab_to_int[word] for word in text]
        for i in range(10):
            ret.append(self.vocab_to_int['<PAD>'])
        return ret
