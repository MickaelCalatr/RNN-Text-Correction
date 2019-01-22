import time
import sys
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

from Sources.Model.Network import Network

class Train(object):
    def __init__(self, conf):
        super(Train, self).__init__()
        self.conf = conf
        self.num_layers = conf.num_layers
        self.batch_size = conf.batch_size
        self.display_step = conf.display_step
        self.epochs = 100
        self.stop = conf.stop
        self.keep_probab = 0.75
        self.dataset = None
        self.total_batches = 100

    def pad_sentence_batch(self, sentence_batch):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [self.dataset.vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def run (self, dataset):
        self.dataset = dataset
        log_string = 'kp={},nl={}'.format(self.keep_probab, self.num_layers)
        network = Network(self.conf)
        model = network.build_graph(self.batch_size, dataset.vocab_to_int)
        self.train(model, log_string, dataset)

    def get_batch(self, vocab_to_int, i):
        while i < self.total_batches:
            pad_label_lengths = []
            pad_line_lengths = []
            label_eos = []
            raw_line, raw_label = self.dataset.get_batch(self.batch_size)
            for line, label in zip(raw_line, raw_label):
                label.append(vocab_to_int['<EOS>'])
                label_eos.append(label)

            pad_label_batch = np.array(self.pad_sentence_batch(label_eos))
            pad_batch = np.array(self.pad_sentence_batch(raw_line))

            # Need the lengths for the _lengths parameters
            for sentence in pad_batch:
                pad_line_lengths.append(len(sentence))
            for sentence in pad_label_batch:
                pad_label_lengths.append(len(sentence))

            yield pad_batch, pad_label_batch, pad_line_lengths, pad_label_lengths


    def get_batches(self, labels, data, vocab_to_int):
        """Batch sentences, noisy sentences, and the lengths of their sentences together.
           With each epoch, sentences will receive new mistakes"""

        for batch_i in range(0, len(labels) // self.batch_size):
            start_i = batch_i * self.batch_size

            lines_batch = data[start_i:start_i + self.batch_size]
            labels_batch = labels[start_i:start_i + self.batch_size]

            sentences_batch_eos = []
            for sentence in labels_batch:
                sentence.append(vocab_to_int['<EOS>'])
                sentences_batch_eos.append(sentence)

            pad_sentences_batch = np.array(self.pad_sentence_batch(sentences_batch_eos))
            pad_sentences_noisy_batch = np.array(self.pad_sentence_batch(lines_batch))

            # Need the lengths for the _lengths parameters

            pad_sentences_noisy_lengths = []
            for sentence in pad_sentences_noisy_batch:
                pad_sentences_noisy_lengths.append(len(sentence))

            pad_sentences_lengths = []
            for sentence in pad_sentences_batch:
                pad_sentences_lengths.append(len(sentence))
            yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths


    def train(self, model, log_string, dataset):
        '''Train the RNN'''
        with tf.Session() as sess:
            if self.conf.verbose:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            else:
                logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

            sess.run(tf.global_variables_initializer())

            # Used to determine when to stop the training early
            testing_loss_summary = []
            testing_acc_summary = []
            # Keep track of which batch iteration is being trained
            iteration = 0
            stop_early = 0
            testing_check = self.display_step * 5 #(len(training_sorted)//batch_size//per_epoch)-1
            self.epochs = 100#len(dataset.train_data) // self.batch_size
            self.total_batches = 100#len(dataset.train_data) // self.batch_size

            logging.info("Training Model: {}".format(log_string))

            train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
            test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))
            for epoch_i in range(1, self.epochs + 1):
                batch_loss = 0
                batch_time = 0

                acc = 0
                step = 0
                i = 0
                for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(self.get_batch(dataset.vocab_to_int, step)):
                    start_time = time.time()
                    summary, loss, _, accuracy = sess.run([model.merged, model.cost, model.train_op, model.accuracy], {model.inputs: input_batch, model.targets: target_batch, model.inputs_length: input_length, model.targets_length: target_length, model.keep_prob: self.conf.keep_prob})
                    step += 1
                    batch_loss += loss
                    end_time = time.time()
                    batch_time += end_time - start_time
                    acc += accuracy
                    # Record the progress of training
                    train_writer.add_summary(summary, iteration)

                    iteration += 1
                    if batch_i % testing_check == 0 and batch_i > 0 and accuracy > 0.85:
                        checkpoint = "./{}{}.ckpt".format(self.conf.directory, log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)
                    if batch_i % self.display_step == 0 and batch_i > 0:
                        logging.info('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}, Accuracy: {:>4.2f}%'
                              .format(epoch_i,
                                      batch_i, self.total_batches,
                                      loss,
                                      batch_time,
                                      100 * (acc / step)))
                        batch_loss = 0
                        batch_time = 0
                        acc = 0
                        step = 0
                    # if batch_i == 2000:
                    #     dataset.current_fd += 1
                    #     if dataset.current_fd == len(dataset.fds):
                    #         dataset.finish()
                    #     dataset.create_dataset(dataset.fds[dataset.current_fd])
                    #     break;

                    #### Testing ####
                    # if batch_i % testing_check == 0 and batch_i > 0:
                    #     batch_loss_testing = 0
                    #     batch_time_testing = 0
                    #     acc_average = 0
                    #     acc = 0
                    #     step = 1
                    #     for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(self.get_batch(dataset.vocab_to_int, batch_i)):
                    #         start_time_testing = time.time()
                    #         summary, loss, acc = sess.run([model.merged,
                    #                                   model.cost, model.accuracy],
                    #                                      {model.inputs: input_batch,
                    #                                       model.targets: target_batch,
                    #                                       model.inputs_length: input_length,
                    #                                       model.targets_length: target_length,
                    #                                       model.keep_prob: 1})
                    #
                    #         batch_loss_testing += loss
                    #         end_time_testing = time.time()
                    #         batch_time_testing += end_time_testing - start_time_testing
                    #         acc_average += acc
                    #         step += 1
                    #         # Record the progress of testing
                    #         test_writer.add_summary(summary, iteration)
                    #
                    #     n_batches_testing = batch_i + 1
                    #     print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}, Accuracy: {:>4.2f}%'
                    #           .format(batch_loss_testing / step,
                    #                   batch_time_testing, 100 * (acc_average / step)))
                    #
                    #     batch_time_testing = 0
                    #
                    #     # If the batch_loss_testing is at a new minimum, save the model
                    #     testing_loss_summary.append(batch_loss_testing)
                    #     testing_acc_summary.append(acc_average / step)
                    #     # if batch_loss_testing <= min(testing_loss_summary):
                    #     #     print('New Record!')
                    #     #     stop_early = 0
                    #     #     checkpoint = "./{}{}.ckpt".format(self.conf.directory, log_string)
                    #     #     saver = tf.train.Saver()
                    #     #     saver.save(sess, checkpoint)


                if stop_early == self.stop:
                    print("Stopping Training.")
                    break
    def to_save(self, l):
        return sum(l) / float(len(l)) > 85
