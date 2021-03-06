# coding: utf-8

# # Creating a Spell Checker

# The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. The data that we will use for this project will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads). Our model is designed using grid search to find the optimal architecture, and hyperparameter values. The best results, as measured by sequence loss with 15% of our data, were created using a two-layered network with a bi-direction RNN in the encoding layer and Bahdanau Attention in the decoding layer. [FloydHub's](https://www.floydhub.com/) GPU service was used to train the model.
#
# The sections of the project are:
# - Loading the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Fixing Custom Sentences
# - Summary

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
from sklearn.model_selection import train_test_split
import json
import os

from Common.dataset import *
from Common.dataset_augmentation import dataset_augmentation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = './Dataset/'
dataset_file = "dataset.json"

training_sorted = []
testing_sorted = []

train_data = []
test_data = []

vocab_to_int = {}
int_to_vocab = {}


# In[72]:

# The default parameters
epochs = 100
batch_size = 32
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75
lengths = []

# Train the model with the desired tuning parameters
def start_train():
    for keep_probability in [0.75]:
        for num_layers in [2]:
            for threshold in [0.95]:
                log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                        num_layers,
                                                        threshold)
                model = build_graph(keep_probability, rnn_size, num_layers, batch_size,
                                    learning_rate, embedding_size, direction)
                train(model, epochs, log_string)

def create_dic(data, labels):
    count = 0
    for line in data:
        for c in line:
            if c not in vocab_to_int:
                vocab_to_int[c] = count
                count += 1
    for line in labels:
        for c in line:
            if c not in vocab_to_int:
                vocab_to_int[c] = count
                count += 1
    codes = ['<PAD>', '<EOS>', '<GO>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1
    for c, value in vocab_to_int.items():
        int_to_vocab[value] = c
    print("Total Vocab: {}".format(len(vocab_to_int)))
    print(sorted(vocab_to_int))

def main():
    print("[1] Loading data...")
    data = load_file(path, dataset_file)
    print("[1] Loading: Done !")

    print("[2] Dataset loading...")
    size, raw_dataset = load_dataset(data)
    print("[2] Dataset loading: Done !")
    print("     Dataset size: " + str(size) + " elements.")

    print("[3] Dataset augmentation...")
    augmentation_size, data, labels = dataset_augmentation(raw_dataset)
    for i in range(5):
        print(i, "DATA = ", data[i])
        print(i, "LABEL = ", labels[i])
    create_dic(data, labels)
    int_sentences = []
    int_labels = []
    for s in data:
        int_sentence = []
        for c in s:
            int_sentence.append(vocab_to_int[c])
        int_sentences.append(int_sentence)
    for s in labels:
        int_sentence = []
        for c in s:
            int_sentence.append(vocab_to_int[c])
        int_labels.append(int_sentence)

    lengths = []
    for sentence in int_sentences:
        lengths.append(len(sentence))
    lengths = pd.DataFrame(lengths, columns=["counts"])

    for i in range(5):
        print(i, "DATA = ", int_sentences[i])
        print(i, "LABEL = ", int_labels[i])
    # In[20]:

    lengths.describe()


    # In[21]:

    # Limit the data we will use to train our model
    max_length = 92
    min_length = 10

    data_sentences = []
    label_sentences = []
    #
    # i = 0
    # for sentence in int_sentences:
    #     if len(sentence) <= max_length and len(sentence) >= min_length:
    #         data_sentences.append(sentence)
    #         label_sentences.append(int_labels[i])
    #     i+=1

    # print("We will use {} to train and test our model.".format(len(good_sentences)))


    # *Note: I decided to not use very long or short sentences because they are not as useful for training our model. Shorter sentences are less likely to include an error and the text is more likely to be repetitive. Longer sentences are more difficult to learn due to their length and increase the training time quite a bit. If you are interested in using this model for more than just a personal project, it would be worth using these longer sentence, and much more training data to create a more accurate model.*
    # data_sentences, label_sentences = randomise_dataset(data_sentences, label)
    # In[22]:
    global train_data, training_sorted, test_data, testing_sorted
    # Split the data into training and testing sentences
    train_data, test_data = train_test_split(int_sentences, test_size = 0.15, shuffle=False)#random_state = 2)
    training_sorted, testing_sorted = train_test_split(int_labels, test_size = 0.15, shuffle=False)#random_state = 2)
    # training, testing = train_test_split(data_sentences, test_size = 0.15, shuffle=False)#random_state = 2)
    # label_training, label_testing = train_test_split(label_sentences, test_size = 0.15, shuffle=False)#random_state = 2)
    #
    # print("Number of training sentences:", len(training))
    # print("Number of testing sentences:", len(testing))

    for i in range(5):
        print(i, "DATA = ", train_data[i])
        print(i, "LABEL = ", training_sorted[i])
    # In[23]:

    # Sort the sentences by length to reduce padding, which will allow the model to train faster
    # global train_data, training_sorted, test_data, testing_sorted
    #
    # for i in range(min_length, max_length+1):
    #     for j in range(0, len(training)):
    #         if len(training[j]) == i:
    #             training_sorted.append(label_training[j])
    #             train_data.append(training[j])
    #     for j in range(0, len(testing)):
    #         if len(testing[j]) == i:
    #             testing_sorted.append(label_testing[j])
    #             test_data.append(testing[j])

    # lines, labels = dataset_convert_to_int(data, labels)
    print("[3] Dataset augmentation: Done !")
    print("     Dataset augmentated of : " + str(int(augmentation_size * 100 / size)) + "%.")

    print("[4] Dataset randomisation...")

    # # Sort the sentences by length to reduce padding, which will allow the model to train faster
    # i = int(len(int_labels) * 0.75) - 1
    # global train_data, training_sorted, test_data, testing_sorted
    # training_sorted = int_labels[:i]
    # train_data = int_sentences[:i]
    # testing_sorted = int_labels[i:]
    # test_data = int_sentences[i:]
    #
    # global lengths
    # for sentence in training_sorted:
    #     lengths.append(len(sentence))
    # lengths = pd.DataFrame(lengths, columns=["counts"])
    # print(lengths)
    # data, labels = randomise_dataset(lines, labels)
    for i in range(5):
        print(i, "DATA = ", train_data[i])
        print(i, "LABEL = ", training_sorted[i])
    print("[4] Dataset randomisation: Done !")

    print("[5] Training...")
    start_train()
    print("[4] Training: Done !")
    print("[5] Testing...")
    test()
    print("[4] Testing: Done !")




def get_batches(sentences, batch_size, threshold):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""

    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        # sentences_batch = sentences[start_i:start_i + batch_size]
        #
        # sentences_batch_noisy = []
        # for sentence in sentences_batch:
        #     sentences_batch_noisy.append(noise_maker(sentence, threshold))
        #
        # sentences_batch_eos = []
        # for sentence in sentences_batch:
        #     sentence.append(vocab_to_int['<EOS>'])
        #     sentences_batch_eos.append(sentence)
        #
        # pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        # pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        # print(pad_sentence_batch)
        # print(pad_sentences_noisy_batch)
        # # Need the lengths for the _lengths parameters
        # pad_sentences_lengths = []
        # for sentence in pad_sentences_batch:
        #     pad_sentences_lengths.append(len(sentence))
        #
        # pad_sentences_noisy_lengths = []
        # for sentence in pad_sentences_noisy_batch:
        #     print(sentence)
        #     pad_sentences_noisy_lengths.append(len(sentence))
        #
        # print(pad_sentences_lengths)
        # print(pad_sentences_noisy_lengths)
        # yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths

        if len(sentences) == len(test_data):
            lines_batch = test_data[start_i:start_i + batch_size]
        else :
            lines_batch = train_data[start_i:start_i + batch_size]
        labels_batch = sentences[start_i:start_i + batch_size]

        sentences_batch_eos = []
        for sentence in labels_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)

        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(lines_batch))

        # Need the lengths for the _lengths parameters

        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))

        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths



letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]

def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''

    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence



# # # Building the Model
#
# # In[63]:

def model_inputs():
    '''Create palceholders for inputs to the model'''

    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


# In[64]:

def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[65]:

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''

    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state


    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]


# In[66]:

def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_target_length):
    '''Create the training logits'''

    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)

        training_logits, _ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)
        # training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
        #                                                        output_time_major=False,
        #                                                        impute_finished=True,
        #                                                        maximum_iterations=max_target_length)
        return training_logits


# In[67]:

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):
    '''Create the inference logits'''

    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _ ,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)
        # inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
        #                                                         output_time_major=False,
        #                                                         impute_finished=True,
        #                                                         maximum_iterations=max_target_length)

        return inference_logits


# In[68]:

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length,
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)
        # dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell, attn_mech, rnn_size)

    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)
    # initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state, _zero_state_tensors(rnn_size, batch_size, tf.float32))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  targets_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[69]:

def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    '''Use the previous functions to create the training and inference logits'''

    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers,
                                           enc_embed_input, keep_prob, direction)

    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits  = decoding_layer(dec_embed_input,
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size,
                                                        inputs_length,
                                                        targets_length,
                                                        max_target_length,
                                                        rnn_size,
                                                        vocab_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers,
                                                        direction)

    return training_logits, inference_logits


# In[70]:

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[71]:

# *Note: This set of values achieved the best results.*


# In[73]:

def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.reset_default_graph()

    # Load the model inputs
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                targets,
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph




# ## Training the Model

# In[74]:

def train(model, epochs, log_string):
    '''Train the RNN'''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0

        display_step = 30 # The progress of the training will be displayed after every 30 batches
        stop_early = 0
        stop = 3 # If the batch_loss_testing does not decrease in 3 consecutive checks, stop training
        per_epoch = 3 # Test the model 3 times per epoch
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))
        for epoch_i in range(1, epochs+1):
            batch_loss = 0
            batch_time = 0
            i = 0
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged, model.cost,model.train_op], {model.inputs: input_batch, model.targets: target_batch, model.inputs_length: input_length, model.targets_length: target_length, model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(training_sorted) // batch_size,
                                  batch_loss / display_step,
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost],
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing,
                                  batch_time_testing))

                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!')
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)
                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

                if stop_early == stop:
                    print("Stopping Training.")
                    break


def text_to_ints(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

# In[ ]:
def tester():
    # Create your own sentence or use one from the dataset
    text = "Coca x6x2 fraise 33 mls       "
    text = text_to_ints(text)

    #random = np.random.randint(0,len(testing_sorted))
    #text = testing_sorted[random]
    #text = noise_maker(text, 0.95)

    checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"

    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

    with tf.Session() as sess:
        # Load saved model
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                     model.inputs_length: [len(text)]*batch_size,
                                                     model.targets_length: [len(text)+1],
                                                     model.keep_prob: [1.0]})[0]

    # Remove the padding from the generated sentence
    pad = vocab_to_int["<PAD>"]

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))

main()
# ## Fixing Custom Sentences

# In[75]:
#


#
# # In[176]:
#
# # Create your own sentence or use one from the dataset
# text = "Spellin is difficult, whch is wyh you need to study everyday."
# text = text_to_ints(text)
#
# #random = np.random.randint(0,len(testing_sorted))
# #text = testing_sorted[random]
# #text = noise_maker(text, 0.95)
#
# checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"
#
# model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)
#
# with tf.Session() as sess:
#     # Load saved model
#     saver = tf.train.Saver()
#     saver.restore(sess, checkpoint)
#
#     #Multiply by batch_size to match the model's input parameters
#     answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
#                                                  model.inputs_length: [len(text)]*batch_size,
#                                                  model.targets_length: [len(text)+1],
#                                                  model.keep_prob: [1.0]})[0]
#
# # Remove the padding from the generated sentence
# pad = vocab_to_int["<PAD>"]
#
# print('\nText')
# print('  Word Ids:    {}'.format([i for i in text]))
# print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))
#
# print('\nSummary')
# print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
# print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


# Examples of corrected sentences:
# - Spellin is difficult, whch is wyh you need to study everyday.
# - Spelling is difficult, which is why you need to study everyday.
#
#
# - The first days of her existence in th country were vrey hard for Dolly.
# - The first days of her existence in the country were very hard for Dolly.
#
#
# - Thi is really something impressiv thaat we should look into right away!
# - This is really something impressive that we should look into right away!

# ## Summary

# I hope that you have found this project to be rather interesting and useful. The example sentences that I have presented above were specifically chosen, and the model will not always be able to make corrections of this quality. Given the amount of data that we are working with, this model still struggles. For it to be more useful, it would require far more training data, and additional parameter tuning. This parameter values that I have above worked best for me, but I expect there are even better values that I was not able to find.
#
# Thanks for reading!
