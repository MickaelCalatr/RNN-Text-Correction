import tensorflow as tf

from tensorflow.python.layers.core import Dense

class DecoderOp:
    def __init__(self, max_target_length, name_scope="RNN_Decoder_Cell"):
        self.max_target_length = max_target_length
        self.name_scope = name_scope

    def add_op(self, dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length,
                                  rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
        '''Create the decoding cell and attention for the training and inference decoding layers'''

        with tf.name_scope(self.name_scope):
            for layer in range(num_layers):
                with tf.variable_scope('decoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                    dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                             input_keep_prob = keep_prob)

        output_layer = Dense(vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.1, stddev=0.1))

        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                      enc_output,
                                                      inputs_length,
                                                      normalize=False,
                                                      name='BahdanauAttention')

        with tf.name_scope("Attention_Wrapper"):
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)

        initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)

        with tf.variable_scope("decode"):
            training_logits = self.add_training_op(dec_cell, dec_embed_input,
                                                      targets_length, initial_state,
                                                      output_layer)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.add_inference_op(dec_cell, embeddings,
                                                        vocab_to_int['<GO>'],
                                                        vocab_to_int['<EOS>'],
                                                        batch_size, initial_state,
                                                        output_layer)

        return training_logits, inference_logits


    def add_training_op(self, dec_cell, dec_embed_input, targets_length, initial_state, output_layer):
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
                                                                    maximum_iterations=self.max_target_length)
            return training_logits


    def add_inference_op(self, dec_cell, embeddings, start_token, end_token, batch_size, initial_state, output_layer):
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
                                                                    maximum_iterations=self.max_target_length)
            return inference_logits
