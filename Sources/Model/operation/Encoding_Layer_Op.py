import tensorflow as tf

class EncodingLayerOp:
    def __init__(self, rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.keep_prob = keep_prob
        self.rnn_inputs = rnn_inputs
        self.sequence_length = sequence_length

    def add_op(self, direction):
        '''Create the encoding layer'''
        if direction == 1:
            return self.direction_one()
        elif direction == 2:
            return self.direction_two()

    def direction_two(self):
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(self.rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            input_keep_prob = self.keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                            input_keep_prob = self.keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            self.rnn_inputs,
                                                                            self.sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]


    def direction_one(self):
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = self.keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,
                                                              self.rnn_inputs,
                                                              self.sequence_length,
                                                              dtype=tf.float32)
            return enc_output, enc_state
