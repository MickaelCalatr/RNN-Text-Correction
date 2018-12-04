import tensorflow as tf

from Sources.Model.operation.Decoder_Op import DecoderOp
from Sources.Model.operation.Encoding_Layer_Op import EncodingLayerOp
from Sources.Model.operation.Encoding_Input_Op import EncodingInputOp


class Seq2seqOp:
    def __init__(self, conf):
        self.rnn_size = conf.rnn_size
        self.num_layers = conf.num_layers
        self.learning_rate = conf.learning_rate
        self.embedding_size = conf.embedding_size
        self.direction = conf.direction
        self.batch_size = conf.batch_size

    def add_op(self, inputs, targets, keep_prob, inputs_length, targets_length, max_target_length,
                      vocab_size, vocab_to_int):
        '''Use the previous functions to create the training and inference logits'''

        enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1, 1))
        enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)

        enc_output, enc_state = EncodingLayerOp(self.rnn_size, inputs_length, self.num_layers, enc_embed_input, keep_prob).add_op(self.direction)

        dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1, 1))

        dec_input = EncodingInputOp(targets, vocab_to_int, self.batch_size).add_op()
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        training_logits, inference_logits = DecoderOp(max_target_length).add_op(dec_embed_input,
                                                            dec_embeddings,
                                                            enc_output,
                                                            enc_state,
                                                            vocab_size,
                                                            inputs_length,
                                                            targets_length,
                                                            self.rnn_size,
                                                            vocab_to_int,
                                                            keep_prob,
                                                            self.batch_size,
                                                            self.num_layers,
                                                            self.direction)

        return training_logits, inference_logits
