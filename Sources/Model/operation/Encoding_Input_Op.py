import tensorflow as tf

class EncodingInputOp:
    def __init__(self, targets, vocab_to_int, batch_size, name_scope="process_encoding"):
        self.name_scope = name_scope
        self.targets = targets
        self.batch_size = batch_size
        self.vocab_to_int = vocab_to_int

    def add_op(self):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

        with tf.name_scope(self.name_scope):
            ending = tf.strided_slice(self.targets, [0, 0], [self.batch_size, -1], [1, 1])
            dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<GO>']), ending], 1)

        return dec_input
