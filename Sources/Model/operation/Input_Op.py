import tensorflow as tf

class InputOp:
    def add_op(self):
        """
        Initialises all input placeholders needed in the graph
        :return: Input and labels placeholders
        """
        with tf.name_scope('inputs'):
            inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        with tf.name_scope('targets'):
            targets = tf.placeholder(tf.int32, [None, None], name='targets')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
        targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
        max_target_length = tf.reduce_max(targets_length, name='max_target_len')

        return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length
