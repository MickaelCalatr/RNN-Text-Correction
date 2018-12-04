import tensorflow as tf

from collections import namedtuple

from Sources.Model.operation.Input_Op import InputOp
from Sources.Model.operation.Seq2seq_Op import Seq2seqOp

class Network(object):
    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf
        self.keep_prob = conf.keep_prob
        self.rnn_size = conf.rnn_size
        self.num_layers = conf.num_layers
        self.learning_rate = conf.learning_rate
        self.embedding_size = conf.embedding_size
        self.direction = conf.direction

    def init_weights_loss(self, targets_length, max_target_length):
        masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
        return masks

    def build_graph(self, batch_size, vocab_to_int):
        tf.reset_default_graph()

        # Load the model inputs
        inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = InputOp().add_op()

        # Create the training and inference logits
        training_logits, inference_logits = Seq2seqOp(self.conf).add_op((tf.reverse(inputs, [-1])),
                                                          targets,
                                                          self.keep_prob,
                                                          inputs_length,
                                                          targets_length,
                                                          max_target_length,
                                                          len(vocab_to_int) + 1,
                                                          vocab_to_int)

        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits.rnn_output, 'logits')

        with tf.name_scope('predictions'):
            predictions = tf.identity(inference_logits.sample_id, name='predictions')
            tf.summary.histogram('predictions', predictions)


        # Create the weights for sequence_loss
        masks = self.init_weights_loss(targets_length, max_target_length)

        with tf.name_scope("cost"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                    targets,
                                                    masks)
            tf.summary.scalar('cost', cost)

        with tf.name_scope("optimze"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        # Merge all of the summaries
        merged = tf.summary.merge_all()

        # Export the nodes
        export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                        'predictions', 'merged', 'train_op','optimizer', 'accuracy']
        Graph = namedtuple('Graph', export_nodes)
        local_dict = locals()
        graph = Graph(*[local_dict[each] for each in export_nodes])

        return graph
