
import tensorflow as tf

from Common.dataset import *
from Common.dataset_augmentation import *
from model_builder import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = './'
dataset_file = "dataset.json"


vocab_to_int = {}
int_to_vocab = {}


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


def test():
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

    text = "Coa x2x6 33 mili BT         "
    final = []

    text = text_to_ints(text)
    # for i in range(10):
    #     text.append(vocab_to_int('<EOS>'))
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


def text_to_ints(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

test()
