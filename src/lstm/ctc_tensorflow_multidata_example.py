#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf
from tensorflow.contrib import rnn

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops

# try:
#     from python_speech_features import mfcc
# except ImportError:
#     print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
#     raise ImportError
from src.lstm.utils import sparse_tuple_from as sparse_tuple_from
from src.lstm.utils import pad_sequences as pad_sequences

import configparser
import logging
import h5py
from math import ceil
# Config the logger.
# Output into log file.
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='../../log/app.log',
                filemode='w')
# Output to the console.
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/ctc/lstm.conf')

# Import data set
# Name of file storing trunk names.
trunk_names_file_name = cp.get('data', 'trunk_names_file_name')
# Name of HDF5 file as training data set.
training_data_file_name = cp.get('data', 'training_data_file_name')
# Read trunk names.
trunk_names_file = open(trunk_names_file_name, 'r')
# Read training data set.
training_data_file = h5py.File(training_data_file_name, 'r')


# def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):
#
#     # Generating different timesteps for each fake data
#     timesteps = np.random.randint(min_size, max_size, (num_examples,))
#
#     # Generating random input
#     inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])
#
#     # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
#     labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])
#
#     return inputs, labels

training_iters = 1
def get_data():

    # # Generating different timesteps for each fake data
    # timesteps = []
    #
    # # Generating random input
    # inputs = []
    #
    # # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
    # labels = []

    all_trunk_names = trunk_names_file.readlines()
    for iter in range(0, training_iters, 1):
        # For each iteration.
        logging.debug("Iter:" + str(iter))
        inputs = []
        labels = []
        seq_len = []
        # Break out of the training iteration while there is no trunk usable.
        if not all_trunk_names:
            break
        # Calculate how many batches can the data set be divided into.
        n_batches = ceil(len(all_trunk_names)/batch_size)
        # Train the RNN(LSTM) model by batch.
        for batch in range(0, n_batches, 1):
            # For each batch.
            # Define two variables to store input data.
            batch_x = []
            batch_y = []
            batch_seq_len = []
            # Traverse all trunks of a batch.
            for trunk in range(0, batch_size, 1):
                # For each trunk in the batch.
                # Calculate the index of current trunk in the whole data set.
                trunk_name_index = n_batches * batch_size + trunk
                # There is a fact that if the number of all trunks
                #   can not be divisible by batch size,
                #   then the last batch can not get enough trunks of batch size.
                # The above fact is equivalent to the fact
                #   that there is at least a trunk
                #   whose index is no less than the number of all trunks.
                if(trunk_name_index >= len(all_trunk_names)):
                    # So some used trunks should be add to the last batch when the "fact" happened.
                    # Select the last trunk to be added into the last batch.
                    trunk_name_index = len(all_trunk_names)-1
                # Get trunk name from all trunk names by trunk name index.
                trunk_name = all_trunk_names[trunk_name_index]
                # Get trunk data by trunk name without line break character.
                # sentence_x is a tensor of shape (n_steps, n_inputs)
                sentence_x = training_data_file['source/' + trunk_name.strip('\n')]
                # sentence_y is a tensor of shape (None)
                sentence_y = training_data_file['target/' + trunk_name.strip('\n')]
                # sentence_len is a tensor of shape (None)
                sentence_len = training_data_file['size/' + trunk_name.strip('\n')]
                # Add current trunk into the batch.
                batch_x.append(sentence_x)
                batch_y.append(sentence_y)
                batch_seq_len.append(sentence_len)
            inputs.append(batch_x)
            labels.append(batch_y)
            seq_len.append(batch_seq_len)
            break
    return inputs, labels, seq_len

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 69
# Accounting the 0th indice +  space + blank label = 28 characters
# num_classes = ord('z') - ord('a') + 1 + 1 + 1
num_classes = 49

# Hyper-parameters
num_epochs = 1
num_hidden = 384
num_layers = 2
batch_size = 16
initial_learning_rate = 1e-2
momentum = 0.9
keep_prob = 1.0

num_examples = 1
# num_batches_per_epoch = int(num_examples/batch_size)
num_batches_per_epoch = 1

# inputs, labels = fake_data(num_examples, num_features, num_classes - 1)
inputs, labels, seq_len = get_data()

# You can preprocess the input data here
train_inputs = inputs

# You can preprocess the target data here
train_targets = labels

train_seq_len = seq_len

# THE MAIN CODE!

# graph = tf.Graph()
# with graph.as_default():


# e.g: log filter bank or MFCC features
# Has size [batch_size, max_stepsize, num_features], but the
# batch_size and max_stepsize can vary along each step
inputs = tf.placeholder(tf.float32, [None, None, num_features])

# Here we use sparse_placeholder that will generate a
# SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32)

# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

# Truncated normal with mean 0 and stdev=0.1
# Tip: Try another initialization
# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
weights = {
    'out':tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
}
# Zero initialization
# Tip: Is tf.zeros_initializer the same?
biases = {
    'out':tf.Variable(tf.constant(0., shape=[num_classes]))
}

with tf.variable_scope("LSTM") as vs:
    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    lstm_cell = rnn.LSTMCell(num_hidden, forget_bias=1.0)
    # Drop out in case of over-fitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stacking rnn cells
    stack = rnn.MultiRNNCell([lstm_cell] * num_layers)

    def RNN(inputs, seq_len, weights, biases):
        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Doing the affine projection
        logits = tf.matmul(outputs, weights['out']) + biases['out']

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        return logits

    pred = RNN(inputs, seq_len, weights, biases)

    loss = ctc_ops.ctc_loss(targets, pred, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate).minimize(cost)
    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = ctc_ops.ctc_greedy_decoder(pred, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))
    # Initialize the saver to save session.
    # dict = {'weight_out': weights['out'], 'biases_out': biases['out']}
    # saver = tf.train.Saver(dict)
    # saved_model_path = cp.get('model', 'saved_model_path')
    # Initialize the saver to save session.
    lstm_variables = [v for v in tf.global_variables()
                        if v.name.startswith(vs.name)]
    saver = tf.train.Saver(lstm_variables)
    saved_model_path = cp.get('model', 'saved_model_path')
    to_save_model_path = cp.get('model', 'to_save_model_path')

    with tf.Session() as sess:
        # Initializate the weights and biases
        init = tf.global_variables_initializer()
        sess.run(init)
        load_path = saver.restore(sess, saved_model_path)
        logging.info("Model restored from file: " + saved_model_path)
        logging.info("Start training!")
        # Restore model weights from previously saved model
        # load_path = saver.restore(session, saved_model_path)
        # logging.info("Model restored from file: " + saved_model_path)

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):

                # Getting the index
                # indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs = train_inputs[batch]
                # batch_train_inputs = train_inputs
                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparse_tuple_from(train_targets[batch])
                # batch_train_targets = sparse_tuple_from(train_targets)

                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len}

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += sess.run(ler, feed_dict=feed)*batch_size


            # Shuffle the data
            # shuffled_indexes = np.random.permutation(num_examples)
            shuffled_indexes = 0
            train_inputs = train_inputs[shuffled_indexes]
            train_targets = train_targets[shuffled_indexes]

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))

        # Decoding all at once. Note that this isn't the best way

        # Padding input to max_time_step of this batch
        batch_train_inputs, batch_train_seq_len = pad_sequences(train_inputs)

        # Converting to sparse representation so as to to feed SparseTensor input
        batch_train_targets = sparse_tuple_from(train_targets)

        feed = {inputs: batch_train_inputs,
                targets: batch_train_targets,
                seq_len: batch_train_seq_len
                }

        # Decoding
        d = sess.run(decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)

        for i, seq in enumerate(dense_decoded):

            seq = [s for s in seq if s != -1]

            logging.debug('Sequence %d' % i)
            logging.debug('\t Original:\n%s' % train_targets[i])
            logging.debug('\t Decoded:\n%s' % seq)