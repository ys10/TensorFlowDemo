'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Author: Yang Shuai
Project: https://github.com/ys10/TensorFlowDemo
'''

from __future__ import print_function

import configparser
import logging
import tensorflow as tf
import h5py
from math import ceil
from tensorflow.contrib import rnn
from src.lstm.utils import pad_sequences as pad_sequences
from src.lstm.utils import sparse_tuple_from as sparse_tuple_from

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops


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
cp.read('../../conf/lstm_with_ctc_data.ini')

# Import data set
# Name of file storing trunk names.
trunk_names_file_name = cp.get('data', 'trunk_names_file_name')
# Name of HDF5 file as training data set.
training_data_file_name = cp.get('data', 'training_data_file_name')
# Read trunk names.
trunk_names_file = open(trunk_names_file_name, 'r')
# Read training data set.
training_data_file = h5py.File(training_data_file_name, 'r')

'''
To classify vector using a recurrent neural network,
we consider every trunk row as a sequence.
Because trunk shape is 200*69,
we will then handle 69 dimension sequences of 200 steps for every sample.
'''
# Parameters
learning_rate = 0.001
batch_size = 1
display_batch = 1
training_iters = 10
# For dropout to prevent over-fitting.
# Neural network will not work with a probability of 1-keep_prob.
keep_prob = 1.0

# Network Parameters
n_input = 69 # data input
n_steps = 777 # time steps
n_hidden = 384 # hidden layer num of features
n_layers = 2 # num of hidden layers
n_classes = 49 # total classes

# tf Graph input
x = tf.placeholder(tf.float32, [None, None, n_input])
y = tf.sparse_placeholder(tf.int32, [batch_size, None])
seq_len = tf.placeholder(tf.int32, [None])

# Define parameters of full connection between the second LSTM layer and output layer.
# Define weights.
weights = {
    # 'out': tf.Variable(tf.random_normal([n_hidden, n_classes], dtype=tf.float64), dtype = tf.float64)
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
# Define biases.
biases = {
    # 'out': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64), dtype = tf.float64)
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Define LSTM as a RNN.
def RNN(x, seq_len, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    # x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    # x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Drop out in case of over-fitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stack two same lstm cell
    cell = rnn.MultiRNNCell([lstm_cell] * n_layers)

    # Initialize the states of LSTM.
    # cell.zero_state(batch_size, dtype = tf.float32)
    # states = tf.zeros([batch_size, n_hidden * n_layers])
    # states = cell.zero_state(batch_size, dtype=tf.float32)

    # Get lstm cell outputs with shape (n_steps, batch_size, n_input).
    # tf.get_variable_scope().reuse_variables()
    outputs, states = tf.nn.dynamic_rnn(cell, x, seq_len, dtype=tf.float32)
    # Permuting batch_size and n_steps.
    # outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, n_hidden])
    # Now, shape of outputs is (batch_size, n_steps, n_input)
    # Linear activation, using rnn inner loop last output
    # The first dim of outputs & weights must be same.
    logits = tf.matmul(outputs, weights['out']) + biases['out']
    logits = tf.reshape(logits, [batch_size, -1, n_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    #
    # logits = tf.reshape(logits, [batch_size, None, n_classes])
    return logits

# Define prediction of RNN(LSTM).
pred = RNN(x, seq_len, weights, biases)

# Define loss and optimizer.
cost = tf.reduce_mean( ctc_ops.ctc_loss(labels = y, inputs = pred, sequence_length = seq_len, time_major=False))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = ctc_ops.ctc_greedy_decoder(pred, seq_len)

# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))

# Configure session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# Initialize the saver to save session.
saver = tf.train.Saver()
saved_model_path = cp.get('model', 'saved_model_path')
to_save_model_path = cp.get('model', 'to_save_model_path')

# Launch the graph
with tf.Session(config=config) as sess:
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Keep training until reach max iterations
    logging.info("Start training!")
    # Read all trunk names.
    all_trunk_names = trunk_names_file.readlines()
    for iter in range(0, training_iters, 1):
        # For each iteration.
        logging.debug("Iter:" + str(iter))
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
            batch_x, _ = pad_sequences(batch_x)
            batch_y = sparse_tuple_from(batch_y)
            # batch_x is a tensor of shape (batch_size, n_steps, n_inputs)
            # batch_y is a tensor of shape (batch_size, n_steps - truncated_step, n_inputs)
            # Run optimization operation (Back-propagation Through Time)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
            # Print accuracy by display_batch.
            if batch % display_batch == 0:
                # Calculate batch accuracy.
                # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                # Calculate batch loss.
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                logging.debug("Iter:" + str(iter) + ",Batch:"+ str(batch)
                      + ", Batch Loss= {:.6f}".format(loss))
                # logging.debug("Iter:" + str(iter) + ",Batch:"+ str(batch)
                #       + ", Batch Loss= {:.6f}".format(loss)
                #       + ", Training Accuracy= {:.5f}".format(acc))
            break;
        # Save session by iteration.
        saver.save(sess, to_save_model_path + str(iter));
        logging.info("Model saved successfully!")
    logging.info("Optimization Finished!")