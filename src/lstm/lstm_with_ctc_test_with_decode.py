'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Author: Yang Shuai
Project: https://github.com/ys10/TensorFlowDemo
'''

from __future__ import print_function

import time, os
import configparser
import logging
import tensorflow as tf
import h5py
from math import ceil
from tensorflow.contrib import rnn
from src.lstm.utils import *

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops


# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/ctc/test.ini')

# Config the logger.
# Output into log file.
log_file_name = cp.get('log', 'log_dir') + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log'
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level = logging.DEBUG,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file_name,
                filemode='w')
# Output to the console.
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Import data set
# Name of file storing trunk names.
trunk_names_file_name = cp.get('data', 'trunk_names_file_name')
# Name of HDF5 file as training data set.
test_data_file_name = cp.get('data', 'test_data_file_name')
# Read trunk names.
trunk_names_file = open(trunk_names_file_name, 'r')
# Read training data set.
test_data_file = h5py.File(test_data_file_name, 'r')
# Output file name.
outpout_data_file_name = cp.get('result', 'output_data_file_dir')+ time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.hdf5'
if not os.path.exists(outpout_data_file_name):
    f = open(outpout_data_file_name, 'w')
    f.close()
# Output files.
outpout_data_file = h5py.File(outpout_data_file_name, 'w')
#
lstm_grp = outpout_data_file.create_group("lstm_output")
linear_grp = outpout_data_file.create_group("linear_output")

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
training_epochs = 1
# For dropout to prevent over-fitting.
# Neural network will not work with a probability of 1-keep_prob.
keep_prob = 1.0

# Network Parameters
n_input = 69 # data input
n_steps = 777 # time steps
n_classes = 49 # total classes

# n_input = 36 # data input
# n_steps = 1496 # time steps
# n_classes = 47 # total classes
n_hidden = 384 # hidden layer num of features
n_layers = 2 # num of hidden layers

#
trunk_name = ''

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, None, n_input])
y = tf.sparse_placeholder(tf.int32, [batch_size, None])
seq_len = tf.placeholder(tf.int32, [None])

# Define parameters of full connection between the second LSTM layer and output layer.
# Define weights.

with tf.variable_scope("LSTM") as vs:
    weights = {
        # 'out': tf.Variable(tf.random_normal([n_hidden, n_classes], dtype=tf.float64), dtype = tf.float64)
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    # Define biases.
    biases = {
        # 'out': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64), dtype = tf.float64)
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Drop out in case of over-fitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stack two same lstm cell
    stack = rnn.MultiRNNCell([lstm_cell] * n_layers)

    # Define output saving function
    def output_data_saving(trunk_name, lstm_grp, linear_grp, greedy_decode_grp, beam_decode_grp, logits, outputs, greedy_decoded, beam_decoded):
        # Sub group.
        logits_array = tensor_to_array(logits)
        linear_grp.create_dataset(trunk_name, shape = logits.shape, data = logits_array, dtype = 'f')
        outputs_array = tensor_to_array(outputs)
        lstm_grp.create_dataset(trunk_name, shape = outputs.shape, data = outputs_array, dtype = 'f')
        # decode_array = tf.cast(decoded[0], tf.int32)
        greedy_decode_grp.create_dataset(trunk_name, shape = greedy_decoded[0].dense_shape, data = greedy_decoded[0].values, dtype = 'i')
        beam_decode_grp.create_dataset(trunk_name, shape = beam_decoded[0].dense_shape, data = beam_decoded[0].values, dtype='i')
        return

    # Define LSTM as a RNN.
    def RNN(x, seq_len, weights, biases):

        # Get lstm cell outputs with shape (n_steps, batch_size, n_input).
        outputs, states = tf.nn.dynamic_rnn(stack, x, seq_len, dtype=tf.float32)
        # Permuting batch_size and n_steps.
        outputs = tf.reshape(outputs, [-1, n_hidden])
        # Now, shape of outputs is (batch_size, n_steps, n_input)
        # Linear activation, using rnn inner loop last output
        # The first dim of outputs & weights must be same.
        logits = tf.matmul(outputs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, -1, n_classes])
        logits = tf.nn.softmax(logits)
        return logits, outputs

    # Define prediction of RNN(LSTM).
    pred, outputs = RNN(x, seq_len, weights, biases)

    # Define loss and optimizer.
    cost = tf.reduce_mean( ctc_ops.ctc_loss(labels = y, inputs = pred, sequence_length = seq_len, time_major=False))

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    beam_decoded, _ = ctc_ops.ctc_beam_search_decoder(tf.transpose(pred, (1, 0, 2)), seq_len)
    greedy_decoded, _ = ctc_ops.ctc_greedy_decoder(tf.transpose(pred, (1, 0, 2)), seq_len)

    # Inaccuracy: label error rate
    beam_ler = tf.reduce_mean(tf.edit_distance(tf.cast(beam_decoded[0], tf.int32), y))
    greedy_ler = tf.reduce_mean(tf.edit_distance(tf.cast(greedy_decoded[0], tf.int32), y))

    # Configure session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Initialize the saver to save session.
    lstm_variables = [v for v in tf.global_variables()
                        if v.name.startswith(vs.name)]
    saver = tf.train.Saver(lstm_variables)
    saved_model_path = cp.get('model', 'saved_model_path')
    # Launch the graph
    with tf.Session(config=config) as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_model_path)
        logging.info("Model restored from file: " + saved_model_path)
        # Keep training until reach max iterations
        logging.info("Start training!")
        # Read all trunk names.
        all_trunk_names = trunk_names_file.readlines()
        for epoch in range(0, training_epochs, 1):
            train_cost = train_greedy_ler = train_beam_ler = 0
            start = time.time()
            # For each iteration.
            logging.debug("epoch:" + str(epoch))
            # Output groups.
            epoch_grp = outpout_data_file.create_group("epoch" + str(epoch))
            lstm_grp = epoch_grp.create_group("lstm_output")
            linear_grp = epoch_grp.create_group("linear_output")
            beam_decode_grp = epoch_grp.create_group("beam_decode")
            greedy_decode_grp = epoch_grp.create_group("greedy_decode")
            # Break out of the training iteration while there is no trunk usable.
            if not all_trunk_names:
                break
            trunk_name = ''
            # Every batch only contains one trunk.
            trunk = 0
            for line in all_trunk_names:
                trunk_name = line.split()[0]
                # print("trunk_name: " + trunk_name)
                # print("length:"+ len(trunk_name))
                # Define two variables to store input data.
                batch_x = []
                batch_y = []
                batch_seq_len = []
                # Get trunk data by trunk name without line break character.
                # sentence_x is a tensor of shape (n_steps, n_inputs)
                sentence_x = test_data_file['source/' + trunk_name.strip('\n')]
                # sentence_y is a tensor of shape (None)
                sentence_y = test_data_file['target/' + trunk_name.strip('\n')]
                # sentence_len is a tensor of shape (None)
                sentence_len = test_data_file['size/' + trunk_name.strip('\n')].value
                # Add current trunk into the batch.
                batch_x.append(sentence_x)
                batch_y.append(sentence_y)
                # batch_seq_len.append(sentence_len)
                # Padding.
                batch_x, batch_seq_len = pad_sequences(batch_x, maxlen=n_steps)
                if(batch_seq_len[0] > n_steps):
                    continue
                batch_y = sparse_tuple_from(batch_y)
                # batch_x is a tensor of shape (batch_size, n_steps, n_inputs)
                # batch_y is a tensor of shape (batch_size, n_steps - truncated_step, n_inputs)
                # Run optimization operation (Back-propagation Through Time)
                feed_dict = {x: batch_x, y: batch_y, seq_len: batch_seq_len}
                # Calculate batch loss.
                #
                linear_outputs  = sess.run(pred, feed_dict)
                lstm_outputs = sess.run(outputs, feed_dict)
                batch_cost = sess.run(cost, feed_dict)
                # ler
                batch_greedy_ler = sess.run(greedy_ler, feed_dict)
                train_greedy_ler += batch_greedy_ler * batch_size
                #
                batch_beam_ler = sess.run(beam_ler, feed_dict)
                train_beam_ler += batch_beam_ler * batch_size
                # decode
                batch_greedy_decode = sess.run(greedy_decoded, feed_dict)
                batch_beam_decode = sess.run(beam_decoded, feed_dict)
                output_data_saving(trunk_name, lstm_grp, linear_grp, beam_decode_grp, greedy_decode_grp, linear_outputs, lstm_outputs, greedy_decoded, beam_decoded)
                logging.debug("Trunk: " + str(trunk) + " name:" + str(trunk_name) + ", cost = {}, time = {:.3f}".format(batch_cost, time.time() - start))
                logging.debug("label: " + str(batch_y))
                logging.debug("linear_outputs: " + str(linear_outputs))
                logging.debug("lstm_outputs: " + str(lstm_outputs))
                logging.debug("beam decode:" + str(batch_beam_decode))
                logging.debug("greedy decode:" + str(batch_greedy_decode))
                logging.debug("Batch beam ler= {:.6f}".format(batch_beam_ler))
                logging.debug("Batch greddy ler= {:.6f}".format(batch_greedy_ler))
                trunk += 1
                # if trunk >=2:
                #     break;
            # break;
        logging.info("Testing Finished!")