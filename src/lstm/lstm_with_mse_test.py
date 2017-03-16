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
import time
import os

# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/mse/test.ini')

# Config the logger.
# Output into log file.
log_file_name = cp.get('log', 'log_dir') + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log'
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= log_file_name ,
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
training_data_file = h5py.File(test_data_file_name, 'r')
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
training_iters = 10
# For dropout to prevent over-fitting.
# Neural network will not work with a probability of 1-keep_prob.
keep_prob = 1.0
# Step of truncated back propagation.
truncated_step = 100

# Network Parameters
n_input = 69 # data input
n_steps = 200 # time steps
n_hidden = 384 # hidden layer num of features
n_layers = 2 # num of hidden layers
n_classes = 49 # total classes

#
trunk_name = ''

# tf Graph input
x = tf.placeholder("float32", [batch_size, n_steps, n_input])
y = tf.placeholder("int32", [batch_size, n_steps, n_classes])

with tf.variable_scope("LSTM") as vs:
    # Define parameters of full connection between the second LSTM layer and output layer.
    # Define weights.
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    # Define biases.
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Drop out in case of over-fitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stack two same lstm cell
    stack = rnn.MultiRNNCell([lstm_cell] * n_layers)

    # Define output saving function
    def output_data_saving(trunk_name, logits, outputs):
        linear_grp.create_dataset(trunk_name, data = logits, dtype = 'f')
        lstm_grp.create_dataset(trunk_name, data = outputs, dtype = 'f')
        return

    # Define LSTM as a RNN.
    def RNN(x, weights, biases, trunk_name):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)
        #
        outputs, states = rnn.static_rnn(stack, x, dtype=tf.float32)
        #
        outputs = tf.reshape(outputs, [-1, n_hidden])
        # Permuting batch_size and n_steps.
        # Now, shape of outputs is (batch_size, n_steps, n_input)
        # Linear activation, using rnn inner loop last output
        logits = tf.matmul(outputs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, -1, n_classes])
        logits = tf.nn.softmax(logits)
        #
        output_data_saving(trunk_name, logits, outputs)
        return logits

    # Define prediction of RNN(LSTM).
    pred = RNN(x, weights, biases, trunk_name)

    # Define loss and optimizer.
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    cost = tf.reduce_mean(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Configure session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Initialize the saver to save session.
    # lstm = tf.Variable(stack , name="lstm")
    # variables_dict = {'weights_out': weights['out'], 'biases_out': biases['out'], 'lstm': stack}
    # saver = tf.train.Saver(variables_dict)
    # saved_model_path = cp.get('model', 'saved_model_path')
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
        for iter in range(0, training_iters, 1):
            start = time.time()
            # For each iteration.
            logging.debug("Iter:" + str(iter))
            # Break out of the training iteration while there is no trunk usable.
            if not all_trunk_names:
                break
            trunk_name = ''
            # Traverse all trunks of a batch.
            for trunk_name in all_trunk_names:
                # Define two variables to store input data.
                batch_x = []
                batch_y = []
                # Get trunk data by trunk name without line break character.
                # trunk_x is a tensor of shape (n_steps, n_inputs)
                trunk_x = training_data_file['source/' + trunk_name.strip('\n')]
                # trunk_y is a tensor of shape (n_steps - truncated_step, n_classes)
                trunk_y = training_data_file['target/' + trunk_name.strip('\n')]
                # Add current trunk into the batch.
                batch_x.append(trunk_x)
                batch_y.append(trunk_y)
                # batch_x is a tensor of shape (batch_size, n_steps, n_inputs)
                # batch_y is a tensor of shape (batch_size, n_steps - truncated_step, n_inputs)
                # Run optimization operation (Back-propagation Through Time)
                feed_dict = {x: batch_x, y: batch_y, trunk_name: trunk_name}
                # Print accuracy by display_batch.
                # Calculate batch accuracy.
                acc = sess.run(accuracy, feed_dict)
                # Calculate batch loss.
                loss = sess.run(cost, feed_dict)
                logging.debug("Trunk name:" + str(trunk_name)
                              + ", Batch Loss= {:.6f}".format(loss)
                              + ", Training Accuracy= {:.5f}".format(acc)
                              + ", time = {:.3f}".format(time.time() - start))
            break;
        logging.info("Testing Finished!")