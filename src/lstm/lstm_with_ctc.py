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
import math
from tensorflow.contrib import rnn
from src.lstm.utils import pad_sequences as pad_sequences
from src.lstm.utils import sparse_tuple_from as sparse_tuple_from

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops


# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/ctc/lstm.ini')

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
learning_rate = 0.0001
batch_size = 16
display_batch = 1
training_iters = 5
# For dropout to prevent over-fitting.
# Neural network will not work with a probability of 1-keep_prob.
keep_prob = 1.0

# Network Parameters
# n_input = 69 # data input
# n_steps = 777 # time steps
# n_classes = 49 # total classes

n_input = 36 # data input
n_steps = 1496 # time steps
n_classes = 47 # total classes
n_hidden = 384 # hidden layer num of features
n_layers = 2 # num of hidden layers

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

        # # Define a lstm cell with tensorflow
        # lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # # Drop out in case of over-fitting.
        # lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # # Stack two same lstm cell
        # cell = rnn.MultiRNNCell([lstm_cell] * n_layers)

        # Initialize the states of LSTM.
        # cell.zero_state(batch_size, dtype = tf.float32)
        # states = tf.zeros([batch_size, n_hidden * n_layers])
        # states = cell.zero_state(batch_size, dtype=tf.float32)

        # Get lstm cell outputs with shape (n_steps, batch_size, n_input).
        outputs, states = tf.nn.dynamic_rnn(stack, x, seq_len, dtype=tf.float32)
        # Permuting batch_size and n_steps.
        # outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, n_hidden])
        # Now, shape of outputs is (batch_size, n_steps, n_input)
        # Linear activation, using rnn inner loop last output
        # The first dim of outputs & weights must be same.
        logits = tf.matmul(outputs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, -1, n_classes])
        # logits = tf.transpose(logits, (1, 0, 2))
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
    decoded, log_prob = ctc_ops.ctc_greedy_decoder(tf.transpose(pred, (1, 0, 2)), seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))

    # Configure session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Initialize the saver to save session.
    lstm_variables = [v for v in tf.global_variables()
                        if v.name.startswith(vs.name)]
    saver = tf.train.Saver(lstm_variables)
    saved_model_path = cp.get('model', 'saved_model_path')
    to_save_model_path = cp.get('model', 'to_save_model_path')


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
            train_cost = train_ler = 0
            start = time.time()
            # For each iteration.
            logging.debug("Iter:" + str(iter))
            # Break out of the training iteration while there is no trunk usable.
            if not all_trunk_names:
                break
            logging.debug("number of trunks:"+str(len(all_trunk_names)))
            # Calculate how many batches can the data set be divided into.
            n_batches = math.floor(len(all_trunk_names)/batch_size)
            # n_batches = math.ceil(len(all_trunk_names) / batch_size)
            logging.debug("n_batches:" + str(n_batches))
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
                    trunk_name_index = batch * batch_size + trunk
                    logging.debug("trunk_name_index: " + str(trunk_name_index))
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
                        logging.info("trunk_name_index >= len(all_trunk_names), trunk_name_index is:"+ str(trunk_name_index)+"len(all_trunk_names):"+str(len(all_trunk_names)))
                    # Get trunk name from all trunk names by trunk name index.
                    trunk_name = all_trunk_names[trunk_name_index]
                    logging.debug("trunk_name: " + trunk_name)
                    # Get trunk data by trunk name without line break character.
                    # sentence_x is a tensor of shape (n_steps, n_inputs)
                    sentence_x = training_data_file['source/' + trunk_name.strip('\n')]
                    # sentence_y is a tensor of shape (None)
                    # sentence_y = training_data_file['target/' + trunk_name.strip('\n')].value
                    sentence_y = training_data_file['target/' + trunk_name.strip('\n')]
                    # sentence_len is a tensor of shape (None)
                    sentence_len = training_data_file['size/' + trunk_name.strip('\n')].value

                    # print(sentence_y.value)
                    # Add current trunk into the batch.
                    batch_x.append(sentence_x)
                    # sentence_y, _ = pad_sequences([sentence_y], maxlen=n_classes)
                    batch_y.append(sentence_y)
                    batch_seq_len.append(sentence_len)
                #     break
                # break
                    # batch_seq_len.append(sentence_len)
                batch_x, _ = pad_sequences(batch_x, maxlen=n_steps, padding='pre')
                batch_y = sparse_tuple_from(batch_y)
                # batch_x is a tensor of shape (batch_size, n_steps, n_inputs)
                # batch_y is a tensor of shape (batch_size, n_steps - truncated_step, n_inputs)
                # Run optimization operation (Back-propagation Through Time)
                feed_dict = {x: batch_x, y: batch_y, seq_len: batch_seq_len}
                batch_cost, _ = sess.run([cost, optimizer], feed_dict)
                train_cost += batch_cost * batch_size
                batch_ler = sess.run(ler, feed_dict)
                train_ler += batch_ler * batch_size
                batch_decode = sess.run(decoded, feed_dict)
                # Print accuracy by display_batch.
                if batch % display_batch == 0:
                    # Calculate batch accuracy.
                    # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    # Calculate batch loss.
                    logging.debug("batch_y: "+str(batch_y))
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    logging.debug("Iter:" + str(iter) + ",Batch:"+ str(batch)
                          + ", Batch Loss= {:.6f}".format(loss) + ", Batch ler= {:.6f}".format(batch_ler))
                    logging.debug("Decode:" + str(batch_decode))
                    # logging.debug("Iter:" + str(iter) + ",Batch:"+ str(batch)
                    #       + ", Batch Loss= {:.6f}".format(loss)
                    #       + ", Training Accuracy= {:.5f}".format(acc))
                # break;
            # Metrics mean
            train_cost /= (batch_size * n_batches)
            train_ler /= (batch_size * n_batches)
            log = "Iter {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(iter+1, training_iters, train_cost, train_ler, time.time() - start))
            # Save session by iteration.
            saver.save(sess, to_save_model_path, global_step=iter);
            logging.info("Model saved successfully to: " + to_save_model_path)
        logging.info("Optimization Finished!")