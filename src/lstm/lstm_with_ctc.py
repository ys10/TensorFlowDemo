'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Author: Yang Shuai
Project: https://github.com/ys10/TensorFlowDemo
'''

from __future__ import print_function

import time, os, random
import configparser
import logging
import tensorflow as tf
import h5py
import math
from tensorflow.contrib import rnn
from src.lstm.utils import *

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
# Name of file storing training names.
training_names_file_name = cp.get('data', 'training_names_file_name')
# Name of HDF5 file as training data set.
training_data_file_name = cp.get('data', 'training_data_file_name')
# Read training names.
training_names_file = open(training_names_file_name, 'r')
# Read training data set.
training_data_file = h5py.File(training_data_file_name, 'r')
# Name of file storing validation names.
validation_names_file_name = cp.get('data', 'validation_names_file_name')
# Name of HDF5 file as validation data set.
validation_data_file_name = cp.get('data', 'validation_data_file_name')
# Read validation names.
validation_names_file = open(validation_names_file_name, 'r')
# Read validation data set.
validation_data_file = h5py.File(validation_data_file_name, 'r')

'''
To classify vector using a recurrent neural network,
we consider every trunk row as a sequence.
Because trunk shape is 200*69,
we will then handle 69 dimension sequences of 200 steps for every sample.
'''
# Parameters
learning_rate = 0.0000001
batch_size = 16
display_batch = 1
save_batch = 10
training_epoch = 70
start_epoch = 40
# For dropout to prevent over-fitting.
# Neural network will not work with a probability of 1-keep_prob.
keep_prob = 1.0

# Network Parameters
# n_input = 69 # data input
# n_steps = 200 # time steps
# n_classes = 49 # total classes

n_input = 69 # data input
n_steps = 777 # time steps
n_classes = 50 # total classes

# n_input = 36 # data input
# n_steps = 1500 # time steps
# n_classes = 47 # total classes
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
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate
    # correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
    saver = tf.train.Saver(lstm_variables, max_to_keep=training_epoch)
    saved_model_path = cp.get('model', 'saved_model_path')
    to_save_model_path = cp.get('model', 'to_save_model_path')

    # Launch the graph
    with tf.Session(config=config) as sess:
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # saver.restore(sess, saved_model_path)
        logging.info("Model restored from file: " + saved_model_path)
        # Keep training until reach max epoch
        logging.info("Start training!")
        # Read all training trunk names.
        all_training_trunk_names = training_names_file.readlines()
        # Read all validation trunk names.
        all_validation_trunk_names = validation_names_file.readlines()
        # Train
        for epoch in range(start_epoch, training_epoch, 1):
            if epoch - start_epoch >= 15:
                learning_rate *= 0.95
            train_cost = train_greedy_ler = train_beam_ler = 0
            start = time.time()
            # For each epoch.
            logging.debug("epoch:" + str(epoch))
            logging.debug("learning_rate: "+ str(learning_rate))
            # Break out of the training epoch while there is no trunk usable.
            if not all_training_trunk_names:
                break
            # Shuffle the trunk name list.
            random.shuffle(all_training_trunk_names)
            logging.debug("number of trunks:"+str(len(all_training_trunk_names)))
            # Calculate how many batches can the data set be divided into.
            training_batches = math.floor(len(all_training_trunk_names)/batch_size)
            # training_batches = math.ceil(len(all_training_trunk_names) / batch_size)
            logging.debug("training_batches:" + str(training_batches))
            # Train the RNN(LSTM) model by batch.
            for batch in range(0, training_batches, 1):
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
                    if(trunk_name_index >= len(all_training_trunk_names)):
                        # So some used trunks should be add to the last batch when the "fact" happened.
                        # Select the last trunk to be added into the last batch.
                        trunk_name_index = len(all_training_trunk_names)-1
                        logging.info("trunk_name_index >= len(all_training_trunk_names), trunk_name_index is:"+ str(trunk_name_index)+"len(all_training_trunk_names):"+str(len(all_training_trunk_names)))
                    # Get trunk name from all trunk names by trunk name index.
                    # trunk_name = all_training_trunk_names[trunk_name_index].split()[0]
                    trunk_name = all_training_trunk_names[trunk_name_index].strip('\n')
                    logging.debug("trunk_name: " + trunk_name)
                    # Get trunk data by trunk name without line break character.
                    # sentence_x is a tensor of shape (n_steps, n_inputs)
                    sentence_x = training_data_file['source/' + trunk_name]
                    # sentence_y is a tensor of shape (None)
                    # sentence_y = training_data_file['target/' + trunk_name.strip('\n')].value
                    sentence_y = training_data_file['target/' + trunk_name]
                    # sentence_len is a tensor of shape (None)
                    sentence_len = training_data_file['size/' + trunk_name].value
                    # print(sentence_y.value)
                    # Add current trunk into the batch.
                    batch_x.append(sentence_x)
                    # sentence_y, _ = pad_sequences([sentence_y], maxlen=n_classes)
                    batch_y.append(sentence_y)
                    batch_seq_len.append(sentence_len)
                batch_x, _ = pad_sequences(batch_x, maxlen=n_steps, padding='pre')
                batch_y = sparse_tuple_from(batch_y)
                # batch_x is a tensor of shape (batch_size, n_steps, n_inputs)
                # batch_y is a tensor of shape (batch_size, n_steps - truncated_step, n_inputs)
                # Run optimization operation (Back-propagation Through Time)
                feed_dict = {x: batch_x, y: batch_y, seq_len: batch_seq_len}
                # train_summary, _, batch_greedy_decode, batch_beam_decode = sess.run([train_merged, optimizer, greedy_decoded, beam_decoded], feed_dict)
                batch_cost, _, batch_greedy_ler, batch_beam_ler, batch_greedy_decode, batch_beam_decode\
                    = sess.run([cost, optimizer, greedy_ler, beam_ler, greedy_decoded, beam_decoded], feed_dict)
                train_cost += batch_cost * batch_size
                # ler
                # batch_greedy_ler = sess.run(greedy_ler, feed_dict)
                train_greedy_ler += batch_greedy_ler * batch_size
                #
                # batch_beam_ler = sess.run(beam_ler, feed_dict)
                train_beam_ler += batch_beam_ler * batch_size
                # decode
                # batch_greedy_decode = sess.run(greedy_decoded, feed_dict)
                # batch_beam_decode = sess.run(beam_decoded, feed_dict)
                #
                # Print accuracy by display_batch.
                if batch % display_batch == 0:
                    # Calculate batch accuracy.
                    # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    # Calculate batch loss.
                    logging.debug("batch_y: "+str(batch_y))
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    logging.debug("epoch:" + str(epoch) + ",Batch:"+ str(batch) + ", Batch Loss= {:.6f}".format(loss)
                                  + ", Batch beam ler= {:.6f}".format(batch_beam_ler) + ", Batch greddy ler= {:.6f}".format(batch_greedy_ler))
                    logging.debug("beam decode:" + str(batch_beam_decode))
                    logging.debug("greddy decode:" + str(batch_greedy_decode))
                break;
            # Metrics mean
            train_cost /= (batch_size * training_batches)
            train_beam_ler /= (batch_size * training_batches)
            train_greedy_ler /= (batch_size * training_batches)
            log = "epoch {}/{}, train_cost = {:.3f}, train_beam_ler = {:.3f}, train_greedy_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(epoch+1, training_epoch, train_cost, train_beam_ler, train_greedy_ler, time.time() - start))
            # TODO
            # Save session by epoch.
            if epoch % save_batch ==0:
                saver.save(sess, to_save_model_path, global_step=epoch);
                logging.info("Model saved successfully to: " + to_save_model_path)
            # Validation
            logging.debug("Validation starting!")
            logging.debug("Number of validation trunks:"+str(len(all_validation_trunk_names)))
            # Calculate how many batches can the data set be divided into.
            validation_batches = math.floor(len(all_validation_trunk_names)/batch_size)
            # training_batches = math.ceil(len(all_training_trunk_names) / batch_size)
            logging.debug("Validation_batches:" + str(validation_batches))
            # set validation parameters.
            validation_cost = validation_greedy_ler = validation_beam_ler = 0
            for batch in range(0, validation_batches, 1):
                # For each batch.
                # Define two variables to store input data.
                batch_x = []
                batch_y = []
                batch_seq_len = []
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
                    if (trunk_name_index >= len(all_validation_trunk_names)):
                        # So some used trunks should be add to the last batch when the "fact" happened.
                        # Select the last trunk to be added into the last batch.
                        trunk_name_index = len(all_validation_trunk_names) - 1
                        logging.info("trunk_name_index >= len(all_validation_trunk_names), trunk_name_index is:" + str(
                            trunk_name_index) + "len(all_validation_trunk_names):" + str(len(all_validation_trunk_names)))
                    # Get trunk name from all trunk names by trunk name index.
                    # trunk_name = all_training_trunk_names[trunk_name_index].split()[0]
                    trunk_name = all_validation_trunk_names[trunk_name_index].strip('\n')
                    logging.debug("trunk_name: " + trunk_name)
                    # Get trunk data by trunk name without line break character.
                    # sentence_x is a tensor of shape (n_steps, n_inputs)
                    sentence_x = validation_data_file['source/' + trunk_name]
                    # sentence_y is a tensor of shape (None)
                    # sentence_y = training_data_file['target/' + trunk_name.strip('\n')].value
                    sentence_y = validation_data_file['target/' + trunk_name]
                    # sentence_len is a tensor of shape (None)
                    sentence_len = validation_data_file['size/' + trunk_name].value
                    # print(sentence_y.value)
                    # Add current trunk into the batch.
                    batch_x.append(sentence_x)
                    # sentence_y, _ = pad_sequences([sentence_y], maxlen=n_classes)
                    batch_y.append(sentence_y)
                    batch_seq_len.append(sentence_len)
                batch_x, batch_seq_len = pad_sequences(batch_x, maxlen=n_steps, padding='pre')
                batch_y = sparse_tuple_from(batch_y)
                feed_dict = {x: batch_x, y: batch_y, seq_len: batch_seq_len}
                # Train with validation set.
                # batch_cost, _ = sess.run([cost, optimizer], feed_dict)
                # Train without validation set.
                batch_cost = sess.run(cost, feed_dict)
                validation_cost += batch_cost * batch_size
                # ler
                batch_greedy_ler = sess.run(greedy_ler, feed_dict)
                validation_greedy_ler += batch_greedy_ler * batch_size
                #
                batch_beam_ler = sess.run(beam_ler, feed_dict)
                validation_beam_ler += batch_beam_ler * batch_size
                # decode
                batch_greedy_decode = sess.run(greedy_decoded, feed_dict)
                batch_beam_decode = sess.run(beam_decoded, feed_dict)
                # Print accuracy by display_batch.
                if batch % display_batch == 0:
                    # Calculate batch accuracy.
                    # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    # Calculate batch loss.
                    logging.debug("batch_y: " + str(batch_y))
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq_len: batch_seq_len})
                    logging.debug("epoch:" + str(epoch) + ",Batch:" + str(batch) + ", Batch Loss= {:.6f}".format(loss)
                                  + ", Batch beam ler= {:.6f}".format(
                        batch_beam_ler) + ", Batch greddy ler= {:.6f}".format(batch_greedy_ler))
                    logging.debug("beam decode:" + str(batch_beam_decode))
                    logging.debug("greddy decode:" + str(batch_greedy_decode))
                break;
            # Metrics mean
            validation_cost /= (batch_size * validation_batches)
            validation_beam_ler /= (batch_size * validation_batches)
            validation_greedy_ler /= (batch_size * validation_batches)
            log = "epoch {}/{}, validation_cost = {:.3f}, validation_beam_ler = {:.3f}, validation_greedy_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(epoch + 1, training_epoch, validation_cost, validation_beam_ler, validation_greedy_ler, time.time() - start))
            logging.debug("Validation end.")
            # break;
        logging.info("Optimization Finished!")