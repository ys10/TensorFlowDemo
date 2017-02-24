from __future__ import print_function

import tensorflow as tf
import numpy, h5py
from tensorflow.contrib import rnn

# Import data set
# Read hdf5 file as data set.
# File storing group name.
group_file_name = '../../tmp/data/train_speechorder_timit.txt'
# HDF5 file as training data set.
training_data_file_name = '../../tmp/data/train-timit.hdf5'
# Read group data.
groups = open(group_file_name, 'r')
# Read training data file.
training_data = h5py.File(training_data_file_name, 'r')

'''
To classify vector using a recurrent neural network,
we consider every trunk row as a sequence.
Because trunk shape is 200*69,
we will then handle 69 sequences of 200 steps for every sample.
'''

# Parameters
learning_rate = 0.001
batch_size = 1
display_step = 1
training_iters = 1
# For dropout.
# 每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
keep_prob = 1.0

# Network Parameters
n_input = 69 # data input
n_steps = 200 # time steps
n_hidden = 384 # hidden layer num of features
n_layers = 2 # num of hidden layers
n_classes = 49 # total classes

# tf Graph input
x = tf.placeholder("float32", [batch_size, n_steps, n_input])
y = tf.placeholder("int32", [batch_size, n_steps, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([batch_size, n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_steps, n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Drop out in case overfitting.
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    # Stack two same lstm cell
    cell = rnn.MultiRNNCell([lstm_cell] * n_layers)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
    #
    outputs = tf.transpose(outputs, [1, 0, 2])
    # print(outputs.__sizeof__())
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Launch the graph
with tf.Session() as sess:
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Keep training until reach max iterations
    for iter in range(0, training_iters, 1):
        # For each epoch.
        # Traverse all batches.
        print("iter:"+str(iter))
        batch = 0;
        while 1:
            print("batch:"+str(batch))
            # Read a batch of data.
            lines = groups.readlines(batch_size);
            if not lines:
                break
            # Traverse the batch.
            step = 1
            for trunk in lines:
                print("step:"+str(step))
                # For each trunk in the batch.
                # Get training data by group name without line break.
                # X is a tensor of shape (n_steps, n_input)
                trunk_x = training_data['source/' + trunk.strip('\n')]
                # Y is a tensor of shape (n_steps, n_classes)
                trunk_y = training_data['target/' + trunk.strip('\n')]
                # Reshape data to get 200 seq of 69 elements
                trunk_x = [trunk_x]
                # Reshape data to get 200 seq of 49 elements
                trunk_y = [trunk_y]
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: trunk_x, y: trunk_y})
                # Print accuracy by display_step.
                #if (step + batch * batch_size) % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: trunk_x, y: trunk_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: trunk_x, y: trunk_y})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                step += 1
            batch += 1
            break
    print("Optimization Finished!")