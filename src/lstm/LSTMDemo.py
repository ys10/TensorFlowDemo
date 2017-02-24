from __future__ import print_function

import tensorflow as tf
import h5py
from math import ceil
from tensorflow.contrib import rnn

# Import data set
# Name of file storing trunk names.
trunk_names_file_name = '../../tmp/data/train_speechorder_timit.txt'
# Name of HDF5 file as training data set.
training_data_file_name = '../../tmp/data/train-timit.hdf5'
# Read trunk names.
trunk_names_file = open(trunk_names_file_name, 'r')
# Read training data set.
training_data_file = h5py.File(training_data_file_name, 'r')

'''
To classify vector using a recurrent neural network,
we consider every trunk row as a sequence.
Because trunk shape is 200*69,
we will then handle 69 sequences of 200 steps for every sample.
'''

# Parameters
learning_rate = 0.001
batch_size = 2
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

    # Get lstm cell output with shape (n_steps, batch_size, n_input)
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
    # Reshaping to (batch_size, n_steps, n_input)
    outputs = tf.transpose(outputs, [1, 0, 2])
    # Linear activation, using rnn inner loop last output
    # The first dim of outputs & weights must be same.
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
        # For each iteration.
        print("iter:"+str(iter))
        # Read all trunk names.
        all_trunk_names = trunk_names_file.readlines();
        if not all_trunk_names:
            break
        n_batches = ceil(len(all_trunk_names)/batch_size)
        for batch in range(0, n_batches, 1):
            # Read a batch of trunk names as lines.
            # Break out of the "while" loop when there is no more batch in current iteration.
            # For each batch.
            print("batch:"+str(batch))
            batch_x = []
            batch_y = []
            # Traverse all trunks.
            for trunk in range(0, batch_size, 1):
                print("trunk:"+str(trunk))
                # For each trunk in the batch as line in lines.
                # Get training data by trunk name without line break character.
                # X is a tensor of shape (n_steps, n_input)
                trunk_name_index = n_batches * batch_size + trunk
                if(trunk_name_index >= len(all_trunk_names)):
                    trunk_name_index = len(all_trunk_names)-1
                trunk_name = all_trunk_names[trunk_name_index]
                trunk_x = training_data_file['source/' + trunk_name.strip('\n')]
                # Y is a tensor of shape (n_steps, n_classes)
                trunk_y = training_data_file['target/' + trunk_name.strip('\n')]
                # Reshape data to get 200 seq of 69 elements
                batch_x.append(trunk_x)
                # Reshape data to get 200 seq of 49 elements
                batch_y.append(trunk_y)
            #
            #batch_x = array(batch_x)
            print("batch_x.len:")
            print(len(batch_x))
            #batch_y = array(batch_y)
            print("batch_y.len:")
            print(len(batch_y))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Print accuracy by display_step.
            # if (step + batch * batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(iter) + ", Batch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            batch += 1
            break
    print("Optimization Finished!")