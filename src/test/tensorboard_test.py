import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
max_steps = 1000
learning_rate = 0.0001
drop_out = 0.9
data_dir = "/tmp/data/"
log_dir = "/log/summary/"

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name = 'x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope("input_reshape"):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(stddev))
        tf.summary.scalar('min', tf.reduce_min(stddev))
        tf.summary.histogram('histogram', var)


