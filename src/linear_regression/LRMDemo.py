import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from src.function import loss_function

# Define iuput & output tensor.
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# Define variable of model.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# The output of model.
a = tf.nn.softmax(tf.matmul(x, W) + b)
# Define loss function.
loss = loss_function.mse
# Define train method.
def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
train = training(loss(a,y), 0.5)

# tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后看看他们是否一致，是就返回true,不是就返回false,这样得到一个boolean数组.
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
# tf.cast将boolean数组转成int数组，最后求平均值，得到分类的准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Put data in the directory /tmp/data .
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# Read data set.
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Build session.
sess = tf.Session()

# Initialize all variables.
tf.initialize_all_variables().run() # 所有变量初始化
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # 获得一批100个数据
    train.run({x: batch_xs, y: batch_ys})   # 给训练模型提供输入和输出
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))