#coding: utf-8
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784], name='input_x')
W = tf.Variable(tf.zeros([784,10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
y = tf.nn.softmax(tf.matmul(x, W) + b, name='output_y')
y_ = tf.placeholder(tf.float32, [None, 10], name='input_y')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in xrange(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    answer = tf.argmax(y, 1, name='answer')
    correct_prediction = tf.equal(answer, tf.argmax(y_, 1), name='correct')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output_y', 'accuracy'])
    with tf.gfile.FastGFile('model/restore_test.pb', mode='wb') as f:
            f.write(output_graph_def.SerializeToString())
    print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
#print y.eval()
