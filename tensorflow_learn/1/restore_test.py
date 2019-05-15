#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
W = tf.Variable(tf.zeros([784,10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
x = tf.placeholder(tf.float32, [None, 784])
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    print tf.all_variables()
    print 'W: ', sess.run(W)
    print 'b: ', sess.run(b)
    print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
