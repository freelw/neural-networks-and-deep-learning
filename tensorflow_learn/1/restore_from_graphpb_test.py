#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 10], name='input_y')
with open('./model/restore_test.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print graph_def
    [accuracy, answer, W] = tf.import_graph_def(graph_def, input_map={'input_x': x, 'input_y': y_}, return_elements=['accuracy:0', 'answer:0', 'W:0'])
with tf.Session() as sess:
    print sess.run([W, answer, accuracy], feed_dict={x:mnist.test.images[0:10], y_:mnist.test.labels[0:10]})
