#coding: utf-8
import tensorflow as tf
W = tf.Variable(tf.zeros([784,10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    print tf.all_variables()
    print 'W: ', sess.run(W)
    print 'b: ', sess.run(b)
