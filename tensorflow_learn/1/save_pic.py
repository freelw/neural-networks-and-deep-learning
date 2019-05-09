#coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

save_dir = 'MNIST_DATA/raw/'

if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in xrange(20):
    image_array = mnist.train.images[i, :]
    image_array = image_array.reshape(28, 28)
    filename = save_dir + 'mnist_train_%d.jpg' % i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)


