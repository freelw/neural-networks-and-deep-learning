from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

print mnist.train.images.shape
print mnist.train.labels.shape

print mnist.validation.images.shape
print mnist.validation.labels.shape

print mnist.test.images.shape
print mnist.test.labels.shape
