import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1),) for y in sizes[1:]]
        self.weights = [np.zeros((y, x),)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases[0][0]=1
        print self.biases
        print self.weights

        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a)+b
        return a

if '__main__' == __name__:
    net = Network([3, 5, 2])
    x = np.zeros((3,1), )
    print 'xxxxx'
    print x
    print x.shape
    print net.feedforward(x)
