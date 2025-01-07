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
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

if '__main__' == __name__:
    #net = Network([3, 5, 2])
    x = np.zeros((3,2), )
    y = np.zeros((2,3), )
    x[0][0] = 1
    x[1][0] = 2
    x[2][0] = 3
    x[0][1] = 6
    x[1][1] = 6
    x[2][1] = 6
    y[0][0] = 4
    y[0][1] = 5
    y[0][2] = 6
    y[1][0] = 7
    y[1][1] = 7
    y[1][2] = 7
    print x
    print y
    print sigmoid_prime(-np.dot(y, x))
