
import network
import numpy as np

def test_feedforward_backprop():
    y = np.zeros((3,1),)
    y[0][0] = 1
    y[1][0] = 0
    y[2][0] = 0

    a = np.zeros((5,1),)
    a[0][0] = 2
    a[1][0] = 1
    a[2][0] = 3
    a[3][0] = 5
    a[4][0] = 4

    print y
    print a
    net = network.Network([5, 4, 3])

    print '-----feedforward------'
    print net.feedforward(a)

    print '-----backprop------'
    delta_nabla_b, delta_nabla_w = net.backprop(a, y)
    print delta_nabla_b
    print delta_nabla_w


def test_update_mini():
    y = np.zeros((3,1),)
    y[0][0] = 1
    y[1][0] = 0
    y[2][0] = 0

    a = np.zeros((5,1),)
    a[0][0] = 2
    a[1][0] = 1
    a[2][0] = 3
    a[3][0] = 5
    a[4][0] = 4

    net = network.Network([5, 4, 3])

    mini_batch = [(a, y)]
    for i in range(2):
        net.update_mini_batch(mini_batch, 0.1)

    print "biases:"
    print net.biases
    print "weights:"
    print net.weights
    

if '__main__' == __name__:
    #test_feedforward_backprop()
    test_update_mini()