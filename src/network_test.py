
import network
import numpy as np

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

#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)