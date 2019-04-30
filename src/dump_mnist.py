import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_len = len(training_data)
training_data_x_len = len(training_data[0][0])
training_data_y_len = len(training_data[0][1])
test_data_len = len(test_data)
test_data_x_len = len(test_data[0][0])
f = open('mnist.dump', 'wb')
f.write('%s' % training_data_len + '\n')
f.write('%s' % training_data_x_len + '\n')
f.write('%s' % training_data_y_len + '\n')
for i in xrange(training_data_len):
    for j in xrange(training_data_x_len):
        f.write('%s' % training_data[i][0][j][0] + ' ')
    f.write('\n')
    for j in xrange(training_data_y_len):
        f.write('%s' % training_data[i][1][j][0] + ' ')
    f.write('\n')
f.write('%s' % test_data_len + '\n')
f.write('%s' % test_data_x_len + '\n')
for i in xrange(test_data_len):
    for j in xrange(test_data_x_len):
        f.write('%s' % test_data[i][0][j][0] + ' ')
    f.write('\n')
    f.write('%s' % test_data[i][1])
    f.write('\n')
f.close()