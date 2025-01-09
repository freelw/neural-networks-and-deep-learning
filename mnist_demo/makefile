all: check_loader mnist_demo test_matrix network_v2_test
mnist_demo:
	g++ main.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o mnist_demo
check_loader:
	g++ check_loader.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o check_loader
test_matrix:
	g++ test_matrix.cpp matrix.cpp network_v2.cpp -o test_matrix
network_v2_test:
	g++ network_v2_test.cpp matrix.cpp network_v2.cpp mnist_loader_base.cpp -o network_v2_test
clean:
	rm mnist_demo check_loader test_matrix network_v2_test
prepare_data:
	cd resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
rebuild: clean all