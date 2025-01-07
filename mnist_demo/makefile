all: check_loader mnist_demo test_matrix
mnist_demo:
	g++ main.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o mnist_demo
check_loader:
	g++ check_loader.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o check_loader
test_matrix:
	g++ test_matrix.cpp matrix.cpp -o test_matrix
clean:
	rm mnist_demo check_loader test_matrix
prepare_data:
	cd resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
rebuild: clean all