all: check_loader mnist_demo
mnist_demo:
	g++ main.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o mnist_demo
check_loader:
	g++ check_loader.cpp mnist_loader.cpp mnist_loader_v2.cpp mnist_loader_base.cpp -o check_loader
clean:
	rm mnist_demo check_loader
prepare_data:
	cd resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
rebuild: clean all