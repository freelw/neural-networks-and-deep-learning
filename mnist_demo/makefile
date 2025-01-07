all:
	g++ main.cpp mnist_loader.cpp network.cpp mnist_loader_v2.cpp -o mnist_demo
clean:
	rm mnist_demo
prepare_data:
	cd resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte