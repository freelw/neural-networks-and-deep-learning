all:
	g++ main.cpp mnist_loader.cpp network.cpp -o mnist_demo
clean:
	rm mnist_demo
init_py_env:
	cd .. && bash init_py_env.sh 
dump_input:
	cd ../src && python dump_mnist.py
prepare: init_py_env dump_input
run:
	./mnist_demo ../src/mnist.dump 