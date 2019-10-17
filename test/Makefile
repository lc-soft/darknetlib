run: test weights
	LD_LIBRARY_PATH=`cd .. && pwd`:$LD_LIBRARY_PATH ./test

test.o: test.c
	gcc -c test.c

test: test.o
	gcc -L../ -o test test.o -ldarknet

yolov3-tiny.weights:
	wget https://pjreddie.com/media/files/yolov3-tiny.weights

weights: yolov3-tiny.weights

all: test run

clean:
	rm test test.o
