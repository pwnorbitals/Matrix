all: build

build:
	g++ -o ./build/matrix.out ./src/main.cpp

clean:
	rm -f ./build/*