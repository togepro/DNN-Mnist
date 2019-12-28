object = main.o network.o  mnist.o util.o

main: $(object)
	g++ -o main $(object)

main.o: network.h

network.o: network.h util.h

mnist.o: mnist.h util.h

util.o: util.h
	g++ -c -o util.o util.cpp -std=c++11


.PHONY : clean

clean:
	rm main $(object)
