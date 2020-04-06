AR := ar
AROPT := rcs
CFLAGS := -g -O0 -I$(JULIA_HIOP_LIBRARY_PATH)/include
LDLIBS := -L$(JULIA_HIOP_LIBRARY_PATH)/lib -L/sandbox/schanen/hiop.jl/deps -lchiopInterface -lhiop
CXX := mpicxx
CC  := mpicc

all: lib test

test: lib test.o 
	$(CC) $(CFLAGS) -o test test.o $(LDLIBS) -lm -lblas -llapack -lstdc++ 

test.o: test.c
	$(CC) $(CFLAGS) -c test.c

lib: chiopInterface.o
	# $(AR) $(AROPT) libchiopInterface.a chiopInterface.o
	$(CC) $(CFLAGS) -shared -o libchiopInterface.so chiopInterface.o

chiopInterface.o: chiopInterface.cpp
	$(CXX) $(CFLAGS) -c -fpic chiopInterface.cpp

clean:
	- $(RM) *.o *.so *.a test
