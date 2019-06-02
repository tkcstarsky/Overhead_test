CC = nvcc
GCC = g++
CCFLAG = -lcublas
OPTION = -O3
OUT1 = loop_test
INPUT1 = loop_test.cu
OUT2 = decoding_test
INPUT2 = decoding_test.cu
OUT0 = auto_loop
INPUT0 = auto_loop.cpp
all: loop csr
auto_loop: $(INPUT0)
	$(GCC) -o $(OUT0) $(INPUT0) 
loop: $(INPUT1)
	$(CC) $(OPTION) -o $(OUT1) $(INPUT1) $(CCFLAG)
decoding: $(INPUT2)
	$(CC) $(OPTION) -o $(OUT2) $(INPUT2) $(CCFLAG)
clean:
	rm -f $(OUT1) $(OUT2) $(OUT0)