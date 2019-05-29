CC = nvcc
CCFLAG = -lcublas
OPTION = -O3
OUT1 = loop_test
INPUT1 = loop_test.cu
OUT2 = decoding_test
INPUT2 = decoding_test.cu
all: loop csr
loop: $(INPUT1)
	$(CC) $(OPTION) -o $(OUT1) $(INPUT1) $(CCFLAG)
decoding: $(INPUT2)
	$(CC) $(OPTION) -o $(OUT2) $(INPUT2) $(CCFLAG)
clean:
	rm -f $(OUT1) $(OUT2) 