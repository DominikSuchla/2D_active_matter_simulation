CC = /usr/local/cuda-9.1/bin/nvcc
CFLAGS = -std=c++11 -arch=sm_35

gpu:
	$(CC) $(CFLAGS) ActFlow_V3-2.cu -I ./ -L /usr/local/cuda-9.1/lib64/ -l cufft -o ActFlow_V3-2
