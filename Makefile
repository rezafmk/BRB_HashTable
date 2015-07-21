default:
	nvcc -o bigdata -arch=sm_30 main.cu
clean:
	rm -f bigdata
