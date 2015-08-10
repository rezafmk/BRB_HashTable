FLAGS= -g -O3
GPUDoubleBuffering=main.cu
GPUL2Version=mainL2.cu


default:normal

normal:
	nvcc $(FLAGS) -arch=sm_30 -o bigdata $(GPUDoubleBuffering) $(Libs) $(Includes)

L2version:
	nvcc $(FLAGS) -arch=sm_30 -o bigdata $(GPUL2Version) $(Libs) $(Includes)
	
clean:
	rm -f bigdata
