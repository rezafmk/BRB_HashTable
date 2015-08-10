GPUMultiPass=multiPass.cu
FLAGS= -g -O3 -Xptxas -v -Xptxas --maxrregcount=32
GPUNormal=doubleBuffering.cu
CPUPthread=pthread.cu
GPUDoubleBuffering=doubleBuffering.cu


default:multipass

multipass:
	nvcc $(FLAGS) -arch=sm_20 -o bigdata $(GPUMultiPass) $(Libs) $(Includes)

doubleBuffering:
	nvcc $(FLAGS) -arch=sm_20 -o bigdata $(GPUDoubleBuffering) $(Libs) $(Includes)
	
pthread:
	nvcc -O3 -arch=sm_20 -o bigdata $(CPUPthread) $(Libs) $(Includes)

clean:
	rm -f bigdata hash normal
