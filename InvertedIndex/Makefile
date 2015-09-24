FLAGS=-Xptxas --maxrregcount=32
default:multipass

multipass:
	nvcc main.cu paging.cu hashtable.cu -arch=sm_30 -o bigdata -rdc=true $(FLAGS) $(GPUMultiPass) $(Libs) $(Includes)
debug:
	nvcc -g -G main.cu paging.cu hashtable.cu -arch=sm_30 -o bigdata -rdc=true $(FLAGS) $(GPUMultiPass) $(Libs) $(Includes)
pthread:
	nvcc -o bigdata -arch=sm_30 pthread.cu -ltcmalloc -I/usr/include/google/
clean:
	rm -f bigdata
