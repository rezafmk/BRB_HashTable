FLAGS=-Xptxas --maxrregcount=32
default:multipass

multipass:
	nvcc main.cu paging.cu hashtable.cu -arch=sm_30 -o bigdata -rdc=true $(FLAGS) $(GPUMultiPass) $(Libs) $(Includes)
debug:
	nvcc -g -G main.cu paging.cu hashtable.cu -arch=sm_30 -o bigdata -rdc=true $(FLAGS) $(GPUMultiPass) $(Libs) $(Includes)
clean:
	rm -f bigdata
