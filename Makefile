default:
	nvcc -o bigdata -arch=sm_30 -rdc=true main.cu paging.cu hashtable.cu 
debug:
	nvcc -g -G -o bigdata -O0 -Xptxas -O0 -arch=sm_30 -rdc=true main.cu paging.cu hashtable.cu 
clean:
	rm -f bigdata
