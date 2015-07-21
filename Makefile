default:
	nvcc -o bigdata -arch=sm_30 -rdc=true main.cu paging.cu hashtable.cu 
clean:
	rm -f bigdata
