
default:multipass

multipass:
	nvcc main.cu -arch=sm_30 -o bigdata $(FLAGS) $(GPUMultiPass) $(Libs) $(Includes)
clean:
	rm -f bigdata
