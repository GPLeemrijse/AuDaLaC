prefix_sum: prefix_sum.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc prefix_sum.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 prefix_sum.o ../../../runtime_components/*.o -o prefix_sum.out -O3