sorting: sorting.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc sorting.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 sorting.o ../../../runtime_components/*.o -o sorting.out -O3
