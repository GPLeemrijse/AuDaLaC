all: sorting.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc sorting.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 sorting.o ../../../compile_resources/*.o -o sorting.out -O3
