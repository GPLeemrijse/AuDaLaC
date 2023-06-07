all: prefix_sum.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc prefix_sum.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 prefix_sum.o ../../../compile_resources/*.o -o prefix_sum.out -O3
