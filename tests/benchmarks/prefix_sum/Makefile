../../../runtime_components/*.o: ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3

prefix_sum.o: prefix_sum.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc prefix_sum.cu -I../../../runtime_components/ -O3

prefix_sum.out: prefix_sum.o
	nvcc -arch=sm_60 prefix_sum.o ../../../runtime_components/*.o -o prefix_sum.out -O3