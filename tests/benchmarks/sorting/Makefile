../../../runtime_components/*.o: ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3

sorting.o: sorting.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc sorting.cu -I../../../runtime_components/ -O3

sorting.out: sorting.o
	nvcc -arch=sm_60 sorting.o ../../../runtime_components/*.o -o sorting.out -O3