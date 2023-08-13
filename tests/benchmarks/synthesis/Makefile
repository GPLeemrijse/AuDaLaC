../../../runtime_components/*.o: ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3

synthesis.o: synthesis.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc synthesis.cu -I../../../runtime_components/ -O3

synthesis.out: synthesis.o
	nvcc -arch=sm_60 synthesis.o ../../../runtime_components/*.o -o synthesis.out -O3