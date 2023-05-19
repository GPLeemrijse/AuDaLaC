all: SPM.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SPM.cu -I../../../compile_resources/ --ptxas-options=-v
	nvcc -arch=sm_60 SPM.o ../../../compile_resources/*.o -o SPM.out
