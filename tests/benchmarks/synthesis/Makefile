synthesis: synthesis.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc synthesis.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 synthesis.o ../../../compile_resources/*.o -o synthesis.out -O3

syn_dbg: synthesis.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G  -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G  -dc synthesis.cu -I../../../compile_resources/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G  synthesis.o ../../../compile_resources/*.o -o synthesis.out
