synthesis: synthesis.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc synthesis.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 synthesis.o ../../../runtime_components/*.o -o synthesis.out -O3

syn_dbg: synthesis.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -g -G  -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/
	nvcc -arch=sm_60 -g -G  -dc synthesis.cu -I../../../runtime_components/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G  synthesis.o ../../../runtime_components/*.o -o synthesis.out
