SPM: SPM.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc SPM.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SPM.o ../../../runtime_components/*.o -o SPM.out -O3
