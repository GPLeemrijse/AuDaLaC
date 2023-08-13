../../../runtime_components/*.o: ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3

SPM.o: SPM.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc SPM.cu -I../../../runtime_components/ -O3

SPM.out: SPM.o
	nvcc -arch=sm_60 SPM.o ../../../runtime_components/*.o -o SPM.out -O3