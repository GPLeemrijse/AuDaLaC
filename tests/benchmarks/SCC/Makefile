../../../runtime_components/*.o: ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3

SCC_FB.o: SCC_FB.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc SCC_FB.cu -I../../../runtime_components/ -O3

SCC_FB.out: SCC_FB.o
	nvcc -arch=sm_60 SCC_FB.o ../../../runtime_components/*.o -o SCC_FB.out -O3


SCC_COL.o: SCC_COL.cu ../../../runtime_components/*.o
	nvcc -arch=sm_60 -dc SCC_COL.cu -I../../../runtime_components/ -O3


SCC_COL.out: SCC_COL.o
	nvcc -arch=sm_60 SCC_COL.o ../../../runtime_components/*.o -o SCC_COL.out -O3