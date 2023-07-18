norm: SPM.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SPM.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SPM.o ../../../compile_resources/*.o -o SPM.out -O3

opt: SPM_opt.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SPM_opt.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SPM_opt.o ../../../compile_resources/*.o -o SPM_opt.out -O3

spm2: SPM2.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SPM2.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SPM2.o ../../../compile_resources/*.o -o SPM2.out -O3

pred: spm_pred.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc spm_pred.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 spm_pred.o ../../../compile_resources/*.o -o spm_pred.out -O3
