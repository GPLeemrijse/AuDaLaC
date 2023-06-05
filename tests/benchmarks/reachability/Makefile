all: reachability.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc reachability.cu -I../../../compile_resources/ --ptxas-options=-v
	nvcc -arch=sm_60 reachability.o ../../../compile_resources/*.o -o reachability.out

debug: reachability.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G -dc reachability.cu -I../../../compile_resources/
	nvcc -arch=sm_60 -g -G reachability.o ../../../compile_resources/*.o -o reachability.out

disasm: reachability.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G --cubin -dc reachability.cu -I../../../compile_resources/
	nvdisasm reachability.cubin > reachability.cubin.disasm
	cuobjdump -sass reachability.cubin > reachability.cubin.dump
	nvcc -arch=sm_60 -g -G --ptx -dc reachability.cu -I../../../compile_resources/ -o reachability.ptx

run: reachability.out
	./reachability.out example1.init

sanitize: reachability.out
	compute-sanitizer reachability.out example1.init > sanitizer_output