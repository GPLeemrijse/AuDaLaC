debug: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G -dc SCC.cu -I../../../compile_resources/
	nvcc -arch=sm_60 -g -G SCC.o ../../../compile_resources/*.o -o SCC.out


all: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SCC.cu -I../../../compile_resources/
	nvcc -arch=sm_60 SCC.o ../../../compile_resources/*.o -o SCC.out
	#(cd fixtures/ && python3 compare_random_graph.py 1000 1000)

disasm: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G --cubin -dc SCC.cu -I../../../compile_resources/
	#nvdisasm SCC.cubin > SCC.cubin.disasm
	cuobjdump -sass SCC.cubin > SCC.cubin.dump


run: ./SCC.out $(init)
	./SCC.out $(init)

golden: SCC_golden.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h $(init)
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SCC_golden.cu -I../../../compile_resources/
	nvcc -arch=sm_60 SCC_golden.o ../../../compile_resources/*.o -o SCC_golden.out
	./SCC_golden.out $(init)


run_golden: ./SCC_golden.out $(init)
	./SCC_golden.out $(init)