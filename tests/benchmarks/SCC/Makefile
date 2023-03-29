all: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SCC.cu -I../../../compile_resources/
	nvcc -arch=sm_60 SCC.o ../../../compile_resources/*.o -o SCC.out
	./SCC.out SCC_1.init

run: ./SCC.out $(init)
	./SCC.out $(init)

golden: SCC_golden.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h $(init)
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SCC_golden.cu -I../../../compile_resources/
	nvcc -arch=sm_60 SCC_golden.o ../../../compile_resources/*.o -o SCC_golden.out
	./SCC_golden.out $(init)


run_golden: ./SCC_golden.out $(init)
	./SCC_golden.out $(init)