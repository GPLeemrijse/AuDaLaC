SCC_FB: SCC_FB.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SCC_FB.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_FB.o ../../../compile_resources/*.o -o SCC_FB.out -O3
	#(cd fixtures/ && python3 compare_random_graph.py 1000 1000)

SCC_COL: SCC_COL.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SCC_COL.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_COL.o ../../../compile_resources/*.o -o SCC_COL.out -O3

debug: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G -dc SCC.cu -I../../../compile_resources/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G SCC.o ../../../compile_resources/*.o -o SCC.out

disasm: SCC.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G --cubin -dc SCC.cu -I../../../compile_resources/
	nvdisasm SCC.cubin > SCC.cubin.disasm
	cuobjdump -sass SCC.cubin > SCC.cubin.dump
	nvcc -arch=sm_60 -g -G --ptx -dc SCC.cu -I../../../compile_resources/ -o SCC.ptx

run: ./SCC.out $(init)
	./SCC.out $(init)

golden: SCC_golden.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h $(init)
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -dc SCC_golden.cu -I../../../compile_resources/
	nvcc -arch=sm_60 SCC_golden.o ../../../compile_resources/*.o -o SCC_golden.out
	./SCC_golden.out $(init)


MP: SCC_MP.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SCC_MP.cu -I../../../compile_resources/ -O3 --ptxas-options=-v
	nvcc -arch=sm_60 SCC_MP.o ../../../compile_resources/*.o -o SCC_MP.out -O3

MP_dbg: SCC_MP.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/
	nvcc -arch=sm_60 -g -G -dc SCC_MP.cu -I../../../compile_resources/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G SCC_MP.o ../../../compile_resources/*.o -o SCC_MP.out


run_golden: ./SCC_golden.out $(init)
	./SCC_golden.out $(init)

sanitize: ./SCC.out
	compute-sanitizer SCC.out /tmp/random.init > sanitizer_output


COL2: SCC_col.cu ../../../compile_resources/*.cu ../../../compile_resources/*.h
	nvcc -arch=sm_60 -dc ../../../compile_resources/*.cu -odir ../../../compile_resources/ -O3
	nvcc -arch=sm_60 -dc SCC_col2.cu -I../../../compile_resources/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_col2.o ../../../compile_resources/*.o -o SCC_col2.out -O3