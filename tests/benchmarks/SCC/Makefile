SCC_FB: SCC_FB.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc SCC_FB.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_FB.o ../../../runtime_components/*.o -o SCC_FB.out -O3
	#(cd fixtures/ && python3 compare_random_graph.py 1000 1000)

SCC_COL: SCC_COL.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc SCC_COL.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_COL.o ../../../runtime_components/*.o -o SCC_COL.out -O3

debug: SCC.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/
	nvcc -arch=sm_60 -g -G -dc SCC.cu -I../../../runtime_components/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G SCC.o ../../../runtime_components/*.o -o SCC.out

disasm: SCC.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/
	nvcc -arch=sm_60 -g -G --cubin -dc SCC.cu -I../../../runtime_components/
	nvdisasm SCC.cubin > SCC.cubin.disasm
	cuobjdump -sass SCC.cubin > SCC.cubin.dump
	nvcc -arch=sm_60 -g -G --ptx -dc SCC.cu -I../../../runtime_components/ -o SCC.ptx

run: ./SCC.out $(init)
	./SCC.out $(init)

golden: SCC_golden.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h $(init)
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/
	nvcc -arch=sm_60 -dc SCC_golden.cu -I../../../runtime_components/
	nvcc -arch=sm_60 SCC_golden.o ../../../runtime_components/*.o -o SCC_golden.out
	./SCC_golden.out $(init)


MP: SCC_MP.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc SCC_MP.cu -I../../../runtime_components/ -O3 --ptxas-options=-v
	nvcc -arch=sm_60 SCC_MP.o ../../../runtime_components/*.o -o SCC_MP.out -O3

MP_dbg: SCC_MP.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -g -G -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/
	nvcc -arch=sm_60 -g -G -dc SCC_MP.cu -I../../../runtime_components/ --ptxas-options=-v
	nvcc -arch=sm_60 -g -G SCC_MP.o ../../../runtime_components/*.o -o SCC_MP.out


run_golden: ./SCC_golden.out $(init)
	./SCC_golden.out $(init)

sanitize: ./SCC.out
	compute-sanitizer SCC.out /tmp/random.init > sanitizer_output


COL2: SCC_col.cu ../../../runtime_components/*.cu ../../../runtime_components/*.h
	nvcc -arch=sm_60 -dc ../../../runtime_components/*.cu -odir ../../../runtime_components/ -O3
	nvcc -arch=sm_60 -dc SCC_col2.cu -I../../../runtime_components/ --ptxas-options=-v -O3
	nvcc -arch=sm_60 SCC_col2.o ../../../runtime_components/*.o -o SCC_col2.out -O3