#ifndef ADL_H
#define ADL_H
#include <string>
#include <assert.h>
#include <tuple>


#define MAX_FP_DEPTH 16

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif


#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      #ifdef __CUDA_ARCH__
         // Device version
         printf("CUDA KERNEL ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
         if (abort) assert(0);
      #else
         // Host version
         fprintf(stderr,"CUDA HOST ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
         if (abort) exit(code);
      #endif
   }
}

#ifdef DEBUG
   #define dbg_assert(pred) assert(pred)
#else
   #define dbg_assert(pred) 
#endif


namespace ADL {

	typedef int32_t IntType;
	typedef uint32_t NatType;
	typedef bool BoolType;
	typedef uint32_t RefType;
	typedef RefType inst_size;


	enum Type {
	    Int,
	    Nat,
	    Bool,
	    Ref
	};

   enum OccupancyStrategy {
      MaxOccupancy = 0,
      OneBlockPerSM = 1,
   };

   static std::tuple<dim3, dim3> get_launch_dims(inst_size nrof_instances, const void* kernel, OccupancyStrategy strat = MaxOccupancy){
      assert(strat == MaxOccupancy);
      int numBlocksPerSm = 0;
      int numThreads = THREADS_PER_BLOCK;

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);
      
      int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
      dim3 dimBlock(numThreads, 1, 1);
      dim3 dimGrid( min(max_blocks, (nrof_instances + numThreads - 1)/numThreads), 1, 1);
      return std::make_tuple(dimGrid, dimBlock);
   }

	static Type parse_type_string(std::string s) {
      if (s == "Int") return Int;
      if (s == "Nat") return Nat;
      if (s == "Bool") return Bool;
      return Ref;
   }

   static constexpr size_t size_of_type(Type t) {
      if (t == Int) return sizeof(IntType);
      if (t == Nat) return sizeof(NatType);
      if (t == Bool) return sizeof(BoolType);
      return sizeof(RefType);
   }

   static const IntType default_Int = 0;
	static const NatType default_Nat = 0;
	static const BoolType default_Bool = false;
	static const RefType default_Ref = 0;

   static constexpr const void* default_value(Type t) {
      if (t == Int) return &default_Int;
      if (t == Nat) return &default_Nat;
      if (t == Bool) return &default_Bool;
      return &default_Ref;
   }
}
#endif