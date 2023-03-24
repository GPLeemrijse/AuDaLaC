#ifndef ADL_H
#define ADL_H
#include <string>

#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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