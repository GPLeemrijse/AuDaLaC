use crate::ast::*;
use crate::as_type_enum;
use crate::Scope;
use crate::MemOrder;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub struct StructManagers<'a> {
	program : &'a Program,
	nrof_structs : u64,
	memorder : MemOrder,
	scope : Scope,
}

impl StructManagers<'_> {
	pub fn new<'a>(program : &'a Program,
			   nrof_structs : u64,
			   memorder : MemOrder,
			   scope : Scope) -> StructManagers<'a> {
		StructManagers {
			program,
			nrof_structs,
			memorder,
			scope
		}
	}

	fn type_as_c(&self, t : &Type) -> String {
		let basic_type = self.basic_type_as_c(t);

		if !self.memorder.is_strong() {
	    	basic_type.to_string()
	    } else {
	    	format!("ATOMIC({basic_type})")
	    }
	}

	fn basic_type_as_c(&self, t : &Type) -> &str {
		use Type::*;
	    match t {
	        Named(..) => "ADL::RefType",
	        String => "ADL::StringType",
	        Nat => "ADL::NatType",
	        Int => "ADL::IntType",
	        Bool => "ADL::BoolType",
	        Null => "ADL::RefType",
	    }
	}
}

impl CompileComponent for StructManagers<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("\"ADL.h\"");
		set.insert("\"init_file.h\"");
		set.insert("\"Struct.h\"");
		set.insert("<cuda/atomic>");
	}

	fn defines(&self) -> Option<String> {
		let scope = self.scope.as_cuda_scope();
		let store_macro = if self.memorder.is_strong() {
			let order = self.memorder.as_cuda_order();
			format!("A.store(B, {order})")
		} else {
			format!("A = B")
		};

		let load_macro = if self.memorder.is_strong() {
			let order = self.memorder.as_cuda_order();
			format!("A.load({order})")
		} else {
			format!("A")
		};

		Some(formatdoc!{"
			#define ATOMIC(T) cuda::atomic<T, {scope}>
			#define STORE(A, B) {store_macro}
			#define LOAD(A) {load_macro}
		"})
	}
	
	fn typedefs(&self) -> Option<String> {
		let mut res = "using namespace cooperative_groups;\n".to_string();
		for strct in &self.program.structs {
			let struct_name = &strct.name;
			let nrof_params = strct.parameters.len();

			let param_decls = strct.parameters.iter()
								.map(|(s, t, _)| format!("{}* {s};", self.type_as_c(t)))
								.reduce(|acc: String, nxt| acc + "\n			" + &nxt).unwrap();
			let param_type_assertions = strct.parameters.iter()
								.enumerate()
								.map(|(idx, (_, t, _))| format!("assert (info->parameter_types[{idx}] == ADL::{});", as_type_enum(t)))
								.reduce(|acc: String, nxt| acc + "\n		" + &nxt).unwrap();
			let params_args = strct.parameters.iter()
								.map(|(s, t, _)| format!("{} _{s}", self.basic_type_as_c(t)))
								.reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();
			let param_sizes = strct.parameters.iter()
								.map(|(_, t, _)| format!("sizeof({})", self.type_as_c(t)))
								.collect::<Vec<String>>()
								.join(",\n\t\t\t");

			let assignments = strct.parameters.iter()
								.map(|(s, _, _)| format!("STORE({s}[slot], _{s});"))
								.reduce(|acc: String, nxt| acc + "\n		" + &nxt).unwrap();

			res.push_str(&formatdoc!{"
				class {struct_name} : public Struct {{
				public:
					{struct_name} (void) : Struct() {{}}
					
					union {{
						void* parameters[{nrof_params}];
						struct {{
							{param_decls}
						}};
					}};

					void assert_correct_info(InitFile::StructInfo* info) {{
						assert (info->name == \"{struct_name}\");
						assert (info->parameter_types.size() == {nrof_params});
						{param_type_assertions}
					}};

					void** get_parameters(void) {{
						return parameters;
					}}

					size_t child_size(void) {{
						return sizeof({struct_name});
					}}

					size_t param_size(uint idx) {{
						static size_t sizes[{nrof_params}] = {{
							{param_sizes}
						}};
						return sizes[idx];
					}}

					__host__ __device__ RefType create_instance({params_args}) {{
						RefType slot = claim_instance();
						{assignments}
						return slot;
					}}
				}};

			"});
		}
		return Some(res);
	}

	fn globals(&self) -> Option<String> {
		let mut res = String::new();
		let mut constr = String::new();
		let mut ptrs = String::new();
		let mut device_constr = String::new();

		let mut struct_names : Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
		struct_names.sort();

		for struct_name in struct_names {
			constr.push_str(&format!{"{struct_name} host_{struct_name} = {struct_name}();\n"});
			ptrs.push_str(&format!{"{struct_name}* host_{struct_name}_ptr = &host_{struct_name};\n"});
			device_constr.push_str(&format!{"{struct_name}* gm_{struct_name};\n"});
		}

		res.push_str(&formatdoc!{"
			{constr}
			{ptrs}
			{device_constr}
		"});

		return Some(res);
	}

	fn functions(&self) -> Option<String> { None }
	fn kernels(&self) -> Option<String> { None }
	
	fn pre_main(&self) -> Option<String> {
		let mut res = String::new();
		let mut registers = String::new();
		let mut inits = String::new();
		let mut to_device = String::new();

		let mut struct_names : Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
		struct_names.sort();

		for (idx, struct_name) in struct_names.iter().enumerate() {
			registers.push_str(&format!{"\tCHECK(cudaHostRegister(&host_{struct_name}, sizeof({struct_name}), cudaHostRegisterDefault));\n"});
			inits.push_str(&format!{"\thost_{struct_name}.initialise(&structs[{idx}], {});\n", self.nrof_structs});
			to_device.push_str(&format!{"\tgm_{struct_name} = ({struct_name}*)host_{struct_name}.to_device();\n"});
		}

		res.push_str(&formatdoc!{"
			{registers}
			{inits}
				CHECK(cudaDeviceSynchronize());

			{to_device}
		"});

		return Some(res);
	}
	
	fn main(&self) -> Option<String> { None }
	fn post_main(&self) -> Option<String> { None }
}