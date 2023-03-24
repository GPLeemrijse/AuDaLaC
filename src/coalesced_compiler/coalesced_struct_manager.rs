use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;
use indoc::{indoc, formatdoc};
use crate::coalesced_compiler::*;

pub struct CoalescedStructManager<'a> {
	program : &'a Program,
	nrof_structs : u64,
}

impl StructManager for CoalescedStructManager<'_> {	
	fn add_includes(&self, set: &mut BTreeSet<std::string::String>) {
		set.insert("\"ADL.h\"".to_string());
		set.insert("\"init_file.h\"".to_string());
		set.insert("\"Struct.h\"".to_string());
	}
	fn defines(&self) -> std::string::String {
		return String::new();
	}
	fn struct_typedef(&self) -> String {
		let mut res = String::new();
		for strct in &self.program.structs {
			let struct_name = &strct.name;
			let nrof_params = strct.parameters.len();
			let param_decls = strct.parameters.iter()
								.map(|(s, t, _)| format!("ADL::{}* {s};", as_c_type(t)))
								.reduce(|acc: String, nxt| acc + "\n			" + &nxt).unwrap();
			let param_type_assertions = strct.parameters.iter()
								.enumerate()
								.map(|(idx, (_, t, _))| format!("assert (info->parameter_types[{idx}] == ADL::{});", as_type_enum(t)))
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

					void assertCorrectInfo(InitFile::StructInfo* info) {{
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
				}};

			"});
		}
		return res;
	}
	fn globals(&self) -> String {
		return String::new();
	}
	fn function_defs(&self) -> String {
		return String::new();
	}
	fn kernels(&self) -> String {
		return String::new();
	}
	fn pre_main(&self) -> String {
		let mut res = String::new();
		let mut constr = String::new();
		let mut inits = String::new();
		let mut to_device = String::new();

		for (idx, strct) in self.program.structs.iter().enumerate() {
			let struct_name = &strct.name;

			constr.push_str(&format!{"	{struct_name} host_{struct_name} = {struct_name}();\n"});
			inits.push_str(&format!{"	host_{struct_name}.initialise(&structs[{idx}], {});\n", self.nrof_structs});
			to_device.push_str(&format!{"	{struct_name}* gm_{struct_name} = ({struct_name}*)host_{struct_name}.to_device();\n"});
		}

		res.push_str(&formatdoc!{"
			{constr}
			{inits}
				CHECK(cudaDeviceSynchronize());

			{to_device}
		"});

		return res;
	}
	fn post_main(&self) -> String {
		return String::new();
	}
}

impl CoalescedStructManager<'_> {
	pub fn new(program: &Program, nrof_structs: u64) -> CoalescedStructManager {
		CoalescedStructManager {
			program,
			nrof_structs
		}
	}
}