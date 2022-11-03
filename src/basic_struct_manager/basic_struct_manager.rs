use std::collections::HashMap;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;
use indoc::{indoc, formatdoc};


pub struct BasicStructManager<'a> {
	program : &'a Program,
}

impl StructManager for BasicStructManager<'_> {
	fn add_includes(&self, set: &mut BTreeSet<String>) {
		set.insert("<stdarg.h>".to_string());
	}

	fn defines(&self) -> String {
		let struct_names = self.program.structs.iter().map(|s| s.name.clone());
		let mut res = String::new();
		for s in struct_names {
			let s_cap : String = s.chars().map(|c| c.to_uppercase().collect::<String>()).collect::<String>();
			res.push_str(&format!("#define MAX_NROF_{}S 100\n", s_cap));
		}
		res
	}

	fn struct_typedef(&self) -> String {
		let mut res = String::new();
		for strct in &self.program.structs {
			let n = &strct.name;
			let params = strct.parameters.iter()
						.map(|(s, t, _)| format!("    {} {s};", t.as_c_type()))
						.reduce(|acc: String, nxt| acc + "\n" + &nxt).unwrap();

			res.push_str(
				&formatdoc!{"
					typedef struct {n} {{
					{params}
					}} {n};

				"}
			);
		}
		res.push_str(
			&indoc!{"
				typedef struct StructManager {
				    void* structs;
				    unsigned int struct_size;
				    unsigned int nrof_active_structs;
				    unsigned int nrof_active_structs_before_launch;
				} StructManager;
			"}
		);
		res
	}

	fn globals(&self) -> String {
		let mut res = String::new();
		for strct in &self.program.structs {
			let n = &strct.name;
			res.push_str(
				&formatdoc!{"
					__managed__ StructManager {n}_manager = {{
						NULL,
						sizeof({n}),
						0,
						0
					}};

				"}
			);
		}
		res
	}

	fn function_defs(&self) -> String {
		indoc! {r#"
			__host__ __device__ void print_struct_manager(StructManager* m){{
			    printf("size:%u, active:%u, active_bl:%u\n", m->struct_size, m->nrof_active_structs, m->nrof_active_structs_before_launch);
			}}

			void destroy_struct_manager(StructManager* self){{
			    cudaFree(self);
			}}

			void ready_struct_manager(StructManager* self){{
			    self->nrof_active_structs_before_launch = self->nrof_active_structs;
			}}
		"#}.to_string()
	}
	fn pre_main(&self) -> std::string::String { todo!() }
	fn post_main(&self) -> std::string::String { todo!() }
}


impl BasicStructManager<'_> {
	pub fn new(program: &Program) -> BasicStructManager {
		BasicStructManager {
			program,
		}
	}
}


#[cfg(test)]
mod tests {
}