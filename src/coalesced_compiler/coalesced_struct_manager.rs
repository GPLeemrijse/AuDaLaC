use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;
use indoc::{formatdoc, indoc};
use crate::coalesced_compiler::*;

pub struct CoalescedStructManager<'a> {
	pub program : &'a Program,
	pub nrof_structs : u64
}

impl StructManager for CoalescedStructManager<'_> {
	fn add_includes(&self, set: &mut BTreeSet<std::string::String>) {
		set.insert("\"ADL.h\"".to_string());
		set.insert("\"init_file.h\"".to_string());
		set.insert("\"Struct.h\"".to_string());
		set.insert("<cooperative_groups.h>".to_string());
	}
	
	fn defines(&self) -> String {
		indoc!("
			using namespace cooperative_groups;

			#define SET_PARAM(P, V, T, I) ({if (I != 0) { T read_val = P; T write_val = V; if (read_val != write_val) {P = write_val; FP->set();}}})
		").to_string()
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
			let params_args = strct.parameters.iter()
								.map(|(s, t, _)| format!("ADL::{} _{s}", as_c_type(t)))
								.reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();
			let assignments = strct.parameters.iter()
								.map(|(s, _, _)| format!("{s}[slot] = _{s};"))
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

					__host__ __device__ RefType create_instance({params_args}) {{
						RefType slot = claim_instance();
						{assignments}
						return slot;
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
		let step_kernels : Vec<String> = self.program.structs.iter()
															 .map(|strct|
															 	strct.steps.iter()
																		   .map(|step| self.step_as_c(strct, step))
															 )
															 .flatten()
															 .collect();
		let print_kernels : Vec<String> = self.program.structs.iter()
												.map(|strct|
													self.print_kernel_as_c(strct)
												)
												.collect();
		return step_kernels.join("\n") + &print_kernels.join("\n")
	}

	fn pre_main(&self) -> String {
		let mut res = String::new();
		let mut constr = String::new();
		let mut ptrs = String::new();
		let mut registers = String::new();
		let mut inits = String::new();
		let mut to_device = String::new();

		let mut struct_names : Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
		struct_names.sort();

		for (idx, struct_name) in struct_names.iter().enumerate() {
			constr.push_str(&format!{"\t{struct_name} host_{struct_name} = {struct_name}();\n"});
			ptrs.push_str(&format!{"\t{struct_name}* host_{struct_name}_ptr = &host_{struct_name};\n"});
			registers.push_str(&format!{"\tCHECK(cudaHostRegister(&host_{struct_name}, sizeof({struct_name}), cudaHostRegisterDefault));\n"});
			inits.push_str(&format!{"\thost_{struct_name}.initialise(&structs[{idx}], {});\n", self.nrof_structs});
			to_device.push_str(&format!{"\t{struct_name}* gm_{struct_name} = ({struct_name}*)host_{struct_name}.to_device();\n"});
		}

		res.push_str(&formatdoc!{"
			{constr}
			{ptrs}
			{registers}
			{inits}
				CHECK(cudaDeviceSynchronize());

			{to_device}
		"});

		return res;
	}
	
	fn post_main(&self) -> String {
		return String::new();
	}

	fn kernel_parameters(&self, _strct: &ADLStruct, step: &Step) -> Vec<String> {
		let mut result : Vec<String> = vec!["FPManager* FP".to_string()];

		result.append(&mut self.program.structs.iter()
											   .map(|s| format!("{}* const {}", s.name, s.name.to_lowercase()))
											   .collect()
		);
		
		let mut constructed_types : Vec<String> = step.constructors()
													  .iter()
													  .map(|c| format!("{}* const host_{}", c, c.to_lowercase()))
													  .collect();
		result.append(&mut constructed_types);
		result
	}

	fn kernel_arguments(&self, _strct: &ADLStruct, step: &Step) -> Vec<String> {
		let mut result : Vec<String> = vec!["&device_FP".to_string()];

		result.append(&mut self.program.structs.iter()
											   .map(|s| format!("&gm_{}", s.name))
											   .collect()
		);
		
		let mut constructed_types : Vec<String> = step.constructors()
													  .iter()
													  .map(|c| format!("&host_{c}_ptr"))
													  .collect();
		result.append(&mut constructed_types);
		result
	}
	fn kernel_name(&self, strct: &ADLStruct, step: &Step) -> String {
		format!("{}_{}", strct.name, step.name)
	}

}

impl CoalescedStructManager<'_> {
	pub fn new<'a>(program: &'a Program, nrof_structs: u64) -> CoalescedStructManager<'a> {
		CoalescedStructManager {
			program,
			nrof_structs,
		}
	}

	fn step_as_c(&self, strct: &ADLStruct, step: &Step) -> String {
		let kernel_name = self.kernel_name(strct, step);
		let part_before_params = format!("__global__ void {kernel_name}(");
		let param_indent = format!(",\n{}{}",
								   "\t".repeat(part_before_params.len() / 4),
								   " ".repeat(part_before_params.len() % 4)
								  ); 

		let kernel_parameters = self.kernel_parameters(strct, step).join(&param_indent);

		let kernel_body = self.statements_as_c(&step.statements, &strct, &step, 1);

		let constructs = step.constructors();

		let sync_constructed = if constructs.is_empty() {
			"".to_string()
		} else {
			let to_sync = constructs.iter()
									.map(|i| format!("{}->sync_nrof_instances(host_{});", i.to_lowercase(), i.to_lowercase()))
									.reduce(|acc, nxt| acc + "\n\t\t" + &nxt)
									.unwrap();
			formatdoc!{"\n
				\tgrid.sync();

				\tif (self == 0) {{
				\t\t{to_sync}
				\t}}
			"}
		};


		



		formatdoc!{"
			{part_before_params}{kernel_parameters}){{
				grid_group grid = this_grid();
				RefType self = blockDim.x * blockIdx.x + threadIdx.x;
				if(!{}->is_active(self)) {{ return; }}
			{kernel_body}{sync_constructed}
			}}
		", strct.name.to_lowercase()}
	}

	fn print_kernel_as_c(&self, strct: &ADLStruct) -> String {
		let kernel_name = format!("{}_print", strct.name);
		let s_name = &strct.name;
		let s_name_lwr = strct.name.to_lowercase();
		let kernel_parameters = format!("{s_name}* {s_name_lwr}");
		let params_as_fmt = strct.parameters.iter()
											.map(|(n, t, _)| format!("{n}={}", as_printf(t)))
											.reduce(|a, n| a + ", " + &n)
											.unwrap();
		let params_as_expr = strct.parameters.iter()
											.map(|(n, _, _)| format!(", {s_name_lwr}->{n}[self]"))
											.reduce(|a, n| a + &n)
											.unwrap();

		formatdoc!{"
			__global__ void {kernel_name}({kernel_parameters}){{
				RefType self = blockDim.x * blockIdx.x + threadIdx.x;
				if(!{s_name_lwr}->is_active(self)) {{ return; }}
				if (self == 0) {{
					printf(\"{s_name}(0): ---------------------------\\n\");
				}} else {{
					printf(\"{s_name}(%u): {params_as_fmt}\\n\", self{params_as_expr});
				}}
			}}
		"}
	}

	fn statements_as_c(&self, statements: &Vec<Stat>, strct: &ADLStruct, step: &Step, indent_lvl: usize) -> String {
		let indent = String::from("\t").repeat(indent_lvl);
		let mut stmt_vec : Vec<String> = Vec::new();

		for stmt in statements {
			use crate::ast::Stat::*;

			let statement_as_string = match stmt {
				IfThen(e, stmts, _) => {
					let cond = self.expression_as_c(e, strct, step);
					let body = self.statements_as_c(stmts, strct, step, indent_lvl + 1);
					format!{"{indent}if {cond} {{{body}\n{indent}}}"}
				},
				Declaration(t, n, e, _) => {
					let t_as_c = as_c_type(t);
					let e_as_c = self.expression_as_c(e, strct, step);
					format!("{indent}{t_as_c} {n} = {e_as_c};")
				},
				Assignment(parts, e, _) => {
					let types = self.get_part_types(parts, strct, step);
					let e_as_c = self.expression_as_c(e, strct, step);
					
					let parts_as_c = self.expression_as_c(
						&Exp::Var(
							parts.to_vec(),
							(0, 0)
						),
						strct,
						step
					);

					let owner = self.expression_as_c(
						&Exp::Var(
							parts[..parts.len()-1].to_vec(),
							(0, 0)
						),
						strct,
						step
					);

					if strct.parameter_by_name(&parts[0]).is_some() {
						let p_type = as_c_type(types.last().unwrap());
						
						format!("{indent}SET_PARAM({parts_as_c}, {e_as_c}, {p_type}, {owner});")
					} else {
						format!("{indent}{parts_as_c} = {e_as_c};")
					}
				},
			};
			stmt_vec.push(statement_as_string);
		}

		return stmt_vec.iter()
					   .fold("".to_string(), |acc: String, nxt| acc + "\n" + nxt);
	}

	fn get_part_types<'a>(&'a self, parts: &Vec<String>, strct: &'a ADLStruct, step: &'a Step) -> Vec<&'a Type> {
		let mut cur_strct : &ADLStruct = strct;
		let mut p_types : Vec<&Type> = Vec::new();
		let declarations : Vec<(&String, &Type)> = step.declarations();
		let local_par_type : Option<&Type> = declarations.iter()
											   .find(|(s, _)| **s == parts[0])
											   .and_then(|(_, t)| Some(*t));
		
		// Figure out if first part is a parameter or local variable
		let start_idx;
		if local_par_type.is_some() {
			let t = local_par_type.unwrap();
			p_types.push(t);
			start_idx = 1;// Skip first part

			if let Some(t_name) = t.name() {
				cur_strct = &self.program.struct_by_name(t_name).unwrap();
			} else {
				debug_assert!(parts.len() == 1);
			}
		} else {
			start_idx = 0;// Use next loop to find types
		}

		for p in &parts[start_idx..] {
			p_types.push(&cur_strct.parameter_by_name(p).unwrap().1);

			if let Type::Named(s) = p_types.last().unwrap() {
				cur_strct = &self.program.struct_by_name(s).unwrap();
			} else {
				debug_assert!(p == parts.last().unwrap());
			}
		}
		p_types
	}

	pub fn expression_as_c(&self, e: &Exp, strct: &ADLStruct, step: &Step) -> String {
	    use Exp::*;
	    use std::iter::zip;
	    
	    match e {
	        BinOp(e1, c, e2, _) => {
	            let e1_comp = self.expression_as_c(e1, strct, step);
	            let e2_comp = self.expression_as_c(e2, strct, step);

	            format!("({e1_comp} {c} {e2_comp})")
	        },
	        UnOp(c, e, _) => {
	            let e_comp = self.expression_as_c(e, strct, step);
	            format!("({c}{e_comp})")
	        },
	        Constructor(n, args, _) => {
	            let args_comp = args.iter()
	                    .map(|e| self.expression_as_c(e, strct, step))
	                    .reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();

	            format!("{}->create_instance({args_comp})", n.to_lowercase())
	        },
	        Var(parts, _) => {
	        	if parts.is_empty() {
	        		// Used for statements_as_c
	        		return "self".to_string();
	        	}
	        	let is_parameter = strct.parameter_by_name(&parts[0]).is_some();
	        	let types = self.get_part_types(parts, strct, step);

	        	let mut previous_c_expr : String;
	        	let mut previous_c_type : &String;
	        	let start_idx;
	        	if is_parameter {
	        		previous_c_expr = "self".to_string();
	        		previous_c_type = &strct.name;
	        		start_idx = 0;
	        	} else {
	        		previous_c_expr = parts[0].clone();
	        		previous_c_type = match types[0].name() {
	        			Some(t_name) => t_name,
	        			None => unreachable!()
	        		};

	        		start_idx = 1;
	        	}

	        	for (t, p) in zip(types[start_idx..].iter(), parts[start_idx..].iter()) {
	        		previous_c_expr = format!("{}->{p}[{previous_c_expr}]", previous_c_type.to_lowercase());
	        		if t.name().is_some() {
	        			previous_c_type = t.name().unwrap();
	        		}
	        	}

	            return format!("({previous_c_expr})");
	        },
	        Lit(l, _) => as_c_literal(l),
	    }
	}
}