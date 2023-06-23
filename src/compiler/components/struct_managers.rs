use crate::analysis::executors;
use crate::as_type_enum;
use crate::compiler::utils::*;
use crate::compiler::CompileComponent;
use crate::parser::ast::*;
use crate::MemOrder;
use crate::Scope;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct StructManagers<'a> {
    program: &'a Program,
    allocated_per_instance: &'a HashMap<String, usize>,
    memorder: &'a MemOrder,
    scope: Scope,
    use_step_parity: bool,
}

impl StructManagers<'_> {
    pub fn new<'a>(
        program: &'a Program,
        allocated_per_instance: &'a HashMap<String, usize>,
        memorder: &'a MemOrder,
        scope: Scope,
        use_step_parity: bool,
    ) -> StructManagers<'a> {
        StructManagers {
            program,
            allocated_per_instance,
            memorder,
            scope,
            use_step_parity,
        }
    }

    fn type_as_c(&self, t: &Type) -> String {
        let basic_type = as_c_type(t);

        if !self.memorder.is_strong() {
            basic_type.to_string()
        } else {
            format!("ATOMIC({basic_type})")
        }
    }

    fn create_instance_function_as_c(&self, strct: &ADLStruct) -> String {
        let mut create_func_parameters: Vec<String> = strct
            .parameters
            .iter()
            .map(|(s, t, _)| format!("{} _{s}", as_c_type(t)))
            .collect();

        let mut assignments = strct
            .parameters
            .iter()
            .map(|(s, _, _)| format!("STORE({s}[slot], _{s});"))
            .collect::<Vec<String>>()
            .join("\n\t\t");

        let claim_instance;
        if self.use_step_parity {
            claim_instance = format!("claim_instance2()");
            create_func_parameters.push("bool* stable".to_string());
            assignments.push_str("\n\t\t*stable = false;");
        } else {
            claim_instance = "claim_instance()".to_string();
        }

        let header = "__device__ RefType create_instance".to_string();
        let signature = format_signature(&header, &create_func_parameters, 1);

        formatdoc! {"
            \t{signature}
            \t\tRefType slot = {claim_instance};
            \t\t{assignments}
            \t\treturn slot;
            \t}}"
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
            let order = self.memorder.as_cuda_order(Some(MemoryOperation::Store));
            format!("A.store(V, {order})")
        } else {
            format!("A = V")
        };

        let load_macro = if self.memorder.is_strong() {
            let order = self.memorder.as_cuda_order(Some(MemoryOperation::Load));
            format!("A.load({order})")
        } else {
            format!("A")
        };

        let acq_order = MemOrder::AcqRel.as_cuda_order(Some(MemoryOperation::Load));
        let rel_order = MemOrder::AcqRel.as_cuda_order(Some(MemoryOperation::Store));

        Some(formatdoc! {"
			#define ATOMIC(T) cuda::atomic<T, {scope}>
			#define STORE(A, V) {store_macro}
			#define LOAD(A) {load_macro}
            
			#define WLOAD(T, A) *((T*)&A)
            #define ACQLOAD(A) A.load({acq_order})
			#define WSTORE(T, A, V) *((T*)&A) = V
            #define RELSTORE(A, V) A.store(V, {rel_order})

		"})
    }

    fn typedefs(&self) -> Option<String> {
        let mut res = String::new();

        for strct in &self.program.structs {
            let struct_name = &strct.name;
            let nrof_params = strct.parameters.len();

            let param_decls = strct
                .parameters
                .iter()
                .map(|(s, t, _)| format!("{}* {s};", self.type_as_c(t)))
                .reduce(|acc: String, nxt| acc + "\n\t" + &nxt)
                .unwrap();

            let param_type_assertions = strct
                .parameters
                .iter()
                .enumerate()
                .map(|(idx, (_, t, _))| {
                    format!(
                        "assert (info->parameter_types[{idx}] == ADL::{});",
                        as_type_enum(t)
                    )
                })
                .reduce(|acc: String, nxt| acc + "\n		" + &nxt)
                .unwrap();

            let param_sizes = strct
                .parameters
                .iter()
                .map(|(_, t, _)| format!("sizeof({})", self.type_as_c(t)))
                .collect::<Vec<String>>()
                .join(",\n\t\t\t");

            let create_func = self.create_instance_function_as_c(strct);

            let first_param = strct
                .parameters
                .iter()
                .next()
                .map_or("NULL".to_string(), |p| format!("&{}", p.0));

            res.push_str(&formatdoc! {"
				class {struct_name} : public Struct {{
				public:
					{struct_name} (void) : Struct() {{}}
					
					{param_decls}

					void assert_correct_info(InitFile::StructInfo* info) {{
						assert (info->name == \"{struct_name}\");
						assert (info->parameter_types.size() == {nrof_params});
						{param_type_assertions}
					}};

					void** get_parameters(void) {{
						return (void**){first_param};
					}}

					size_t child_size(void) {{
						return sizeof({struct_name});
					}}

					size_t param_size(uint idx) {{
						static const size_t sizes[{nrof_params}] = {{
							{param_sizes}
						}};
						return sizes[idx];
					}}

				{create_func}
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

        let mut struct_names: Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
        struct_names.sort();

        for struct_name in struct_names {
            constr.push_str(&format! {"{struct_name} host_{struct_name} = {struct_name}();\n"});
            ptrs.push_str(
                &format! {"{struct_name}* host_{struct_name}_ptr = &host_{struct_name};\n"},
            );
            device_constr.push_str(&format!{"__device__ {struct_name}* __restrict__ {};\n", struct_name.to_lowercase()});
        }

        res.push_str(&formatdoc! {"
			{constr}
			{ptrs}
			{device_constr}
		"});

        return Some(res);
    }

    fn functions(&self) -> Option<String> {
        None
    }
    fn kernels(&self) -> Option<String> {
        None
    }

    fn pre_main(&self) -> Option<String> {
        let mut res = String::new();
        let mut registers = String::new();
        let mut inits = String::new();
        let mut to_device = String::new();
        let mut memcpy = String::new();

        let mut struct_names: Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
        struct_names.sort();

        let mut max_executor_str = "0".to_string();
        let executors = executors(self.program);

        for (idx, struct_name) in struct_names.iter().enumerate() {
            let s_name_lwr = struct_name.to_lowercase();
            let allocated_space: Option<&usize> = self.allocated_per_instance.get(*struct_name);

            let n_insts;
            if allocated_space.is_none() {
                println!(
                    "Warning: did not find number of instances to allocate for {struct_name}."
                );
                n_insts = format!("structs[{idx}].nrof_instances + 1");
            } else {
                n_insts = allocated_space.unwrap().to_string();
            }

            registers.push_str(&format!{"\tCHECK(cudaHostRegister(&host_{struct_name}, sizeof({struct_name}), cudaHostRegisterDefault));\n"});
            inits.push_str(
                &format! {"\thost_{struct_name}.initialise(&structs[{idx}], {n_insts});\n"},
            );
            to_device.push_str(&format!{"\t{struct_name} * const loc_{s_name_lwr} = ({struct_name}*)host_{struct_name}.to_device();\n"});
            memcpy.push_str(&format!{"\tCHECK(cudaMemcpyToSymbol({s_name_lwr}, &loc_{s_name_lwr}, sizeof({struct_name} * const)));\n"});

            if executors.get(struct_name).is_some() {
                max_executor_str = if max_executor_str == "0".to_string() {
                    n_insts
                } else {
                    format!("max({n_insts}, {max_executor_str})")
                };
            }
        }

        let max_instances =
            format!("\tinst_size max_nrof_executing_instances = {max_executor_str};");

        res.push_str(&formatdoc! {"
			{registers}
			{inits}
			{max_instances}
				CHECK(cudaDeviceSynchronize());

			{to_device}
			{memcpy}
		"});

        return Some(res);
    }

    fn main(&self) -> Option<String> {
        None
    }
    fn post_main(&self) -> Option<String> {
        None
    }
}
