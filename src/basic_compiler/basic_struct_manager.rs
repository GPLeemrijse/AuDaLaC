use crate::compiler::StructManager;
use crate::parser::ast::*;
use crate::basic_compiler::*;
use indoc::{formatdoc, indoc};
use std::collections::BTreeSet;

pub struct BasicStructManager<'a> {
    program: &'a Program,
    nrof_structs: usize,
}

impl StructManager for BasicStructManager<'_> {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<stdarg.h>");
    }

    fn defines(&self) -> String {
        let struct_names = self.program.structs.iter().map(|s| s.name.clone());
        let mut res = "#define SET_PARAM(I, T, P, V) ({if (I != 0) { T read_val = P; if (read_val != V) {P = V; clear_stability_stack();}}})\n".to_string();
        for s in struct_names {
            let s_cap: String = s
                .chars()
                .map(|c| c.to_uppercase().collect::<String>())
                .collect::<String>();
            res.push_str(&format!(
                "#define MAX_NROF_{}S {}\n",
                s_cap, self.nrof_structs
            ));
        }
        res
    }

    fn struct_typedef(&self) -> String {
        let mut res = String::new();

        // Add declarations
        for strct in &self.program.structs {
            let n = &strct.name;
            res.push_str(&formatdoc! {"
					struct {n};
				"});
        }

        // Add implementations
        for strct in &self.program.structs {
            let n = &strct.name;
            let params = strct
                .parameters
                .iter()
                .map(|(s, t, _)| format!("    {} {s};", as_c_type(t)))
                .reduce(|acc: String, nxt| acc + "\n" + &nxt)
                .unwrap();

            res.push_str(&formatdoc! {"
					typedef struct {n} {{
					{params}
					}} {n};
				"});
        }
        res.push_str(&indoc! {"
				typedef struct StructManager {
				    void* structs;
				    unsigned int nrof_active_structs;
				    unsigned int nrof_active_structs_before_launch;
				} StructManager;
			"});
        res
    }

    fn globals(&self) -> String {
        let mut res = String::new();
        for strct in &self.program.structs {
            let n = &strct.name;
            res.push_str(&formatdoc! {"
					__managed__ StructManager {n}_manager = {{
						NULL,
						0,
						0
					}};
				"});
        }
        res
    }

    fn function_defs(&self) -> String {
        let mut res = indoc! {r#"
			__host__ __device__ void print_struct_manager(StructManager* m){
			    printf("active:%u, active_bl:%u\n", m->nrof_active_structs, m->nrof_active_structs_before_launch);
			}

			void destroy_struct_manager(StructManager* self){
			    cudaFree(self->structs);
			}

			void ready_struct_manager(StructManager* self){
			    self->nrof_active_structs_before_launch = self->nrof_active_structs;
			}

		"#}.to_string();

        for strct in &self.program.structs {
            let n = &strct.name;
            let n_cap: String = n
                .chars()
                .map(|c| c.to_uppercase().collect::<String>())
                .collect::<String>();
            let params = strct
                .parameters
                .iter()
                .map(|(s, t, _)| format!(", {} {s}", as_c_type(t)))
                .reduce(|acc: String, nxt| acc + &nxt)
                .unwrap();
            let set_params = strct
                .parameters
                .iter()
                .map(|(s, _, _)| format!("    result->{s} = {s};\n"))
                .reduce(|acc: String, nxt| acc + &nxt)
                .unwrap();
            let fmt = strct
                .parameters
                .iter()
                .map(|(s, t, _)| format!(", {s}={}", as_printf(t)))
                .reduce(|acc: String, nxt| acc + &nxt)
                .unwrap();
            let cast_params = strct
                .parameters
                .iter()
                .map(|(s, _, _)| format!(", m->{s}"))
                .reduce(|acc: String, nxt| acc + &nxt)
                .unwrap();

            res.push_str(&formatdoc! {r#"
					{n}* host_create_{n}(StructManager* manager{params}) {{
					    unsigned int idx = manager->nrof_active_structs;
					    manager->nrof_active_structs++;

					    {n}* result = &(({n}*)manager->structs)[idx];
					{set_params}
					    return result;
					}}

					__device__ {n}* create_{n}(StructManager* manager{params}) {{
					    unsigned int idx = atomicInc(&manager->nrof_active_structs, MAX_NROF_{n_cap}S);
					    {n}* result = &(({n}*)manager->structs)[idx];
					{set_params}
					    return result;
					}}

					__host__ __device__ void print_{n}({n}* m){{
					    printf("{n}: %p{fmt}\n", m{cast_params});
					}}

					__global__ void {n}_print() {{
					    int i = blockDim.x * blockIdx.x + threadIdx.x;
					    if(i >= {n}_manager.nrof_active_structs_before_launch)
					        return;

					    {n}* self = &(({n}*){n}_manager.structs)[i];

					    print_{n}(self);
					}}
				"#});
        }
        res
    }

    fn pre_main(&self) -> String {
        let mut res = String::new();
        for strct in &self.program.structs {
            let n = &strct.name;
            let n_cap: String = n
                .chars()
                .map(|c| c.to_uppercase().collect::<String>())
                .collect::<String>();

            res.push_str(&formatdoc! {"
				cudaMallocManaged(&{n}_manager.structs, MAX_NROF_{n_cap}S * sizeof({n}));
			"});
        }
        res.push('\n');

        for strct in &self.program.structs {
            let n = &strct.name;
            let param_defaults = strct
                .parameters
                .iter()
                .map(|(_, t, _)| format!(", {}", as_c_default(t)))
                .reduce(|acc: String, nxt| acc + &nxt)
                .unwrap();

            res.push_str(&formatdoc! {"
				{n}* null_{n} = host_create_{n}(&{n}_manager{param_defaults});
			"});
        }
        res.push('\n');

        for strct in &self.program.structs {
            let n = &strct.name;
            for (p_name, t, _) in &strct.parameters {
                if let Type::Named(p_type) = t {
                    res.push_str(&formatdoc! {"
						null_{n}->{p_name} = null_{p_type};
					"});
                }
            }
        }

        res
    }

    fn post_main(&self) -> String {
        let mut res = String::new();
        for strct in &self.program.structs {
            let n = &strct.name;
            res.push_str(&formatdoc! {"
				destroy_struct_manager(&{n}_manager);
			"})
        }
        res
    }

    fn kernels(&self) -> std::string::String {
        let mut res = String::new();

        for strct in &self.program.structs {
            let strct_name = &strct.name;
            for step in &strct.steps {
                let step_name = &step.name;
                let body = self.make_body(&step.statements, &strct.parameters);
                res.push_str(&formatdoc! {"
					__global__ void {strct_name}_{step_name}() {{
					    int i = blockDim.x * blockIdx.x + threadIdx.x;
					    if(i >= {strct_name}_manager.nrof_active_structs_before_launch)
					    	return;

					    {strct_name}* self = &(({strct_name}*){strct_name}_manager.structs)[i];

					    {body}
					}}
				"});
            }
        }
        res
    }

    fn kernel_parameters(&self, _: &ADLStruct, _: &Step) -> Vec<String> {
        todo!()
    }
    fn kernel_arguments(&self, _: &ADLStruct, _: &Step) -> Vec<String> {
        todo!()
    }
    fn kernel_name(&self, strct: &ADLStruct, step: &Step) -> String {
        format!("{}_{}", strct.name, step.name)
    }
}

impl BasicStructManager<'_> {
    pub fn new(program: &Program, nrof_structs: usize) -> BasicStructManager {
        BasicStructManager {
            program,
            nrof_structs,
        }
    }

    pub fn make_body(
        &self,
        statements: &Vec<Stat>,
        parameters: &Vec<(String, Type, Loc)>,
    ) -> String {
        let mut res = String::new();
        let is_param = |p: &String| parameters.iter().any(|(s, _, _)| s == p);

        for stmt in statements {
            use crate::parser::ast::Stat::*;
            res.push_str(&match stmt {
                IfThen(e, stmts_true, stmts_false, _) => {
                    let cond = as_c_expression(e, self.program, &is_param, None);
                    let body_true = self.make_body(&stmts_true, parameters);
                    let body_false = self.make_body(&stmts_false, parameters);

                    let else_block = if body_false == "" {
                        "".to_string()
                    } else {
                        format!("else {{\n\t{body_false}\n}}")
                    };

                    formatdoc! {"
							if {cond} {{
								{body_true}
							}}{else_block}
						"}
                }
                Declaration(t, n, e, _) => {
                    let t_comp = as_c_type(t);
                    let e_comp = as_c_expression(e, self.program, &is_param, None);
                    format!("{t_comp} {n} = {e_comp};\n")
                }
                Assignment(parts_exp, e, _) => {
                    let parts = parts_exp.get_parts();
                    let p = parts.join("->");
                    let e_comp = as_c_expression(e, self.program, &is_param, None);

                    if is_param(&parts[0]) {
                        let p_type = as_c_type(&self.get_param_type(parts, parameters));
                        format!("SET_PARAM(i, {p_type}, self->{p}, {e_comp});\n")
                    } else {
                        format!("{p} = {e_comp};\n")
                    }
                }
                InlineCpp(cpp) => cpp.to_string(),
            });
        }

        res
    }

    fn get_param_type(&self, parts: &Vec<String>, parameters: &Vec<(String, Type, Loc)>) -> Type {
        let mut params: &Vec<(String, Type, Loc)> = parameters;
        let mut p_type: &Type = &parameters
            .iter()
            .find(|(s, _, _)| s == &parts[0])
            .unwrap()
            .1;
        for p in &parts[1..] {
            if let Type::Named(s) = p_type {
                params = &self
                    .program
                    .structs
                    .iter()
                    .find(|strct| strct.name == *s)
                    .unwrap()
                    .parameters;
            }
            p_type = &params.iter().find(|(s, _, _)| s == p).unwrap().1;
        }
        p_type.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::ast::Exp::*;
    use crate::parser::ast::Literal::*;
    use crate::parser::ast::Schedule::*;
    use crate::parser::ast::Stat::*;
    use crate::parser::ast::Type::*;
    use crate::parser::ast::*;
    use crate::BasicStructManager;

    #[test]
    fn test_nested_constructors() {
        let mut program: Program = Program {
            inline_global_cpp: Vec::new(),
            structs: Vec::new(),
            schedule: Box::new(StepCall("test".to_string(), (1, 2))),
        };

        let step1 = Step {
            name: "step1".to_string(),
            statements: vec![Declaration(
                Named("strctA".to_string()),
                "e1".to_string(),
                Box::new(Constructor(
                    "strctA".to_string(),
                    vec![
                        Lit(NullLit, (0, 0)),
                        Constructor(
                            "strctB".to_string(),
                            vec![Lit(NullLit, (0, 0)), Lit(NullLit, (0, 0))],
                            (0, 0),
                        ),
                    ],
                    (0, 0),
                )),
                (0, 0),
            )],
            loc: (0, 0),
        };

        let strct_a = ADLStruct {
            name: "strctA".to_string(),
            parameters: vec![
                ("P1".to_string(), Named("strctA".to_string()), (0, 0)),
                ("P2".to_string(), Named("strctB".to_string()), (0, 0)),
            ],
            steps: vec![step1.clone()],
            loc: (0, 0),
        };

        let strct_b = ADLStruct {
            name: "strctB".to_string(),
            parameters: vec![
                ("P1".to_string(), Named("strctA".to_string()), (0, 0)),
                ("P2".to_string(), Named("strctB".to_string()), (0, 0)),
            ],
            steps: vec![step1.clone()],
            loc: (0, 0),
        };

        program.structs.push(strct_a);
        program.structs.push(strct_b);

        let basic_sm: BasicStructManager = BasicStructManager::new(&program, 10);

        let body = basic_sm.make_body(
            &program.structs[0].steps[0].statements,
            &program.structs[0].parameters,
        );
        assert_eq!(
			"strctA* e1 = create_strctA(&strctA_manager, (strctA*)strctA_manager.structs, create_strctB(&strctB_manager, (strctA*)strctA_manager.structs, (strctB*)strctB_manager.structs));\n", body);

        let body = basic_sm.make_body(
            &program.structs[1].steps[0].statements,
            &program.structs[1].parameters,
        );
        assert_eq!(
			"strctA* e1 = create_strctA(&strctA_manager, (strctA*)strctA_manager.structs, create_strctB(&strctB_manager, (strctA*)strctA_manager.structs, (strctB*)strctB_manager.structs));\n", body);
    }
}
