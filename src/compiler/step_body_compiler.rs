use crate::analysis::racing_parameters;
use crate::compiler::utils::*;
use crate::parser::ast::*;
use indoc::formatdoc;
use std::collections::HashMap;
use std::collections::HashSet;

pub struct StepBodyCompiler<'a> {
    var_exp_type_info: &'a HashMap<*const Exp, Vec<Type>>,
    use_step_parity: bool,
    print_unstable: bool,
    ld_st_weakly_for_non_racing: bool,
}

impl StepBodyCompiler<'_> {
    pub fn new<'a>(
        type_info: &'a HashMap<*const Exp, Vec<Type>>,
        use_step_parity: bool,
        print_unstable: bool,
        ld_st_weakly_for_non_racing: bool,
    ) -> StepBodyCompiler<'a> {
        StepBodyCompiler {
            var_exp_type_info: type_info,
            use_step_parity,
            print_unstable,
            ld_st_weakly_for_non_racing,
        }
    }

    pub fn step_body_as_c(&self, program: &Program, strct: &ADLStruct, step: &Step) -> String {
        self.statements_as_c(
            &step.statements,
            strct,
            step,
            1,
            &racing_parameters(step, program, strct, self.var_exp_type_info),
        )
    }

    fn statements_as_c(
        &self,
        statements: &Vec<Stat>,
        strct: &ADLStruct,
        step: &Step,
        indent_lvl: usize,
        racing_params: &HashSet<(&String, &String)>,
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);

        statements
            .iter()
            .map(|stmt| match stmt {
                Stat::IfThen(..) => self.if_stmt_as_c(stmt, strct, step, indent_lvl, racing_params),
                Stat::Declaration(..) => {
                    self.decl_as_c(stmt, strct, step, indent_lvl, racing_params)
                }
                Stat::Assignment(..) => {
                    self.assignment_as_c(stmt, strct, step, indent_lvl, racing_params)
                }
                Stat::InlineCpp(cpp) => format!("{indent}{cpp}"),
            })
            .fold("".to_string(), |acc: String, nxt| acc + "\n" + &nxt)
    }

    fn if_stmt_as_c(
        &self,
        stmt: &Stat,
        strct: &ADLStruct,
        step: &Step,
        indent_lvl: usize,
        racing_params: &HashSet<(&String, &String)>,
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::IfThen(e, stmts_true, stmts_false, _) = stmt {
            let cond = self.expression_as_c(e, strct, step, racing_params);
            let body_true =
                self.statements_as_c(stmts_true, strct, step, indent_lvl + 1, racing_params);
            let body_false =
                self.statements_as_c(stmts_false, strct, step, indent_lvl + 1, racing_params);

            let else_block = if body_false == "" {
                "".to_string()
            } else {
                format!(" else {{{body_false}\n{indent}}}")
            };

            format! {"{indent}if ({cond}) {{{body_true}\n{indent}}}{else_block}"}
        } else {
            unreachable!()
        }
    }

    fn decl_as_c(
        &self,
        stmt: &Stat,
        strct: &ADLStruct,
        step: &Step,
        indent_lvl: usize,
        racing_params: &HashSet<(&String, &String)>,
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::Declaration(t, n, e, _) = stmt {
            let t_as_c = as_c_type(t);
            let e_as_c = self.expression_as_c(e, strct, step, racing_params);
            format!("{indent}{t_as_c} {n} = {e_as_c};")
        } else {
            unreachable!()
        }
    }

    fn assignment_as_c(
        &self,
        stmt: &Stat,
        strct: &ADLStruct,
        step: &Step,
        indent_lvl: usize,
        racing_params: &HashSet<(&String, &String)>,
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::Assignment(lhs_exp, rhs_exp, _) = stmt {
            let (lhs_as_c, owner, is_parameter) = self.var_exp_as_c(lhs_exp, strct, racing_params);
            let rhs_as_c = self.expression_as_c(rhs_exp, strct, step, racing_params);

            if is_parameter {
                let (owner_exp, owner_type) =
                    owner.expect("Expected an owner for a parameter assignment.");
                let parts_vec = lhs_exp.get_parts();
                let param = parts_vec.last().unwrap();
                let owner_type_name = owner_type
                    .name()
                    .expect("non-named type as owner is illegal");
                let owner_type_name_lwr = owner_type_name.to_lowercase();

                let types = self
                    .var_exp_type_info
                    .get(&(&**lhs_exp as *const Exp))
                    .expect("Could not find var expression in type info.");
                let par_type_name = as_c_type(types.last().unwrap());

                let param_str_param = if self.print_unstable {
                    format!(", \"{owner_type_name}.{param}\"")
                } else {
                    "".to_string()
                };

                let is_racing = racing_params.contains(&(owner_type_name, param));

                let store_op = if !is_racing && self.ld_st_weakly_for_non_racing {
                    "WSetParam"
                } else {
                    "SetParam"
                };

                formatdoc! {"
                    {indent}// {lhs_exp} := {rhs_exp};
                    {indent}{store_op}<{par_type_name}>({owner_exp}, {owner_type_name_lwr}->{param}, {rhs_as_c}, stable{param_str_param});"
                }
            } else {
                format!("{indent}{lhs_as_c} = {rhs_as_c};")
            }
        } else {
            unreachable!()
        }
    }

    fn expression_as_c<'a>(
        &self,
        e: &Exp,
        strct: &ADLStruct,
        step: &'a Step,
        racing_params: &HashSet<(&String, &String)>,
    ) -> String {
        use Exp::*;

        match e {
            BinOp(e1, c, e2, _) => {
                let e1_comp = self.expression_as_c(e1, strct, step, racing_params);
                let e2_comp = self.expression_as_c(e2, strct, step, racing_params);

                format!("({e1_comp} {c} {e2_comp})")
            }
            UnOp(c, e, _) => {
                let e_comp = self.expression_as_c(e, strct, step, racing_params);
                format!("({c}{e_comp})")
            }
            Constructor(n, args, _) => {
                let mut arg_expressions = args
                    .iter()
                    .map(|e| self.expression_as_c(e, strct, step, racing_params))
                    .collect::<Vec<String>>();

                if self.use_step_parity {
                    arg_expressions.push("stable".to_string());
                }

                let args = arg_expressions.join(", ");

                format!("{}->create_instance({args})", n.to_lowercase())
            }
            Var(..) => {
                self.var_exp_as_c(e, strct, racing_params).0 // Not interested in the owner or is_parameter
            }
            Lit(l, _) => as_c_literal(l),
        }
    }

    fn var_exp_as_c(
        &self,
        exp: &Exp,
        strct: &ADLStruct,
        racing_params: &HashSet<(&String, &String)>,
    ) -> (String, Option<(String, Type)>, bool) {
        use std::iter::zip;
        if let Exp::Var(parts, _) = exp {
            let types = self
                .var_exp_type_info
                .get(&(exp as *const Exp))
                .expect("Could not find var expression in type info.");
            let is_param_owned_by_self = strct.parameter_by_name(&parts[0]).is_some();
            let is_param_owned_by_local_var = !is_param_owned_by_self && parts.len() > 1;

            let (exp_as_c, owner);

            // For parameters, the 'self' part is implicit.
            if is_param_owned_by_self {
                (exp_as_c, owner) = self.stitch_parts_and_types(
                    zip(
                        ["self".to_string()].iter().chain(parts.iter()),
                        [Type::Named(strct.name.clone())].iter().chain(types.iter()),
                    ),
                    racing_params,
                );
            } else {
                (exp_as_c, owner) =
                    self.stitch_parts_and_types(zip(parts.iter(), types.iter()), racing_params);
            }
            return (
                exp_as_c,
                owner,
                is_param_owned_by_self || is_param_owned_by_local_var,
            );
        }
        panic!("Expected Var expression.");
    }

    /* Returns the full expression for evaluating the field-type pairs in parts,
       and, if applicable, returns the penultimate partial result (the parameter owner).
    */
    fn stitch_parts_and_types<'a, I>(
        &self,
        mut parts: I,
        racing_params: &HashSet<(&String, &String)>,
    ) -> (String, Option<(String, Type)>)
    where
        I: Iterator<Item = (&'a String, &'a Type)>,
    {
        let (p0, mut previous_c_type) = parts
            .next()
            .expect("Supply at least one part-type pair to get_var_expr_as_c.");
        let mut previous_c_expr = p0.clone();
        let mut owner = None;

        let mut peekable = parts.peekable();

        while let Some((id, id_type)) = peekable.next() {
            // non-named types are only allowed at the end. validate_ast should catch this.
            if previous_c_type.name().is_none() && peekable.peek().is_some() {
                panic!("non-named type not at the end of var expression.");
            }

            // Store the penultimate value in owner
            if peekable.peek().is_none() {
                owner = Some((previous_c_expr.clone(), previous_c_type.clone()));
            }

            let prev_struct_name: &String = previous_c_type.name().unwrap();
            let is_racing = racing_params.contains(&(prev_struct_name, id));

            let load_op = if !is_racing && self.ld_st_weakly_for_non_racing {
                format!("WLOAD({}, ", as_c_type(id_type))
            } else {
                "LOAD(".to_string()
            };

            previous_c_expr = format!(
                "{load_op}{}->{id}[{previous_c_expr}])",
                prev_struct_name.to_lowercase()
            );
            previous_c_type = id_type;
        }

        (previous_c_expr, owner)
    }

    pub fn functions(&self) -> Vec<String> {
        if self.ld_st_weakly_for_non_racing {
            vec![self.set_param_function(), self.weak_set_param_function()]
        } else {
            vec![self.set_param_function()]
        }
    }

    fn set_param_function(&self) -> String {
        if self.print_unstable {
            formatdoc!("
				template<typename T>
				__device__ void SetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable, const char* par_str) {{
					if (owner != 0){{
						T old_val = LOAD(params[owner]);
						if (old_val != new_val){{
							STORE(params[owner], new_val);
							*stable = false;
							printf(\"%s was set (%u).\\n\", par_str, owner);
						}}
					}}
				}}
			")
        } else {
            formatdoc!("
				template<typename T>
				__device__ void SetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {{
					if (owner != 0){{
						T old_val = LOAD(params[owner]);
						if (old_val != new_val){{
							STORE(params[owner], new_val);
							*stable = false;
						}}
					}}
				}}
			")
        }
    }

    fn weak_set_param_function(&self) -> String {
        if self.print_unstable {
            formatdoc!("
                template<typename T>
                __device__ void WSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable, const char* par_str) {{
                    if (owner != 0){{
                        T old_val = WLOAD(T, params[owner]);
                        if (old_val != new_val){{
                            WSTORE(T, params[owner], new_val);
                            *stable = false;
                            printf(\"%s was set (%u).\\n\", par_str, owner);
                        }}
                    }}
                }}
            ")
        } else {
            formatdoc!("
                template<typename T>
                __device__ void WSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {{
                    if (owner != 0){{
                        T old_val = WLOAD(T, params[owner]);
                        if (old_val != new_val){{
                            WSTORE(T, params[owner], new_val);
                            *stable = false;
                        }}
                    }}
                }}
            ")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::*;
    use crate::StepBodyCompiler;
    use indoc::formatdoc;

    #[test]
    fn test_local_variable_as_parameter() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1) {
                step1 {
                    S1 p2_alias := p2;
                    p2 := p2_alias;
                    p2_alias.p1 := 12;
                }
            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, false);

        let expected = formatdoc!(
            "

            \tRefType p2_alias = LOAD(s1->p2[self]);
            \t// p2 := p2_alias;
            \tSetParam<RefType>(self, s1->p2, p2_alias, stable);
            \t// p2_alias.p1 := 12;
            \tSetParam<NatType>(p2_alias, s1->p1, 12, stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }

    #[test]
    fn test_local_variable_as_parameter_with_weak_loads_and_stores_1() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1) {
                step1 {
                    S1 p2_alias := p2;
                    p2 := p2_alias;
                    p2_alias.p1 := 12;
                }
            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let racing_params = step1.racing_parameters(&program, &s1, &type_info);
        assert!(racing_params.contains(&(&"S1".to_string(), &"p1".to_string())));
        assert!(racing_params.len() == 1);

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true);

        let expected = formatdoc!(
            "

            \tRefType p2_alias = WLOAD(RefType, s1->p2[self]);
            \t// p2 := p2_alias;
            \tWSetParam<RefType>(self, s1->p2, p2_alias, stable);
            \t// p2_alias.p1 := 12;
            \tSetParam<NatType>(p2_alias, s1->p1, 12, stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }

    #[test]
    fn test_local_variable_as_parameter_with_weak_loads_and_stores_2() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1) {
                step1 {
                    S1 p2_alias := p2;
                    p2_alias.p1 := 12;
                }
            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true);

        let expected = formatdoc!(
            "

            \tRefType p2_alias = WLOAD(RefType, s1->p2[self]);
            \t// p2_alias.p1 := 12;
            \tSetParam<NatType>(p2_alias, s1->p1, 12, stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }

    #[test]
    fn test_weak_stores_1() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1) {
                step1 {
                    p2 := this;
                }
            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true);

        let expected = formatdoc!(
            "

            \t// p2 := this;
            \tWSetParam<RefType>(self, s1->p2, self, stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }

    #[test]
    fn test_weak_stores_with_constructor() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1) {
                step1 {
                    S1 new_s1 := S1(0, null);

                    p2.p1 := 1;
                }
            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true);

        let expected = formatdoc!(
            "
            
            \tRefType new_s1 = s1->create_instance(0, 0, stable);
            \t// p2.p1 := 1;
            \tSetParam<NatType>(WLOAD(RefType, s1->p2[self]), s1->p1, 1, stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }
}
