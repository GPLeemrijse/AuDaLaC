use crate::analysis::get_new_label_params;
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
    memorder: &'a MemOrder
}


struct StepCompilationInfo<'a> {
    pub owner: &'a ADLStruct,
    pub racing_params: &'a HashSet<(&'a String, &'a String)>,
    pub new_label_params: &'a HashSet<(&'a String, &'a String)>,
    pub new_label_param_assignments: &'a HashSet<*const Stat>
}

impl StepBodyCompiler<'_> {
    pub fn new<'a>(
        type_info: &'a HashMap<*const Exp, Vec<Type>>,
        use_step_parity: bool,
        print_unstable: bool,
        ld_st_weakly_for_non_racing: bool,
        memorder: &'a MemOrder
    ) -> StepBodyCompiler<'a> {
        StepBodyCompiler {
            var_exp_type_info: type_info,
            use_step_parity,
            print_unstable,
            ld_st_weakly_for_non_racing,
            memorder
        }
    }

    pub fn step_body_as_c(&self, program: &Program, owner: &ADLStruct, step: &Step) -> String {
        let racing_params = racing_parameters(step, program, owner, self.var_exp_type_info);
        let (new_label_params, new_label_param_assignments) = get_new_label_params(owner, self.var_exp_type_info, step);

        let info = StepCompilationInfo {
            owner: owner,
            racing_params: &racing_params,
            new_label_params: &new_label_params,
            new_label_param_assignments: &new_label_param_assignments
        };

        
        self.statements_as_c(
            &step.statements,
            1,
            &info
        )
    }

    fn statements_as_c(
        &self,
        statements: &Vec<Stat>,
        indent_lvl: usize,
        info: &StepCompilationInfo
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);

        statements
            .iter()
            .map(|stmt| match stmt {
                Stat::IfThen(..) => self.if_stmt_as_c(stmt, indent_lvl, info),
                Stat::Declaration(..) => {
                    self.decl_as_c(stmt, indent_lvl, info)
                }
                Stat::Assignment(..) => {
                    self.assignment_as_c(stmt, indent_lvl, info)
                }
                Stat::InlineCpp(cpp) => format!("{indent}{cpp}"),
            })
            .fold("".to_string(), |acc: String, nxt| acc + "\n" + &nxt)
    }

    fn if_stmt_as_c(
        &self,
        stmt: &Stat,
        indent_lvl: usize,
        info: &StepCompilationInfo
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::IfThen(e, stmts_true, stmts_false, _) = stmt {
            let cond = self.expression_as_c(e, info);
            let body_true =
                self.statements_as_c(stmts_true, indent_lvl + 1, info);
            let body_false =
                self.statements_as_c(stmts_false, indent_lvl + 1, info);

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
        indent_lvl: usize,
        info: &StepCompilationInfo
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::Declaration(t, n, e, _) = stmt {
            let t_as_c = as_c_type(t);
            let e_as_c = self.expression_as_c(e, info);
            format!("{indent}{t_as_c} {n} = {e_as_c};")
        } else {
            unreachable!()
        }
    }

    fn assignment_as_c(
        &self,
        stmt: &Stat,
        indent_lvl: usize,
        info: &StepCompilationInfo
    ) -> String {
        let indent = String::from("\t").repeat(indent_lvl);
        if let Stat::Assignment(lhs_exp, rhs_exp, _) = stmt {
            let (lhs_as_c, owner, is_parameter) = self.var_exp_as_c(lhs_exp, info);
            let rhs_as_c = self.expression_as_c(rhs_exp, info);

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

                let is_storing_fresh_label = info.new_label_param_assignments.contains(&(stmt as *const Stat));
                let is_racing = info.racing_params.contains(&(owner_type_name, param));

                let store_op = if !is_racing && self.ld_st_weakly_for_non_racing {
                    "WeakSetParam"
                } else {
                    if is_storing_fresh_label && self.memorder == &MemOrder::Relaxed {
                        "RelSetParam"
                    } else {
                        "SetParam"
                    }
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
        info: &StepCompilationInfo
    ) -> String {
        use Exp::*;

        match e {
            BinOp(e1, c, e2, _) => {
                let e1_comp = self.expression_as_c(e1, info);
                let e2_comp = self.expression_as_c(e2, info);

                format!("({e1_comp} {c} {e2_comp})")
            }
            UnOp(c, e, _) => {
                let e_comp = self.expression_as_c(e, info);
                format!("({c}{e_comp})")
            }
            Constructor(n, args, _) => {
                let mut arg_expressions = args
                    .iter()
                    .map(|e| self.expression_as_c(e, info))
                    .collect::<Vec<String>>();

                if self.use_step_parity {
                    arg_expressions.push("stable".to_string());
                }

                let args = arg_expressions.join(", ");

                format!("{}->create_instance({args})", n.to_lowercase())
            }
            Var(..) => {
                self.var_exp_as_c(e, info).0 // Not interested in the owner or is_parameter
            }
            Lit(l, _) => as_c_literal(l),
        }
    }

    fn var_exp_as_c(
        &self,
        exp: &Exp,
        info: &StepCompilationInfo
    ) -> (String, Option<(String, Type)>, bool) {
        use std::iter::zip;
        if let Exp::Var(parts, _) = exp {
            let types = self
                .var_exp_type_info
                .get(&(exp as *const Exp))
                .expect("Could not find var expression in type info.");
            let is_param_owned_by_self = info.owner.parameter_by_name(&parts[0]).is_some();
            let is_param_owned_by_local_var = !is_param_owned_by_self && parts.len() > 1;

            let (exp_as_c, owner);

            // For parameters, the 'self' part is implicit.
            if is_param_owned_by_self {
                (exp_as_c, owner) = self.stitch_parts_and_types(
                    zip(
                        ["self".to_string()].iter().chain(parts.iter()),
                        [Type::Named(info.owner.name.clone())].iter().chain(types.iter()),
                    ),
                    info
                );
            } else {
                (exp_as_c, owner) =
                    self.stitch_parts_and_types(zip(parts.iter(), types.iter()), info);
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
        info: &StepCompilationInfo
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
            let is_racing = info.racing_params.contains(&(prev_struct_name, id));
            let potentially_fresh_label = info.new_label_params.contains(&(prev_struct_name, id));
            
            let load_op = if !is_racing && self.ld_st_weakly_for_non_racing {
                format!("WLOAD({}, ", as_c_type(id_type))
            } else {
                if potentially_fresh_label && self.memorder == &MemOrder::Relaxed {
                    "ACQLOAD(".to_string()
                } else {
                    "LOAD(".to_string()
                }
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
        let mut result = Vec::new();
        result.push(self.set_param_function());

        if self.ld_st_weakly_for_non_racing {
            result.push(self.weak_set_param_function());
        }

        if self.memorder == &MemOrder::Relaxed {
            result.push(self.release_set_param_function());
        }
        return result;
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
                __device__ void WeakSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable, const char* par_str) {{
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
                __device__ void WeakSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {{
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

    fn release_set_param_function(&self) -> String {
        if self.print_unstable {
            formatdoc!("
                template<typename T>
                __device__ void RelSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable, const char* par_str) {{
                    if (owner != 0){{
                        T old_val = ACQLOAD(params[owner]);
                        if (old_val != new_val){{
                            RELSTORE(params[owner], new_val);
                            *stable = false;
                            printf(\"%s was set (%u).\\n\", par_str, owner);
                        }}
                    }}
                }}
            ")
        } else {
            formatdoc!("
                template<typename T>
                __device__ void RelSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {{
                    if (owner != 0){{
                        T old_val = ACQLOAD(params[owner]);
                        if (old_val != new_val){{
                            RELSTORE(params[owner], new_val);
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
    use crate::analysis::racing_parameters;
    use crate::parser::*;
    use crate::StepBodyCompiler;
    use indoc::formatdoc;
    use crate::MemOrder;

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

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, false, &MemOrder::Relaxed);

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

        let racing_params = racing_parameters(step1, &program, &s1, &type_info);
        assert!(racing_params.contains(&(&"S1".to_string(), &"p1".to_string())));
        assert!(racing_params.len() == 1);

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

        let expected = formatdoc!(
            "

            \tRefType p2_alias = WLOAD(RefType, s1->p2[self]);
            \t// p2 := p2_alias;
            \tWeakSetParam<RefType>(self, s1->p2, p2_alias, stable);
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

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

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

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

        let expected = formatdoc!(
            "

            \t// p2 := this;
            \tWeakSetParam<RefType>(self, s1->p2, self, stable);"
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

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

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

    #[test]
    fn test_acq_load_with_fresh_label() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(ref: S1) {
                step1 {
                    ref := S1(ref);
                }
            }
            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

        let expected = formatdoc!(
            "

            \t// ref := S1(ref);
            \tWeakSetParam<RefType>(self, s1->ref, s1->create_instance(WLOAD(RefType, s1->ref[self]), stable), stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }

    #[test]
    fn test_acq_load_with_fresh_label2() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(ref: S1, ref2: S1) {
                step1 {
                    S1 old_ref := ref2;
                    ref2 := S1(ref, null);
                }
            }
            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

        let expected = formatdoc!(
            "

            \tRefType old_ref = WLOAD(RefType, s1->ref2[self]);
            \t// ref2 := S1(ref, null);
            \tWeakSetParam<RefType>(self, s1->ref2, s1->create_instance(WLOAD(RefType, s1->ref[self]), 0, stable), stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }


    #[test]
    fn test_acq_load_with_fresh_label3() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(ref: S1, ref2: S1) {
                step1 {
                    S1 old_ref := ref;
                    ref2.ref2 := S1(ref.ref2, null);
                }
            }
            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());

        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();

        let step_compiler = StepBodyCompiler::new(&type_info, true, false, true, &MemOrder::Relaxed);

        let expected = formatdoc!(
            "

            \tRefType old_ref = WLOAD(RefType, s1->ref[self]);
            \t// ref2.ref2 := S1(ref.ref2, null);
            \tRelSetParam<RefType>(ACQLOAD(s1->ref2[self]), s1->ref2, s1->create_instance(ACQLOAD(s1->ref2[WLOAD(RefType, s1->ref[self])]), 0, stable), stable);"
        );

        let received = step_compiler.step_body_as_c(&program, &s1, &step1);

        if expected != received {
            eprintln!("Expected:\n---\n{expected}");
            eprintln!("\nReceived:\n---\n{received}");
        }
        assert_eq!(expected, received);
    }
}
