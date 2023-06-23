use crate::analysis::*;
use std::collections::{HashSet, HashMap};
use crate::parser::ast::*;

pub fn racing_parameters<'a>(
    step: &'a Step,
    program: &'a Program,
    owner_strct: &'a ADLStruct,
    var_exp_type_info: &'a HashMap<*const Exp, Vec<Type>>,
) -> HashSet<(&'a String, &'a String)> {
    let written_params = written_parameters(step, program, owner_strct, var_exp_type_info);
    let ext_ref_params = externally_referenced_parameters(step, var_exp_type_info);

    written_params
        .intersection(&ext_ref_params)
        .map(|n| *n)
        .collect()
}

// Returns a unique vector of possibly constructed struct types
pub fn constructors<'a>(step: &'a Step) -> HashSet<&'a String> {
    use Exp::*;
    let mut result = HashSet::new();

    visit_step::<HashSet<&String>, (), ()>(
        step,
        &mut result,
        |_, _, _| {},
        &None,
        |e, set, _| {
            if let Constructor(s, _, _) = e {
                set.insert(s);
            }
        },
        &None,
    );
    result
}

// Returns all declared variable names and their type
pub fn declarations<'a>(step: &'a Step) -> HashSet<(&'a String, &'a Type)> {
    use Stat::*;
    let mut result = HashSet::new();
    visit_step::<HashSet<(&String, &Type)>, (), ()>(
        step,
        &mut result,
        |stat, set, _| {
            if let Declaration(t, s, _, _) = stat {
                set.insert((&s, &t));
            }
        },
        &None,
        |_, _, _| {},
        &None,
    );
    result
}


fn written_parameters<'a>(
    step: &'a Step,
    program: &'a Program,
    owner_strct: &'a ADLStruct,
    var_exp_type_info: &'a HashMap<*const Exp, Vec<Type>>,
) -> HashSet<(&'a String, &'a String)> {
    use Exp::*;
    use Stat::*;

    let mut result = HashSet::new();

    visit_step::<
        HashSet<(&String, &String)>,
        (&'a ADLStruct, &'a HashMap<*const Exp, Vec<Type>>),
        &'a Program
    >(
        step,
        &mut result,
        |stat, set, args| {
            let (strct, info) = args.unwrap();
            if let Assignment(lhs, _, _) = stat {
                if let Var(parts, _) = &**lhs {
                    assert!(!parts.is_empty());
                    let is_param_owned_by_self = strct.parameter_by_name(&parts[0]).is_some();

                    if parts.len() == 1 && is_param_owned_by_self {
                        set.insert((&strct.name, parts.last().unwrap()));
                    } else if parts.len() > 1 {
                        let types = info.get(
                                &(&**lhs as *const Exp)
                            )
                            .expect("Could not find var expression in type info.");

                        set.insert((types[types.len()-2].name().unwrap(), parts.last().unwrap()));
                    }
                } else {
                    unreachable!()
                }
            }
        },
        &Some((owner_strct, var_exp_type_info)),
        |exp, set, args| {
            let program = args.unwrap();
            if let Constructor(cons_type, _, _) = exp {
                for (p_name, _, _) in &program.struct_by_name(cons_type).unwrap().parameters {
                    set.insert((cons_type, &p_name));
                }
            }
        },
        &Some(program)
    );
    return result;
}

fn externally_referenced_parameters<'a>(
    step: &'a Step,
    var_exp_type_info: &'a HashMap<*const Exp, Vec<Type>>,
) -> HashSet<(&'a String, &'a String)> {
    use Stat::*;

    let mut result = HashSet::new();
    visit_step::<
        HashSet<(&String, &String)>,
        &'a HashMap<*const Exp, Vec<Type>>,
        &'a HashMap<*const Exp, Vec<Type>>
    >(
        step,
        &mut result,
        |stat, set, args| {
            if let Assignment(lhs, _, _) = stat {
                get_ext_refs_from_exp(lhs, set, args);
            }
        },
        &Some(var_exp_type_info),
        get_ext_refs_from_exp,
        &Some(var_exp_type_info),
    );
    return result;
}

fn get_ext_refs_from_exp<'a>(
    exp: &'a Exp,
    set: &mut HashSet<(&'a String, &'a String)>,
    info: &Option<&'a HashMap<*const Exp, Vec<Type>>>,
) {
    if let Exp::Var(parts, _) = exp {
        /* If of the form a.b or a.b.c etc. then (a, b)
        and (b, c) are written via a reference.*/
        if parts.len() > 1 {
            let types = info
                .unwrap()
                .get(&(&*exp as *const Exp))
                .expect("Could not find var expression in type info.");

            set.insert((
                types[types.len() - 2].name().unwrap(),
                parts.last().unwrap(),
            ));
        }
    }
}

// fn get_vars_with_new_label<'a>(strct: &'a ADLStruct, step: &'a Step) -> HashSet<&'a String> {
//     use Stat::*;
//     use Exp::*;

//     fixpoint_visit_step::<&String, &ADLStruct>(
//         step,
//         |stat, set, strct_opt| {
//             match stat {
//                 Declaration(_, lhs, rhs, _) => {
//                     // If lhs is declared with the value of a new label, lhs is also a new label variable 
//                     if rhs.is_local_var(strct_opt.unwrap()) {
//                         if set.contains(&rhs.get_parts()[0]) {
//                             set.insert(lhs);
//                         }
//                     }
//                 },
//                 Assignment(lhs, rhs, _) => {
//                     if lhs.is_local_var(strct_opt.unwrap()) && rhs.is_local_var(strct_opt.unwrap()) {
//                         if set.contains(&rhs.get_parts()[0]) {
//                             set.insert(&rhs.get_parts()[0]);
//                         }
//                     }
//                 },
//                 _ => {}
//             }
//         },
//         &Some(strct),
//         |_, _, _| (),
//         &Some(strct)
//     )
// }


#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::analysis::step_analysis::written_parameters;
    use crate::analysis::racing_parameters;
    use crate::parser::*;

    #[test]
    fn test_racing_parameters() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1, p3 : S2) {
                step1 {
                    S1 new_s1 := S1(0, null, null);
                }

                step3 {
                    S1 new_s1 := S1(0, null, null);
                    new_s1.p1 := 0;
                }

                step4 {
                    S1 p2_alias := p2;
                    p2 := p2_alias;
                    p2_alias.p1 := 12;
                }
            }

            struct S2 (p1 : S1) {
                step2 {
                    p1.p3.p1 := p1;
                }
            }

            step1 < step2",
            )
            .unwrap();

        let (_, type_info) = validate_ast(&program);
        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let s2 = program.struct_by_name(&"S2".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();
        let step2 = s2.step_by_name(&"step2".to_string()).unwrap();
        let step3 = s1.step_by_name(&"step3".to_string()).unwrap();
        let step4 = s1.step_by_name(&"step4".to_string()).unwrap();

        let rp = racing_parameters(step1, &program, &s1, &type_info);
        assert!(rp.is_empty());

        let rp = racing_parameters(step2, &program, &s2, &type_info);
        assert!(rp.contains(&(&"S2".to_string(), &"p1".to_string())));
        assert!(rp.len() == 1);

        let rp = racing_parameters(step3, &program, &s1, &type_info);
        assert!(rp.contains(&(&"S1".to_string(), &"p1".to_string())));
        assert!(rp.len() == 1);

        let rp = racing_parameters(step4, &program, &s1, &type_info);
        assert!(rp.contains(&(&"S1".to_string(), &"p1".to_string())));
        assert!(rp.len() == 1);
    }

    #[test]
    fn test_written_parameters() {
        let program = ProgramParser::new()
            .parse(
                "struct S1(p1 : Nat, p2: S1, p3 : S2) {
                step1 {
                    p1 := 123;
                }

                step3 {
                    p2 := this;
                }
            }
            struct S2 (p1 : S1) {
                step2 {
                    p1.p2.p3 := this;
                    p1.p1 := 456;
                }
            }

            step1 < step2",
            )
            .unwrap();

        let (_, type_info) = validate_ast(&program);
        let s1 = program.struct_by_name(&"S1".to_string()).unwrap();
        let s2 = program.struct_by_name(&"S2".to_string()).unwrap();
        let step1 = s1.step_by_name(&"step1".to_string()).unwrap();
        let step2 = s2.step_by_name(&"step2".to_string()).unwrap();
        let step3 = s1.step_by_name(&"step3".to_string()).unwrap();

        assert_eq!(
            written_parameters(step1, &program, &s1, &type_info),
            HashSet::from([(&"S1".to_string(), &"p1".to_string())])
        );

        assert_eq!(
            written_parameters(step2, &program, &s2, &type_info),
            HashSet::from([
                (&"S1".to_string(), &"p1".to_string()),
                (&"S1".to_string(), &"p3".to_string())
            ])
        );

        assert_eq!(
            written_parameters(step3, &program, &s1, &type_info),
            HashSet::from([(&"S1".to_string(), &"p2".to_string())])
        );
    }
}