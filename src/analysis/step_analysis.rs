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

    let mut result = HashSet::new();
    visit_step::<
        HashSet<(&String, &String)>,
        (),
        &'a HashMap<*const Exp, Vec<Type>>
    >(
        step,
        &mut result,
        |_, _, _| {},
        &None,
        |exp, set, info| {
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
        },
        &Some(var_exp_type_info),
    );
    return result;
}

struct NewLabelInfo<'a> {
    pub strct: &'a ADLStruct,
    pub type_info: &'a HashMap<*const Exp, Vec<Type>>,
    pub nl_params : HashSet<(&'a String, &'a String)>,
    pub nl_assignments : HashSet<*const Stat>
}

type VarMap<'a> = HashMap<&'a String, bool>;

pub fn get_new_label_params<'a>(
    strct: &'a ADLStruct,
    type_info: &'a HashMap<*const Exp, Vec<Type>>,
    step: &'a Step
) -> (HashSet<(&'a String, &'a String)>, HashSet<*const Stat>) {

    let mut nl_info = NewLabelInfo {
        strct,
        type_info,
        nl_params: HashSet::new(),
        nl_assignments: HashSet::new()
    };

    let mut var_env : VarMap = HashMap::new();
    for s in &step.statements {
        get_nl_stat(&s, &mut var_env, &mut nl_info);
    }

    return (nl_info.nl_params, nl_info.nl_assignments);
}

fn get_nl_stat<'a>(stat: &'a Stat, var_env : &mut VarMap<'a>, nl_info : &mut NewLabelInfo<'a>){
    use Stat::*;
    match stat {
        IfThen(..) => get_nl_ifthen(stat, var_env, nl_info),
        Declaration(..) => get_nl_declaration(stat, var_env, nl_info),
        Assignment(..) => get_nl_assignment(stat, var_env, nl_info),
        InlineCpp(..) => {}
    }
}

fn get_nl_ifthen<'a>(ifthen: &'a Stat, var_env : &mut VarMap<'a>, nl_info : &mut NewLabelInfo<'a>){
    if let Stat::IfThen(_, s_true, s_false, _) = ifthen {

        let mut var_env_true = var_env.clone();
        for s in s_true {
            get_nl_stat(&s, &mut var_env_true, nl_info);
        }

        let mut var_env_false = var_env.clone();
        for s in s_false {
            get_nl_stat(&s, &mut var_env_false, nl_info);
        }
        merge_var_envs(var_env, &var_env_true, &var_env_false);

    } else {
        unreachable!();
    }
}

fn get_nl_declaration<'a>(decl: &'a Stat, var_env : &mut VarMap<'a>, nl_info : &mut NewLabelInfo<'a>){
    if let Stat::Declaration(_, lhs, rhs, _) = decl {
        let rhs_is_new_label = rhs.is_local_var(nl_info.strct)
                                &&
                               *var_env.get(&rhs.get_parts()[0]).unwrap_or(&false);

        let rhs_is_constructed = matches!(**rhs, Exp::Constructor(..));

        // lhs is a nl_var if rhs is a nl_var or rhs is a constructor
        var_env.insert(lhs, rhs_is_new_label || rhs_is_constructed);
    } else {
        unreachable!();
    }
}

fn get_nl_assignment<'a>(assignment: &'a Stat, var_env : &mut VarMap<'a>, nl_info : &mut NewLabelInfo<'a>){
    if let Stat::Assignment(lhs, rhs, _) = assignment {
        let rhs_is_new_label = rhs.is_local_var(nl_info.strct)
                                &&
                               *var_env.get(&rhs.get_parts()[0]).unwrap_or(&false);
        let rhs_is_constructed = matches!(**rhs, Exp::Constructor(..));
        let lhs_is_loc_var = lhs.is_local_var(nl_info.strct);
        let lhs_local_variable = if lhs_is_loc_var {
            Some(&lhs.get_parts()[0])
        } else {
            None
        };
        let lhs_param = if !lhs_is_loc_var {
            let types = nl_info.type_info
                .get(&(&**lhs as *const Exp))
                .expect("Could not find var expression in type info.");
            let parts = lhs.get_parts();
            let par_name = parts.last().unwrap();

            let strct_name = if parts.len() == 1 {
                &nl_info.strct.name
            } else {
                types[types.len() - 2].name().unwrap()
            };

            Some((
                strct_name,
                par_name,
            ))
        } else {
            None
        };

        // If assigning to a local variable
        if let Some(lhs_var) = lhs_local_variable {
            // lhs is a nl_var if rhs is a nl_var or rhs is a constructor
            var_env.insert(lhs_var, rhs_is_new_label || rhs_is_constructed);
        }

        // If assigning to a parameter
        else if let Some(lhs_par) = lhs_param {
            if rhs_is_new_label || rhs_is_constructed {
                nl_info.nl_assignments.insert(assignment as *const Stat);
                nl_info.nl_params.insert(lhs_par);
            }
        } else {
            unreachable!();
        }
    } else {
        unreachable!();
    }
}

fn merge_var_envs<'a>(var_env : &mut VarMap<'a>, a : &VarMap<'a>, b : &VarMap<'a>){
    let all_keys : HashSet<&'a String> = a.keys()
                                           .cloned()
                                           .collect::<HashSet<&String>>()
                                           .union(&b.keys().cloned().collect())
                                           .cloned()
                                           .collect();
    for k in all_keys {
        let nl_in_a = *a.get(k).unwrap_or(&false);
        let nl_in_b = *a.get(k).unwrap_or(&false);
        let nl_in_old = *var_env.get(k).unwrap_or(&false);

        // A variable holds a new_label if it does in at least one of the branches
        var_env.insert(k, nl_in_a || nl_in_b || nl_in_old);
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::get_new_label_params;
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

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());
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

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());
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

    #[test]
    fn test_new_label_pars() {
        let program = ProgramParser::new()
            .parse(
                "struct Node(n: Node, o : other) {
                step1 {
                    Node new := Node(null, null);
                    Node new2 := new;
                    Node new3 := new2;
                    n := new3;
                }

                step2 {
                    Node new := Node(null, null);
                    Node new2 := new;
                    Node new3 := new2;
                    n := this;
                }

                step3 {
                    Node new2 := null;
                    Node new3 := null;
                    Node new := Node(null, null);
                    new2 := new;
                    new3 := new2;
                    n := new3;
                }

                step4 {
                    Node new2 := null;
                    Node new3 := null;
                    Node new := Node(null, null);
                    new2 := new;
                    new3 := new2;
                    new3 := this;
                    n := new3;
                    o.x := new2;
                }

                step5 {
                    other new_other := other(Node(null, null));
                    o := new_other;
                    o := null;
                    o := new_other;
                }

                step6 {
                    o.x.n.o := other(null);
                    o := null;
                }
            }

            struct other(x : Node) {

            }

            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());
        let strct = program.struct_by_name(&"Node".to_string()).unwrap();
        let step1 = strct.step_by_name(&"step1".to_string()).unwrap();
        let step2 = strct.step_by_name(&"step2".to_string()).unwrap();
        let step3 = strct.step_by_name(&"step3".to_string()).unwrap();
        let step4 = strct.step_by_name(&"step4".to_string()).unwrap();
        let step5 = strct.step_by_name(&"step5".to_string()).unwrap();
        let step6 = strct.step_by_name(&"step6".to_string()).unwrap();

        let (np, np_stores) = get_new_label_params(strct, &type_info, step1);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 1);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step2);
        assert_eq!(np, HashSet::new());
        assert_eq!(np_stores.len(), 0);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step3);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 1);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step4);
        assert_eq!(np, HashSet::from_iter([(&"other".to_string(), &"x".to_string())]));
        assert_eq!(np_stores.len(), 1);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step5);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"o".to_string())]));
        assert_eq!(np_stores.len(), 2);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step6);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"o".to_string())]));
        assert_eq!(np_stores.len(), 1);
    }

    #[test]
    fn test_new_label_pars_scopes() {
        let program = ProgramParser::new()
            .parse(
                "struct Node(n: Node, o : other) {
                step1 {
                    Node new := null;
                    if n != null then {
                        new := Node(null, null);
                    }
                    n := new;
                }
                step2 {
                    Node new := Node(null, null);
                    if n != null then {
                        new := null;
                    }
                    n := new;
                }
                step3 {
                    Node new := Node(null, null);
                    if n != null then {
                        new := null;
                    }
                    if n != null then {
                        n := new;
                    }
                    n := new;
                }
                step4 {
                    Node new := Node(null, null);
                    if n != null then {
                        new := null;
                    }
                    if n != null then {
                        new := null;
                        n := new;
                        o.x.o := other(null);
                    }
                    n := new;
                }
                step5 {
                    Node new := null;
                    if n != null then {
                        new := Node(null, null);
                        n := new;
                    }
                    n := new;
                }
            }
            struct other(x : Node) {}
            step1",
            )
            .unwrap();

        let (errors, type_info) = validate_ast(&program);
        assert!(errors.is_empty());
        let strct = program.struct_by_name(&"Node".to_string()).unwrap();
        let step1 = strct.step_by_name(&"step1".to_string()).unwrap();
        let step2 = strct.step_by_name(&"step2".to_string()).unwrap();
        let step3 = strct.step_by_name(&"step3".to_string()).unwrap();
        let step4 = strct.step_by_name(&"step4".to_string()).unwrap();
        let step5 = strct.step_by_name(&"step5".to_string()).unwrap();

        let (np, np_stores) = get_new_label_params(strct, &type_info, step1);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 1);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step2);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 1);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step3);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 2);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step4);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string()), (&"Node".to_string(), &"o".to_string())]));
        assert_eq!(np_stores.len(), 2);

        let (np, np_stores) = get_new_label_params(strct, &type_info, step5);
        assert_eq!(np, HashSet::from_iter([(&"Node".to_string(), &"n".to_string())]));
        assert_eq!(np_stores.len(), 2);
    }
}