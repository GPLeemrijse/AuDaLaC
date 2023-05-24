use crate::parser::ast::{Exp, Program, Stat, Step, Type};
use indoc::formatdoc;
use std::collections::HashMap;
use std::io::Write;

pub fn generate_init_file(ast: &Program, mut writer: Box<dyn Write>) {
    let nrof_structs = ast.structs.len();
    let mut struct_decls_vec: Vec<String> = ast
        .structs
        .iter()
        .map(|s| {
            format!(
                "{} {}",
                s.name,
                s.parameters
                    .iter()
                    .map(|p| p.1.to_string())
                    .reduce(|acc: String, nxt| acc + " " + &nxt)
                    .unwrap()
            )
            .to_string()
        })
        .collect();

    struct_decls_vec.sort();

    let struct_decls_str: String = struct_decls_vec
        .into_iter()
        .reduce(|acc: String, nxt| acc + "\n" + &nxt)
        .unwrap();

    writer
        .write(
            formatdoc!(
                "ADL structures {nrof_structs}
		{struct_decls_str}
		"
            )
            .as_bytes(),
        )
        .expect("Could not write to writer!");

    let structs = &ast.structs;
    let steps: Vec<&Step> = structs
        .iter()
        .map(|s| &s.steps)
        .flatten()
        .filter(|s| s.name == "init")
        .collect();

    if steps.len() != 1 {
        panic!("To create an init file, exactly 1 struct should implement an 'init' step.");
    }

    let statements = &steps[0].statements;

    // instance name -> (paramater values, instance number, struct name)
    let mut insts: HashMap<&String, (Vec<i64>, usize, &String)> =
        HashMap::with_capacity(statements.len());
    //							  name ->  (nrof_instances, struct_nr)
    let mut struct_info: HashMap<&String, (usize, usize)> = HashMap::with_capacity(structs.len());

    // Set instance count to 1 for each structure (accomodating for null)
    for (s_idx, s) in structs.iter().enumerate() {
        struct_info.insert(&s.name, (1, s_idx));
    }

    // Do it in two passes:
    // - Read all declarations
    // - Fill all parameters

    for stmt in statements {
        if let Stat::Declaration(Type::Named(t1), id, exp, _) = stmt {
            if let Exp::Constructor(t2, param_exprs, constr_loc) = &**exp {
                if t1 != t2 {
                    panic!("Invalid type @{:?}.", constr_loc)
                }
                let (inst_count, _) = struct_info.get_mut(t1).unwrap();

                // See if id is already in map
                if !insts.contains_key(id) {
                    // If not, insert blank instance
                    insts.insert(id, (Vec::with_capacity(param_exprs.len()), *inst_count, t1));
                    *inst_count += 1;
                } else {
                    panic!("Double usage of id {}.", id);
                }
            } else {
                panic!("Only constructor expressions are allowed.");
            }
        } else {
            panic!("Init files can only contain declarations of named types.");
        }
    }

    for stmt in statements {
        if let Stat::Declaration(_, id, exp, _) = stmt {
            if let Exp::Constructor(_, param_exprs, _) = &**exp {
                for p in param_exprs {
                    let p_value = match p {
                        // If the expression is a literal we return the corresponding value
                        Exp::Lit(lit, _) => {
                            use crate::parser::ast::Literal::*;
                            match lit {
                                NatLit(nat) => *nat as i64,
                                IntLit(int) => *int as i64,
                                BoolLit(b) => {
                                    if *b {
                                        1
                                    } else {
                                        0
                                    }
                                }
                                NullLit => 0,
                                _ => {
                                    panic!("Type of literal not supported.");
                                }
                            }
                        }
                        // If the expression is a variable, we return the instance idx of that variable
                        Exp::Var(ids, _) => {
                            if ids.len() != 1 {
                                panic!("Variable should have exactly 1 part.");
                            }

                            insts.get(&ids[0]).unwrap().1 as i64 // .1 == idx
                        }
                        _ => panic!("Constructor parameters can only be literals or references."),
                    };
                    let (param_values, _, _) = insts.get_mut(id).unwrap();
                    param_values.push(p_value);
                }
            }
        }
    }

    let mut instances: Vec<(Vec<i64>, usize, &String)> = insts.into_values().collect();

    // Add null-instances
    for s in structs {
        instances.push((vec![0; s.parameters.len()], 0, &s.name));
    }
    instances.sort_by(|a, b| (a.2, a.1).cmp(&(b.2, b.1)));

    for idx in 0..instances.len() {
        if idx == 0 || instances[idx - 1].2 != instances[idx].2 {
            writer
                .write(
                    format! {
                        "{} instances {}\n",
                        instances[idx].2,
                        struct_info.get(&instances[idx].2).unwrap().0
                    }
                    .as_bytes(),
                )
                .expect("Failed to write.");
        }
        writer
            .write(
                format!(
                    "{}\n",
                    instances[idx]
                        .0
                        .iter()
                        .map(|int| int.to_string())
                        .reduce(|acc: String, nxt| acc + " " + &nxt)
                        .unwrap()
                )
                .to_string()
                .as_bytes(),
            )
            .expect("Failed to write.");
    }

    writer.flush().expect("Could not flush writer!");
}
