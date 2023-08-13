use crate::analysis::step_calls;
use std::collections::HashMap;
use crate::frontend::ast::*;
use std::collections::HashSet;

pub fn executors(program: &Program) -> HashSet<&String> {
    let s2s = get_step_to_structs(program);
    let mut calls = Vec::new();
    step_calls(&program.schedule, &mut calls);

    let mut result = HashSet::new();
    for c in calls {
        match c {
            (None, step) => s2s.get(step).unwrap().iter().for_each(|strct| {
                result.insert(&strct.name);
            }),
            (Some(strct), _) => {
                result.insert(strct);
            }
        }
    }
    result
}

pub fn get_step_to_structs(program: &Program) -> HashMap<&String, Vec<&ADLStruct>> {
    let mut s2s: HashMap<&String, Vec<&ADLStruct>> = HashMap::new();

    for strct in &program.structs {
        for step in &strct.steps {
            s2s.entry(&step.name)
                .and_modify(|v| v.push(strct))
                .or_insert(vec![strct]);
        }
    }
    return s2s;
}