use crate::ast::*;
use crate::compiler::ScheduleManager;
use indoc::{formatdoc, indoc};
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct BasicScheduleManager<'a> {
    program: &'a Program,
    step_to_structs: HashMap<String, Vec<String>>,
}

impl ScheduleManager for BasicScheduleManager<'_> {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<stdio.h>");
    }

    fn defines(&self) -> String {
        format!(
            "#define FP_DEPTH {}\n",
            self.program.schedule.fixpoint_depth() + 1
        )
    }

    fn struct_typedef(&self) -> String {
        indoc! {"
			typedef struct FixpointManager {
			    bool stack[FP_DEPTH];
			    unsigned int current_level;
			} FixpointManager;
		"}
        .to_string()
    }

    fn globals(&self) -> String {
        indoc! {"
			__managed__ FixpointManager fixpoint_manager = {0};
		"}
        .to_string()
    }

    fn function_defs(&self) -> String {
        indoc! {r#"
			__device__ void clear_stability_stack(){
			    for(int i = 0; i < fixpoint_manager.current_level; i++){
			        fixpoint_manager.stack[i] = false;
			    }
			}

			__host__ void push_stability_stack(){
			    fixpoint_manager.current_level++;
			}

			__host__ void pop_stability_stack(){
			    fixpoint_manager.current_level--;
			}

			__host__ void reset_current_iteration_stability_stack(){
			    fixpoint_manager.stack[fixpoint_manager.current_level-1] = true;
			}

			__host__ void print_stability_stack(){
			    for(int i = 0; i < fixpoint_manager.current_level; i++){
			        printf("%s, ", fixpoint_manager.stack[i] ? "true" : "false");
				}
			    printf("\n");
			}

			__host__ bool fixpoint_reached_stability_stack(){
			    return fixpoint_manager.stack[fixpoint_manager.current_level-1] == true;
			}
		"#}
        .to_string()
    }

    fn run_schedule(&self) -> String {
        let mut result = String::new();
        self.print_schedule(&self.program.schedule, 1, &mut result);
        result
    }
}

impl BasicScheduleManager<'_> {
    pub fn new(program: &Program) -> BasicScheduleManager {
        BasicScheduleManager {
            program,
            step_to_structs: BasicScheduleManager::get_step_to_structs(program),
        }
    }

    fn print_schedule(&self, sched: &Schedule, indent_lvl: usize, res: &mut String) {
        use crate::ast::Schedule::*;
        let indent = " ".repeat(indent_lvl * 4);
        let tpb = 512;

        match sched {
            StepCall(s, _) => {
                let readies = self
                    .program
                    .structs
                    .iter()
                    .map(|s| format!("{indent}ready_struct_manager(&{}_manager);", s.name.clone()))
                    .reduce(|acc: String, nxt| acc + "\n" + &nxt)
                    .unwrap();

                for strct in self.step_to_structs.get(s).unwrap() {
                    res.push_str(&formatdoc! {"
						{readies}
						{indent}{strct}_{s}<<<({strct}_manager.nrof_active_structs + {tpb} - 1)/{tpb}, {tpb}>>>();
						{indent}cudaDeviceSynchronize();
					"});
                }
            }
            Sequential(s1, s2, _) => {
                self.print_schedule(s1, indent_lvl, res);
                self.print_schedule(s2, indent_lvl, res);
            }
            TypedStepCall(t, s, _) => {
                let readies = self
                    .program
                    .structs
                    .iter()
                    .map(|s| format!("{indent}ready_struct_manager(&{}_manager);", s.name.clone()))
                    .reduce(|acc: String, nxt| acc + "\n" + &nxt)
                    .unwrap();

                res.push_str(&formatdoc! {"
					{readies}
					{indent}{t}_{s}<<<({t}_manager.nrof_active_structs + {tpb} - 1)/{tpb}, {tpb}>>>();
					{indent}cudaDeviceSynchronize();
				"});
            }
            Fixpoint(s, _) => {
                let mut sched = String::new();
                self.print_schedule(s, indent_lvl + 1, &mut sched);

                res.push_str(&formatdoc! {"
					push_stability_stack();
					{indent}do{{
					{indent}{indent}reset_current_iteration_stability_stack();
					
					{sched}
					{indent}}}
					{indent}while(!fixpoint_reached_stability_stack());
					{indent}pop_stability_stack();
				"});
            }
        }
    }

    fn get_step_to_structs(program: &Program) -> HashMap<String, Vec<String>> {
        let mut s2s: HashMap<String, Vec<String>> = HashMap::new();

        for strct in &program.structs {
            let steps = strct.steps.iter().map(|s| s.name.clone());
            for step in steps {
                s2s.entry(step)
                    .and_modify(|v| v.push(strct.name.clone()))
                    .or_insert(vec![strct.name.clone()]);
            }
        }
        s2s
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::Schedule::*;
    use crate::ast::*;
    use crate::BasicScheduleManager;
    use std::collections::HashMap;

    #[test]
    fn test_get_step_to_structs() {
        let r = BasicScheduleManager::get_step_to_structs(&Program {
            inline_global_cpp: Vec::new(),
            structs: vec![
                ADLStruct {
                    name: "a".to_string(),
                    parameters: Vec::new(),
                    steps: vec![Step {
                        name: "s1".to_string(),
                        statements: Vec::new(),
                        loc: (0, 0),
                    }],
                    loc: (0, 0),
                },
                ADLStruct {
                    name: "b".to_string(),
                    parameters: Vec::new(),
                    steps: vec![
                        Step {
                            name: "s1".to_string(),
                            statements: Vec::new(),
                            loc: (0, 0),
                        },
                        Step {
                            name: "s2".to_string(),
                            statements: Vec::new(),
                            loc: (0, 0),
                        },
                    ],
                    loc: (0, 0),
                },
            ],
            schedule: Box::new(StepCall("sc".to_string(), (0, 0))),
        });
        let r2: HashMap<String, Vec<String>> = HashMap::from([
            ("s1".to_string(), vec!["a".to_string(), "b".to_string()]),
            ("s2".to_string(), vec!["b".to_string()]),
        ]);
        assert_eq!(r, r2);
    }
}
