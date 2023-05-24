use crate::compiler::compilation_traits::*;
use crate::compiler::components::WorkDivisor;
use crate::compiler::StepBodyCompiler;
use crate::parser::ast::*;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct OnHostSchedule<'a> {
    program: &'a Program,
    fp: &'a dyn FPStrategy,
    step_to_structs: HashMap<&'a String, Vec<&'a ADLStruct>>,
    step_transpiler: &'a StepBodyCompiler<'a>,
    work_divisor: &'a WorkDivisor<'a>,
}

impl OnHostSchedule<'_> {
    pub fn new<'a>(
        program: &'a Program,
        fp: &'a dyn FPStrategy,
        step_transpiler: &'a StepBodyCompiler<'_>,
        work_divisor: &'a WorkDivisor<'a>,
    ) -> OnHostSchedule<'a> {
        OnHostSchedule {
            program,
            fp,
            step_to_structs: program.get_step_to_structs(),
            step_transpiler: step_transpiler,
            work_divisor,
        }
    }

    fn step_call_function_as_c(&self, strct: &ADLStruct, step: &Step) -> String {
        let strct_name = &strct.name;
        let strct_name_lwr = strct.name.to_lowercase();
        let step_name = &step.name;
        let kernel_name = format!("{strct_name}_{step_name}");

        formatdoc!(
            "
			void launch_{kernel_name}(inst_size nrof_instances) {{
				inst_size nrof_threads = (nrof_instances + I_PER_THREAD - 1) / I_PER_THREAD;
				void* args[] = {{}};
				auto dims = ADL::get_launch_dims(nrof_threads, (void*){kernel_name});

				CHECK(
					cudaLaunchCooperativeKernel(
						(void*){kernel_name},
						std::get<0>(dims),
						std::get<1>(dims),
						args
					)
				);
			}}
		"
        )
    }
}

impl CompileComponent for OnHostSchedule<'_> {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<cooperative_groups.h>");
    }

    fn defines(&self) -> Option<String> {
        None
    }

    fn typedefs(&self) -> Option<String> {
        Some("using namespace cooperative_groups;".to_string())
    }

    fn globals(&self) -> Option<String> {
        let mut result = self.program.inline_global_cpp.join("\n");
        result.push_str("\n");
        result.push_str(&self.fp.global_decl());
        Some(result)
    }

    fn functions(&self) -> Option<String> {
        Some(
            self.program
                .structs
                .iter()
                .map(|strct| {
                    strct
                        .steps
                        .iter()
                        .map(|step| self.step_call_function_as_c(strct, step))
                })
                .flatten()
                .collect::<Vec<String>>()
                .join("\n"),
        )
    }

    fn kernels(&self) -> Option<String> {
        None
    }

    fn pre_main(&self) -> Option<String> {
        None
    }

    fn main(&self) -> Option<String> {
        None
    }

    fn post_main(&self) -> Option<String> {
        None
    }
}
